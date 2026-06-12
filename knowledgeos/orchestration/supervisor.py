from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import time
from typing import Any
import uuid

from ..agents.academic_research import AcademicResearchAgent
from ..agents.evaluation import EvaluationAgent
from ..agents.memory import MemoryAgent
from ..agents.planner import PlannerAgent
from ..agents.retrieval import RetrievalAgent
from ..agents.synthesis import SynthesisAgent
from ..agents.web_research import WebResearchAgent
from ..analytics.service import AnalyticsService, MetricsEvent
from ..config import Settings, settings
from ..evaluation.evaluator import Evaluator
from ..llm import DeterministicLLMClient, LLMClient, OpenAIAdapter, VoyageEmbeddingAdapter
from ..memory.store import MemoryStore
from ..retrieval.hybrid import HybridRetriever
from ..storage.sqlite_store import KnowledgeOSSQLiteStore
from ..types import AgentOutcome, AgentStatus, ResearchRequest, ResearchResponse, ResearchState, TraceEvent


@dataclass(slots=True)
class OrchestrationResult:
    response: ResearchResponse
    outcomes: dict[str, AgentOutcome] = field(default_factory=dict)
    execution_ms: float = 0.0


class KnowledgeOSOrchestrator:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        retriever: HybridRetriever | None = None,
        memory_store: MemoryStore | None = None,
        analytics: AnalyticsService | None = None,
        store: KnowledgeOSSQLiteStore | None = None,
        config: Settings = settings,
    ) -> None:
        self.config = config
        self.store = store or KnowledgeOSSQLiteStore(config.local_state_path)
        self.llm = llm or OpenAIAdapter(model=config.openai_model)
        self.embedding_client = VoyageEmbeddingAdapter(model=config.voyage_model)
        self.retriever = retriever or HybridRetriever(self.embedding_client)
        self.memory_store = memory_store or MemoryStore(self.embedding_client, halflife_hours=config.memory_decay_halflife_hours, sink=self.store)
        self.analytics = analytics or AnalyticsService(sink=self.store)
        self.analytics.hydrate(self.store.load_metrics())
        self.memory_store.hydrate(self.store.load_memories())
        self.planner = PlannerAgent(self.llm)
        self.retrieval = RetrievalAgent(self.retriever)
        self.web = WebResearchAgent()
        self.academic = AcademicResearchAgent()
        self.memory = MemoryAgent(self.memory_store)
        self.evaluator = EvaluationAgent(Evaluator())
        self.synthesis = SynthesisAgent(self.llm)

    def route(self, request: ResearchRequest) -> list[str]:
        route = ["planner", "memory", "retrieval", "synthesis", "evaluation"]
        if request.include_web:
            route.insert(3, "web_research")
        if request.include_academic:
            route.insert(4 if request.include_web else 3, "academic_research")
        return route

    def run(self, request: ResearchRequest) -> OrchestrationResult:
        state = ResearchState(request=request)
        trace_id = str(uuid.uuid4())
        state.artifacts["trace_id"] = trace_id
        start = time.perf_counter()
        state.trace.append(TraceEvent(trace_id=trace_id, agent_name="supervisor", event_type="start", message="Orchestration started"))

        outcomes: dict[str, AgentOutcome] = {}
        outcomes["planner"] = self.planner.run(state)

        parallel_agents = [("memory", self.memory), ("retrieval", self.retrieval)]
        if request.include_web:
            parallel_agents.append(("web_research", self.web))
        if request.include_academic:
            parallel_agents.append(("academic_research", self.academic))

        with ThreadPoolExecutor(max_workers=len(parallel_agents) or 1) as executor:
            future_map = {executor.submit(agent.run, state): name for name, agent in parallel_agents}
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    outcomes[name] = future.result()
                except Exception as exc:
                    state.errors.append(f"{name}: {exc}")
                    state.trace.append(TraceEvent(trace_id=trace_id, agent_name=name, event_type="error", message=str(exc), metadata={"agent": name}))
                    outcomes[name] = AgentOutcome(agent_name=name, status=AgentStatus.FAILED, confidence=0.0, error=str(exc))

        state.citations = list({(c.source_id, c.chunk_id): c for c in state.citations}.values())

        outcomes["synthesis"] = self.synthesis.run(state)
        outcomes["evaluation"] = self.evaluator.run(state)

        low_confidence = state.scores.get("confidence", 0.0) < 0.55 or state.scores.get("hallucination_risk", 1.0) > 0.45
        if low_confidence and state.retry_count < self.config.max_agent_retries:
            state.retry_count += 1
            state.trace.append(
                TraceEvent(
                    trace_id=trace_id,
                    agent_name="supervisor",
                    event_type="retry",
                    message="Evaluation triggered retry",
                    metadata={"retry_count": state.retry_count, "scores": state.scores},
                )
            )
            for name, agent in parallel_agents:
                try:
                    outcomes[name] = agent.run(state)
                except Exception as exc:
                    state.errors.append(f"{name}: {exc}")
                    outcomes[name] = AgentOutcome(agent_name=name, status=AgentStatus.FAILED, confidence=0.0, error=str(exc))
            outcomes["synthesis"] = self.synthesis.run(state)
            outcomes["evaluation"] = self.evaluator.run(state)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        token_estimate = len(request.query.split()) * 10 + len(state.final_answer.split())
        self.analytics.record(
            MetricsEvent(
                name="research",
                user_id=request.user_id,
                session_id=request.session_id,
                duration_ms=elapsed_ms,
                token_count=token_estimate,
                cost_usd=round(token_estimate * 0.00001, 6),
                citations=state.citations,
                metadata={"scores": state.scores, "retry_count": state.retry_count, "route": self.route(request)},
            )
        )
        state.trace.append(TraceEvent(trace_id=trace_id, agent_name="supervisor", event_type="complete", message="Orchestration finished", metadata={"elapsed_ms": elapsed_ms}))
        response = ResearchResponse(
            answer=state.final_answer,
            citations=state.citations,
            confidence=state.scores.get("confidence", 0.0),
            scores=state.scores,
            trace=state.trace,
            artifacts=state.artifacts,
        )
        for event in state.trace:
            self.store.append_trace(event)
        return OrchestrationResult(response=response, outcomes=outcomes, execution_ms=elapsed_ms)
