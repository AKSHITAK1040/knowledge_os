from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
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
from ..llm import DeterministicLLMClient, GroqAdapter, LLMClient, OpenAIAdapter, VoyageEmbeddingAdapter
from ..memory.store import MemoryStore
from ..retrieval.hybrid import HybridRetriever
from ..storage.sqlite_store import KnowledgeOSSQLiteStore
from ..types import AgentOutcome, AgentStatus, QueryIntent, ResearchRequest, ResearchResponse, ResearchState, TraceEvent
from ..utils import dedupe_citations


@dataclass(slots=True)
class OrchestrationResult:
    response: ResearchResponse
    outcomes: dict[str, AgentOutcome] = field(default_factory=dict)
    execution_ms: float = 0.0


class KnowledgeOSOrchestrator:
    """Enterprise Supervisor Multi-Agent Orchestrator with Dynamic Routing and Reflection."""

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
        self.llm = llm or GroqAdapter(api_key=config.groq_api_key, model=config.groq_model)
        self.embedding_client = VoyageEmbeddingAdapter(model=config.voyage_model)
        self.retriever = retriever or HybridRetriever(self.embedding_client, sink=self.store)
        self.memory_store = memory_store or MemoryStore(
            self.embedding_client,
            halflife_hours=config.memory_decay_halflife_hours,
            sink=self.store,
        )
        self.analytics = analytics or AnalyticsService(sink=self.store)

        # Hydrate stored state
        try:
            self.analytics.hydrate(self.store.load_metrics())
            self.memory_store.hydrate(self.store.load_memories())
            stored_chunks = self.store.load_chunks()
            if stored_chunks:
                self.retriever.hydrate(stored_chunks)
        except Exception:
            pass

        # Specialized Sub-Agents
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
        start_time = time.perf_counter()

        state.trace.append(
            TraceEvent(
                trace_id=trace_id,
                agent_name="supervisor",
                event_type="orchestration_start",
                message=f"Supervisor started multi-agent orchestration for query: {request.query}",
                metadata={"user_id": request.user_id, "session_id": request.session_id},
            )
        )

        outcomes: dict[str, AgentOutcome] = {}

        # 1. Planner Stage
        outcomes["planner"] = self.planner.run(state)

        # 2. Parallel Gathering Stage (Memory, Local Retrieval, Web, Academic)
        parallel_agents: list[tuple[str, Any]] = [("memory", self.memory), ("retrieval", self.retrieval)]
        if request.include_web:
            parallel_agents.append(("web_research", self.web))
        if request.include_academic:
            parallel_agents.append(("academic_research", self.academic))

        with ThreadPoolExecutor(max_workers=len(parallel_agents) or 1) as executor:
            future_to_name = {executor.submit(agent.run, state): name for name, agent in parallel_agents}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    outcome = future.result()
                    outcomes[name] = outcome
                except Exception as exc:
                    state.errors.append(f"{name}: {exc}")
                    state.trace.append(
                        TraceEvent(
                            trace_id=trace_id,
                            agent_name=name,
                            event_type="agent_error",
                            message=str(exc),
                            metadata={"agent": name},
                        )
                    )
                    outcomes[name] = AgentOutcome(agent_name=name, status=AgentStatus.FAILED, confidence=0.0, error=str(exc))

        # Deduplicate citations and evidence
        state.citations = dedupe_citations(state.citations)
        state.evidence = dedupe_citations(state.evidence)

        # 3. Synthesis Stage
        outcomes["synthesis"] = self.synthesis.run(state)

        # 4. Evaluation Stage
        outcomes["evaluation"] = self.evaluator.run(state)

        # 5. Critique & Self-Correction Reflection Loop
        needs_retry = (
            state.scores.get("confidence", 0.0) < 0.50
            or state.scores.get("hallucination_risk", 1.0) > 0.48
            or state.scores.get("grounding", 0.0) < 0.45
        )

        if needs_retry and state.retry_count < self.config.max_agent_retries:
            state.retry_count += 1
            state.trace.append(
                TraceEvent(
                    trace_id=trace_id,
                    agent_name="supervisor",
                    event_type="reflection_retry",
                    message=f"Self-correction loop triggered iteration {state.retry_count} with critique: {state.critique_notes}",
                    metadata={"retry_count": state.retry_count, "critique": state.critique_notes, "scores": state.scores},
                )
            )

            # Re-execute targeted retrieval & synthesis with critique
            outcomes["retrieval"] = self.retrieval.run(state)
            outcomes["synthesis"] = self.synthesis.run(state)
            outcomes["evaluation"] = self.evaluator.run(state)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        token_estimate = len(request.query.split()) * 12 + len(state.final_answer.split())
        cost_usd = round(token_estimate * 0.00001, 6)

        # Record Analytics Metric
        self.analytics.record(
            MetricsEvent(
                name="research",
                user_id=request.user_id,
                session_id=request.session_id,
                duration_ms=elapsed_ms,
                token_count=token_estimate,
                cost_usd=cost_usd,
                citations=state.citations,
                metadata={
                    "scores": state.scores,
                    "retry_count": state.retry_count,
                    "route": self.route(request),
                    "intent": state.intent.value,
                },
            )
        )

        state.trace.append(
            TraceEvent(
                trace_id=trace_id,
                agent_name="supervisor",
                event_type="orchestration_complete",
                message=f"Orchestration completed in {elapsed_ms}ms with confidence {state.scores.get('confidence', 0.0)}",
                duration_ms=elapsed_ms,
                metadata={"elapsed_ms": elapsed_ms, "scores": state.scores},
            )
        )

        # Persist Traces to durable SQLite store
        for event in state.trace:
            try:
                self.store.append_trace(event)
            except Exception:
                pass

        response = ResearchResponse(
            answer=state.final_answer,
            citations=state.citations,
            confidence=state.scores.get("confidence", 0.0),
            scores=state.scores,
            trace=state.trace,
            artifacts=state.artifacts,
        )

        return OrchestrationResult(response=response, outcomes=outcomes, execution_ms=elapsed_ms)
