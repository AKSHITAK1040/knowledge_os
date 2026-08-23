from __future__ import annotations

from ..llm import LLMClient
from ..types import AgentOutcome, AgentStatus, QueryIntent, ResearchState
from ..utils import tokenize
from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__()
        self.llm = llm

    def _run(self, state: ResearchState) -> AgentOutcome:
        request = state.request

        # 1. Infer Intent if not explicitly specified
        if request.intent:
            state.intent = request.intent
        else:
            raw_intent = self.llm.generate(request.query, metadata={"task": "intent"})
            try:
                state.intent = QueryIntent(raw_intent)
            except Exception:
                state.intent = QueryIntent.DEEP_RESEARCH

        # 2. Query Decomposition into sub-queries
        sub_queries = [request.query]
        tokens = tokenize(request.query)
        if len(tokens) > 6:
            sub_queries.append(" ".join(tokens[: len(tokens) // 2]))
            sub_queries.append(" ".join(tokens[len(tokens) // 2 :]))
        state.sub_queries = sub_queries

        # 3. Dynamic Plan Construction
        plan_steps = [
            f"1. Deconstruct query intent: {state.intent.value.upper()}.",
            "2. Retrieve internal knowledge base chunks and episodic user memory in parallel.",
        ]
        if request.include_web:
            plan_steps.append("3. Execute parallel web intelligence search for current facts.")
        if request.include_academic:
            plan_steps.append("4. Query academic repositories for peer-reviewed literature.")
        plan_steps.append("5. Cross-rerank all candidate citations with Reciprocal Rank Fusion (RRF).")
        plan_steps.append("6. Synthesize grounded answer with inline citations and executive summary.")
        plan_steps.append("7. Evaluate RAG Triad (Faithfulness, Context Precision, Answer Relevance).")

        state.plan = plan_steps

        self.trace(
            state,
            "plan_created",
            f"Planner formulated {len(plan_steps)}-step execution strategy for intent: {state.intent.value}",
            metadata={"intent": state.intent.value, "steps": plan_steps, "sub_queries": sub_queries},
        )

        return self.outcome(
            state,
            status=AgentStatus.SUCCEEDED,
            confidence=0.95,
            message="Plan and intent resolved",
            artifacts={"plan": plan_steps, "intent": state.intent.value, "sub_queries": sub_queries},
        )
