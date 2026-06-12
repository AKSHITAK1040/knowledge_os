from __future__ import annotations

from ..llm import LLMClient
from ..types import AgentOutcome, AgentStatus, ResearchState
from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__()
        self.llm = llm

    def _run(self, state: ResearchState) -> AgentOutcome:
        request = state.request
        plan = [
            "Clarify the research objective and extract key entities.",
            "Retrieve internal knowledge and user memory.",
        ]
        if request.include_web:
            plan.append("Parallelize web research for current evidence.")
        if request.include_academic:
            plan.append("Parallelize academic research for peer-reviewed evidence.")
        plan.append("Synthesize with citations and evaluate grounding.")
        state.plan = plan
        self.trace(state, "plan_created", "Planner created multi-step research plan", metadata={"steps": plan})
        return self.outcome(state, status=AgentStatus.SUCCEEDED, confidence=0.9, message="Plan generated", artifacts={"plan": plan})

