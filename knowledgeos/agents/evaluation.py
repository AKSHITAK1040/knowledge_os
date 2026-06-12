from __future__ import annotations

from ..evaluation.evaluator import Evaluator
from ..types import AgentOutcome, AgentStatus, ResearchState
from .base import BaseAgent


class EvaluationAgent(BaseAgent):
    name = "evaluation"

    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__()
        self.evaluator = evaluator

    def _run(self, state: ResearchState) -> AgentOutcome:
        report = self.evaluator.evaluate(state.request.query, state.final_answer or state.draft, state.citations, state.retrieved_chunks)
        state.scores = {
            "relevance": report.relevance,
            "grounding": report.grounding,
            "citation_quality": report.citation_quality,
            "hallucination_risk": report.hallucination_risk,
            "confidence": report.confidence,
        }
        state.artifacts["evaluation_notes"] = report.notes
        self.trace(state, "evaluation_complete", "Evaluator scored the draft", metadata=state.scores)
        return self.outcome(
            state,
            confidence=report.confidence,
            message=report.recommendation,
            artifacts={"evaluation": state.scores, "notes": report.notes},
        )

