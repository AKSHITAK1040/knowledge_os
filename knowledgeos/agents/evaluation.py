from __future__ import annotations

from ..evaluation.evaluator import Evaluator
from ..types import AgentOutcome, ResearchState
from .base import BaseAgent


class EvaluationAgent(BaseAgent):
    name = "evaluation"

    def __init__(self, evaluator: Evaluator | None = None) -> None:
        super().__init__()
        self.evaluator = evaluator or Evaluator()

    def _run(self, state: ResearchState) -> AgentOutcome:
        report = self.evaluator.evaluate(
            state.request.query,
            state.final_answer or state.draft,
            state.citations,
            state.retrieved_chunks,
        )
        state.scores = {
            "relevance": report.relevance,
            "grounding": report.grounding,
            "citation_quality": report.citation_quality,
            "context_precision": report.context_precision,
            "hallucination_risk": report.hallucination_risk,
            "confidence": report.confidence,
        }
        state.critique_notes = report.notes
        state.artifacts["evaluation_notes"] = report.notes

        self.trace(
            state,
            "evaluation_complete",
            f"Evaluator scored response with recommendation: {report.recommendation}",
            metadata=state.scores,
        )

        return self.outcome(
            state,
            confidence=report.confidence,
            message=report.recommendation,
            artifacts={"evaluation": state.scores, "notes": report.notes},
        )
