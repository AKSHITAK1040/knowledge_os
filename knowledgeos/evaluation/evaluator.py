from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..types import Citation, KnowledgeChunk
from ..utils import score_overlap, tokenize


@dataclass(slots=True)
class EvaluationReport:
    relevance: float
    grounding: float
    citation_quality: float
    hallucination_risk: float
    recommendation: str
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        base = (self.relevance + self.grounding + self.citation_quality) / 3
        return max(0.0, min(1.0, base * (1.0 - self.hallucination_risk)))


class Evaluator:
    def evaluate(self, query: str, answer: str, citations: Iterable[Citation], evidence: Iterable[KnowledgeChunk]) -> EvaluationReport:
        citations = list(citations)
        evidence = list(evidence)
        relevance = max(0.0, min(1.0, score_overlap(query, answer)))
        grounding = self._grounding_score(answer, evidence)
        citation_quality = self._citation_quality(citations, answer)
        hallucination_risk = max(0.0, 1.0 - ((grounding + citation_quality) / 2))
        notes = []
        if not citations:
            notes.append("No citations attached.")
        if grounding < 0.5:
            notes.append("Answer is weakly grounded in retrieved evidence.")
        if citation_quality < 0.5:
            notes.append("Citation support is thin or poorly attributed.")
        recommendation = "retry" if max(1.0 - relevance, hallucination_risk) > 0.45 else "accept"
        return EvaluationReport(
            relevance=round(relevance, 4),
            grounding=round(grounding, 4),
            citation_quality=round(citation_quality, 4),
            hallucination_risk=round(hallucination_risk, 4),
            recommendation=recommendation,
            notes=notes,
        )

    def _grounding_score(self, answer: str, evidence: list[KnowledgeChunk]) -> float:
        if not answer or not evidence:
            return 0.0
        best = 0.0
        for chunk in evidence:
            best = max(best, score_overlap(answer, chunk.text))
        return max(0.0, min(1.0, best))

    def _citation_quality(self, citations: list[Citation], answer: str) -> float:
        if not citations:
            return 0.0
        answer_tokens = set(tokenize(answer))
        if not answer_tokens:
            return 0.0
        supported = 0
        for citation in citations:
            excerpt_tokens = set(tokenize(citation.excerpt or ""))
            supported += 1 if len(answer_tokens & excerpt_tokens) > 0 else 0
        return supported / len(citations)

