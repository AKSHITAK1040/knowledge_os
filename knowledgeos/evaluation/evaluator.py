from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..types import Citation, KnowledgeChunk
from ..utils import extract_claims, score_overlap, score_semantic_grounding, tokenize


@dataclass(slots=True)
class EvaluationReport:
    relevance: float
    grounding: float
    citation_quality: float
    context_precision: float
    hallucination_risk: float
    recommendation: str
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        # Balanced harmonic composite
        core = (self.relevance * 0.35) + (self.grounding * 0.40) + (self.citation_quality * 0.25)
        risk_penalty = 1.0 - (self.hallucination_risk * 0.5)
        return round(max(0.0, min(1.0, core * risk_penalty)), 4)


class Evaluator:
    """Enterprise RAG Triad and Hallucination Guardrail Evaluator."""

    def evaluate(
        self,
        query: str,
        answer: str,
        citations: Iterable[Citation],
        evidence: Iterable[KnowledgeChunk],
    ) -> EvaluationReport:
        citations_list = list(citations)
        evidence_list = list(evidence)
        evidence_texts = [f"{c.metadata.get('title', '')} {c.text}" for c in evidence_list]
        for cit in citations_list:
            if cit.excerpt:
                evidence_texts.append(f"{cit.title} {cit.excerpt}")

        # 1. Answer Relevance to User Query
        answer_relevance = self._answer_relevance(query, answer)

        # 2. Context Precision (how relevant is the retrieved evidence to the query)
        context_precision = self._context_precision(query, evidence_texts)

        # 3. Faithfulness / Grounding (sentence-level claim verification)
        grounding = self._grounding_score(answer, evidence_texts)

        # 4. Citation Quality & Attribution
        citation_quality = self._citation_quality(citations_list, answer)

        # 5. Hallucination Risk Index
        hallucination_risk = round(max(0.0, min(1.0, 1.0 - (grounding * 0.65 + citation_quality * 0.35))), 4)

        # Formulate actionable critique notes
        notes: list[str] = []
        if not citations_list:
            notes.append("Critical: No citations attached to the response.")
        if grounding < 0.55:
            notes.append("Grounding Alert: Claims in the answer deviate from retrieved source text.")
        if citation_quality < 0.50:
            notes.append("Attribution Alert: Citations are weakly linked to generated claims.")
        if answer_relevance < 0.45:
            notes.append("Relevance Alert: Synthesized response does not directly resolve the query intent.")

        recommendation = "retry" if (hallucination_risk > 0.48 or grounding < 0.50) else "accept"

        return EvaluationReport(
            relevance=round(answer_relevance, 4),
            grounding=round(grounding, 4),
            citation_quality=round(citation_quality, 4),
            context_precision=round(context_precision, 4),
            hallucination_risk=hallucination_risk,
            recommendation=recommendation,
            notes=notes,
        )

    def _answer_relevance(self, query: str, answer: str) -> float:
        if not answer or not query:
            return 0.0
        overlap = score_overlap(query, answer)
        query_terms = set(tokenize(query))
        answer_terms = set(tokenize(answer))
        term_coverage = len(query_terms & answer_terms) / max(1, len(query_terms))
        return min(1.0, max(0.0, 0.4 * overlap + 0.6 * term_coverage))

    def _context_precision(self, query: str, evidence_texts: Sequence[str]) -> float:
        if not evidence_texts or not query:
            return 0.0
        scores = [score_overlap(query, text) for text in evidence_texts]
        relevant = sum(1 for s in scores if s > 0.15)
        return relevant / max(1, len(evidence_texts))

    def _grounding_score(self, answer: str, evidence_texts: Sequence[str]) -> float:
        if not answer or not evidence_texts:
            return 0.0
        claims = extract_claims(answer)
        if not claims:
            return 0.0
        claim_scores = [score_semantic_grounding(claim, evidence_texts) for claim in claims]
        return sum(claim_scores) / max(1, len(claim_scores))

    def _citation_quality(self, citations: list[Citation], answer: str) -> float:
        if not citations:
            return 0.0
        answer_tokens = set(tokenize(answer))
        if not answer_tokens:
            return 0.0

        valid = 0
        for citation in citations:
            title_tokens = set(tokenize(citation.title))
            excerpt_tokens = set(tokenize(citation.excerpt or ""))
            matched = len(answer_tokens & (title_tokens | excerpt_tokens)) > 0
            if matched:
                valid += 1

        return valid / max(1, len(citations))
