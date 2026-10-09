from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

from knowledgeos.evaluation.evaluator import Evaluator
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.types import Citation, ResearchRequest
from knowledgeos.utils import tokenize


@dataclass(slots=True)
class BenchmarkCase:
    query: str
    expected_terms: list[str]
    expected_citations: list[str] = field(default_factory=list)
    user_id: str = "benchmark-user"
    session_id: str = "benchmark-session"
    include_web: bool = False
    include_academic: bool = False


@dataclass(slots=True)
class BenchmarkResult:
    query: str
    retrieval_precision: float
    citation_accuracy: float
    grounding: float
    context_precision: float
    answer_relevance: float
    latency_ms: float
    cost_usd: float
    confidence: float


class BenchmarkSuite:
    """Standardized Benchmark Suite for evaluating RAG, Orchestration, and Grounding."""

    def __init__(self, orchestrator: KnowledgeOSOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.evaluator = Evaluator()

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        start = time.perf_counter()
        result = self.orchestrator.run(
            ResearchRequest(
                query=case.query,
                user_id=case.user_id,
                session_id=case.session_id,
                include_web=case.include_web,
                include_academic=case.include_academic,
            )
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        retrieval_precision = self._retrieval_precision(case.expected_terms, result.response.citations)
        citation_accuracy = self._citation_accuracy(case.expected_citations, result.response.citations)

        scores = result.response.scores
        token_count = len(case.query.split()) * 12 + len(result.response.answer.split())
        cost_usd = round(token_count * 0.00001, 6)

        return BenchmarkResult(
            query=case.query,
            retrieval_precision=round(retrieval_precision, 4),
            citation_accuracy=round(citation_accuracy, 4),
            grounding=round(scores.get("grounding", 0.0), 4),
            context_precision=round(scores.get("context_precision", 0.0), 4),
            answer_relevance=round(scores.get("relevance", 0.0), 4),
            latency_ms=elapsed_ms,
            cost_usd=cost_usd,
            confidence=round(result.response.confidence, 4),
        )

    def run(self, cases: Iterable[BenchmarkCase]) -> dict[str, object]:
        results = [self.run_case(case) for case in cases]
        return {
            "cases": [asdict(result) for result in results],
            "summary": {
                "retrieval_precision": round(statistics.mean(result.retrieval_precision for result in results), 4) if results else 0.0,
                "citation_accuracy": round(statistics.mean(result.citation_accuracy for result in results), 4) if results else 0.0,
                "grounding": round(statistics.mean(result.grounding for result in results), 4) if results else 0.0,
                "context_precision": round(statistics.mean(result.context_precision for result in results), 4) if results else 0.0,
                "answer_relevance": round(statistics.mean(result.answer_relevance for result in results), 4) if results else 0.0,
                "latency_ms": round(statistics.mean(result.latency_ms for result in results), 2) if results else 0.0,
                "cost_usd": round(sum(result.cost_usd for result in results), 6),
                "confidence": round(statistics.mean(result.confidence for result in results), 4) if results else 0.0,
            },
        }

    def _retrieval_precision(self, expected_terms: list[str], citations: list[Citation]) -> float:
        if not expected_terms or not citations:
            return 0.0
        expected = set(term.lower() for term in expected_terms)
        found_terms = set()
        for citation in citations:
            found_terms.update(tokenize(citation.title))
            found_terms.update(tokenize(citation.excerpt or ""))
        return len(expected & found_terms) / len(expected)

    def _citation_accuracy(self, expected_citations: list[str], citations: list[Citation]) -> float:
        if not expected_citations or not citations:
            return 0.0
        expected = set(term.lower() for term in expected_citations)
        found = {citation.title.lower() for citation in citations}
        return len(expected & found) / len(expected)
