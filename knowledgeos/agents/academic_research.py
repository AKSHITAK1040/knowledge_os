from __future__ import annotations

from ..research.sources import AcademicSearchClient
from ..types import AgentOutcome, AgentStatus, ResearchState
from .base import BaseAgent


class AcademicResearchAgent(BaseAgent):
    name = "academic_research"

    def __init__(self, search_client: AcademicSearchClient | None = None) -> None:
        super().__init__()
        self.search_client = search_client or AcademicSearchClient()

    def _run(self, state: ResearchState) -> AgentOutcome:
        try:
            results = self.search_client.search(state.request.query, max_results=5)
        except Exception as exc:
            self.trace(state, "academic_research_failed", "Academic research failed", metadata={"error": str(exc)})
            return self.outcome(state, status=AgentStatus.FAILED, confidence=0.1, message="Academic research failed", error=str(exc))
        citations = [result.to_citation() for result in results]
        state.evidence.extend(citations)
        state.citations.extend(citations)
        self.trace(state, "academic_research_complete", "Academic research completed", metadata={"results": len(results)})
        return self.outcome(state, confidence=0.7 if citations else 0.2, message="Collected academic evidence", evidence=citations, artifacts={"results": [citation.title for citation in citations]})

