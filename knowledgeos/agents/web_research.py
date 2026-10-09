from __future__ import annotations

from ..research.sources import DuckDuckGoSearchClient
from ..types import AgentOutcome, AgentStatus, ResearchState
from .base import BaseAgent


class WebResearchAgent(BaseAgent):
    name = "web_research"

    def __init__(self, search_client: DuckDuckGoSearchClient | None = None) -> None:
        super().__init__()
        self.search_client = search_client or DuckDuckGoSearchClient()

    def _run(self, state: ResearchState) -> AgentOutcome:
        try:
            results = self.search_client.search(state.request.query, max_results=5)
        except Exception as exc:
            self.trace(state, "web_research_failed", "Web research failed", metadata={"error": str(exc)})
            return self.outcome(state, status=AgentStatus.FAILED, confidence=0.1, message="Web research failed", error=str(exc))
        citations = [result.to_citation(score=round(max(0.60, 0.85 - idx * 0.03), 2)) for idx, result in enumerate(results)]
        self.trace(state, "web_research_complete", f"Web research completed with {len(citations)} citations", metadata={"results": len(results)})
        return self.outcome(
            state,
            confidence=0.75 if citations else 0.2,
            message="Collected web evidence",
            evidence=citations,
            artifacts={"results": [citation.title for citation in citations]},
        )

