from __future__ import annotations

from ..retrieval.hybrid import HybridRetriever
from ..types import AgentOutcome, AgentStatus, ResearchState
from .base import BaseAgent


class RetrievalAgent(BaseAgent):
    name = "retrieval"

    def __init__(self, retriever: HybridRetriever) -> None:
        super().__init__()
        self.retriever = retriever

    def _run(self, state: ResearchState) -> AgentOutcome:
        bundle = self.retriever.search(state.request.query, top_k=state.request.top_k)
        state.retrieved_chunks = bundle.chunks
        state.citations.extend(bundle.citations)
        state.artifacts["retrieval"] = bundle.diagnostics
        self.trace(state, "retrieval_complete", "Hybrid retrieval completed", metadata=bundle.diagnostics)
        return self.outcome(state, confidence=bundle.confidence, message="Retrieved internal evidence", evidence=bundle.citations, artifacts=bundle.diagnostics)

