from __future__ import annotations

from ..retrieval.hybrid import HybridRetriever
from ..types import AgentOutcome, AgentStatus, ResearchState
from ..utils import dedupe_citations
from .base import BaseAgent


class RetrievalAgent(BaseAgent):
    name = "retrieval"

    def __init__(self, retriever: HybridRetriever) -> None:
        super().__init__()
        self.retriever = retriever

    def _run(self, state: ResearchState) -> AgentOutcome:
        top_k = state.request.top_k or 8
        queries_to_run = state.sub_queries or [state.request.query]

        collected_chunks = list(state.retrieved_chunks)
        collected_citations = list(state.citations)
        diagnostics = {}

        for q in queries_to_run[:3]:
            bundle = self.retriever.search(q, top_k=top_k)
            collected_chunks.extend(bundle.chunks)
            collected_citations.extend(bundle.citations)
            diagnostics.update(bundle.diagnostics)

        # Deduplicate chunks by id
        seen_chunk_ids: set[str] = set()
        deduped_chunks = []
        for chunk in collected_chunks:
            if chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.id)
                deduped_chunks.append(chunk)

        state.retrieved_chunks = deduped_chunks
        state.citations = dedupe_citations(collected_citations)
        state.evidence.extend(state.citations)
        state.artifacts["retrieval"] = diagnostics

        self.trace(
            state,
            "retrieval_complete",
            f"Hybrid retrieval fetched {len(deduped_chunks)} chunks across {len(queries_to_run)} sub-queries",
            metadata=diagnostics,
        )

        return self.outcome(
            state,
            confidence=0.85 if deduped_chunks else 0.3,
            message=f"Retrieved {len(deduped_chunks)} internal evidence chunks",
            evidence=state.citations,
            artifacts=diagnostics,
        )
