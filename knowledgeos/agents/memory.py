from __future__ import annotations

from ..memory.store import MemoryStore
from ..types import AgentOutcome, AgentStatus, ResearchState
from .base import BaseAgent


class MemoryAgent(BaseAgent):
    name = "memory"

    def __init__(self, memory_store: MemoryStore) -> None:
        super().__init__()
        self.memory_store = memory_store

    def _run(self, state: ResearchState) -> AgentOutcome:
        request = state.request
        session_memory = self.memory_store.recall_session(request.session_id)
        user_memory = self.memory_store.recall_user(request.user_id)
        semantic_memory = self.memory_store.recall_semantic(request.query, top_k=5)
        state.session_memory = session_memory
        state.user_memory = user_memory
        summary = self.memory_store.summarize(semantic_memory)
        if summary:
            self.memory_store.remember(
                user_id=request.user_id,
                session_id=request.session_id,
                summary=f"Research context recalled for query: {request.query}",
                payload={"summary": summary, "query": request.query},
                importance=0.6,
            )
        state.artifacts["memory_summary"] = summary
        self.trace(state, "memory_recalled", "Memory context loaded", metadata={"session_items": len(session_memory), "user_items": len(user_memory), "semantic_items": len(semantic_memory)})
        return self.outcome(state, confidence=0.75, message="Loaded session, user, and semantic memory", artifacts={"memory_summary": summary})

