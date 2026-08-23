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

        all_recalled = session_memory + user_memory + semantic_memory
        summary = self.memory_store.summarize(all_recalled)
        state.artifacts["memory_summary"] = summary

        # Remember query context
        self.memory_store.remember(
            user_id=request.user_id,
            session_id=request.session_id,
            summary=f"Researched topic: {request.query}",
            payload={"query": request.query, "intent": state.intent.value},
            importance=0.65,
        )

        self.trace(
            state,
            "memory_recalled",
            f"Loaded episodic memory: {len(session_memory)} session, {len(user_memory)} user, {len(semantic_memory)} semantic items",
            metadata={
                "session_count": len(session_memory),
                "user_count": len(user_memory),
                "semantic_count": len(semantic_memory),
            },
        )

        return self.outcome(
            state,
            confidence=0.80,
            message="Episodic memory synchronized",
            artifacts={"memory_summary": summary},
        )
