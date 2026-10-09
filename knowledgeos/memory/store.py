from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Iterable

from ..llm import EmbeddingClient
from ..types import MemoryItem
from ..utils import (
    cosine_similarity,
    deterministic_embedding,
    memory_decay_score,
    stable_hash,
    summarize_text,
    tokenize,
)


class SessionMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = defaultdict(list)

    def add(self, item: MemoryItem) -> None:
        # Avoid duplicate ids
        self._items[item.session_id] = [m for m in self._items[item.session_id] if m.id != item.id]
        self._items[item.session_id].append(item)

    def get(self, session_id: str) -> list[MemoryItem]:
        return list(self._items.get(session_id, []))

    def remove(self, memory_id: str) -> None:
        for sid in list(self._items.keys()):
            self._items[sid] = [m for m in self._items[sid] if m.id != memory_id]
            if not self._items[sid]:
                self._items.pop(sid, None)

    def clear(self, session_id: str) -> None:
        self._items.pop(session_id, None)


class UserMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = defaultdict(list)

    def add(self, item: MemoryItem) -> None:
        self._items[item.user_id] = [m for m in self._items[item.user_id] if m.id != item.id]
        self._items[item.user_id].append(item)

    def get(self, user_id: str) -> list[MemoryItem]:
        return list(self._items.get(user_id, []))

    def remove(self, memory_id: str) -> None:
        for uid in list(self._items.keys()):
            self._items[uid] = [m for m in self._items[uid] if m.id != memory_id]
            if not self._items[uid]:
                self._items.pop(uid, None)

    def clear(self, user_id: str) -> None:
        self._items.pop(user_id, None)


class SemanticMemoryStore:
    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self.embedding_client = embedding_client
        self._items: list[tuple[MemoryItem, list[float]]] = []

    def add(self, item: MemoryItem) -> None:
        text = f"{item.summary} {item.payload}"
        embedding = self.embedding_client.embed(text) if self.embedding_client else deterministic_embedding(text)
        self._items = [pair for pair in self._items if pair[0].id != item.id]
        self._items.append((item, embedding))

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        if not self._items:
            return []
        query_embedding = self.embedding_client.embed(query) if self.embedding_client else deterministic_embedding(query)
        scored = []
        for item, embedding in self._items:
            score = cosine_similarity(query_embedding, embedding)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def remove(self, memory_id: str) -> None:
        self._items = [pair for pair in self._items if pair[0].id != memory_id]


class MemoryStore:
    """Enterprise Hierarchical Memory Store with decay scoring and persistence."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        halflife_hours: float = 72.0,
        sink: Any | None = None,
    ) -> None:
        self.halflife_hours = halflife_hours
        self.session = SessionMemoryStore()
        self.user = UserMemoryStore()
        self.semantic = SemanticMemoryStore(embedding_client)
        self.sink = sink

    def hydrate(self, items: Iterable[MemoryItem]) -> None:
        for item in items:
            self.session.add(item)
            self.user.add(item)
            self.semantic.add(item)

    def remember(
        self,
        *,
        user_id: str,
        session_id: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryItem:
        payload = payload or {}
        item_id = stable_hash(f"{user_id}:{session_id}:{summary}:{time.time()}")
        item = MemoryItem(
            id=item_id,
            user_id=user_id,
            session_id=session_id,
            summary=summarize_text(summary, 80),
            payload=payload,
            importance=importance,
            created_at=time.time(),
            last_accessed_at=time.time(),
        )
        self.session.add(item)
        self.user.add(item)
        self.semantic.add(item)

        if self.sink is not None and hasattr(self.sink, "upsert_memory"):
            try:
                self.sink.upsert_memory(item)
            except Exception:
                pass

        return item

    def recall_session(self, session_id: str) -> list[MemoryItem]:
        items = self.session.get(session_id)
        now = time.time()
        for item in items:
            item.last_accessed_at = now
        return items

    def recall_user(self, user_id: str) -> list[MemoryItem]:
        items = self.user.get(user_id)
        now = time.time()
        for item in items:
            item.last_accessed_at = now
        return items

    def recall_semantic(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        items = self.semantic.search(query, top_k=top_k)
        now = time.time()
        for item in items:
            item.last_accessed_at = now
        return items

    def delete_memory(self, memory_id: str) -> bool:
        self.session.remove(memory_id)
        self.user.remove(memory_id)
        self.semantic.remove(memory_id)
        if self.sink is not None and hasattr(self.sink, "delete_memory"):
            return self.sink.delete_memory(memory_id)
        return True

    def clear_user(self, user_id: str) -> None:
        self.user.clear(user_id)

    def summarize(self, items: Iterable[MemoryItem], max_items: int = 5) -> str:
        selected = list(items)[:max_items]
        if not selected:
            return ""
        lines = [f"- [{item.session_id}] {item.summary}" for item in selected]
        return "\n".join(lines)

    def compress(self, user_id: str) -> list[MemoryItem]:
        items = self.user.get(user_id)
        if not items:
            return []
        buckets: dict[str, list[MemoryItem]] = defaultdict(list)
        for item in items:
            key_tokens = sorted(set(tokenize(item.summary)))[:4]
            key = " ".join(key_tokens)
            buckets[key].append(item)

        compressed: list[MemoryItem] = []
        for key, bucket in buckets.items():
            if len(bucket) == 1:
                compressed.append(bucket[0])
                continue
            combined_summary = "; ".join(item.summary for item in bucket[:3])
            merged_item = MemoryItem(
                id=stable_hash(f"compressed:{user_id}:{key}:{len(bucket)}"),
                user_id=user_id,
                session_id=bucket[-1].session_id,
                summary=summarize_text(combined_summary, 80),
                payload={"compressed_from": [item.id for item in bucket]},
                importance=max(item.importance for item in bucket),
            )
            compressed.append(merged_item)
            if self.sink is not None and hasattr(self.sink, "upsert_memory"):
                try:
                    self.sink.upsert_memory(merged_item)
                except Exception:
                    pass
        return compressed

    def decay_scores(self, user_id: str) -> dict[str, float]:
        return {
            item.id: memory_decay_score(item, self.halflife_hours)
            for item in self.user.get(user_id)
        }
