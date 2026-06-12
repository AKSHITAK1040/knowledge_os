from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import time
from typing import Iterable

from ..llm import EmbeddingClient
from ..types import MemoryItem
from ..utils import deterministic_embedding, memory_decay_score, stable_hash, summarize_text, tokenize


class SessionMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = defaultdict(list)

    def add(self, item: MemoryItem) -> None:
        self._items[item.session_id].append(item)

    def get(self, session_id: str) -> list[MemoryItem]:
        return list(self._items.get(session_id, []))


class UserMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = defaultdict(list)

    def add(self, item: MemoryItem) -> None:
        self._items[item.user_id].append(item)

    def get(self, user_id: str) -> list[MemoryItem]:
        return list(self._items.get(user_id, []))


class SemanticMemoryStore:
    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self.embedding_client = embedding_client
        self._items: list[tuple[MemoryItem, list[float]]] = []

    def add(self, item: MemoryItem) -> None:
        text = f"{item.summary} {item.payload}"
        embedding = self.embedding_client.embed(text) if self.embedding_client else deterministic_embedding(text)
        self._items.append((item, embedding))

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        query_embedding = self.embedding_client.embed(query) if self.embedding_client else deterministic_embedding(query)
        scored = []
        for item, embedding in self._items:
            score = sum(a * b for a, b in zip(query_embedding, embedding))
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]


class MemoryStore:
    def __init__(self, embedding_client: EmbeddingClient | None = None, halflife_hours: float = 72.0, sink: object | None = None) -> None:
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

    def remember(self, *, user_id: str, session_id: str, summary: str, payload: dict[str, object] | None = None, importance: float = 0.5) -> MemoryItem:
        payload = payload or {}
        item = MemoryItem(
            id=stable_hash(f"{user_id}:{session_id}:{summary}:{len(payload)}:{time.time()}"),
            user_id=user_id,
            session_id=session_id,
            summary=summarize_text(summary, 80),
            payload=payload,
            importance=importance,
        )
        self.session.add(item)
        self.user.add(item)
        self.semantic.add(item)
        if self.sink is not None and hasattr(self.sink, "upsert_memory"):
            self.sink.upsert_memory(item)
        return item

    def recall_session(self, session_id: str) -> list[MemoryItem]:
        items = self.session.get(session_id)
        timestamp = time.time()
        for item in items:
            item.last_accessed_at = timestamp
        return items

    def recall_user(self, user_id: str) -> list[MemoryItem]:
        items = self.user.get(user_id)
        timestamp = time.time()
        for item in items:
            item.last_accessed_at = timestamp
        return items

    def recall_semantic(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        items = self.semantic.search(query, top_k=top_k)
        timestamp = time.time()
        for item in items:
            item.last_accessed_at = timestamp
        return items

    def summarize(self, items: Iterable[MemoryItem], max_items: int = 4) -> str:
        selected = list(items)[:max_items]
        if not selected:
            return ""
        lines = [f"- {item.summary}" for item in selected]
        return "\n".join(lines)

    def compress(self, user_id: str) -> list[MemoryItem]:
        items = self.user.get(user_id)
        buckets: dict[str, list[MemoryItem]] = defaultdict(list)
        for item in items:
            key_tokens = sorted(set(tokenize(item.summary)))[:5]
            key = " ".join(key_tokens)
            buckets[key].append(item)
        compressed: list[MemoryItem] = []
        for key, bucket in buckets.items():
            if len(bucket) == 1:
                compressed.append(bucket[0])
                continue
            summary = "; ".join(item.summary for item in bucket[:3])
            compressed.append(
                MemoryItem(
                    id=stable_hash(f"compressed:{user_id}:{key}:{len(bucket)}"),
                    user_id=user_id,
                    session_id=bucket[-1].session_id,
                    summary=summarize_text(summary, 80),
                    payload={"compressed_from": [item.id for item in bucket]},
                    importance=max(item.importance for item in bucket),
                )
            )
        return compressed

    def decay_scores(self, user_id: str) -> dict[str, float]:
        return {
            item.id: memory_decay_score(item, self.halflife_hours)
            for item in self.user.get(user_id)
        }
