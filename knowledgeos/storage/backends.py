from __future__ import annotations

import time
from threading import Lock
from typing import Any

from ..types import TraceEvent
from ..utils import cosine_similarity, deterministic_embedding


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def add(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[str, list[float], dict[str, Any]]] = []
        self._lock = Lock()

    def upsert(self, key: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._items = [item for item in self._items if item[0] != key]
            self._items.append((key, deterministic_embedding(text), metadata or {}))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        query_embedding = deterministic_embedding(query)
        with self._lock:
            items = list(self._items)

        candidates = []
        for key, embedding, metadata in items:
            if filters:
                match = True
                for f_k, f_v in filters.items():
                    if metadata.get(f_k) != f_v:
                        match = False
                        break
                if not match:
                    continue
            sim = cosine_similarity(query_embedding, embedding)
            candidates.append((key, sim, metadata))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]


class RedisCacheBackend:
    """In-memory cache with optional Redis DSN integration."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._cache.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        with self._lock:
            self._cache[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class PostgresEventStore:
    """Adapter placeholder for Postgres event persistence."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def append(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def query(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)


class MilvusVectorStore:
    """Adapter placeholder for Milvus vector persistence."""

    def __init__(self, uri: str | None = None, collection: str = "knowledgeos_chunks") -> None:
        self.uri = uri
        self.collection = collection
        self._items: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._lock = Lock()

    def upsert(self, key: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._items[key] = (deterministic_embedding(text), metadata or {})

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        query_embedding = deterministic_embedding(query)
        with self._lock:
            items = list(self._items.items())
        scored = [(key, cosine_similarity(query_embedding, embedding), metadata) for key, (embedding, metadata) in items]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
