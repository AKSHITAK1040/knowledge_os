from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from ..types import MemoryItem


class KnowledgeOSSQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_event_id TEXT,
                    agent_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    duration_ms REAL NOT NULL,
                    token_count INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    citations TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    importance REAL NOT NULL
                )
                """
            )

    def append_trace(self, event: Any) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                (event_id, trace_id, parent_event_id, agent_name, event_type, message, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.trace_id,
                    event.parent_event_id,
                    event.agent_name,
                    event.event_type,
                    event.message,
                    event.timestamp,
                    json.dumps(event.metadata),
                ),
            )

    def list_traces(self, trace_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM traces"
        params: tuple[Any, ...] = ()
        if trace_id:
            query += " WHERE trace_id = ?"
            params = (trace_id,)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params = params + (limit,)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata"])} for row in rows]

    def record_metric(self, event: Any) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics
                (name, timestamp, user_id, session_id, duration_ms, token_count, cost_usd, metadata, citations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.name,
                    event.timestamp,
                    event.user_id,
                    event.session_id,
                    event.duration_ms,
                    event.token_count,
                    event.cost_usd,
                    json.dumps(event.metadata),
                    json.dumps([asdict(citation) for citation in event.citations]),
                ),
            )

    def list_metrics(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata"]), "citations": json.loads(row["citations"])} for row in rows]

    def load_metrics(self, limit: int = 1000) -> list[Any]:
        from ..analytics.service import MetricsEvent
        from ..types import Citation, SourceKind

        events = []
        for row in self.list_metrics(limit=limit):
            citations = [
                Citation(
                    source_id=item.get("source_id", ""),
                    title=item.get("title", ""),
                    url=item.get("url"),
                    chunk_id=item.get("chunk_id"),
                    excerpt=item.get("excerpt"),
                    score=float(item.get("score", 0.0)),
                    source_kind=SourceKind(item.get("source_kind", SourceKind.INTERNAL.value)),
                )
                for item in row["citations"]
            ]
            events.append(
                MetricsEvent(
                    name=row["name"],
                    timestamp=row["timestamp"],
                    user_id=row["user_id"] or "",
                    session_id=row["session_id"] or "",
                    duration_ms=row["duration_ms"],
                    token_count=row["token_count"],
                    cost_usd=row["cost_usd"],
                    metadata=row["metadata"],
                    citations=citations,
                )
            )
        return events

    def upsert_memory(self, item: MemoryItem) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, user_id, session_id, summary, payload, created_at, last_accessed_at, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.user_id,
                    item.session_id,
                    item.summary,
                    json.dumps(item.payload),
                    item.created_at,
                    item.last_accessed_at,
                    item.importance,
                ),
            )

    def list_memories(self, user_id: str | None = None, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM memories"
        filters: list[str] = []
        params: list[Any] = []
        if user_id:
            filters.append("user_id = ?")
            params.append(user_id)
        if session_id:
            filters.append("session_id = ?")
            params.append(session_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY last_accessed_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]

    def load_memories(self, user_id: str | None = None, session_id: str | None = None, limit: int = 100) -> list[Any]:
        from ..types import MemoryItem

        return [
            MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                summary=row["summary"],
                payload=row["payload"],
                created_at=row["created_at"],
                last_accessed_at=row["last_accessed_at"],
                importance=row["importance"],
            )
            for row in self.list_memories(user_id=user_id, session_id=session_id, limit=limit)
        ]
