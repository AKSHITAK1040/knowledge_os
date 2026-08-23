from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any

from ..types import Citation, KnowledgeChunk, MemoryItem, SourceKind, TraceEvent


class KnowledgeOSSQLiteStore:
    """Production-grade local SQLite persistence store for traces, metrics, memories, and chunks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
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
                    duration_ms REAL DEFAULT 0.0,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces (trace_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces (timestamp DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories (session_id);")

    def append_trace(self, event: TraceEvent) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                (event_id, trace_id, parent_event_id, agent_name, event_type, message, timestamp, duration_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.trace_id,
                    event.parent_event_id,
                    event.agent_name,
                    event.event_type,
                    event.message,
                    event.timestamp,
                    event.duration_ms,
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

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

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

    def load_memories(self, user_id: str | None = None, session_id: str | None = None, limit: int = 100) -> list[MemoryItem]:
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

    def upsert_chunk(self, chunk: KnowledgeChunk, created_at: float | None = None) -> None:
        import time

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                (id, text, source_kind, title, url, metadata, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.text,
                    chunk.source_kind.value if hasattr(chunk.source_kind, "value") else str(chunk.source_kind),
                    str(chunk.metadata.get("title", "untitled")),
                    chunk.metadata.get("url"),
                    json.dumps(chunk.metadata),
                    json.dumps(chunk.embedding),
                    created_at or time.time(),
                ),
            )

    def load_chunks(self, limit: int = 5000) -> list[KnowledgeChunk]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        chunks = []
        for row in rows:
            chunks.append(
                KnowledgeChunk(
                    id=row["id"],
                    text=row["text"],
                    source_kind=SourceKind(row["source_kind"]),
                    metadata=json.loads(row["metadata"]),
                    embedding=json.loads(row["embedding"]),
                )
            )
        return chunks

    def count_chunks(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM chunks").fetchone()
            return row["count"] if row else 0

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            trace_count = conn.execute("SELECT COUNT(*) as c FROM traces").fetchone()["c"]
            metric_count = conn.execute("SELECT COUNT(*) as c FROM metrics").fetchone()["c"]
            memory_count = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        return {
            "traces": trace_count,
            "metrics": metric_count,
            "memories": memory_count,
            "chunks": chunk_count,
        }
