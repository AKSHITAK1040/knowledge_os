from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import time
import uuid


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SourceKind(str, Enum):
    INTERNAL = "internal"
    WEB = "web"
    ACADEMIC = "academic"
    MEMORY = "memory"


@dataclass(slots=True)
class Citation:
    source_id: str
    title: str
    url: str | None = None
    chunk_id: str | None = None
    excerpt: str | None = None
    score: float = 0.0
    source_kind: SourceKind = SourceKind.INTERNAL


@dataclass(slots=True)
class KnowledgeChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_kind: SourceKind = SourceKind.INTERNAL
    embedding: list[float] = field(default_factory=list)


@dataclass(slots=True)
class MemoryItem:
    id: str
    user_id: str
    session_id: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    importance: float = 0.5


@dataclass(slots=True)
class TraceEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_event_id: str | None = None
    agent_name: str = ""
    event_type: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentOutcome:
    agent_name: str
    status: AgentStatus
    confidence: float = 0.0
    message: str = ""
    evidence: list[Citation] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class ResearchRequest:
    query: str
    user_id: str
    session_id: str
    top_k: int = 8
    include_web: bool = True
    include_academic: bool = True
    require_citations: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = [asdict(c) for c in self.citations]
        data["trace"] = [asdict(t) for t in self.trace]
        return data


@dataclass(slots=True)
class ResearchState:
    request: ResearchRequest
    plan: list[str] = field(default_factory=list)
    session_memory: list[MemoryItem] = field(default_factory=list)
    user_memory: list[MemoryItem] = field(default_factory=list)
    retrieved_chunks: list[KnowledgeChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    evidence: list[Citation] = field(default_factory=list)
    draft: str = ""
    final_answer: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

