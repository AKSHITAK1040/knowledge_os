from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


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


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    DEEP_RESEARCH = "deep_research"
    ACADEMIC_SURVEY = "academic_survey"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    MEMORY_RECALL = "memory_recall"
    GENERAL_SYNTHESIS = "general_synthesis"


@dataclass(slots=True)
class Citation:
    source_id: str
    title: str
    url: str | None = None
    chunk_id: str | None = None
    excerpt: str | None = None
    score: float = 0.0
    source_kind: SourceKind = SourceKind.INTERNAL
    offset_start: int | None = None
    offset_end: int | None = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KnowledgeChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_kind: SourceKind = SourceKind.INTERNAL
    embedding: list[float] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


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

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_event_id: str | None = None
    agent_name: str = ""
    event_type: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


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

    def asdict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [asdict(e) for e in self.evidence]
        data["trace"] = [asdict(t) for t in self.trace]
        return data


@dataclass(slots=True)
class ResearchRequest:
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    top_k: int = 8
    include_web: bool = True
    include_academic: bool = True
    require_citations: bool = True
    intent: QueryIntent | None = None
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
    intent: QueryIntent = QueryIntent.GENERAL_SYNTHESIS
    plan: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    session_memory: list[MemoryItem] = field(default_factory=list)
    user_memory: list[MemoryItem] = field(default_factory=list)
    retrieved_chunks: list[KnowledgeChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    evidence: list[Citation] = field(default_factory=list)
    draft: str = ""
    final_answer: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    critique_notes: list[str] = field(default_factory=list)
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
