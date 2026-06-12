from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatRequest:
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    top_k: int = 8
    include_web: bool = True
    include_academic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestRequest:
    text: str | None = None
    path: str | None = None
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryQuery:
    user_id: str
    session_id: str | None = None
    query: str | None = None

