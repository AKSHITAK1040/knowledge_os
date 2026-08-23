from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research query or prompt", examples=["Explain multi-agent supervisor architecture"])
    user_id: str = Field("anonymous", description="Unique user or tenant identifier")
    session_id: str = Field("default", description="Conversation session ID")
    top_k: int = Field(8, ge=1, le=50, description="Max knowledge chunks to retrieve")
    include_web: bool = Field(True, description="Enable parallel web intelligence search")
    include_academic: bool = Field(True, description="Enable parallel academic literature search")
    intent: str | None = Field(None, description="Optional explicit query intent (e.g. factoid, deep_research, academic_survey)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary query metadata")


class IngestRequest(BaseModel):
    text: str | None = Field(None, description="Raw text or Markdown to ingest")
    path: str | None = Field(None, description="Local path to file (PDF, DOCX, TXT, MD, CSV, JSON)")
    url: str | None = Field(None, description="Web URL to fetch and ingest")
    title: str | None = Field(None, description="Document title")
    source_kind: str = Field("internal", description="Source kind: internal, web, academic, memory")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class BatchIngestRequest(BaseModel):
    documents: list[IngestRequest] = Field(..., min_length=1, description="List of documents to ingest in batch")


class MemoryQuery(BaseModel):
    user_id: str = Field(..., description="Target user ID")
    session_id: str | None = Field(None, description="Target session ID")
    query: str | None = Field(None, description="Semantic search query")
    top_k: int = Field(5, ge=1, le=20, description="Max memory items to return")


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(..., description="Target user ID")
    session_id: str = Field("default", description="Session ID")
    summary: str = Field(..., min_length=1, description="Memory factual summary")
    payload: dict[str, Any] = Field(default_factory=dict, description="Structured memory payload")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Memory importance score")
