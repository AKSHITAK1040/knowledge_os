from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from ..analytics.service import AnalyticsService
from ..api.models import ChatRequest, IngestRequest, MemoryQuery
from ..config import settings
from ..ingestion.pipeline import IngestionPipeline
from ..orchestration.supervisor import KnowledgeOSOrchestrator
from ..types import ResearchRequest


app = FastAPI(title="KnowledgeOS API", version="0.1.0")

orchestrator = KnowledgeOSOrchestrator(config=settings)
ingestion_pipeline = IngestionPipeline(orchestrator.retriever)
analytics: AnalyticsService = orchestrator.analytics

_request_log: dict[str, deque[float]] = defaultdict(deque)


def _authenticate(request: Request) -> dict[str, str]:
    expected = os.getenv("KNOWLEDGEOS_API_KEY", "dev-key")
    provided = request.headers.get("x-api-key", "dev-key")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "role": request.headers.get("x-role", "reader"),
        "user_id": request.headers.get("x-user-id", "anonymous"),
    }


def _enforce_rate_limit(request: Request) -> None:
    identity = request.headers.get("x-user-id", request.client.host if request.client else "anonymous")
    now = time.time()
    window = 60.0
    queue = _request_log[identity]
    while queue and now - queue[0] > window:
        queue.popleft()
    if len(queue) >= settings.api_rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    queue.append(now)


@app.middleware("http")
async def middleware(request: Request, call_next):
    _enforce_rate_limit(request)
    return await call_next(request)


@app.get("/health")
def health(_: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "persisted_traces": len(orchestrator.store.list_traces(limit=1_000)),
        "persisted_metrics": len(orchestrator.store.list_metrics(limit=1_000)),
    }


@app.post("/chat")
def chat(payload: ChatRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    request = ResearchRequest(
        query=payload.query,
        user_id=payload.user_id or auth["user_id"],
        session_id=payload.session_id,
        top_k=payload.top_k,
        include_web=payload.include_web,
        include_academic=payload.include_academic,
        metadata=payload.metadata,
    )
    result = orchestrator.run(request)
    trace_id = result.response.trace[0].trace_id if result.response.trace else None
    return result.response.asdict() | {"execution_ms": result.execution_ms, "trace_id": trace_id}


@app.post("/research")
def research(payload: ChatRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    request = ResearchRequest(
        query=payload.query,
        user_id=payload.user_id or auth["user_id"],
        session_id=payload.session_id,
        top_k=payload.top_k,
        include_web=payload.include_web,
        include_academic=payload.include_academic,
        metadata=payload.metadata,
    )
    result = orchestrator.run(request)
    return {
        "response": result.response.asdict(),
        "outcomes": {name: asdict(outcome) for name, outcome in result.outcomes.items()},
        "execution_ms": result.execution_ms,
    }


@app.post("/ingest")
def ingest(payload: IngestRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    if auth["role"] not in {"admin", "writer"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    report = ingestion_pipeline.ingest_payload(asdict(payload))
    return asdict(report)


@app.get("/memory")
def memory(query: MemoryQuery = Depends(), auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    if auth["role"] not in {"admin", "writer", "reader"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    session_items = orchestrator.memory_store.recall_session(query.session_id) if query.session_id else []
    user_items = orchestrator.memory_store.recall_user(query.user_id)
    semantic_items = orchestrator.memory_store.recall_semantic(query.query, top_k=5) if query.query else []
    return {
        "session": [asdict(item) for item in session_items],
        "user": [asdict(item) for item in user_items],
        "semantic": [asdict(item) for item in semantic_items],
    }


@app.get("/analytics")
def get_analytics(auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    if auth["role"] not in {"admin", "analyst"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return analytics.summary()


@app.get("/traces")
def traces(trace_id: str | None = None, limit: int = 100, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    if auth["role"] not in {"admin", "analyst"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return {"items": orchestrator.store.list_traces(trace_id=trace_id, limit=limit)}
