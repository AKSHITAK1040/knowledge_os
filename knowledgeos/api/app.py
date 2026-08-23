from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from ..analytics.service import AnalyticsService
from ..api.models import BatchIngestRequest, ChatRequest, IngestRequest, MemoryCreateRequest, MemoryQuery
from ..config import settings
from ..ingestion.pipeline import IngestionPipeline
from ..orchestration.supervisor import KnowledgeOSOrchestrator
from ..types import QueryIntent, ResearchRequest


app = FastAPI(
    title="KnowledgeOS API",
    description="Enterprise Multi-Agent Research & Knowledge Intelligence Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = KnowledgeOSOrchestrator(config=settings)
ingestion_pipeline = IngestionPipeline(orchestrator.retriever)
analytics: AnalyticsService = orchestrator.analytics

_request_log: dict[str, deque[float]] = defaultdict(deque)


def _authenticate(request: Request) -> dict[str, str]:
    expected = os.getenv("KNOWLEDGEOS_API_KEY", "dev-key")
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
    api_key = request.headers.get("x-api-key", "") or token or "dev-key"

    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key or Bearer token")
    return {
        "role": request.headers.get("x-role", "admin"),
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


@app.get("/")
@app.get("/health")
def health(_: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    stats = orchestrator.store.stats()
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "version": "1.0.0",
        "stats": stats,
        "persisted_traces": stats["traces"],
        "persisted_metrics": stats["metrics"],
    }


@app.post("/chat")
@app.post("/v1/chat")
def chat(payload: ChatRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    intent_enum = None
    if payload.intent:
        try:
            intent_enum = QueryIntent(payload.intent)
        except Exception:
            pass

    request = ResearchRequest(
        query=payload.query,
        user_id=payload.user_id if payload.user_id != "anonymous" else auth["user_id"],
        session_id=payload.session_id,
        top_k=payload.top_k,
        include_web=payload.include_web,
        include_academic=payload.include_academic,
        intent=intent_enum,
        metadata=payload.metadata,
    )
    result = orchestrator.run(request)
    trace_id = result.response.trace[0].trace_id if result.response.trace else None
    return result.response.asdict() | {
        "execution_ms": result.execution_ms,
        "trace_id": trace_id,
        "plan": result.response.artifacts.get("plan", []),
    }


@app.post("/research")
@app.post("/v1/research")
def research(payload: ChatRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    intent_enum = None
    if payload.intent:
        try:
            intent_enum = QueryIntent(payload.intent)
        except Exception:
            pass

    request = ResearchRequest(
        query=payload.query,
        user_id=payload.user_id if payload.user_id != "anonymous" else auth["user_id"],
        session_id=payload.session_id,
        top_k=payload.top_k,
        include_web=payload.include_web,
        include_academic=payload.include_academic,
        intent=intent_enum,
        metadata=payload.metadata,
    )
    result = orchestrator.run(request)
    return {
        "response": result.response.asdict(),
        "outcomes": {name: outcome.asdict() for name, outcome in result.outcomes.items()},
        "execution_ms": result.execution_ms,
        "trace_id": result.response.trace[0].trace_id if result.response.trace else None,
    }


@app.post("/ingest")
@app.post("/v1/ingest")
def ingest(payload: IngestRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    if auth["role"] not in {"admin", "writer"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    try:
        report = ingestion_pipeline.ingest_payload(payload.model_dump())
        return asdict(report)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/v1/ingest/batch")
def ingest_batch(payload: BatchIngestRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    if auth["role"] not in {"admin", "writer"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    reports = ingestion_pipeline.ingest_batch([doc.model_dump() for doc in payload.documents])
    return {
        "total_documents": len(reports),
        "total_chunks": sum(r.chunks_created for r in reports),
        "reports": [asdict(r) for r in reports],
    }


@app.get("/memory")
@app.get("/v1/memory")
def get_memory(
    user_id: str = Query(..., description="Target user ID"),
    session_id: str | None = Query(None, description="Optional session ID"),
    query: str | None = Query(None, description="Optional semantic search query"),
    top_k: int = Query(5, description="Max semantic results"),
    auth: dict[str, str] = Depends(_authenticate),
) -> dict[str, Any]:
    session_items = orchestrator.memory_store.recall_session(session_id) if session_id else []
    user_items = orchestrator.memory_store.recall_user(user_id)
    semantic_items = orchestrator.memory_store.recall_semantic(query, top_k=top_k) if query else []
    decay_scores = orchestrator.memory_store.decay_scores(user_id)

    return {
        "session": [asdict(item) for item in session_items],
        "user": [asdict(item) for item in user_items],
        "semantic": [asdict(item) for item in semantic_items],
        "decay_scores": decay_scores,
    }


@app.post("/memory")
@app.post("/v1/memory")
def create_memory(payload: MemoryCreateRequest, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    item = orchestrator.memory_store.remember(
        user_id=payload.user_id,
        session_id=payload.session_id,
        summary=payload.summary,
        payload=payload.payload,
        importance=payload.importance,
    )
    return asdict(item)


@app.delete("/memory/{memory_id}")
@app.delete("/v1/memory/{memory_id}")
def delete_memory(memory_id: str, auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    success = orchestrator.memory_store.delete_memory(memory_id)
    return {"deleted": success, "memory_id": memory_id}


@app.get("/analytics")
@app.get("/v1/analytics")
def get_analytics(auth: dict[str, str] = Depends(_authenticate)) -> dict[str, Any]:
    return analytics.summary()


@app.get("/traces")
@app.get("/v1/traces")
def get_traces(
    trace_id: str | None = Query(None, description="Optional trace ID filter"),
    limit: int = Query(100, ge=1, le=1000, description="Max traces to retrieve"),
    auth: dict[str, str] = Depends(_authenticate),
) -> dict[str, Any]:
    return {"items": orchestrator.store.list_traces(trace_id=trace_id, limit=limit)}


@app.get("/v1/chunks")
def get_chunks(
    limit: int = Query(100, ge=1, le=1000, description="Max chunks to retrieve"),
    auth: dict[str, str] = Depends(_authenticate),
) -> dict[str, Any]:
    chunks = orchestrator.store.load_chunks(limit=limit)
    return {
        "total_persisted": orchestrator.store.count_chunks(),
        "items": [c.asdict() for c in chunks],
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus OpenMetrics telemetry endpoint for MLOps observability."""
    from fastapi.responses import PlainTextResponse

    summary = analytics.summary()
    stats = orchestrator.store.stats()

    lines = [
        "# HELP knowledgeos_requests_total Total multi-agent research requests processed",
        "# TYPE knowledgeos_requests_total counter",
        f"knowledgeos_requests_total {summary.get('event_count', 0)}",
        "# HELP knowledgeos_average_latency_ms Average multi-agent execution duration in ms",
        "# TYPE knowledgeos_average_latency_ms gauge",
        f"knowledgeos_average_latency_ms {summary.get('average_latency_ms', 0.0)}",
        "# HELP knowledgeos_p95_latency_ms P95 multi-agent execution duration in ms",
        "# TYPE knowledgeos_p95_latency_ms gauge",
        f"knowledgeos_p95_latency_ms {summary.get('p95_latency_ms', 0.0)}",
        "# HELP knowledgeos_token_consumption_total Total LLM tokens consumed",
        "# TYPE knowledgeos_token_consumption_total counter",
        f"knowledgeos_token_consumption_total {summary.get('token_consumption', 0)}",
        "# HELP knowledgeos_cost_usd_total Estimated total LLM cost in USD",
        "# TYPE knowledgeos_cost_usd_total counter",
        f"knowledgeos_cost_usd_total {summary.get('cost_usd', 0.0)}",
        "# HELP knowledgeos_persisted_chunks_count Total knowledge chunks indexed",
        "# TYPE knowledgeos_persisted_chunks_count gauge",
        f"knowledgeos_persisted_chunks_count {stats.get('chunks', 0)}",
        "# HELP knowledgeos_persisted_memories_count Total episodic memories stored",
        "# TYPE knowledgeos_persisted_memories_count gauge",
        f"knowledgeos_persisted_memories_count {stats.get('memories', 0)}",
        "# HELP knowledgeos_active_users_count Active research users",
        "# TYPE knowledgeos_active_users_count gauge",
        f"knowledgeos_active_users_count {summary.get('active_users', 0)}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

