# API Reference

## `POST /chat`

Submits a research query and returns the synthesized answer with citations and scores.

### Request fields

- `query`
- `user_id`
- `session_id`
- `top_k`
- `include_web`
- `include_academic`

## `POST /research`

Returns the full orchestration result, including agent outcomes and trace metadata.

## `POST /ingest`

Ingests text, files, or URLs into the knowledge base.

## `GET /memory`

Returns session, user, and semantic memory matches.

## `GET /analytics`

Returns platform metrics such as latency, token usage, and top cited documents.

## `GET /traces`

Returns persisted execution traces, optionally filtered by `trace_id`.

## `GET /health`

Returns a lightweight service health payload.
