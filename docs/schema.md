# Database Schema

This implementation is intentionally storage-agnostic in code and ready for PostgreSQL, Redis, and Milvus backing stores.

## Suggested tables

- `users`
- `sessions`
- `research_requests`
- `agent_runs`
- `citations`
- `memory_items`
- `documents`
- `chunks`
- `analytics_events`

## Suggested indices

- `research_requests(user_id, created_at)`
- `agent_runs(request_id, agent_name, created_at)`
- `memory_items(user_id, session_id, created_at)`
- `chunks(document_id, chunk_index)`

