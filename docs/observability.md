# Observability

KnowledgeOS emits structured trace events for each agent and supervisor transition.

## Tracing features

- Agent start, completion, retry, and error events
- Call graph-friendly trace IDs and parent-child relationships
- Metadata for routing, retries, and score snapshots
- Analytics events for latency, token estimates, and cost estimates
- Durable trace and metrics persistence via the local SQLite store
- API access to run history through `GET /traces`

## Suggested production sinks

- OpenTelemetry for distributed traces
- Loki or Elasticsearch for structured logs
- Prometheus for latency and error metrics
- Postgres for durable run history
