from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..types import Citation


@dataclass(slots=True)
class MetricsEvent:
    name: str
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    session_id: str = ""
    duration_ms: float = 0.0
    token_count: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)


class AnalyticsService:
    def __init__(self, sink: Any | None = None) -> None:
        self.events: list[MetricsEvent] = []
        self.sink = sink

    def record(self, event: MetricsEvent) -> None:
        self.events.append(event)
        if self.sink is not None:
            self.sink.record_metric(event)

    def summary(self) -> dict[str, Any]:
        if not self.events:
            return {
                "event_count": 0,
                "average_latency_ms": 0.0,
                "token_consumption": 0,
                "cost_usd": 0.0,
                "top_cited_documents": [],
                "active_users": 0,
            }
        latencies = [event.duration_ms for event in self.events]
        tokens = sum(event.token_count for event in self.events)
        cost = round(sum(event.cost_usd for event in self.events), 6)
        citation_counter: Counter[str] = Counter()
        active_users = {event.user_id for event in self.events if event.user_id}
        for event in self.events:
            for citation in event.citations:
                citation_counter[citation.title] += 1
        return {
            "event_count": len(self.events),
            "average_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
            "token_consumption": tokens,
            "cost_usd": cost,
            "top_cited_documents": citation_counter.most_common(10),
            "active_users": len(active_users),
        }

    def hydrate(self, events: list[MetricsEvent]) -> None:
        self.events = list(events)
