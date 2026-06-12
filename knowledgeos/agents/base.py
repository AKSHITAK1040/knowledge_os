from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any

from ..types import AgentOutcome, AgentStatus, ResearchState, TraceEvent


@dataclass(slots=True)
class AgentContext:
    user_id: str
    session_id: str
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self._last_run_at = 0.0

    def trace(self, state: ResearchState, event_type: str, message: str, *, metadata: dict[str, Any] | None = None, parent_event_id: str | None = None) -> TraceEvent:
        trace_id = state.artifacts.get("trace_id") or (state.trace[0].trace_id if state.trace else TraceEvent().trace_id)
        event = TraceEvent(
            trace_id=trace_id,
            parent_event_id=parent_event_id,
            agent_name=self.name,
            event_type=event_type,
            message=message,
            metadata=metadata or {},
        )
        state.trace.append(event)
        return event

    def outcome(self, state: ResearchState, *, status: AgentStatus = AgentStatus.SUCCEEDED, confidence: float = 0.0, message: str = "", evidence=None, artifacts=None, error: str | None = None) -> AgentOutcome:
        return AgentOutcome(
            agent_name=self.name,
            status=status,
            confidence=confidence,
            message=message,
            evidence=list(evidence or []),
            artifacts=dict(artifacts or {}),
            trace=list(state.trace),
            error=error,
        )

    def run(self, state: ResearchState) -> AgentOutcome:
        self._last_run_at = time.time()
        return self._run(state)

    @abstractmethod
    def _run(self, state: ResearchState) -> AgentOutcome:
        raise NotImplementedError
