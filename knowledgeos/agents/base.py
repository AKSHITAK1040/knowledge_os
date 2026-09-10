from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any

from ..types import AgentOutcome, AgentStatus, Citation, ResearchState, TraceEvent


from threading import Lock


@dataclass(slots=True)
class AgentContext:
    user_id: str
    session_id: str
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"
    _trace_lock = Lock()

    def __init__(self) -> None:
        self._last_run_at = 0.0

    def trace(
        self,
        state: ResearchState,
        event_type: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        with self._trace_lock:
            trace_id = state.artifacts.get("trace_id") or (state.trace[0].trace_id if state.trace else TraceEvent().trace_id)
            event = TraceEvent(
                trace_id=trace_id,
                parent_event_id=parent_event_id,
                agent_name=self.name,
                event_type=event_type,
                message=message,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )
            state.trace.append(event)
        return event

    def outcome(
        self,
        state: ResearchState,
        *,
        status: AgentStatus = AgentStatus.SUCCEEDED,
        confidence: float = 0.0,
        message: str = "",
        evidence: list[Citation] | None = None,
        artifacts: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgentOutcome:
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
        start = time.perf_counter()
        self._last_run_at = time.time()
        try:
            res = self._run(state)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.trace(state, "step_complete", f"Agent '{self.name}' completed execution", duration_ms=duration_ms)
            return res
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.trace(state, "step_error", f"Agent '{self.name}' failed: {exc}", duration_ms=duration_ms, metadata={"error": str(exc)})
            state.errors.append(f"{self.name}: {exc}")
            return self.outcome(state, status=AgentStatus.FAILED, confidence=0.0, error=str(exc))

    @abstractmethod
    def _run(self, state: ResearchState) -> AgentOutcome:
        raise NotImplementedError
