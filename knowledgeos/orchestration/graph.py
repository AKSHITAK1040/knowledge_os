from __future__ import annotations

from typing import Any

from ..types import ResearchRequest, ResearchState
from .supervisor import KnowledgeOSOrchestrator

try:  # pragma: no cover - optional dependency
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - fallback when langgraph is unavailable
    END = None
    StateGraph = None


def build_langgraph(orchestrator: KnowledgeOSOrchestrator):
    """Build a LangGraph-compatible research graph when the dependency is available.

    The custom supervisor remains the operational path, but this helper documents the
    same workflow in LangGraph terms for teams that want visual graph execution.
    """

    if StateGraph is None:
        return None

    graph = StateGraph(dict)

    def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        return state

    def terminal_node(state: dict[str, Any]) -> dict[str, Any]:
        return state

    graph.add_node("planner", planner_node)
    graph.add_node("terminal", terminal_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "terminal")
    graph.add_edge("terminal", END)
    return graph.compile()

