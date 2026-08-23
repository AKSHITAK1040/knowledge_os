from __future__ import annotations

from typing import Any, TypedDict

from ..types import Citation, ResearchRequest, ResearchState, SourceKind
from .supervisor import KnowledgeOSOrchestrator

try:  # pragma: no cover
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = None
    StateGraph = None


class GraphState(TypedDict):
    query: str
    user_id: str
    session_id: str
    include_web: bool
    include_academic: bool
    plan: list[str]
    citations: list[dict[str, Any]]
    answer: str
    scores: dict[str, float]
    retry_count: int


def build_langgraph(orchestrator: KnowledgeOSOrchestrator):
    """Build an executable LangGraph-compatible research StateGraph.
    
    Provides a visual, compiled LangGraph execution graph for teams integrating
    with LangGraph Studio or LangSmith.
    """
    if StateGraph is None:
        return None

    graph = StateGraph(GraphState)

    def planner_node(state: GraphState) -> dict[str, Any]:
        req = ResearchRequest(
            query=state["query"],
            user_id=state.get("user_id", "anonymous"),
            session_id=state.get("session_id", "default"),
            include_web=state.get("include_web", True),
            include_academic=state.get("include_academic", True),
        )
        res_state = ResearchState(request=req)
        orchestrator.planner.run(res_state)
        return {"plan": res_state.plan}

    def research_node(state: GraphState) -> dict[str, Any]:
        req = ResearchRequest(
            query=state["query"],
            user_id=state.get("user_id", "anonymous"),
            session_id=state.get("session_id", "default"),
            include_web=state.get("include_web", True),
            include_academic=state.get("include_academic", True),
        )
        res_state = ResearchState(request=req, plan=state.get("plan", []))
        orchestrator.memory.run(res_state)
        orchestrator.retrieval.run(res_state)
        if req.include_web:
            orchestrator.web.run(res_state)
        if req.include_academic:
            orchestrator.academic.run(res_state)
        return {"citations": [c.asdict() for c in res_state.citations]}

    def synthesis_node(state: GraphState) -> dict[str, Any]:
        req = ResearchRequest(
            query=state["query"],
            user_id=state.get("user_id", "anonymous"),
            session_id=state.get("session_id", "default"),
        )
        res_state = ResearchState(request=req, plan=state.get("plan", []))
        for c_dict in state.get("citations", []):
            res_state.citations.append(
                Citation(
                    source_id=c_dict.get("source_id", "s1"),
                    title=c_dict.get("title", "source"),
                    url=c_dict.get("url"),
                    excerpt=c_dict.get("excerpt"),
                    score=float(c_dict.get("score", 0.0)),
                    source_kind=SourceKind(c_dict.get("source_kind", "internal")),
                )
            )
        orchestrator.synthesis.run(res_state)
        return {"answer": res_state.final_answer}

    def evaluation_node(state: GraphState) -> dict[str, Any]:
        req = ResearchRequest(
            query=state["query"],
            user_id=state.get("user_id", "anonymous"),
            session_id=state.get("session_id", "default"),
        )
        res_state = ResearchState(
            request=req,
            final_answer=state.get("answer", ""),
            draft=state.get("answer", ""),
        )
        for c_dict in state.get("citations", []):
            res_state.citations.append(
                Citation(
                    source_id=c_dict.get("source_id", "s1"),
                    title=c_dict.get("title", "source"),
                    url=c_dict.get("url"),
                    excerpt=c_dict.get("excerpt"),
                    score=float(c_dict.get("score", 0.0)),
                    source_kind=SourceKind(c_dict.get("source_kind", "internal")),
                )
            )
        orchestrator.evaluator.run(res_state)
        current_retries = state.get("retry_count", 0)
        return {
            "scores": res_state.scores,
            "retry_count": current_retries + 1,
        }

    def should_retry(state: GraphState) -> str:
        scores = state.get("scores", {})
        if scores.get("confidence", 0.0) < 0.45 and state.get("retry_count", 0) <= 1:
            return "retry"
        return "accept"

    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("evaluator", evaluation_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", "evaluator")
    graph.add_conditional_edges("evaluator", should_retry, {"retry": "synthesis", "accept": END})

    return graph.compile()
