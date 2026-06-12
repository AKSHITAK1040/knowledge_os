from __future__ import annotations

from dataclasses import asdict

if __package__ in {None, ""}:  # pragma: no cover - streamlit script execution bootstrap
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - runtime optional dependency
    raise SystemExit("Streamlit is required to run the dashboard") from exc

from knowledgeos.analytics.service import AnalyticsService
from knowledgeos.config import settings
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.types import ResearchRequest


st.set_page_config(page_title="KnowledgeOS", page_icon="KOS", layout="wide")

orchestrator = KnowledgeOSOrchestrator(config=settings)
analytics: AnalyticsService = orchestrator.analytics

st.title("KnowledgeOS")
st.caption("Multi-agent research, memory, retrieval, and evaluation")

with st.sidebar:
    st.header("Research Controls")
    user_id = st.text_input("User ID", value="analyst")
    session_id = st.text_input("Session ID", value="demo-session")
    include_web = st.toggle("Include web research", value=True)
    include_academic = st.toggle("Include academic research", value=True)
    top_k = st.slider("Top K", min_value=3, max_value=15, value=8)

query = st.text_area("Ask KnowledgeOS", value="What is the state of multi-agent research systems for enterprise knowledge management?")

if st.button("Research", type="primary"):
    request = ResearchRequest(
        query=query,
        user_id=user_id,
        session_id=session_id,
        top_k=top_k,
        include_web=include_web,
        include_academic=include_academic,
    )
    result = orchestrator.run(request)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Response")
        st.write(result.response.answer)
        st.subheader("Citations")
        for citation in result.response.citations:
            st.markdown(f"- **{citation.title}** ({citation.source_kind}) {citation.url or ''}")
            if citation.excerpt:
                st.caption(citation.excerpt)
    with col2:
        st.subheader("Scores")
        st.json(result.response.scores)
        st.subheader("Trace")
        st.json([asdict(event) for event in result.response.trace[-20:]])

st.divider()
summary = analytics.summary()
metric_cols = st.columns(4)
metric_cols[0].metric("Events", summary.get("event_count", 0))
metric_cols[1].metric("Latency ms", summary.get("average_latency_ms", 0.0))
metric_cols[2].metric("Tokens", summary.get("token_consumption", 0))
metric_cols[3].metric("Cost USD", summary.get("cost_usd", 0.0))
