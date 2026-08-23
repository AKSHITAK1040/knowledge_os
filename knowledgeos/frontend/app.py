from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise SystemExit("Streamlit is required to run the dashboard") from exc

from knowledgeos.analytics.service import AnalyticsService
from knowledgeos.config import settings
from knowledgeos.ingestion.pipeline import IngestionPipeline
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.types import QueryIntent, ResearchRequest, SourceKind

# Streamlit Page Config
st.set_page_config(
    page_title="KnowledgeOS | Enterprise Multi-Agent Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
    }
    .citation-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #EEF2FF;
        color: #4F46E5;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Core Services (cached in session state for fast responsiveness)
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = KnowledgeOSOrchestrator(config=settings)
    st.session_state.ingestion = IngestionPipeline(st.session_state.orchestrator.retriever)

orchestrator: KnowledgeOSOrchestrator = st.session_state.orchestrator
ingestion: IngestionPipeline = st.session_state.ingestion
analytics: AnalyticsService = orchestrator.analytics

# Header
st.markdown('<div class="main-header">⚡ KnowledgeOS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-Style Multi-Agent Research, Long-Term Memory, Hybrid RRF Retrieval & RAG Triad Evaluation</div>',
    unsafe_allow_html=True,
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Agent Controls")
    user_id = st.text_input("User ID", value="enterprise-analyst")
    session_id = st.text_input("Session ID", value="research-session-1")
    research_mode = st.selectbox(
        "Research Mode",
        options=["Deep Multi-Agent", "Academic Survey", "Comparative Analysis", "Quick Factoid"],
        index=0,
    )
    include_web = st.toggle("🌐 Parallel Web Intelligence", value=True)
    include_academic = st.toggle("📚 Academic Literature (arXiv)", value=True)
    top_k = st.slider("Retrieval Top K", min_value=3, max_value=20, value=8)

    st.divider()
    stats = orchestrator.store.stats()
    st.markdown("### 🗄️ System State")
    st.markdown(f"- **Persisted Chunks:** `{stats['chunks']}`")
    st.markdown(f"- **Episodic Memories:** `{stats['memories']}`")
    st.markdown(f"- **Recorded Traces:** `{stats['traces']}`")
    st.markdown(f"- **Analytics Events:** `{stats['metrics']}`")

# Workspace Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔬 Research Studio",
        "📥 Knowledge Ingestion Hub",
        "🧠 Memory Explorer",
        "⚖️ RAG Guardrails & Eval",
        "📊 Observability & Analytics",
    ]
)

# ----------------------------------------------------
# TAB 1: RESEARCH STUDIO
# ----------------------------------------------------
with tab1:
    st.subheader("Interactive Research Studio")

    intent_map = {
        "Deep Multi-Agent": QueryIntent.DEEP_RESEARCH,
        "Academic Survey": QueryIntent.ACADEMIC_SURVEY,
        "Comparative Analysis": QueryIntent.COMPARATIVE_ANALYSIS,
        "Quick Factoid": QueryIntent.FACTOID,
    }

    query = st.text_area(
        "Enter Research Query or Prompt",
        value="What is the architecture and performance of hybrid multi-agent research platforms with RRF retrieval?",
        height=90,
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🚀 Execute Research", type="primary", use_container_width=True)

    if run_btn and query.strip():
        with st.spinner("Multi-agent team orchestrating research plan, hybrid retrieval, and grounding..."):
            req = ResearchRequest(
                query=query.strip(),
                user_id=user_id,
                session_id=session_id,
                top_k=top_k,
                include_web=include_web,
                include_academic=include_academic,
                intent=intent_map[research_mode],
            )
            result = orchestrator.run(req)
            st.session_state["latest_result"] = result

    if "latest_result" in st.session_state:
        res = st.session_state["latest_result"]
        resp = res.response

        st.markdown("---")
        # Executive Metrics Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Execution Time", f"{res.execution_ms} ms")
        m_col2.metric("Confidence Score", f"{int(resp.confidence * 100)}%")
        m_col3.metric("Grounding", f"{int(resp.scores.get('grounding', 0.0) * 100)}%")
        m_col4.metric("Context Precision", f"{int(resp.scores.get('context_precision', 0.0) * 100)}%")
        m_col5.metric("Hallucination Risk", f"{int(resp.scores.get('hallucination_risk', 0.0) * 100)}%")

        out_col1, out_col2 = st.columns([3, 2])

        with out_col1:
            st.markdown("### 📄 Synthesized Research Report")
            st.markdown(resp.answer)

            if resp.citations:
                st.markdown("### 📚 Source Citations")
                for idx, citation in enumerate(resp.citations, start=1):
                    with st.expander(f"[{idx}] {citation.title} ({citation.source_kind.value.upper()}) - Score: {citation.score:.3f}"):
                        if citation.url:
                            st.markdown(f"**URL:** [{citation.url}]({citation.url})")
                        st.markdown(f"**Supporting Excerpt:**\n> {citation.excerpt or 'Full document referenced.'}")

        with out_col2:
            st.markdown("### 📋 Multi-Agent Execution Plan")
            plan_steps = resp.artifacts.get("plan", [])
            for step in plan_steps:
                st.markdown(f"- {step}")

            st.markdown("### 🔍 Agent Execution Outcomes")
            for name, outcome in res.outcomes.items():
                status_icon = "✅" if outcome.status.value == "succeeded" else "⚠️"
                with st.expander(f"{status_icon} Agent: `{name.upper()}` (Confidence: {int(outcome.confidence * 100)}%)"):
                    st.write(f"**Status:** {outcome.status.value}")
                    st.write(f"**Message:** {outcome.message}")
                    if outcome.artifacts:
                        st.json(outcome.artifacts)

# ----------------------------------------------------
# TAB 2: INGESTION HUB
# ----------------------------------------------------
with tab2:
    st.subheader("📥 Knowledge Ingestion & Document Intelligence")
    ingest_type = st.radio("Ingestion Mode", ["File Upload", "Web URL Scraper", "Raw Text Paste"], horizontal=True)

    if ingest_type == "File Upload":
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, Markdown, Text, CSV, JSON)",
            type=["pdf", "docx", "md", "txt", "csv", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest Uploaded Files", type="primary"):
            for f in uploaded_files:
                temp_path = Path(".knowledgeos/uploads") / f.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_bytes(f.read())
                report = ingestion.ingest_file(temp_path)
                st.success(f"Ingested '{f.name}': {report.chunks_created} semantic chunks created!")

    elif ingest_type == "Web URL Scraper":
        url_input = st.text_input("Enter URL to scrape and index", value="https://en.wikipedia.org/wiki/Information_retrieval")
        if st.button("Fetch & Index URL", type="primary") and url_input:
            with st.spinner("Scraping web page and building semantic index..."):
                try:
                    report = ingestion.ingest_url(url_input)
                    st.success(f"Indexed URL '{report.source}': {report.chunks_created} chunks added!")
                except Exception as exc:
                    st.error(f"Failed to ingest URL: {exc}")

    elif ingest_type == "Raw Text Paste":
        text_title = st.text_input("Document Title", value="Enterprise Architecture Guidelines")
        text_body = st.text_area("Paste Content (Plain Text or Markdown)", height=150)
        if st.button("Index Text Document", type="primary") and text_body:
            report = ingestion.ingest_text(text_body, title=text_title)
            st.success(f"Indexed document '{text_title}': {report.chunks_created} chunks added!")

    st.divider()
    st.markdown("### 🗂️ Indexed Knowledge Chunks")
    chunks = orchestrator.store.load_chunks(limit=50)
    if chunks:
        chunk_data = [
            {
                "ID": c.id[:8] + "...",
                "Title": c.metadata.get("title", "untitled"),
                "Source": c.source_kind.value,
                "Length (Chars)": len(c.text),
                "Preview": c.text[:80] + "...",
            }
            for c in chunks
        ]
        st.dataframe(chunk_data, use_container_width=True)
    else:
        st.info("No documents indexed yet. Upload a document or index a URL above.")

# ----------------------------------------------------
# TAB 3: MEMORY EXPLORER
# ----------------------------------------------------
with tab3:
    st.subheader("🧠 Hierarchical Episodic Memory Explorer")
    mem_u_col, mem_s_col = st.columns([1, 1])
    with mem_u_col:
        mem_user = st.text_input("Lookup User ID", value=user_id, key="mem_user_lookup")
    with mem_s_col:
        mem_query = st.text_input("Semantic Memory Search", placeholder="Type concept to search memory...", key="mem_query_search")

    mem_col1, mem_col2 = st.columns([3, 2])

    with mem_col1:
        st.markdown("#### User & Session Memories")
        user_mems = orchestrator.memory_store.recall_user(mem_user)
        if mem_query.strip():
            semantic_mems = orchestrator.memory_store.recall_semantic(mem_query.strip(), top_k=5)
            st.markdown("**Semantic Search Hits:**")
            for m in semantic_mems:
                st.info(f"📌 [{m.session_id}] {m.summary} (Importance: {m.importance})")

        if user_mems:
            for idx, m in enumerate(user_mems):
                with st.expander(f"Memory #{idx+1}: {m.summary[:60]}..."):
                    st.write(f"**Session:** {m.session_id}")
                    st.write(f"**Importance:** {m.importance}")
                    st.write(f"**Payload:** {m.payload}")
                    if st.button(f"🗑️ Delete Memory", key=f"del_mem_{m.id}"):
                        orchestrator.memory_store.delete_memory(m.id)
                        st.rerun()
        else:
            st.info(f"No memories recorded for user '{mem_user}' yet.")

    with mem_col2:
        st.markdown("#### ⚡ Memory Decay & Compression")
        decay_scores = orchestrator.memory_store.decay_scores(mem_user)
        if decay_scores:
            st.write("**Ebbinghaus Decay Scores:**")
            st.json(decay_scores)

        if st.button("🗜️ Compress & Consolidate Memories", use_container_width=True):
            compressed = orchestrator.memory_store.compress(mem_user)
            st.success(f"Consolidated into {len(compressed)} core long-term memory schemas!")
            st.rerun()

        st.markdown("#### ➕ Add Manual Memory Item")
        new_summary = st.text_input("Factual Summary", key="new_mem_summary")
        new_imp = st.slider("Importance", 0.1, 1.0, 0.7, 0.1, key="new_mem_imp")
        if st.button("Save Memory Item", key="save_manual_mem") and new_summary:
            orchestrator.memory_store.remember(user_id=mem_user, session_id=session_id, summary=new_summary, importance=new_imp)
            st.success("Memory item persisted!")
            st.rerun()

# ----------------------------------------------------
# TAB 4: EVALUATION & GUARDRAILS
# ----------------------------------------------------
with tab4:
    st.subheader("⚖️ RAG Triad & Hallucination Guardrails")
    st.markdown(
        """
        KnowledgeOS automatically evaluates every synthesized response against the **RAG Triad** framework:
        - **Faithfulness / Grounding:** Verifies that every assertion is strictly supported by evidence.
        - **Context Precision:** Measures retrieval relevance and noise reduction.
        - **Answer Relevance:** Ensures prompt alignment and semantic completeness.
        - **Citation Attribution:** Validates that inline citations link to substantiated sources.
        """
    )

    if "latest_result" in st.session_state:
        res = st.session_state["latest_result"]
        scores = res.response.scores
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.metric("Grounding / Faithfulness", f"{scores.get('grounding', 0.0):.2f}")
        e_col2.metric("Context Precision", f"{scores.get('context_precision', 0.0):.2f}")
        e_col3.metric("Answer Relevance", f"{scores.get('relevance', 0.0):.2f}")
        e_col4.metric("Citation Quality", f"{scores.get('citation_quality', 0.0):.2f}")

        critique_notes = res.response.artifacts.get("evaluation_notes", [])
        if critique_notes:
            st.markdown("#### 🔍 Automated Evaluator Critique Notes:")
            for n in critique_notes:
                st.warning(f"⚠️ {n}")
        else:
            st.success("✅ Evaluator confirmed all claims are fully grounded and verified.")
    else:
        st.info("Execute a research query in the Research Studio to view live RAG Triad scores.")

# ----------------------------------------------------
# TAB 5: OBSERVABILITY & ANALYTICS
# ----------------------------------------------------
with tab5:
    st.subheader("📊 System Observability & Analytics")
    summary = analytics.summary()

    a_col1, a_col2, a_col3, a_col4 = st.columns(4)
    a_col1.metric("Total Events", summary.get("event_count", 0))
    a_col2.metric("Avg Latency", f"{summary.get('average_latency_ms', 0.0)} ms")
    a_col3.metric("P95 Latency", f"{summary.get('p95_latency_ms', 0.0)} ms")
    a_col4.metric("Estimated Cost", f"${summary.get('cost_usd', 0.0):.5f}")

    st.markdown("---")
    st.markdown("### 🏆 Top Cited Knowledge Documents")
    top_docs = summary.get("top_cited_documents", [])
    if top_docs:
        doc_df = [{"Document Title": title, "Citation Count": count} for title, count in top_docs]
        st.dataframe(doc_df, use_container_width=True)
    else:
        st.info("No citation events recorded yet.")

    st.markdown("---")
    st.markdown("### 🛰️ Live Multi-Agent Execution Trace Waterfall")
    traces = orchestrator.store.list_traces(limit=25)
    if traces:
        trace_table = [
            {
                "Agent": t["agent_name"].upper(),
                "Event": t["event_type"],
                "Duration": f"{t.get('duration_ms', 0.0)} ms",
                "Message": t["message"],
                "Timestamp": time.strftime("%H:%M:%S", time.localtime(t["timestamp"])),
            }
            for t in traces
        ]
        st.dataframe(trace_table, use_container_width=True)
    else:
        st.info("No trace events recorded.")
