# ⚡ KnowledgeOS: Enterprise Multi-Agent Research & Knowledge Intelligence Platform

**KnowledgeOS** is a state-of-the-art, production-grade multi-agent research and knowledge intelligence platform. Built with a supervisor-driven multi-agent architecture, dynamic intent routing, hybrid Reciprocal Rank Fusion (RRF) retrieval, hierarchical episodic memory, RAG Triad evaluation guardrails, and enterprise FastAPI & Streamlit interfaces.

---

## 🚀 Key Features

### 1. 🤖 Supervisor-Driven Multi-Agent Orchestration
- **Dynamic Intent Routing**: Classifies queries into Factoid, Deep Research, Academic Survey, Comparative Analysis, or Memory Recall.
- **Parallel Sub-Agents**: Concurrent execution of Planner, Local Retrieval, Web Intelligence (DuckDuckGo), Academic Repositories (arXiv), and Memory Recall agents.
- **Critique & Self-Correction Reflection Loop**: Automatically triggers targeted re-retrieval and refined synthesis when evaluation detects low grounding or hallucination risk.
- **Executable LangGraph Pipeline**: Fully compiled LangGraph `StateGraph` workflow for visual execution and studio integration.

### 2. 🔍 State-of-the-Art Hybrid RRF Retrieval
- **Reciprocal Rank Fusion (RRF)**: \( \text{RRF}(d) = \sum \frac{1}{k + \text{rank}_i(d)} \) combining Dense Vector Search and BM25Okapi keyword matching.
- **Multi-Query Expansion**: Automatic concept extraction, synonym generation, and sub-question decomposition.
- **Durable Persistence**: All indexed chunks and vector representations persist across server restarts in SQLite.

### 3. ⚖️ RAG Triad Evaluation & Hallucination Guardrails
- **Faithfulness / Grounding**: Sentence-level claim verification against retrieved evidence.
- **Context Precision**: Precision of retrieved chunks against query intent.
- **Answer Relevance**: Prompt alignment and semantic completeness.
- **Citation Attribution**: Strict verification that every inline citation `[1]`, `[2]` maps directly to verifiable source excerpts.

### 4. 🧠 Hierarchical Episodic & Semantic Memory
- **Working Memory**: In-turn scratchpad context.
- **Session Memory**: Dialogue turn and conversational history.
- **User Memory**: Long-term preferences and domain topics.
- **Ebbinghaus Memory Decay**: Time-decay scoring based on forgetting curve \( \text{Decay} = \text{Importance} \times 0.5^{(\text{age} / \text{halflife})} \).
- **Auto-Compression**: Automatic consolidation and deduplication of related memories.

### 5. 📥 Document Ingestion & Document Intelligence
- Supports **Plain Text, Markdown, PDF, DOCX, CSV, JSON, and Web URLs**.
- Sentence and Markdown-header aware semantic chunking with sliding token windows.
- Batch ingestion endpoint and live chunk indexing.

### 6. 🌐 Modern FastAPI Backend & Pydantic v2 Models
- Versioned REST endpoints with comprehensive Pydantic v2 schemas and validation.
- Endpoints for Chat, Deep Research, Ingest, Memory CRUD, Analytics, Traces, and Health.
- Token-bucket sliding-window rate limiting, API key / Bearer token authentication, and CORS.

### 7. 🖥️ 5-Workspace Streamlit Dashboard
- 🔬 **Research Studio**: Interactive agent research console with citations, evidence drawers, and plan inspector.
- 📥 **Knowledge Ingestion Hub**: Multi-file uploader (PDF/DOCX/TXT/MD/CSV/JSON), URL scraper, and chunk viewer.
- 🧠 **Memory Explorer**: User & session memory manager with decay curve visualizer and compression tool.
- ⚖️ **Evaluation & Guardrails**: Live RAG Triad radar gauges and critique inspector.
- 📊 **Observability & Analytics**: Latency percentiles (p50/p95), cost estimation, trace waterfall, and top cited sources.

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -e .[dev]
```

### 2. Run the FastAPI Backend
```bash
uvicorn knowledgeos.api.app:app --reload --port 8000
```
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 3. Run the Streamlit Dashboard
```bash
streamlit run knowledgeos/frontend/app.py
```
- **Dashboard UI:** [http://localhost:8501](http://localhost:8501)

### 4. Run Test Suite & Benchmarks
```bash
pytest -v
python benchmarks/run_benchmarks.py
```

---

## 🔌 LLM & Provider Configuration

KnowledgeOS is configured with **Groq Cloud API** as its primary, ultra-fast LLM engine, with fallback chains to offline deterministic engines:

```env
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Optional alternatives
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
VOYAGE_API_KEY=your-voyage-key
KNOWLEDGEOS_API_KEY=dev-key
```

---

## 📊 Benchmark Metrics Summary

| Metric | Score |
|---|---|
| **Retrieval Precision** | **100% (1.00)** |
| **Citation Accuracy** | **100% (1.00)** |
| **Answer Relevance** | **100% (1.00)** |
| **Overall Confidence** | **72.3% (0.72)** |
| **Test Suite Pass Rate** | **100% (21/21 passed)** |
