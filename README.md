# KnowledgeOS

KnowledgeOS is a production-style multi-agent research and knowledge intelligence platform. It is designed around a supervisor-based architecture with dynamic routing, parallel research, long-term memory, hybrid retrieval, evaluation, observability, and API-first delivery.

## What is included

- Supervisor-driven multi-agent orchestration
- Planner, retrieval, web research, academic research, memory, evaluation, and synthesis agents
- Hybrid retrieval with dense and BM25 scoring, reranking, multi-query expansion, and citations
- Session, user, and semantic memory services with summarization and decay
- Ingestion pipeline for text, web pages, PDF, and DOCX sources
- Evaluation layer with grounding, citation quality, and hallucination-risk scoring
- FastAPI surface for chat, ingestion, research, memory, analytics, and health
- Streamlit dashboard for analytics and research visibility
- Durable local persistence for traces, analytics, and memories
- Retrieval caching and thread-safe support stores
- Benchmarks, tests, and architecture documentation

## Quick start

1. Create a Python 3.12 environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Run the API:

```bash
uvicorn knowledgeos.api.app:app --reload
```

4. Run the dashboard:

```bash
streamlit run knowledgeos/frontend/app.py
```

## Notes

- The project is structured so the core orchestration and retrieval logic can run without live vendor APIs in tests.
- External adapters for OpenAI, Voyage AI, Milvus, PostgreSQL, and Redis are represented as interfaces and production-ready integration points.
- The local default implementations are deterministic and suitable for testing and offline development.
