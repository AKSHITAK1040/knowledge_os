# Deployment Guide

## Local

```bash
pip install -e .[dev]
uvicorn knowledgeos.api.app:app --reload
streamlit run knowledgeos/frontend/app.py
```

## Containerization

- Create separate API and frontend images.
- Mount configuration via environment variables.
- Add Postgres, Redis, and Milvus as managed dependencies.

## Production notes

- Put the API behind a reverse proxy.
- Use a real identity provider for authentication and RBAC.
- Replace the deterministic adapters with OpenAI and Voyage credentials.
- Persist analytics and traces to durable storage.

