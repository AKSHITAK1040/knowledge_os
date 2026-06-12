# Design Decisions

## Why supervisor-based orchestration?

It gives explicit control over routing, retries, traceability, and agent composition.

## Why hybrid retrieval?

Dense retrieval and BM25 catch different relevance signals. Reranking improves precision and citation quality.

## Why a dedicated evaluation agent?

Evaluation should influence the response pipeline before the final answer is returned.

## Why deterministic fallbacks?

They keep the project testable and runnable without live vendor services.

