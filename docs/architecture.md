# Architecture

```mermaid
flowchart TD
  U[User Query] --> S[Supervisor Agent]
  S --> P[Planner Agent]
  S --> M[Memory Agent]
  S --> R[Retrieval Agent]
  S --> W[Web Research Agent]
  S --> A[Academic Research Agent]
  R --> E[Evaluation Agent]
  W --> E
  A --> E
  M --> E
  E --> X[Synthesis Agent]
  X --> O[Final Response with Citations]
```

## Notes

- The supervisor handles routing, retries, and trace emission.
- Retrieval is hybrid by default: dense + BM25 + reranking.
- Memory is split into session, user, and semantic layers.
- Evaluation is a first-class agent, not an afterthought.

