from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.framework import BenchmarkCase, BenchmarkSuite
from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.memory.store import MemoryStore
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.retrieval.hybrid import HybridRetriever
from knowledgeos.storage.sqlite_store import KnowledgeOSSQLiteStore


def main() -> None:
    benchmark_db = Path(".knowledgeos/benchmark.sqlite3")
    if benchmark_db.exists():
        benchmark_db.unlink(missing_ok=True)
    store = KnowledgeOSSQLiteStore(benchmark_db)

    llm = DeterministicLLMClient()
    retriever = HybridRetriever(llm, sink=store)
    retriever.add_chunk(
        "KnowledgeOS combines supervisor orchestration, hybrid retrieval, and long-term memory for enterprise research workflows.",
        title="KnowledgeOS Overview",
    )
    retriever.add_chunk(
        "The RAG Triad framework evaluates faithfulness, answer relevance, and context precision to eliminate hallucinations.",
        title="RAG Triad & Guardrails",
    )
    retriever.add_chunk(
        "Reciprocal Rank Fusion (RRF) merges dense vector search with BM25 keyword rankings for superior retrieval accuracy.",
        title="Hybrid RRF Retrieval",
    )

    orchestrator = KnowledgeOSOrchestrator(
        llm=llm,
        retriever=retriever,
        memory_store=MemoryStore(llm, sink=store),
        store=store,
    )
    suite = BenchmarkSuite(orchestrator)
    report = suite.run(
        [
            BenchmarkCase(
                query="How does KnowledgeOS support enterprise research quality and grounding?",
                expected_terms=["knowledgeos", "retrieval", "memory"],
                expected_citations=["KnowledgeOS Overview"],
            ),
            BenchmarkCase(
                query="Explain the supervisor architecture and RRF retrieval of KnowledgeOS.",
                expected_terms=["supervisor", "orchestration", "retrieval"],
                expected_citations=["KnowledgeOS Overview", "Hybrid RRF Retrieval"],
                session_id="bench-2",
            ),
            BenchmarkCase(
                query="What are the evaluation guardrails for hallucination prevention?",
                expected_terms=["rag", "triad", "guardrails", "faithfulness"],
                expected_citations=["RAG Triad & Guardrails"],
                session_id="bench-3",
            ),
        ]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
