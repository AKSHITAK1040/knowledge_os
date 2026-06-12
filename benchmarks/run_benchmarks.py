from __future__ import annotations

import json

if __package__ in {None, ""}:  # pragma: no cover - script execution path bootstrap
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.memory.store import MemoryStore
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.retrieval.hybrid import HybridRetriever
from benchmarks.framework import BenchmarkCase, BenchmarkSuite


def main() -> None:
    llm = DeterministicLLMClient()
    retriever = HybridRetriever(llm)
    retriever.add_chunk(
        "KnowledgeOS combines supervisor orchestration, hybrid retrieval, and long-term memory for enterprise research workflows.",
        title="KnowledgeOS overview",
    )
    orchestrator = KnowledgeOSOrchestrator(llm=llm, retriever=retriever, memory_store=MemoryStore(llm))
    suite = BenchmarkSuite(orchestrator)
    report = suite.run(
        [
            BenchmarkCase(
                query="How does KnowledgeOS support enterprise research quality and grounding?",
                expected_terms=["knowledgeos", "retrieval", "memory"],
                expected_citations=["KnowledgeOS overview"],
            ),
            BenchmarkCase(
                query="Explain the supervisor architecture of KnowledgeOS.",
                expected_terms=["supervisor", "orchestration"],
                expected_citations=["KnowledgeOS overview"],
                session_id="bench-2",
            ),
        ]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
