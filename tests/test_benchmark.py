from __future__ import annotations

import unittest

from benchmarks.framework import BenchmarkCase, BenchmarkSuite
from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.memory.store import MemoryStore
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.retrieval.hybrid import HybridRetriever


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_suite_returns_summary(self) -> None:
        llm = DeterministicLLMClient()
        retriever = HybridRetriever(llm)
        retriever.add_chunk("KnowledgeOS overview for enterprise research.", title="KnowledgeOS overview")
        orchestrator = KnowledgeOSOrchestrator(llm=llm, retriever=retriever, memory_store=MemoryStore(llm))
        suite = BenchmarkSuite(orchestrator)
        report = suite.run([BenchmarkCase(query="What is KnowledgeOS?", expected_terms=["knowledgeos"], expected_citations=["KnowledgeOS overview"])])
        self.assertIn("summary", report)
        self.assertIn("cases", report)
        self.assertGreaterEqual(report["summary"]["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()

