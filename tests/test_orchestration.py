from __future__ import annotations

import unittest

from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.memory.store import MemoryStore
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator
from knowledgeos.retrieval.hybrid import HybridRetriever
from knowledgeos.types import ResearchRequest


class FakeSearchClient:
    def search(self, query: str, max_results: int = 5):
        from knowledgeos.research.sources import WebEvidence

        return [WebEvidence(title="Test source", url="https://example.com/test", excerpt=f"Evidence for {query}")]


class OrchestrationTests(unittest.TestCase):
    def test_full_research_run(self) -> None:
        llm = DeterministicLLMClient()
        retriever = HybridRetriever(llm)
        retriever.add_chunk(
            "KnowledgeOS combines supervisor orchestration, hybrid retrieval, memory, and evaluation.",
            title="KnowledgeOS overview",
        )
        orchestrator = KnowledgeOSOrchestrator(llm=llm, retriever=retriever, memory_store=MemoryStore(llm))
        orchestrator.web.search_client = FakeSearchClient()
        orchestrator.academic.search_client = FakeSearchClient()
        result = orchestrator.run(
            ResearchRequest(
                query="How does KnowledgeOS improve research workflows?",
                user_id="tester",
                session_id="session-1",
                include_web=True,
                include_academic=True,
            )
        )
        self.assertGreaterEqual(len(result.response.trace), 1)
        self.assertGreaterEqual(len(result.response.citations), 1)
        self.assertIn("confidence", result.response.scores)
        trace_ids = {event.trace_id for event in result.response.trace}
        self.assertEqual(len(trace_ids), 1)


if __name__ == "__main__":
    unittest.main()
