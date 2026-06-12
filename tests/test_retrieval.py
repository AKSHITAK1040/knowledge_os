from __future__ import annotations

import unittest

from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.retrieval.hybrid import HybridRetriever, expand_query


class RetrievalTests(unittest.TestCase):
    def test_expand_query_creates_variants(self) -> None:
        variants = expand_query("multi agent research platform")
        self.assertGreaterEqual(len(variants), 2)

    def test_hybrid_retriever_returns_citations(self) -> None:
        llm = DeterministicLLMClient()
        retriever = HybridRetriever(llm)
        retriever.add_chunk(
            "KnowledgeOS uses supervisor based orchestration with hybrid retrieval, memory, and evaluation.",
            title="KnowledgeOS architecture",
        )
        bundle = retriever.search("How does KnowledgeOS handle orchestration and retrieval?", top_k=3)
        self.assertGreaterEqual(len(bundle.chunks), 1)
        self.assertGreaterEqual(len(bundle.citations), 1)
        self.assertGreater(bundle.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

