from __future__ import annotations

import unittest

from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.retrieval.bm25 import BM25Index
from knowledgeos.retrieval.dense import DenseIndex
from knowledgeos.retrieval.hybrid import HybridRetriever, expand_query
from knowledgeos.types import KnowledgeChunk
from knowledgeos.utils import reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_expand_query_creates_variants(self) -> None:
        variants = expand_query("multi agent research platform")
        self.assertGreaterEqual(len(variants), 2)

    def test_rrf_scoring(self) -> None:
        list1 = ["doc1", "doc2", "doc3"]
        list2 = ["doc2", "doc1", "doc4"]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        self.assertIn("doc1", scores)
        self.assertIn("doc2", scores)
        self.assertGreater(scores["doc1"], scores["doc3"])

    def test_bm25_index(self) -> None:
        index = BM25Index()
        chunk = KnowledgeChunk(id="c1", text="Machine learning and artificial intelligence agents")
        index.add(chunk)
        results = index.search("artificial intelligence", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, "c1")

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
