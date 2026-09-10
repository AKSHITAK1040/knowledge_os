from __future__ import annotations

import unittest

from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.memory.store import MemoryStore


class MemoryTests(unittest.TestCase):
    def test_memory_recall_and_compress(self) -> None:
        store = MemoryStore(DeterministicLLMClient())
        store.remember(user_id="u1", session_id="s1", summary="research on enterprise knowledge", payload={"topic": "knowledge"})
        store.remember(user_id="u1", session_id="s2", summary="research on enterprise knowledge with citations", payload={"topic": "knowledge"})
        recalled = store.recall_user("u1")
        self.assertEqual(len(recalled), 2)
        compressed = store.compress("u1")
        self.assertGreaterEqual(len(compressed), 1)

    def test_decay_scores_present(self) -> None:
        store = MemoryStore(DeterministicLLMClient())
        item = store.remember(user_id="u2", session_id="s1", summary="important memory", payload={}, importance=0.9)
        scores = store.decay_scores("u2")
        self.assertIn(item.id, scores)
        self.assertGreater(scores[item.id], 0.0)

    def test_delete_memory(self) -> None:
        store = MemoryStore(DeterministicLLMClient())
        item = store.remember(user_id="u3", session_id="s1", summary="temporary note", payload={})
        self.assertEqual(len(store.recall_user("u3")), 1)
        self.assertEqual(len(store.recall_session("s1")), 1)
        store.delete_memory(item.id)
        # Verify user, session, and semantic search no longer return it
        self.assertEqual(len(store.recall_user("u3")), 0)
        self.assertEqual(len(store.recall_session("s1")), 0)
        results = store.recall_semantic("temporary note", top_k=5)
        self.assertFalse(any(m.id == item.id for m in results))


if __name__ == "__main__":
    unittest.main()
