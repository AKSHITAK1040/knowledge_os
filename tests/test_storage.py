from __future__ import annotations

import tempfile
import unittest

from knowledgeos.analytics.service import MetricsEvent
from knowledgeos.storage.sqlite_store import KnowledgeOSSQLiteStore
from knowledgeos.types import Citation, MemoryItem, TraceEvent


class StorageTests(unittest.TestCase):
    def test_sqlite_store_persists_core_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeOSSQLiteStore(f"{tmpdir}/knowledgeos.sqlite3")
            trace = TraceEvent(agent_name="test", event_type="start", message="hello")
            store.append_trace(trace)
            metric = MetricsEvent(
                name="research",
                user_id="u1",
                session_id="s1",
                duration_ms=12.5,
                token_count=42,
                cost_usd=0.001,
                citations=[Citation(source_id="1", title="doc")],
            )
            store.record_metric(metric)
            memory = MemoryItem(id="m1", user_id="u1", session_id="s1", summary="memory")
            store.upsert_memory(memory)

            self.assertEqual(len(store.list_traces()), 1)
            self.assertEqual(len(store.list_metrics()), 1)
            self.assertEqual(len(store.list_memories()), 1)
            hydrated = store.load_metrics()
            self.assertEqual(hydrated[0].name, "research")


if __name__ == "__main__":
    unittest.main()

