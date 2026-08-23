from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from knowledgeos.ingestion.pipeline import IngestionPipeline
from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.retrieval.hybrid import HybridRetriever


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = HybridRetriever(DeterministicLLMClient())
        self.pipeline = IngestionPipeline(self.retriever)

    def test_text_ingestion(self) -> None:
        report = self.pipeline.ingest_text(
            "KnowledgeOS provides multi-agent research. It integrates hybrid retrieval and evaluation.",
            title="Overview Text",
        )
        self.assertEqual(report.source, "Overview Text")
        self.assertGreaterEqual(report.chunks_created, 1)

    def test_csv_ingestion(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Topic,Description\nAI,Artificial Intelligence Agents\nRAG,Retrieval Augmented Generation\n")
            f_path = f.name

        try:
            report = self.pipeline.ingest_file(f_path)
            self.assertGreaterEqual(report.chunks_created, 1)
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_json_ingestion(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"key": "concept1", "text": "Supervisor Orchestrator"}, {"key": "concept2", "text": "LangGraph"}], f)
            f_path = f.name

        try:
            report = self.pipeline.ingest_file(f_path)
            self.assertGreaterEqual(report.chunks_created, 1)
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_batch_ingestion(self) -> None:
        items = [
            {"text": "First batch document text", "title": "Doc 1"},
            {"text": "Second batch document text", "title": "Doc 2"},
        ]
        reports = self.pipeline.ingest_batch(items)
        self.assertEqual(len(reports), 2)


if __name__ == "__main__":
    unittest.main()
