from __future__ import annotations

import unittest

from knowledgeos.evaluation.evaluator import Evaluator
from knowledgeos.types import Citation, KnowledgeChunk, SourceKind


class EvaluationTests(unittest.TestCase):
    def test_evaluator_scores_are_bounded(self) -> None:
        evaluator = Evaluator()
        citation = Citation(source_id="1", title="Doc", excerpt="grounded excerpt", source_kind=SourceKind.INTERNAL)
        evidence = [KnowledgeChunk(id="1", text="grounded excerpt with supporting context")]
        report = evaluator.evaluate("grounded excerpt", "grounded excerpt with answer", [citation], evidence)
        self.assertGreaterEqual(report.relevance, 0.0)
        self.assertLessEqual(report.relevance, 1.0)
        self.assertGreaterEqual(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()

