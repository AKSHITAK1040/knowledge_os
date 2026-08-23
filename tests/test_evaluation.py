from __future__ import annotations

import unittest

from knowledgeos.evaluation.evaluator import Evaluator
from knowledgeos.types import Citation, KnowledgeChunk, SourceKind


class EvaluationTests(unittest.TestCase):
    def test_evaluator_rag_triad_scores(self) -> None:
        evaluator = Evaluator()
        citation = Citation(source_id="1", title="KnowledgeOS Doc", excerpt="grounded excerpt with supporting context", source_kind=SourceKind.INTERNAL)
        evidence = [KnowledgeChunk(id="1", text="grounded excerpt with supporting context and architecture insights")]
        report = evaluator.evaluate("What is the architecture?", "The architecture has a grounded excerpt with supporting context [1].", [citation], evidence)
        
        self.assertGreaterEqual(report.relevance, 0.0)
        self.assertLessEqual(report.relevance, 1.0)
        self.assertGreaterEqual(report.grounding, 0.0)
        self.assertLessEqual(report.grounding, 1.0)
        self.assertGreaterEqual(report.context_precision, 0.0)
        self.assertGreaterEqual(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)
        self.assertIn(report.recommendation, ["accept", "retry"])


if __name__ == "__main__":
    unittest.main()
