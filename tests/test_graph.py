from __future__ import annotations

import unittest

from knowledgeos.llm import DeterministicLLMClient
from knowledgeos.orchestration.graph import build_langgraph
from knowledgeos.orchestration.supervisor import KnowledgeOSOrchestrator


class GraphTests(unittest.TestCase):
    def test_build_and_invoke_langgraph(self) -> None:
        llm = DeterministicLLMClient()
        orchestrator = KnowledgeOSOrchestrator(llm=llm)
        graph = build_langgraph(orchestrator)
        if graph is not None:
            initial_state = {
                "query": "Explain multi-agent supervisor architecture",
                "user_id": "tester",
                "session_id": "test-session",
                "include_web": False,
                "include_academic": False,
                "plan": [],
                "citations": [],
                "answer": "",
                "scores": {},
                "retry_count": 0,
            }
            final_state = graph.invoke(initial_state)
            self.assertIn("plan", final_state)
            self.assertIn("answer", final_state)
            self.assertIn("scores", final_state)


if __name__ == "__main__":
    unittest.main()
