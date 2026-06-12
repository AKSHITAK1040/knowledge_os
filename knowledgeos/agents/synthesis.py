from __future__ import annotations

from ..llm import LLMClient
from ..types import AgentOutcome, AgentStatus, ResearchState
from ..utils import dedupe_citations, summarize_text
from .base import BaseAgent


class SynthesisAgent(BaseAgent):
    name = "synthesis"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__()
        self.llm = llm

    def _run(self, state: ResearchState) -> AgentOutcome:
        evidence_lines = []
        for citation in dedupe_citations(state.citations)[:8]:
            evidence_lines.append(f"[{citation.title}] {citation.excerpt or ''}")
        memory_lines = state.artifacts.get("memory_summary", "")
        evaluation_notes = "\n".join(state.artifacts.get("evaluation_notes", []))
        prompt = (
            f"Query: {state.request.query}\n"
            f"Plan: {' | '.join(state.plan)}\n"
            f"Evidence:\n" + "\n".join(evidence_lines) + "\n"
            f"Memory:\n{memory_lines}\n"
            f"Reflection notes:\n{evaluation_notes}\n"
            "Write a grounded, cited response with clear source attribution."
        )
        answer = self.llm.generate(prompt, metadata={"task": "synthesis"})
        if not state.citations:
            answer = f"{answer}\n\nNo citations were available, so this answer should be treated as provisional."
        state.draft = answer
        state.final_answer = answer
        self.trace(state, "synthesis_complete", "Synthesized final response", metadata={"citations": len(state.citations), "preview": summarize_text(answer, 40)})
        return self.outcome(state, confidence=0.82, message="Synthesized response", artifacts={"answer": answer})
