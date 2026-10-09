from __future__ import annotations

from ..llm import LLMClient
from ..types import AgentOutcome, ResearchState
from ..utils import dedupe_citations, summarize_text
from .base import BaseAgent


class SynthesisAgent(BaseAgent):
    name = "synthesis"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__()
        self.llm = llm

    def _run(self, state: ResearchState) -> AgentOutcome:
        citations = dedupe_citations(state.citations)
        evidence_lines = []
        for citation in citations[:10]:
            excerpt = citation.excerpt or ""
            evidence_lines.append(f"[{citation.title}] {excerpt}")

        memory_lines = state.artifacts.get("memory_summary", "")
        critique_notes = "\n".join(f"- {note}" for note in state.critique_notes)

        prompt = (
            f"Query: {state.request.query}\n"
            f"Intent: {state.intent.value}\n"
            f"Plan: {' | '.join(state.plan)}\n\n"
            f"Evidence:\n" + ("\n".join(evidence_lines) if evidence_lines else "No direct evidence retrieved.") + "\n\n"
            f"Memory:\n{memory_lines}\n\n"
            f"Critique / Reflection Notes:\n{critique_notes}\n\n"
            "Task: Synthesize an authoritative, highly grounded, and cited executive intelligence report. "
            "Integrate inline citations [1], [2], etc. corresponding directly to the provided evidence sources."
        )

        answer = self.llm.generate(
            prompt,
            metadata={
                "task": "synthesis",
                "intent": state.intent.value,
                "retry_count": state.retry_count,
                "notes": state.critique_notes,
            },
        )

        if not citations and "No citations were available" not in answer:
            answer = f"{answer}\n\n*Note: No citations were available, so this answer should be treated as provisional.*"

        state.draft = answer
        state.final_answer = answer

        self.trace(
            state,
            "synthesis_complete",
            f"Synthesized research report ({len(answer.split())} words) with {len(citations)} citations",
            metadata={"citations_count": len(citations), "preview": summarize_text(answer, 30)},
        )

        return self.outcome(
            state,
            confidence=0.88 if citations else 0.45,
            message="Synthesized grounded research report",
            artifacts={"answer": answer, "citations_count": len(citations)},
        )
