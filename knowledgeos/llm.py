from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

from .utils import deterministic_embedding


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.2, metadata: dict[str, Any] | None = None) -> str:
        raise NotImplementedError


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class DeterministicLLMClient(LLMClient, EmbeddingClient):
    """Offline deterministic adapter for tests and local development."""

    def generate(self, prompt: str, *, temperature: float = 0.2, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        prompt_preview = " ".join(prompt.strip().split())[:220]
        task = metadata.get("task", "response")
        if task == "plan":
            return (
                "Plan:\n"
                "1. Inspect the query intent.\n"
                "2. Retrieve internal, web, academic, and memory evidence.\n"
                "3. Evaluate grounding and citations.\n"
                "4. Synthesize a cited answer."
            )
        if task == "synthesis":
            return f"Answer built from available evidence: {prompt_preview}"
        if task == "evaluation":
            return "relevance=0.88 grounding=0.84 citation_quality=0.86 hallucination_risk=0.18"
        return f"{task}: {prompt_preview}"

    def embed(self, text: str) -> list[float]:
        return deterministic_embedding(text)


class OpenAIAdapter(LLMClient, EmbeddingClient):
    """Production adapter for OpenAI-compatible models.

    This adapter is intentionally thin and can be wired to the official SDK or an internal gateway.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._fallback = DeterministicLLMClient()

    def generate(self, prompt: str, *, temperature: float = 0.2, metadata: dict[str, Any] | None = None) -> str:
        if not self.api_key:
            return self._fallback.generate(prompt, temperature=temperature, metadata=metadata)
        # Placeholder for SDK integration. The fallback keeps the project functional offline.
        return self._fallback.generate(prompt, temperature=temperature, metadata=metadata)

    def embed(self, text: str) -> list[float]:
        return deterministic_embedding(text)


class VoyageEmbeddingAdapter(EmbeddingClient):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.model = model or os.getenv("VOYAGE_MODEL", "voyage-3-large")

    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            return deterministic_embedding(text)
        # Production integration hook: call Voyage AI embeddings API here.
        return deterministic_embedding(text)

