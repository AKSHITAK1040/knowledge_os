from __future__ import annotations

from ..llm import EmbeddingClient
from ..types import KnowledgeChunk
from ..utils import cosine_similarity


class DenseIndex:
    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client
        self.chunks: list[KnowledgeChunk] = []

    def add(self, chunk: KnowledgeChunk) -> None:
        if not chunk.embedding:
            chunk.embedding = self.embedding_client.embed(chunk.text)
        self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 8) -> list[tuple[KnowledgeChunk, float]]:
        query_embedding = self.embedding_client.embed(query)
        scored = [(chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in self.chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

