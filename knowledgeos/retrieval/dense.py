from __future__ import annotations

from typing import Sequence

from ..llm import EmbeddingClient
from ..types import KnowledgeChunk
from ..utils import cosine_similarity


class DenseIndex:
    """Dense vector embedding index supporting batch ingestion and similarity scoring."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client
        self.chunks: list[KnowledgeChunk] = []

    def add(self, chunk: KnowledgeChunk) -> None:
        if not chunk.embedding:
            chunk.embedding = self.embedding_client.embed(f"{chunk.metadata.get('title', '')} {chunk.text}")
        self.chunks.append(chunk)

    def add_batch(self, chunks: Sequence[KnowledgeChunk]) -> None:
        to_embed = [f"{c.metadata.get('title', '')} {c.text}" for c in chunks if not c.embedding]
        if to_embed:
            embeddings = self.embedding_client.embed_batch(to_embed)
            e_idx = 0
            for chunk in chunks:
                if not chunk.embedding:
                    chunk.embedding = embeddings[e_idx]
                    e_idx += 1
        for chunk in chunks:
            self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 8) -> list[tuple[KnowledgeChunk, float]]:
        if not self.chunks:
            return []
        query_embedding = self.embedding_client.embed(query)
        scored = [(chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in self.chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
