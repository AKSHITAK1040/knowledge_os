from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..llm import EmbeddingClient
from ..types import Citation, KnowledgeChunk, SourceKind
from ..utils import dedupe_citations, score_overlap, stable_hash, summarize_text
from ..storage.backends import RedisCacheBackend
from .bm25 import BM25Index
from .dense import DenseIndex


def expand_query(query: str) -> list[str]:
    variants = {query.strip()}
    tokens = [token for token in query.replace("?", "").split() if token]
    if len(tokens) > 3:
        variants.add(" ".join(tokens[: max(3, len(tokens) // 2)]))
        variants.add(" ".join(tokens[-max(3, len(tokens) // 2) :]))
    if "research" not in query.lower():
        variants.add(f"{query} research evidence")
    if "cite" not in query.lower():
        variants.add(f"{query} with citations")
    return [variant for variant in variants if variant]


@dataclass(slots=True)
class RetrievalBundle:
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, embedding_client: EmbeddingClient, cache_backend: RedisCacheBackend | None = None) -> None:
        self.embedding_client = embedding_client
        self.dense = DenseIndex(embedding_client)
        self.bm25 = BM25Index()
        self.cache_backend = cache_backend or RedisCacheBackend()
        self._source_counter = 0

    def add_chunk(self, text: str, *, title: str, url: str | None = None, source_kind: SourceKind = SourceKind.INTERNAL, metadata: dict[str, object] | None = None) -> KnowledgeChunk:
        metadata = metadata or {}
        chunk = KnowledgeChunk(
            id=stable_hash(f"{title}:{self._source_counter}:{text[:64]}"),
            text=text,
            metadata={"title": title, "url": url, **metadata},
            source_kind=source_kind,
            embedding=self.embedding_client.embed(text),
        )
        self._source_counter += 1
        self.dense.add(chunk)
        self.bm25.add(chunk)
        return chunk

    def ingest_chunks(self, chunks: Iterable[KnowledgeChunk]) -> None:
        for chunk in chunks:
            if not chunk.embedding:
                chunk.embedding = self.embedding_client.embed(chunk.text)
            self.dense.add(chunk)
            self.bm25.add(chunk)

    def search(self, query: str, top_k: int = 8) -> RetrievalBundle:
        cache_key = f"retrieval:{top_k}:{query.strip().lower()}"
        cached = self.cache_backend.get(cache_key)
        if isinstance(cached, RetrievalBundle):
            return cached

        expanded_queries = expand_query(query)
        dense_scores: dict[str, tuple[KnowledgeChunk, float]] = {}
        bm25_scores: dict[str, tuple[KnowledgeChunk, float]] = {}

        for variant in expanded_queries:
            for chunk, score in self.dense.search(variant, top_k=max(top_k, 10)):
                current = dense_scores.get(chunk.id)
                if current is None or score > current[1]:
                    dense_scores[chunk.id] = (chunk, score)
            for chunk, score in self.bm25.search(variant, top_k=max(top_k, 10)):
                current = bm25_scores.get(chunk.id)
                if current is None or score > current[1]:
                    bm25_scores[chunk.id] = (chunk, score)

        merged: dict[str, dict[str, object]] = {}
        for chunk_id, (chunk, score) in dense_scores.items():
            merged.setdefault(chunk_id, {"chunk": chunk, "dense": 0.0, "bm25": 0.0})
            merged[chunk_id]["dense"] = max(merged[chunk_id]["dense"], score)
        for chunk_id, (chunk, score) in bm25_scores.items():
            merged.setdefault(chunk_id, {"chunk": chunk, "dense": 0.0, "bm25": 0.0})
            merged[chunk_id]["bm25"] = max(merged[chunk_id]["bm25"], score)

        ranked: list[tuple[KnowledgeChunk, float]] = []
        for item in merged.values():
            chunk = item["chunk"]
            score = 0.65 * float(item["dense"]) + 0.35 * float(item["bm25"]) + 0.1 * score_overlap(query, chunk.text)
            ranked.append((chunk, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        selected = ranked[:top_k]
        citations = [
            Citation(
                source_id=chunk.id,
                title=str(chunk.metadata.get("title", chunk.id)),
                url=chunk.metadata.get("url"),
                chunk_id=chunk.id,
                excerpt=summarize_text(chunk.text, 40),
                score=round(score, 4),
                source_kind=chunk.source_kind,
            )
            for chunk, score in selected
        ]
        normalized = [max(0.0, min(1.0, score)) for _, score in selected]
        confidence = sum(normalized) / len(normalized) if normalized else 0.0
        result = RetrievalBundle(
            chunks=[chunk for chunk, _ in selected],
            citations=dedupe_citations(citations),
            confidence=round(confidence, 4),
            diagnostics={
                "expanded_queries": float(len(expanded_queries)),
                "dense_hits": float(len(dense_scores)),
                "bm25_hits": float(len(bm25_scores)),
            },
        )
        self.cache_backend.set(cache_key, result, ttl_seconds=120)
        return result
