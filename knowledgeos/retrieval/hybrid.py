from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Sequence

from ..llm import EmbeddingClient
from ..storage.backends import RedisCacheBackend
from ..types import Citation, KnowledgeChunk, SourceKind
from ..utils import dedupe_citations, reciprocal_rank_fusion, score_overlap, stable_hash, summarize_text, tokenize
from .bm25 import BM25Index
from .dense import DenseIndex


def expand_query(query: str) -> list[str]:
    """Expand query with entity extraction, sub-queries, and research variants."""
    clean = query.strip()
    if not clean:
        return []
    variants: set[str] = {clean}
    tokens = tokenize(clean)

    if len(tokens) > 3:
        # Key concept unigrams & bigrams
        variants.add(" ".join(tokens[: max(3, len(tokens) // 2)]))
        variants.add(" ".join(tokens[-max(3, len(tokens) // 2) :]))

    # Domain context variants
    lower = clean.lower()
    if "research" not in lower:
        variants.add(f"{clean} research analysis")
    if "evidence" not in lower:
        variants.add(f"{clean} verified evidence")
    if "architecture" in lower or "system" in lower:
        variants.add(f"{clean} technical architecture and design")

    return [v for v in variants if v]


@dataclass(slots=True)
class RetrievalBundle:
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    """Enterprise Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF) and persistence."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        cache_backend: RedisCacheBackend | None = None,
        sink: Any | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.dense = DenseIndex(embedding_client)
        self.bm25 = BM25Index()
        self.cache_backend = cache_backend or RedisCacheBackend()
        self.sink = sink
        self._source_counter = 0

    def add_chunk(
        self,
        text: str,
        *,
        title: str,
        url: str | None = None,
        source_kind: SourceKind = SourceKind.INTERNAL,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeChunk:
        metadata = metadata or {}
        chunk_id = stable_hash(f"{title}:{self._source_counter}:{text[:64]}")
        chunk = KnowledgeChunk(
            id=chunk_id,
            text=text,
            metadata={"title": title, "url": url, **metadata},
            source_kind=source_kind,
            embedding=self.embedding_client.embed(f"{title} {text}"),
        )
        self._source_counter += 1
        self.dense.add(chunk)
        self.bm25.add(chunk)

        if self.sink is not None and hasattr(self.sink, "upsert_chunk"):
            try:
                self.sink.upsert_chunk(chunk)
            except Exception:
                pass

        return chunk

    def ingest_chunks(self, chunks: Iterable[KnowledgeChunk]) -> None:
        chunk_list = list(chunks)
        for chunk in chunk_list:
            if not chunk.embedding:
                chunk.embedding = self.embedding_client.embed(f"{chunk.metadata.get('title', '')} {chunk.text}")
            self.dense.add(chunk)
            self.bm25.add(chunk)
            if self.sink is not None and hasattr(self.sink, "upsert_chunk"):
                try:
                    self.sink.upsert_chunk(chunk)
                except Exception:
                    pass

    def hydrate(self, chunks: Iterable[KnowledgeChunk]) -> None:
        """Hydrate in-memory indexes from disk/database chunks."""
        for chunk in chunks:
            self.dense.add(chunk)
            self.bm25.add(chunk)

    def search(self, query: str, top_k: int = 8) -> RetrievalBundle:
        cache_key = f"retrieval:{top_k}:{query.strip().lower()}"
        cached = self.cache_backend.get(cache_key)
        if isinstance(cached, RetrievalBundle):
            return cached

        start_time = time.perf_counter()
        expanded_queries = expand_query(query)
        dense_rankings: list[list[str]] = []
        bm25_rankings: list[list[str]] = []
        chunk_map: dict[str, KnowledgeChunk] = {}
        dense_hit_count = 0
        bm25_hit_count = 0

        for variant in expanded_queries:
            dense_hits = self.dense.search(variant, top_k=max(top_k * 2, 10))
            if dense_hits:
                dense_rankings.append([c.id for c, _ in dense_hits])
                dense_hit_count += len(dense_hits)
                for c, _ in dense_hits:
                    chunk_map[c.id] = c

            bm25_hits = self.bm25.search(variant, top_k=max(top_k * 2, 10))
            if bm25_hits:
                bm25_rankings.append([c.id for c, _ in bm25_hits])
                bm25_hit_count += len(bm25_hits)
                for c, _ in bm25_hits:
                    chunk_map[c.id] = c

        all_rankings = dense_rankings + bm25_rankings
        rrf_scores = reciprocal_rank_fusion(all_rankings, k=60)

        # Cross-reranking with token overlap
        scored_candidates: list[tuple[KnowledgeChunk, float]] = []
        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                overlap = score_overlap(query, f"{chunk.metadata.get('title', '')} {chunk.text}")
                final_score = rrf_score + 0.15 * overlap
                scored_candidates.append((chunk, final_score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        selected = scored_candidates[:top_k]

        citations: list[Citation] = []
        for chunk, score in selected:
            # Find best excerpt snippet
            title = str(chunk.metadata.get("title", chunk.id))
            url = chunk.metadata.get("url")
            excerpt = summarize_text(chunk.text, 50)
            citations.append(
                Citation(
                    source_id=chunk.id,
                    title=title,
                    url=url,
                    chunk_id=chunk.id,
                    excerpt=excerpt,
                    score=round(score, 4),
                    source_kind=chunk.source_kind,
                    offset_start=0,
                    offset_end=len(chunk.text),
                )
            )

        confidence = round(min(1.0, sum(s for _, s in selected) / max(1, len(selected)) * 15.0), 4) if selected else 0.0
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        bundle = RetrievalBundle(
            chunks=[chunk for chunk, _ in selected],
            citations=dedupe_citations(citations),
            confidence=confidence,
            diagnostics={
                "expanded_queries": float(len(expanded_queries)),
                "dense_hits": float(dense_hit_count),
                "bm25_hits": float(bm25_hit_count),
                "rrf_candidates": float(len(rrf_scores)),
                "latency_ms": elapsed_ms,
            },
        )
        self.cache_backend.set(cache_key, bundle, ttl_seconds=120)
        return bundle
