from __future__ import annotations

from collections import Counter
import math
from typing import Sequence

from ..types import KnowledgeChunk
from ..utils import tokenize


class BM25Index:
    """BM25Okapi inverted index with title and body weighting."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[KnowledgeChunk] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.doc_freq: Counter[str] = Counter()
        self.avgdl: float = 0.0

    def add(self, chunk: KnowledgeChunk) -> None:
        # Give extra weight to title tokens
        title = str(chunk.metadata.get("title", ""))
        title_tokens = tokenize(title) * 2
        body_tokens = tokenize(chunk.text)
        all_tokens = title_tokens + body_tokens
        
        terms = Counter(all_tokens)
        doc_len = len(all_tokens)
        
        self.chunks.append(chunk)
        self.term_freqs.append(terms)
        self.doc_lengths.append(doc_len)
        
        for term in terms:
            self.doc_freq[term] += 1
            
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))

    def add_batch(self, chunks: Sequence[KnowledgeChunk]) -> None:
        for chunk in chunks:
            self.add(chunk)

    def score(self, query: str, chunk_index: int) -> float:
        query_terms = tokenize(query)
        if not query_terms or chunk_index >= len(self.chunks):
            return 0.0

        tf = self.term_freqs[chunk_index]
        dl = self.doc_lengths[chunk_index] or 1
        total_docs = len(self.chunks)
        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue
            df = self.doc_freq[term]
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = tf[term] * (self.k1 + 1.0)
            denominator = tf[term] + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * (numerator / denominator)

        return max(0.0, score)

    def search(self, query: str, top_k: int = 8) -> list[tuple[KnowledgeChunk, float]]:
        if not self.chunks:
            return []
        scored = [(chunk, self.score(query, i)) for i, chunk in enumerate(self.chunks)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
