from __future__ import annotations

from collections import Counter
import math

from ..types import KnowledgeChunk
from ..utils import tokenize


class BM25Index:
    def __init__(self) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_freq: Counter[str] = Counter()

    def add(self, chunk: KnowledgeChunk) -> None:
        terms = Counter(tokenize(chunk.text))
        self.chunks.append(chunk)
        self.term_freqs.append(terms)
        for term in terms:
            self.doc_freq[term] += 1

    def score(self, query: str, chunk_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return 0.0
        tf = self.term_freqs[chunk_index]
        avgdl = sum(sum(counter.values()) for counter in self.term_freqs) / max(1, len(self.term_freqs))
        dl = sum(tf.values()) or 1
        score = 0.0
        total_docs = len(self.chunks)
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * (dl / avgdl))
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 8) -> list[tuple[KnowledgeChunk, float]]:
        scored = [(chunk, self.score(query, i)) for i, chunk in enumerate(self.chunks)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

