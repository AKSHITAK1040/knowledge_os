from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import math
import re
import time
from typing import Iterable

from .types import Citation, KnowledgeChunk, MemoryItem


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deterministic_embedding(text: str, dimensions: int = 32) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        for index in range(dimensions):
            vector[index] += digest[index % len(digest)] / 255.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 8) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return dot / (left_norm * right_norm)


def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def summarize_text(text: str, max_words: int = 60) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " ..."


def score_overlap(query: str, text: str) -> float:
    query_tokens = Counter(tokenize(query))
    text_tokens = Counter(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    common = sum(min(query_tokens[token], text_tokens[token]) for token in query_tokens)
    return common / max(1, len(query_tokens))


def now_hours() -> float:
    return time.time() / 3600.0


def dedupe_citations(citations: Iterable[Citation]) -> list[Citation]:
    seen: set[tuple[str, str | None]] = set()
    result: list[Citation] = []
    for citation in citations:
        key = (citation.source_id, citation.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


def merge_metadata(*items: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for item in items:
        merged.update(item)
    return merged


def memory_decay_score(item: MemoryItem, halflife_hours: float) -> float:
    age_hours = max(0.0, now_hours() - (item.last_accessed_at / 3600.0))
    if halflife_hours <= 0:
        return item.importance
    decay = 0.5 ** (age_hours / halflife_hours)
    return round(item.importance * decay, 6)

