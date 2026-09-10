from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import math
import re
import time
from typing import Any, Iterable, Sequence

from .types import Citation, KnowledgeChunk, MemoryItem

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u00C0-\u017F]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


def tokenize(text: str) -> list[str]:
    """Tokenize a string into lowercase alphanumeric tokens."""
    if not text:
        return []
    return [token.lower() for token in TOKEN_RE.findall(text)]


def stable_hash(text: str) -> str:
    """Deterministic 16-character SHA-256 hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deterministic_embedding(text: str, dimensions: int = 64) -> list[float]:
    """Deterministic semantic embedding using character n-grams and token hashes."""
    if not text:
        return [0.0] * dimensions
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dimensions

    # Unigram + Bigram feature hashing
    features: list[str] = list(tokens)
    for i in range(len(tokens) - 1):
        features.append(f"{tokens[i]}_{tokens[i+1]}")

    feature_counts = Counter(features)
    for feat, count in feature_counts.items():
        # TF weight
        tf = 1.0 + math.log(count)
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        for index in range(dimensions):
            byte_val = digest[index % len(digest)]
            sign = 1.0 if (byte_val & 1) == 1 else -1.0
            vector[index] += sign * tf * (byte_val / 255.0)

    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [round(v / norm, 8) for v in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(v * v for v in left[:size])) or 1e-9
    right_norm = math.sqrt(sum(v * v for v in right[:size])) or 1e-9
    sim = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Compute Reciprocal Rank Fusion (RRF) scores across multiple ranked lists.
    
    RRF(d) = sum(1 / (k + rank(d)))
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> list[str]:
    """Chunk text respecting sentence boundaries and token limits."""
    if not text or not text.strip():
        return []
    
    # Check if text contains markdown headers
    sections = [s.strip() for s in MARKDOWN_HEADER_RE.split(text) if s.strip()]
    if len(sections) > 1:
        chunks: list[str] = []
        for sec in sections:
            chunks.extend(chunk_text(sec, max_tokens=max_tokens, overlap=overlap))
        return chunks if chunks else [text.strip()]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        tokens = tokenize(text)
        return [" ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max(1, max_tokens - overlap))]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sent_tokens = len(tokenize(sentence))
        if current_length + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # keep overlap sentences
            overlap_chunk = []
            overlap_length = 0
            for prev_sent in reversed(current_chunk):
                prev_len = len(tokenize(prev_sent))
                if overlap_length + prev_len <= overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_length += prev_len
                else:
                    break
            current_chunk = overlap_chunk + [sentence]
            current_length = overlap_length + sent_tokens
        else:
            current_chunk.append(sentence)
            current_length += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text.strip()]


def extract_claims(text: str) -> list[str]:
    """Extract distinct sentences/claims from generated text for verification."""
    if not text:
        return []
    cleaned = re.sub(r"\[\d+\]", "", text)  # remove inline citation tags like [1]
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if len(tokenize(s)) >= 3]
    return sentences if sentences else [text.strip()]


def summarize_text(text: str, max_words: int = 60) -> str:
    """Summarize text by truncating to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " ..."


def stem_token(token: str) -> str:
    """Lightweight rule-based suffix stemming for English tokens."""
    for suffix in ("ions", "ion", "ing", "ies", "es", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def score_overlap(query: str, text: str) -> float:
    """Calculate token overlap coefficient between query/claim and reference text with stemming."""
    query_tokens = Counter(stem_token(t) for t in tokenize(query))
    text_tokens = Counter(stem_token(t) for t in tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    common = sum(min(query_tokens[token], text_tokens[token]) for token in query_tokens)
    return common / max(1, len(query_tokens))


def score_semantic_grounding(claim: str, evidence_texts: Sequence[str]) -> float:
    """Score how well a claim is supported by a collection of evidence texts."""
    if not claim or not evidence_texts:
        return 0.0
    best_score = 0.0
    claim_emb = deterministic_embedding(claim)
    claim_tokens = Counter(tokenize(claim))

    for evidence in evidence_texts:
        ev_tokens = Counter(tokenize(evidence))
        common = sum(min(claim_tokens[token], ev_tokens[token]) for token in claim_tokens)
        token_overlap = common / max(1, len(claim_tokens))
        ev_emb = deterministic_embedding(evidence)
        dense_sim = cosine_similarity(claim_emb, ev_emb)
        combined = 0.5 * token_overlap + 0.5 * dense_sim
        if combined > best_score:
            best_score = combined

    return max(0.0, min(1.0, best_score))


def now_hours() -> float:
    """Current timestamp in hours."""
    return time.time() / 3600.0


def dedupe_citations(citations: Iterable[Citation]) -> list[Citation]:
    """Deduplicate citations by source_id and chunk_id."""
    seen: set[tuple[str, str | None]] = set()
    result: list[Citation] = []
    for citation in citations:
        key = (citation.source_id, citation.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


def merge_metadata(*items: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dictionaries into a single metadata dictionary."""
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged


def memory_decay_score(item: MemoryItem, halflife_hours: float) -> float:
    """Calculate exponential memory decay score based on Ebbinghaus forgetting curve."""
    age_hours = max(0.0, now_hours() - (item.last_accessed_at / 3600.0))
    if halflife_hours <= 0:
        return item.importance
    decay = 0.5 ** (age_hours / halflife_hours)
    return round(item.importance * decay, 6)
