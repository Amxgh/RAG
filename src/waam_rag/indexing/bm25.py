"""A lightweight BM25 implementation for sparse retrieval."""

from __future__ import annotations

import math
from collections import Counter

from waam_rag.schemas import ChunkRecord
from waam_rag.utils.text import lexical_tokens


class BM25Index:
    """In-memory BM25 over chunk texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0

    def rebuild(self, chunks: list[ChunkRecord]) -> None:
        self.doc_freqs = Counter()
        self.term_freqs = {}
        self.doc_lengths = {}
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.doc_count = len(chunks)

        total_length = 0
        for chunk in chunks:
            tokens = lexical_tokens(chunk.text)
            frequencies = Counter(tokens)
            self.term_freqs[chunk.chunk_id] = frequencies
            self.doc_lengths[chunk.chunk_id] = len(tokens)
            total_length += len(tokens)
            for term in frequencies:
                self.doc_freqs[term] += 1
        self.avg_doc_len = total_length / self.doc_count if self.doc_count else 0.0

    def search(self, query: str, candidate_chunks: list[ChunkRecord], top_k: int) -> list[tuple[ChunkRecord, float]]:
        query_terms = lexical_tokens(query)
        scores: list[tuple[ChunkRecord, float]] = []
        for chunk in candidate_chunks:
            score = self._score_document(chunk.chunk_id, query_terms)
            if score > 0:
                scores.append((chunk, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _score_document(self, chunk_id: str, query_terms: list[str]) -> float:
        frequencies = self.term_freqs.get(chunk_id)
        if not frequencies:
            return 0.0

        doc_len = self.doc_lengths.get(chunk_id, 0)
        score = 0.0
        for term in query_terms:
            freq = frequencies.get(term, 0)
            if not freq:
                continue
            doc_freq = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * numerator / denominator
        return score
