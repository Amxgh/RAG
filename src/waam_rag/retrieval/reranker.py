"""Optional reranking for high-precision citation-ready passages."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from waam_rag.config import Settings
from waam_rag.retrieval.fusion import ScoredCandidate
from waam_rag.utils.text import lexical_tokens

LOGGER = logging.getLogger(__name__)


class Reranker(Protocol):
    """Protocol for optional rerankers."""

    def rerank(self, query: str, candidates: list[ScoredCandidate], top_k: int) -> list[ScoredCandidate]:
        """Rerank candidates."""


class HeuristicReranker:
    """Fallback reranker based on lexical overlap and defect emphasis."""

    def rerank(self, query: str, candidates: list[ScoredCandidate], top_k: int) -> list[ScoredCandidate]:
        query_tokens = set(lexical_tokens(query))
        for candidate in candidates:
            chunk_tokens = set(lexical_tokens(candidate.chunk.text))
            overlap = len(query_tokens.intersection(chunk_tokens)) / max(len(query_tokens), 1)
            phrase_boost = 0.1 if candidate.chunk.defect_terms else 0.0
            candidate.rerank_score = candidate.score * 0.7 + overlap * 0.3 + phrase_boost
            candidate.score = candidate.rerank_score
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker with a graceful fallback path."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredCandidate], top_k: int) -> list[ScoredCandidate]:
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
        if scores.size:
            min_score = float(scores.min())
            max_score = float(scores.max())
            denom = max(max_score - min_score, 1e-6)
            normalized = (scores - min_score) / denom
        else:
            normalized = scores

        for candidate, model_score in zip(candidates, normalized, strict=True):
            candidate.rerank_score = float(model_score)
            candidate.score = candidate.score * 0.35 + float(model_score) * 0.65
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


def build_reranker(settings: Settings) -> Reranker | None:
    """Construct a reranker when enabled."""

    if not settings.reranker_enabled:
        return None
    try:
        return CrossEncoderReranker(settings.reranker_model)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning(
            "cross_encoder_unavailable",
            extra={"model": settings.reranker_model, "error": str(exc)},
        )
        return HeuristicReranker()
