"""Embedding backends with a local deterministic fallback."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from waam_rag.config import Settings

LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for pluggable embedding backends."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""


class HashingEmbedder:
    """Deterministic local embedder used for local runs and tests."""

    def __init__(self, n_features: int = 1024) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return matrix.astype(np.float32).toarray()

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbedder:
    """Lazy sentence-transformer backend for stronger semantic retrieval."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    """Create the configured embedder with graceful fallback."""

    backend = settings.embedding_backend.lower()
    if backend in {"sentence_transformer", "sentence-transformer", "st"}:
        try:
            LOGGER.info(
                "loading_sentence_transformer",
                extra={"model": settings.embedding_model},
            )
            return SentenceTransformerEmbedder(
                model_name=settings.embedding_model,
                batch_size=settings.embedding_batch_size,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "sentence_transformer_unavailable",
                extra={"model": settings.embedding_model, "error": str(exc)},
            )
    return HashingEmbedder()
