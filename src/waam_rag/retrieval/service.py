"""Hybrid retrieval orchestration."""

from __future__ import annotations

import time

import numpy as np

from waam_rag.config import Settings
from waam_rag.indexing.bm25 import BM25Index
from waam_rag.indexing.embeddings import Embedder
from waam_rag.indexing.repository import DocumentRepository
from waam_rag.retrieval.fusion import ScoredCandidate, apply_domain_boosts, reciprocal_rank_fusion
from waam_rag.retrieval.query_builder import QueryBuilder
from waam_rag.retrieval.reranker import HeuristicReranker, Reranker
from waam_rag.schemas import ChunkRecord, QueryFilters, QueryRequest


class HybridRetriever:
    """Dense + sparse retrieval with metadata-aware fusion and optional reranking."""

    def __init__(
        self,
        settings: Settings,
        repository: DocumentRepository,
        embedder: Embedder,
        bm25_index: BM25Index,
        query_builder: QueryBuilder,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.embedder = embedder
        self.bm25_index = bm25_index
        self.query_builder = query_builder
        self.reranker = reranker

    def refresh(self) -> None:
        chunks = [chunk for chunk, _ in self.repository.get_chunks(include_embeddings=False)]
        self.bm25_index.rebuild(chunks)

    def retrieve(
        self,
        request: QueryRequest,
    ) -> tuple[str, list[ScoredCandidate], dict[str, float]]:
        timings: dict[str, float] = {}
        bundle = self.query_builder.build(request)
        top_k = request.top_k or self.settings.top_k
        filters = self._effective_filters(request, bundle.expanded_terms)

        dense_start = time.perf_counter()
        candidates = self.repository.get_chunks(filters, include_embeddings=True)
        dense_results = self._dense_search(bundle.dense_query, candidates, self.settings.dense_top_k)
        timings["dense_search"] = round((time.perf_counter() - dense_start) * 1000, 2)

        sparse_results: list[tuple[ChunkRecord, float]] = []
        sparse_start = time.perf_counter()
        if self.settings.bm25_enabled:
            sparse_candidates = [chunk for chunk, _ in candidates]
            sparse_results = self.bm25_index.search(
                bundle.lexical_query,
                sparse_candidates,
                top_k=self.settings.sparse_top_k,
            )
        timings["sparse_search"] = round((time.perf_counter() - sparse_start) * 1000, 2)

        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_results, sparse_results, self.settings.fusion_rrf_k)
        fused = apply_domain_boosts(fused, bundle, self.settings.section_boosts)
        timings["fusion"] = round((time.perf_counter() - fusion_start) * 1000, 2)

        rerank_enabled = request.enable_rerank if request.enable_rerank is not None else self.settings.reranker_enabled
        rerank_start = time.perf_counter()
        ranked = fused[: self.settings.retrieve_candidates]
        if rerank_enabled and ranked:
            reranker = self.reranker or HeuristicReranker()
            ranked = reranker.rerank(
                bundle.lexical_query,
                ranked[: self.settings.reranker_top_n],
                top_k=top_k,
            )
        else:
            ranked = ranked[:top_k]
        timings["rerank"] = round((time.perf_counter() - rerank_start) * 1000, 2)
        timings["total"] = round(sum(timings.values()), 2)
        return bundle.query_summary, ranked[:top_k], timings

    def _dense_search(
        self,
        query: str,
        candidates: list[tuple[ChunkRecord, np.ndarray | None]],
        top_k: int,
    ) -> list[tuple[ChunkRecord, float]]:
        if not candidates:
            return []
        query_vector = self.embedder.embed_query(query).astype(np.float32)
        query_norm = np.linalg.norm(query_vector) or 1.0

        scored: list[tuple[ChunkRecord, float]] = []
        for chunk, embedding in candidates:
            if embedding is None:
                continue
            denom = (np.linalg.norm(embedding) or 1.0) * query_norm
            score = float(np.dot(query_vector, embedding) / denom)
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _effective_filters(self, request: QueryRequest, expanded_terms: list[str]) -> QueryFilters | None:
        filters = request.filters.model_copy(deep=True) if request.filters else QueryFilters()
        if not filters.defect_terms and request.defect_name:
            filters.defect_terms = expanded_terms[:1]
        return filters
