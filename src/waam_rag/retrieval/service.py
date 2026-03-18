"""Hybrid retrieval orchestration."""

from __future__ import annotations

import logging
import time

import numpy as np

from waam_rag.config import Settings
from waam_rag.indexing.bm25 import BM25Index
from waam_rag.indexing.embeddings import Embedder
from waam_rag.indexing.repository import DocumentRepository
from waam_rag.retrieval.fusion import ScoredCandidate, apply_answerability_reranking, reciprocal_rank_fusion
from waam_rag.retrieval.query_builder import QueryBuilder
from waam_rag.retrieval.reranker import HeuristicReranker, Reranker
from waam_rag.schemas import ChunkRecord, QueryBundle, QueryFilters, QueryRequest, RetrievalDiagnostics

LOGGER = logging.getLogger(__name__)


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
    ) -> tuple[QueryBundle, list[ScoredCandidate], RetrievalDiagnostics]:
        bundle = self.query_builder.build(request)
        top_k = request.top_k or self.settings.top_k
        filters = self._effective_filters(request, bundle.expanded_terms)
        all_candidates = self.repository.get_chunks(filters, include_embeddings=True)
        candidate_chunks_considered = len(all_candidates)
        candidates, removed_reference_chunks = self._apply_reference_filtering(all_candidates)

        dense_result_sets: list[tuple[str, list[tuple[ChunkRecord, float]]]] = []
        sparse_result_sets: list[tuple[str, list[tuple[ChunkRecord, float]]]] = []
        subqueries = bundle.subqueries or {"primary": bundle.lexical_query}

        dense_start = time.perf_counter()
        for intent, subquery in subqueries.items():
            dense_result_sets.append(
                (
                    f"dense:{intent}",
                    self._dense_search(subquery, candidates, self.settings.dense_top_k),
                )
            )
        dense_ms = round((time.perf_counter() - dense_start) * 1000, 2)

        sparse_start = time.perf_counter()
        if self.settings.bm25_enabled:
            sparse_candidates = [chunk for chunk, _ in candidates]
            for intent, subquery in subqueries.items():
                sparse_result_sets.append(
                    (
                        f"sparse:{intent}",
                        self.bm25_index.search(
                            subquery,
                            sparse_candidates,
                            top_k=self.settings.sparse_top_k,
                        ),
                    )
                )
        sparse_ms = round((time.perf_counter() - sparse_start) * 1000, 2)

        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion([*dense_result_sets, *sparse_result_sets], self.settings.fusion_rrf_k)
        fused = apply_answerability_reranking(fused, bundle, self.settings)
        fusion_ms = round((time.perf_counter() - fusion_start) * 1000, 2)

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
        rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 2)
        diagnostics = RetrievalDiagnostics(
            dense_search_ms=dense_ms,
            sparse_search_ms=sparse_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            total_ms=round(dense_ms + sparse_ms + fusion_ms + rerank_ms, 2),
            candidate_chunks_considered=candidate_chunks_considered,
            reference_heavy_chunks_removed=removed_reference_chunks,
            subqueries_used=list(subqueries),
        )
        if self.settings.answerability_debug and ranked:
            LOGGER.info(
                "retrieval_debug",
                extra={
                    "subqueries": list(subqueries),
                    "top_chunk_id": ranked[0].chunk.chunk_id,
                    "top_feature_scores": ranked[0].feature_scores,
                    "top_debug_reasons": ranked[0].debug_reasons,
                },
            )
        return bundle, ranked[:top_k], diagnostics

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

    def _apply_reference_filtering(
        self,
        candidates: list[tuple[ChunkRecord, np.ndarray | None]],
    ) -> tuple[list[tuple[ChunkRecord, np.ndarray | None]], int]:
        if not self.settings.reference_detection_enabled:
            return candidates, 0
        kept: list[tuple[ChunkRecord, np.ndarray | None]] = []
        removed = 0
        for chunk, embedding in candidates:
            if self.settings.exclude_reference_chunks and chunk.is_reference_heavy:
                removed += 1
                continue
            kept.append((chunk, embedding))
        return kept, removed
