"""Hybrid fusion and metadata-aware score boosts."""

from __future__ import annotations

from dataclasses import dataclass

from waam_rag.schemas import ChunkRecord, QueryBundle
from waam_rag.utils.text import normalize_for_match


@dataclass
class ScoredCandidate:
    chunk: ChunkRecord
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    dense_results: list[tuple[ChunkRecord, float]],
    sparse_results: list[tuple[ChunkRecord, float]],
    rrf_k: int,
) -> list[ScoredCandidate]:
    """Fuse dense and sparse rankings with reciprocal rank fusion."""

    by_chunk_id: dict[str, ScoredCandidate] = {}
    for results, field in ((dense_results, "dense_score"), (sparse_results, "sparse_score")):
        for rank, (chunk, raw_score) in enumerate(results, start=1):
            candidate = by_chunk_id.setdefault(chunk.chunk_id, ScoredCandidate(chunk=chunk, score=0.0))
            candidate.score += 1.0 / (rrf_k + rank)
            setattr(candidate, field, raw_score)
    return sorted(by_chunk_id.values(), key=lambda item: item.score, reverse=True)


def apply_domain_boosts(
    candidates: list[ScoredCandidate],
    query_bundle: QueryBundle,
    section_boosts: dict[str, float],
) -> list[ScoredCandidate]:
    """Boost chunks that align with the defect, process metadata, and section quality."""

    need_mitigation = any(
        token in query_bundle.lexical_query.lower()
        for token in ("mitigation", "reduce", "avoid", "strategy", "recommend")
    )
    query_params = {
        fragment.split()[0].lower().replace(" ", "_")
        for fragment in query_bundle.process_fragments
        if fragment
    }

    for candidate in candidates:
        chunk = candidate.chunk
        if query_bundle.defect_name and query_bundle.defect_name in chunk.defect_terms:
            candidate.score += 0.15
        if query_bundle.process_types and set(query_bundle.process_types).intersection(chunk.process_types):
            candidate.score += 0.08
        if query_bundle.materials and set(query_bundle.materials).intersection(chunk.materials):
            candidate.score += 0.06
        if query_params:
            matches = len(query_params.intersection(chunk.process_parameters))
            candidate.score += min(matches * 0.03, 0.12)
        if need_mitigation and {"mitigation", "recommendation", "result"}.intersection(chunk.evidence_types):
            candidate.score += 0.08

        section_key = normalize_for_match(chunk.section or "")
        for label, boost in section_boosts.items():
            if label in section_key:
                candidate.score += boost
                break
    return sorted(candidates, key=lambda item: item.score, reverse=True)
