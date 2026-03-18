"""Hybrid fusion and metadata-aware score boosts."""

from __future__ import annotations

from dataclasses import dataclass, field

from waam_rag.config import Settings
from waam_rag.domain import PARAMETER_ALIASES
from waam_rag.schemas import ChunkRecord, QueryBundle
from waam_rag.utils.text import normalize_for_match


@dataclass
class ScoredCandidate:
    chunk: ChunkRecord
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float | None = None
    feature_scores: dict[str, float] = field(default_factory=dict)
    debug_reasons: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    result_sets: list[tuple[str, list[tuple[ChunkRecord, float]]]],
    rrf_k: int,
) -> list[ScoredCandidate]:
    """Fuse dense and sparse rankings with reciprocal rank fusion."""

    by_chunk_id: dict[str, ScoredCandidate] = {}
    for source_name, results in result_sets:
        score_field = "dense_score" if source_name.startswith("dense") else "sparse_score"
        for rank, (chunk, raw_score) in enumerate(results, start=1):
            candidate = by_chunk_id.setdefault(chunk.chunk_id, ScoredCandidate(chunk=chunk, score=0.0))
            candidate.score += 1.0 / (rrf_k + rank)
            setattr(candidate, score_field, max(getattr(candidate, score_field), raw_score))
    for candidate in by_chunk_id.values():
        candidate.feature_scores["rrf_score"] = round(candidate.score, 4)
    return sorted(by_chunk_id.values(), key=lambda item: item.score, reverse=True)


def apply_answerability_reranking(
    candidates: list[ScoredCandidate],
    query_bundle: QueryBundle,
    settings: Settings,
) -> list[ScoredCandidate]:
    """Compute feature-based final ranking scores for evidence utility."""

    dense_normalized = _normalize_feature(candidates, "dense_score")
    sparse_normalized = _normalize_feature(candidates, "sparse_score")

    for candidate in candidates:
        chunk = candidate.chunk
        normalized_text = normalize_for_match(chunk.text)
        parameter_match_ratio = _parameter_match_ratio(query_bundle.parameter_categories, chunk.process_parameters)
        direct_defect_match = _direct_defect_match(query_bundle, chunk, normalized_text)
        mitigation_signal = _signal_score(
            normalized_text,
            chunk.evidence_types,
            cue_words=("mitigation", "reduce", "reduced", "optimiz", "recommended", "minimiz"),
            target_roles=("mitigation", "recommendation", "result"),
        )
        parameter_relation = _parameter_relation_score(
            normalized_text,
            parameter_match_ratio=parameter_match_ratio,
            parameter_categories=query_bundle.parameter_categories,
        )
        experimental_signal = _signal_score(
            normalized_text,
            chunk.evidence_types,
            cue_words=("experiment", "observed", "measured", "results", "reduced porosity", "improved density"),
            target_roles=("result",),
            source_type_bonus=0.3 if chunk.review_or_experimental == "experimental" else 0.0,
        )
        mechanism_signal = _signal_score(
            normalized_text,
            chunk.evidence_types,
            cue_words=("mechanism", "cause", "due to", "because", "promotes", "gas entrapment"),
            target_roles=("cause", "mechanism"),
        )
        material_process_match = _material_process_match(query_bundle, chunk)
        reference_penalty = chunk.reference_contamination_score
        generic_background_penalty = _generic_background_penalty(chunk, settings)
        evidence_directness = chunk.evidence_directness_score
        section_boost = _section_boost(chunk.section, settings.section_boosts)

        dense_score = dense_normalized[candidate.chunk.chunk_id]
        sparse_score = sparse_normalized[candidate.chunk.chunk_id]
        final_score = (
            settings.dense_score_weight * dense_score
            + settings.sparse_score_weight * sparse_score
            + settings.direct_defect_match_weight * direct_defect_match
            + settings.mitigation_signal_weight * mitigation_signal
            + settings.parameter_relation_weight * parameter_relation
            + settings.experimental_evidence_weight * experimental_signal
            + settings.mechanism_signal_weight * mechanism_signal
            + settings.material_process_match_weight * material_process_match
            + settings.evidence_directness_weight * evidence_directness
            + section_boost
            - settings.reference_penalty_weight * reference_penalty
            - settings.generic_background_penalty_weight * generic_background_penalty
        )

        candidate.feature_scores = {
            **candidate.feature_scores,
            "dense_score": round(dense_score, 4),
            "sparse_score": round(sparse_score, 4),
            "direct_defect_match_score": round(direct_defect_match, 4),
            "mitigation_signal_score": round(mitigation_signal, 4),
            "parameter_relation_score": round(parameter_relation, 4),
            "experimental_evidence_score": round(experimental_signal, 4),
            "mechanism_signal_score": round(mechanism_signal, 4),
            "material_process_match_score": round(material_process_match, 4),
            "evidence_directness_score": round(evidence_directness, 4),
            "reference_penalty_score": round(reference_penalty, 4),
            "generic_background_penalty": round(generic_background_penalty, 4),
            "section_boost": round(section_boost, 4),
            "final_score": round(final_score, 4),
        }
        candidate.debug_reasons = _build_debug_reasons(candidate.feature_scores)
        candidate.score = final_score
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _normalize_feature(candidates: list[ScoredCandidate], field_name: str) -> dict[str, float]:
    raw_scores = [getattr(candidate, field_name) for candidate in candidates]
    min_score = min(raw_scores, default=0.0)
    max_score = max(raw_scores, default=0.0)
    denom = max(max_score - min_score, 1e-6)
    return {
        candidate.chunk.chunk_id: (getattr(candidate, field_name) - min_score) / denom if raw_scores else 0.0
        for candidate in candidates
    }


def _direct_defect_match(query_bundle: QueryBundle, chunk: ChunkRecord, normalized_text: str) -> float:
    if query_bundle.defect_name and query_bundle.defect_name in chunk.defect_terms:
        return 1.0
    if any(term and term in normalized_text for term in query_bundle.expanded_terms):
        return 0.6
    return 0.0


def _signal_score(
    normalized_text: str,
    evidence_types: list[str],
    *,
    cue_words: tuple[str, ...],
    target_roles: tuple[str, ...],
    source_type_bonus: float = 0.0,
) -> float:
    keyword_hits = sum(1 for cue in cue_words if cue in normalized_text)
    role_hits = len(set(target_roles).intersection(evidence_types))
    score = min(keyword_hits / max(len(cue_words), 1), 0.6) + min(role_hits * 0.3, 0.6) + source_type_bonus
    return max(0.0, min(score, 1.0))


def _parameter_match_ratio(requested_parameters: list[str], chunk_parameters: list[str]) -> float:
    if not requested_parameters:
        return 0.0
    overlap = set(requested_parameters).intersection(chunk_parameters)
    return len(overlap) / max(len(set(requested_parameters)), 1)


def _parameter_relation_score(
    normalized_text: str,
    *,
    parameter_match_ratio: float,
    parameter_categories: list[str],
) -> float:
    relation_terms = ("increase", "decrease", "higher", "lower", "too high", "too low", "window", "range", "effect")
    relation_hits = sum(1 for term in relation_terms if term in normalized_text)
    alias_hits = 0
    for parameter in parameter_categories:
        aliases = PARAMETER_ALIASES.get(parameter, ())
        if any(alias in normalized_text for alias in aliases):
            alias_hits += 1
    alias_score = alias_hits / max(len(parameter_categories), 1) if parameter_categories else 0.0
    relation_score = min(relation_hits / len(relation_terms), 0.5)
    return max(0.0, min(parameter_match_ratio * 0.5 + alias_score * 0.3 + relation_score * 0.2, 1.0))


def _material_process_match(query_bundle: QueryBundle, chunk: ChunkRecord) -> float:
    score = 0.0
    if query_bundle.process_types and set(query_bundle.process_types).intersection(chunk.process_types):
        score += 0.6
    if query_bundle.materials and set(query_bundle.materials).intersection(chunk.materials):
        score += 0.4
    return max(0.0, min(score, 1.0))


def _generic_background_penalty(chunk: ChunkRecord, settings: Settings) -> float:
    section_key = normalize_for_match(chunk.section or "")
    intro_penalty = 0.2 if any(token in section_key for token in ("introduction", "abstract", "background")) else 0.0
    return max(0.0, min(chunk.generic_background_score + intro_penalty, 1.0))


def _section_boost(section: str | None, section_boosts: dict[str, float]) -> float:
    section_key = normalize_for_match(section or "")
    for label, boost in section_boosts.items():
        if label in section_key:
            return boost
    return 0.0


def _build_debug_reasons(feature_scores: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if feature_scores.get("direct_defect_match_score", 0.0) >= 0.8:
        reasons.append("direct defect match")
    if feature_scores.get("mitigation_signal_score", 0.0) >= 0.35:
        reasons.append("mitigation signal")
    if feature_scores.get("parameter_relation_score", 0.0) >= 0.3:
        reasons.append("parameter relation")
    if feature_scores.get("experimental_evidence_score", 0.0) >= 0.3:
        reasons.append("experimental evidence")
    if feature_scores.get("mechanism_signal_score", 0.0) >= 0.25:
        reasons.append("mechanism signal")
    if feature_scores.get("reference_penalty_score", 0.0) >= 0.5:
        reasons.append("reference penalty")
    if feature_scores.get("generic_background_penalty", 0.0) >= 0.35:
        reasons.append("generic background penalty")
    return reasons
