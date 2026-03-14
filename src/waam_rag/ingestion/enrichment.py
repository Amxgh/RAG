"""Heuristic metadata enrichment for defect-oriented retrieval."""

from __future__ import annotations

import re

from waam_rag.domain import (
    DEFECT_SYNONYMS,
    EVIDENCE_ROLE_PATTERNS,
    MATERIAL_PATTERNS,
    PROCESS_PARAMETER_PATTERNS,
    PROCESS_TYPES,
)
from waam_rag.schemas import ChunkRecord
from waam_rag.utils.text import normalize_for_match


class MetadataEnricher:
    """Attach lightweight domain metadata to chunks."""

    def enrich_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        return [self.enrich_chunk(chunk) for chunk in chunks]

    def enrich_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        normalized = normalize_for_match(chunk.text)
        defect_terms = self._match_taxonomy(normalized, DEFECT_SYNONYMS)
        process_types = self._match_taxonomy(normalized, PROCESS_TYPES)
        materials = self._match_taxonomy(normalized, MATERIAL_PATTERNS)
        parameter_mentions = self._extract_parameter_mentions(normalized)
        evidence_types = self._extract_evidence_types(normalized)
        return chunk.model_copy(
            update={
                "defect_terms": defect_terms,
                "process_types": process_types,
                "materials": materials,
                "process_parameters": sorted(parameter_mentions),
                "parameter_mentions": parameter_mentions,
                "evidence_types": evidence_types,
                "metadata": {
                    **chunk.metadata,
                    "defect_terms": defect_terms,
                    "process_types": process_types,
                    "materials": materials,
                    "process_parameters": sorted(parameter_mentions),
                    "evidence_types": evidence_types,
                },
            }
        )

    def _match_taxonomy(self, normalized_text: str, taxonomy: dict[str, list[str]]) -> list[str]:
        hits: list[str] = []
        for canonical, variants in taxonomy.items():
            if any(re.search(pattern, normalized_text) for pattern in variants):
                hits.append(canonical)
        return sorted(set(hits))

    def _extract_parameter_mentions(self, normalized_text: str) -> dict[str, list[str]]:
        mentions: dict[str, list[str]] = {}
        for parameter, patterns in PROCESS_PARAMETER_PATTERNS.items():
            hits: list[str] = []
            for pattern in patterns:
                hits.extend(match.group(0) for match in re.finditer(pattern, normalized_text))
            if hits:
                mentions[parameter] = sorted(set(hits))
        return mentions

    def _extract_evidence_types(self, normalized_text: str) -> list[str]:
        labels: list[str] = []
        for label, patterns in EVIDENCE_ROLE_PATTERNS.items():
            if any(re.search(pattern, normalized_text) for pattern in patterns):
                labels.append(label)
        return sorted(set(labels))
