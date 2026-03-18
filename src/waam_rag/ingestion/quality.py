"""Chunk-quality analysis for reference contamination and evidence readiness."""

from __future__ import annotations

import re

from waam_rag.config import Settings
from waam_rag.domain import GENERIC_BACKGROUND_PATTERNS, REFERENCE_LIKE_PATTERNS, REFERENCE_SECTION_PATTERNS
from waam_rag.schemas import ChunkRecord
from waam_rag.utils.text import estimate_token_count, normalize_for_match, normalize_whitespace, sentence_split


class ChunkQualityAnalyzer:
    """Trim bibliography tails and annotate chunks with quality signals."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def process_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        return [self.process_chunk(chunk) for chunk in chunks]

    def process_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        cleaned_text, reference_tail_trimmed = self._trim_reference_tail(chunk.text)
        reference_score = self.reference_contamination_score(
            cleaned_text,
            section=chunk.section,
            subsection=chunk.subsection,
        )
        generic_background_score = self.generic_background_score(cleaned_text)
        updated_metadata = {
            **chunk.metadata,
            "reference_tail_trimmed": reference_tail_trimmed,
            "reference_contamination_score": round(reference_score, 4),
            "generic_background_score": round(generic_background_score, 4),
        }
        return chunk.model_copy(
            update={
                "text": cleaned_text,
                "token_count": estimate_token_count(cleaned_text),
                "reference_contamination_score": reference_score,
                "is_reference_heavy": reference_score >= self.settings.reference_exclusion_threshold,
                "generic_background_score": generic_background_score,
                "metadata": updated_metadata,
            }
        )

    def reference_contamination_score(
        self,
        text: str,
        *,
        section: str | None = None,
        subsection: str | None = None,
    ) -> float:
        if not self.settings.reference_detection_enabled:
            return 0.0

        normalized = normalize_for_match(text)
        lower_section = normalize_for_match(" ".join(part for part in (section, subsection) if part))
        section_signal = 1.0 if any(token in lower_section for token in REFERENCE_SECTION_PATTERNS) else 0.0

        doi_hits = len(re.findall(r"\bdoi\b", normalized))
        url_hits = len(re.findall(r"https?://", text.lower()))
        etal_hits = len(re.findall(r"\bet al\b", normalized))
        year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
        semicolon_hits = text.count(";")
        citation_marker_hits = len(re.findall(r"\[[0-9,\-\s]+\]|\([A-Z][A-Za-z]+,\s*(?:19|20)\d{2}\)", text))
        author_year_hits = len(re.findall(r"[A-Z][A-Za-z\-]+(?:\s+et al\.)?,?\s*\(?\d{4}\)?", text))
        explanatory_ratio = self._explanatory_sentence_ratio(text)

        score = (
            0.3 * section_signal
            + 0.18 * min(doi_hits, 2) / 2
            + 0.12 * min(url_hits, 2) / 2
            + 0.16 * min(etal_hits, 3) / 3
            + 0.1 * min(semicolon_hits, 6) / 6
            + 0.12 * min(citation_marker_hits, 4) / 4
            + 0.16 * min(author_year_hits, 4) / 4
            + 0.08 * min(year_hits, 6) / 6
            + 0.08 * (1 - explanatory_ratio)
        )
        if doi_hits and (etal_hits >= 2 or author_year_hits >= 2):
            score = max(score, 0.8)
        return max(0.0, min(score, 1.0))

    def generic_background_score(self, text: str) -> float:
        normalized = normalize_for_match(text)
        hits = sum(1 for pattern in GENERIC_BACKGROUND_PATTERNS if re.search(pattern, normalized))
        sentences = sentence_split(text)
        if not sentences:
            return 0.0
        return max(0.0, min((hits / max(len(sentences), 1)) * 1.5, 1.0))

    def _trim_reference_tail(self, text: str) -> tuple[str, bool]:
        if not self.settings.post_chunk_trim_reference_tails:
            return normalize_whitespace(text), False

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        if not paragraphs:
            return normalize_whitespace(text), False

        trimmed = False
        keep = list(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            normalized = normalize_for_match(paragraph)
            if any(token in normalized for token in REFERENCE_SECTION_PATTERNS):
                keep = paragraphs[:index]
                trimmed = True
                break

        while keep and self._paragraph_reference_score(keep[-1]) >= 0.7:
            keep.pop()
            trimmed = True

        if trimmed and not keep:
            return normalize_whitespace(text), False

        cleaned = normalize_whitespace("\n\n".join(keep or paragraphs))
        return cleaned, trimmed

    def _paragraph_reference_score(self, paragraph: str) -> float:
        normalized = normalize_for_match(paragraph)
        pattern_hits = sum(len(re.findall(pattern, normalized)) for pattern in REFERENCE_LIKE_PATTERNS)
        year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", paragraph))
        semicolon_hits = paragraph.count(";")
        explanatory_ratio = self._explanatory_sentence_ratio(paragraph)
        score = (
            0.4 * min(pattern_hits, 4) / 4
            + 0.25 * min(year_hits, 4) / 4
            + 0.2 * min(semicolon_hits, 4) / 4
            + 0.15 * (1 - explanatory_ratio)
        )
        return max(0.0, min(score, 1.0))

    def _explanatory_sentence_ratio(self, text: str) -> float:
        sentences = sentence_split(text)
        if not sentences:
            return 0.0
        explanatory_hits = 0
        for sentence in sentences:
            lowered = normalize_for_match(sentence)
            if re.search(r"\b(is|are|can|may|should|results|observed|reduce|increase|because|due to)\b", lowered):
                explanatory_hits += 1
        return explanatory_hits / len(sentences)
