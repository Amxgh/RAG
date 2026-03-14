"""Section-aware structure extraction for research papers."""

from __future__ import annotations

import re

from waam_rag.domain import SECTION_KEYWORDS
from waam_rag.schemas import PageText, StructuredBlock
from waam_rag.utils.text import normalize_for_match, normalize_whitespace


class StructureExtractor:
    """Detect section headings and attach them to paragraph blocks."""

    def __init__(self, exclude_references: bool = True) -> None:
        self.exclude_references = exclude_references

    def extract_blocks(self, pages: list[PageText]) -> list[StructuredBlock]:
        blocks: list[StructuredBlock] = []
        current_section = "Front Matter"
        current_subsection: str | None = None

        for page in pages:
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", page.text) if part.strip()]
            for paragraph in paragraphs:
                if self._is_heading(paragraph):
                    heading = normalize_whitespace(paragraph)
                    heading_level = self._heading_level(heading)
                    canonical = self.canonical_section_name(heading)
                    if heading_level == 1:
                        current_section = heading
                        current_subsection = None
                    else:
                        current_subsection = heading
                    excluded = self.exclude_references and canonical == "references"
                    blocks.append(
                        StructuredBlock(
                            text=heading,
                            page_number=page.page_number,
                            section=current_section,
                            subsection=current_subsection,
                            is_heading=True,
                            excluded=excluded,
                        )
                    )
                    continue

                excluded = self.exclude_references and self.canonical_section_name(current_section) == "references"
                blocks.append(
                    StructuredBlock(
                        text=normalize_whitespace(paragraph),
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                        is_heading=False,
                        excluded=excluded,
                    )
                )
        return blocks

    def canonical_section_name(self, value: str | None) -> str:
        normalized = normalize_for_match(value or "")
        for canonical, variants in SECTION_KEYWORDS.items():
            if any(variant in normalized for variant in variants):
                return canonical
        return normalized

    def _is_heading(self, paragraph: str) -> bool:
        normalized = normalize_whitespace(paragraph)
        if len(normalized.split()) > 14:
            return False
        if len(normalized) > 140:
            return False
        if normalized.endswith("."):
            return False
        if self.canonical_section_name(normalized) in SECTION_KEYWORDS:
            return True
        return bool(
            re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,/\-&()]+$", normalized)
            or re.match(r"^[A-Z][A-Za-z0-9 ,/\-&()]+$", normalized)
            or normalized.isupper()
        )

    def _heading_level(self, heading: str) -> int:
        if re.match(r"^\d+\.\d+", heading):
            return 2
        if re.match(r"^\d+\s+", heading):
            return 1
        return 1 if len(heading.split()) <= 6 else 2
