"""Text cleaning helpers for PDF extracted pages."""

from __future__ import annotations

import math
import re
from collections import Counter

from waam_rag.schemas import PageText
from waam_rag.utils.text import normalize_whitespace


class TextCleaner:
    """Remove marginal noise and repair broken PDF text."""

    def clean_pages(self, pages: list[PageText]) -> list[PageText]:
        headers, footers = self._detect_repeated_marginals(pages)
        cleaned_pages: list[PageText] = []
        for page in pages:
            cleaned = self._clean_page_text(page.text, headers, footers)
            cleaned_pages.append(PageText(page_number=page.page_number, text=cleaned))
        return cleaned_pages

    def _detect_repeated_marginals(self, pages: list[PageText]) -> tuple[set[str], set[str]]:
        top_counter: Counter[str] = Counter()
        bottom_counter: Counter[str] = Counter()
        for page in pages:
            lines = [self._normalize_line(line) for line in page.text.splitlines() if self._normalize_line(line)]
            for candidate in lines[:2]:
                if not self._is_probable_page_number(candidate):
                    top_counter[candidate] += 1
            for candidate in lines[-2:]:
                if not self._is_probable_page_number(candidate):
                    bottom_counter[candidate] += 1

        threshold = max(2, math.ceil(len(pages) * 0.35))
        headers = {line for line, count in top_counter.items() if count >= threshold and len(line) < 120}
        footers = {line for line, count in bottom_counter.items() if count >= threshold and len(line) < 120}
        return headers, footers

    def _clean_page_text(self, text: str, headers: set[str], footers: set[str]) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        raw_lines = text.splitlines()
        filtered_lines: list[str] = []
        for line in raw_lines:
            normalized = self._normalize_line(line)
            if not normalized:
                filtered_lines.append("")
                continue
            if normalized in headers or normalized in footers:
                continue
            if self._is_probable_page_number(normalized):
                continue
            filtered_lines.append(normalized)

        paragraphs: list[str] = []
        buffer: list[str] = []
        for line in filtered_lines:
            if not line:
                if buffer:
                    paragraphs.append(self._merge_lines(buffer))
                    buffer = []
                continue
            if self._looks_like_heading(line):
                if buffer:
                    paragraphs.append(self._merge_lines(buffer))
                    buffer = []
                paragraphs.append(line)
                continue
            buffer.append(line)
        if buffer:
            paragraphs.append(self._merge_lines(buffer))

        return normalize_whitespace("\n\n".join(paragraphs))

    def _merge_lines(self, lines: list[str]) -> str:
        merged = ""
        for line in lines:
            if not merged:
                merged = line
            elif merged.endswith("-"):
                merged = f"{merged[:-1]}{line}"
            else:
                merged = f"{merged} {line}"
        return normalize_whitespace(merged)

    def _normalize_line(self, line: str) -> str:
        return normalize_whitespace(line)

    def _looks_like_heading(self, line: str) -> bool:
        if len(line.split()) > 14:
            return False
        if len(line) > 120:
            return False
        if line.endswith("."):
            return False
        return bool(
            re.match(r"^(?:\d+(?:\.\d+)*)?\s*[A-Z][A-Za-z0-9 ,/\-&()]+$", line)
            or line.isupper()
        )

    def _is_probable_page_number(self, line: str) -> bool:
        lowered = line.lower()
        if re.fullmatch(r"\d{1,4}", lowered):
            return True
        if re.fullmatch(r"page\s+\d{1,4}", lowered):
            return True
        if re.fullmatch(r"[ivxlcdm]+", lowered):
            return True
        return False
