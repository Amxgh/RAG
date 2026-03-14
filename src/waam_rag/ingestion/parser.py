"""PDF parsing strategies for research papers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from waam_rag.schemas import PageText, ParsedDocument
from waam_rag.utils.files import sha1_file, stable_doc_id
from waam_rag.utils.text import normalize_whitespace

LOGGER = logging.getLogger(__name__)


class ParserError(RuntimeError):
    """Raised when a PDF cannot be parsed by any available backend."""


class ResearchPaperParser:
    """Parse PDFs using a primary extractor with fallbacks."""

    def __init__(self, enable_ocr_fallback: bool = False) -> None:
        self.enable_ocr_fallback = enable_ocr_fallback

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        attempts: list[tuple[str, Exception]] = []
        for name, parser in (("pypdf", self._parse_with_pypdf), ("pymupdf", self._parse_with_pymupdf)):
            try:
                pages, metadata = parser(path)
                if not pages:
                    raise ParserError(f"{name} extracted no pages")
                return self._build_document(path, pages, metadata)
            except Exception as exc:  # pragma: no cover - backend specific
                attempts.append((name, exc))
                LOGGER.warning("pdf_parse_backend_failed", extra={"backend": name, "file": str(path), "error": str(exc)})

        if self.enable_ocr_fallback:
            raise ParserError(
                f"OCR fallback is enabled but not implemented. Backend errors: {self._format_attempts(attempts)}"
            )
        raise ParserError(f"Failed to parse {path.name}: {self._format_attempts(attempts)}")

    def _parse_with_pypdf(self, path: Path) -> tuple[list[PageText], dict[str, Any]]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [
            PageText(page_number=index + 1, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages)
        ]
        metadata = {key.lstrip("/").lower(): value for key, value in (reader.metadata or {}).items()}
        return pages, metadata

    def _parse_with_pymupdf(self, path: Path) -> tuple[list[PageText], dict[str, Any]]:
        import fitz

        document = fitz.open(path)
        pages = [
            PageText(page_number=index + 1, text=document.load_page(index).get_text("text"))
            for index in range(document.page_count)
        ]
        return pages, document.metadata or {}

    def _build_document(
        self,
        path: Path,
        pages: list[PageText],
        raw_metadata: dict[str, Any],
    ) -> ParsedDocument:
        checksum = sha1_file(path)
        doc_id = stable_doc_id(path)
        title = self._infer_title(raw_metadata, pages, path)
        authors = self._infer_authors(raw_metadata, pages)
        year = self._infer_year(raw_metadata, pages, path)
        venue = self._infer_venue(raw_metadata, pages)
        return ParsedDocument(
            doc_id=doc_id,
            checksum=checksum,
            source_file=str(path.resolve()),
            file_name=path.name,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            page_count=len(pages),
            raw_metadata=raw_metadata,
            pages=pages,
        )

    def _infer_title(
        self, raw_metadata: dict[str, Any], pages: list[PageText], path: Path
    ) -> str | None:
        metadata_title = normalize_whitespace(str(raw_metadata.get("title") or ""))
        if metadata_title:
            return metadata_title

        first_page_lines = self._front_matter_lines(pages)
        if not first_page_lines:
            return path.stem
        title_candidates: list[str] = []
        for line in first_page_lines[:6]:
            lowered = line.lower()
            if lowered.startswith("abstract"):
                break
            if self._looks_like_author_line(line):
                break
            if 6 < len(line) < 200:
                title_candidates.append(line)
        if title_candidates:
            return " ".join(title_candidates[:2]).strip()
        return first_page_lines[0]

    def _infer_authors(self, raw_metadata: dict[str, Any], pages: list[PageText]) -> list[str]:
        metadata_author = normalize_whitespace(str(raw_metadata.get("author") or ""))
        if metadata_author:
            return self._split_authors(metadata_author)

        lines = self._front_matter_lines(pages)
        if len(lines) < 2:
            return []
        for candidate in lines[1:5]:
            if "abstract" in candidate.lower():
                break
            if "," in candidate or " and " in candidate.lower():
                return self._split_authors(candidate)
        return []

    def _infer_year(
        self, raw_metadata: dict[str, Any], pages: list[PageText], path: Path
    ) -> int | None:
        for value in (
            raw_metadata.get("creationdate"),
            raw_metadata.get("moddate"),
            raw_metadata.get("subject"),
        ):
            if value:
                match = re.search(r"(19|20)\d{2}", str(value))
                if match:
                    return int(match.group())
        front_text = "\n".join(self._front_matter_lines(pages)[:15])
        for source in (front_text, path.stem):
            match = re.search(r"(19|20)\d{2}", source)
            if match:
                return int(match.group())
        return None

    def _infer_venue(self, raw_metadata: dict[str, Any], pages: list[PageText]) -> str | None:
        for key in ("subject", "keywords", "producer"):
            value = normalize_whitespace(str(raw_metadata.get(key) or ""))
            if value:
                return value
        lines = self._front_matter_lines(pages)
        for candidate in lines[:12]:
            lowered = candidate.lower()
            if any(token in lowered for token in ("journal", "conference", "proceedings", "transactions")):
                return candidate
        return None

    def _front_matter_lines(self, pages: list[PageText]) -> list[str]:
        if not pages:
            return []
        lines = [normalize_whitespace(line) for line in pages[0].text.splitlines()]
        return [line for line in lines if line]

    def _split_authors(self, raw_text: str) -> list[str]:
        cleaned = raw_text.replace(" and ", ",")
        return [part.strip() for part in cleaned.split(",") if part.strip()]

    def _format_attempts(self, attempts: list[tuple[str, Exception]]) -> str:
        return "; ".join(f"{name}: {exc}" for name, exc in attempts) or "no parser attempts"

    def _looks_like_author_line(self, line: str) -> bool:
        if "," in line:
            parts = [part.strip() for part in line.split(",") if part.strip()]
            return len(parts) >= 2 and all(" " in part for part in parts)
        tokens = line.split()
        if 2 <= len(tokens) <= 6 and tokens[0][0].isupper() and tokens[1][0].isupper():
            return "abstract" not in line.lower()
        return False
