"""Citation formatting utilities."""

from __future__ import annotations

from waam_rag.schemas import ChunkRecord


def format_citation(chunk: ChunkRecord, style: str = "author_year_pages") -> str:
    """Build a citation string from chunk metadata."""

    pages = _format_pages(chunk.page_numbers or [chunk.start_page, chunk.end_page])
    if style == "title_pages":
        label = chunk.title or chunk.source_file.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        return f"[{label}, {pages}]"

    author_label = _author_label(chunk.authors, fallback=chunk.title or chunk.source_file)
    if style == "author_year":
        if chunk.year:
            return f"[{author_label}, {chunk.year}]"
        return f"[{author_label}]"

    if chunk.year:
        return f"[{author_label}, {chunk.year}, {pages}]"
    return f"[{author_label}, {pages}]"


def format_short_citation(chunk: ChunkRecord, style: str = "author_year_pages") -> str:
    """Build a short citation variant for compact context packs."""

    if style == "title_pages":
        label = chunk.title or "Untitled paper"
        return f"[{label}, p. {chunk.start_page}]"
    author_label = _author_label(chunk.authors, fallback=chunk.title or "Untitled paper")
    if chunk.year:
        return f"[{author_label}, {chunk.year}]"
    return f"[{author_label}]"


def _author_label(authors: list[str], fallback: str) -> str:
    if not authors:
        return fallback
    surname = authors[0].split()[-1]
    if len(authors) > 1:
        return f"{surname} et al."
    return surname


def _format_pages(pages: list[int]) -> str:
    unique = sorted(set(pages))
    if not unique:
        return "p. ?"
    if len(unique) == 1:
        return f"p. {unique[0]}"
    return f"pp. {unique[0]}-{unique[-1]}"
