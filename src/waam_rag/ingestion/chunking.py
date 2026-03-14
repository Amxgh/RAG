"""Scientific chunking tuned for research-paper retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from waam_rag.schemas import ChunkRecord, ParsedDocument, StructuredBlock
from waam_rag.utils.text import estimate_token_count, page_list, sentence_split, truncate


@dataclass
class _ChunkFragment:
    text: str
    page_number: int
    section: str | None
    subsection: str | None
    token_count: int


class ScientificChunker:
    """Create section-aware chunks that preserve coherent reasoning units."""

    def __init__(
        self,
        chunk_size_tokens: int = 700,
        chunk_overlap_tokens: int = 120,
        generate_chunk_summary: bool = False,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.generate_chunk_summary = generate_chunk_summary

    def chunk_document(
        self,
        document: ParsedDocument,
        blocks: list[StructuredBlock],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        buffer: list[_ChunkFragment] = []
        buffer_tokens = 0
        current_key: tuple[str | None, str | None] | None = None
        chunk_index = 0

        for block in blocks:
            if block.is_heading or block.excluded or not block.text.strip():
                continue
            key = (block.section, block.subsection)
            fragments = self._fragment_block(block)
            for fragment in fragments:
                if buffer and (key != current_key or buffer_tokens + fragment.token_count > self.chunk_size_tokens):
                    chunks.append(self._build_chunk(document, buffer, chunk_index))
                    chunk_index += 1
                    if key == current_key:
                        buffer = self._overlap_fragments(buffer)
                        buffer_tokens = sum(item.token_count for item in buffer)
                    else:
                        buffer = []
                        buffer_tokens = 0

                buffer.append(fragment)
                buffer_tokens += fragment.token_count
                current_key = key

        if buffer:
            chunks.append(self._build_chunk(document, buffer, chunk_index))
        return chunks

    def _fragment_block(self, block: StructuredBlock) -> list[_ChunkFragment]:
        token_count = estimate_token_count(block.text)
        if token_count <= self.chunk_size_tokens:
            return [
                _ChunkFragment(
                    text=block.text,
                    page_number=block.page_number,
                    section=block.section,
                    subsection=block.subsection,
                    token_count=token_count,
                )
            ]

        fragments: list[_ChunkFragment] = []
        sentence_buffer: list[str] = []
        buffer_tokens = 0
        for sentence in sentence_split(block.text):
            sentence_tokens = estimate_token_count(sentence)
            if sentence_buffer and buffer_tokens + sentence_tokens > self.chunk_size_tokens:
                text = " ".join(sentence_buffer).strip()
                fragments.append(
                    _ChunkFragment(
                        text=text,
                        page_number=block.page_number,
                        section=block.section,
                        subsection=block.subsection,
                        token_count=estimate_token_count(text),
                    )
                )
                sentence_buffer = []
                buffer_tokens = 0
            sentence_buffer.append(sentence)
            buffer_tokens += sentence_tokens
        if sentence_buffer:
            text = " ".join(sentence_buffer).strip()
            fragments.append(
                _ChunkFragment(
                    text=text,
                    page_number=block.page_number,
                    section=block.section,
                    subsection=block.subsection,
                    token_count=estimate_token_count(text),
                )
            )
        return fragments

    def _overlap_fragments(self, fragments: list[_ChunkFragment]) -> list[_ChunkFragment]:
        selected: list[_ChunkFragment] = []
        token_total = 0
        for fragment in reversed(fragments):
            selected.insert(0, fragment)
            token_total += fragment.token_count
            if token_total >= self.chunk_overlap_tokens:
                break
        return selected

    def _build_chunk(
        self,
        document: ParsedDocument,
        fragments: list[_ChunkFragment],
        chunk_index: int,
    ) -> ChunkRecord:
        start_page = min(fragment.page_number for fragment in fragments)
        end_page = max(fragment.page_number for fragment in fragments)
        page_numbers = sorted({fragment.page_number for fragment in fragments})
        section = fragments[0].section
        subsection = fragments[0].subsection
        text = "\n\n".join(fragment.text for fragment in fragments).strip()
        summary = truncate(text, 180) if self.generate_chunk_summary else None
        return ChunkRecord(
            chunk_id=f"{document.doc_id}:{chunk_index:04d}",
            doc_id=document.doc_id,
            source_file=document.source_file,
            title=document.title,
            authors=document.authors,
            year=document.year,
            venue=document.venue,
            start_page=start_page,
            end_page=end_page,
            page_numbers=page_numbers or page_list(start_page, end_page),
            section=section,
            subsection=subsection,
            text=text,
            token_count=estimate_token_count(text),
            summary=summary,
            metadata={"page_range": [start_page, end_page]},
        )
