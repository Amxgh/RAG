from __future__ import annotations

from waam_rag.ingestion.chunking import ScientificChunker
from waam_rag.schemas import ParsedDocument, StructuredBlock


def test_chunker_preserves_sections_pages_and_overlap() -> None:
    document = ParsedDocument(
        doc_id="doc-1",
        checksum="abc",
        source_file="paper.pdf",
        file_name="paper.pdf",
        title="Paper",
        authors=["Alice Smith"],
        year=2024,
        venue="WAAM Conf",
        page_count=2,
        pages=[],
    )
    blocks = [
        StructuredBlock(
            text="Porosity is caused by unstable shielding gas and excessive heat input in WAAM aluminum builds.",
            page_number=1,
            section="Results and Discussion",
            subsection=None,
        ),
        StructuredBlock(
            text="Reducing heat input and stabilizing wire feed speed lowered pore formation in the experiments.",
            page_number=1,
            section="Results and Discussion",
            subsection=None,
        ),
        StructuredBlock(
            text="These mitigation findings were consistent across multiple deposition trials on page two.",
            page_number=2,
            section="Results and Discussion",
            subsection=None,
        ),
    ]

    chunker = ScientificChunker(chunk_size_tokens=18, chunk_overlap_tokens=6)
    chunks = chunker.chunk_document(document, blocks)

    assert len(chunks) >= 2
    assert chunks[0].section == "Results and Discussion"
    assert chunks[0].start_page == 1
    assert chunks[-1].end_page == 2
    assert "heat input" in chunks[1].text
