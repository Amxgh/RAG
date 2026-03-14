from __future__ import annotations

from waam_rag.citations import format_citation, format_short_citation
from waam_rag.schemas import ChunkRecord


def test_citation_formatting_is_page_aware() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_file="paper.pdf",
        title="A Study on Porosity",
        authors=["Alice Smith", "Bob Lee"],
        year=2022,
        venue="WAAM Journal",
        start_page=4,
        end_page=5,
        page_numbers=[4, 5],
        section="Results",
        subsection=None,
        text="Porosity mitigation text.",
        token_count=4,
    )

    assert format_citation(chunk) == "[Smith et al., 2022, pp. 4-5]"
    assert format_short_citation(chunk) == "[Smith et al., 2022]"
