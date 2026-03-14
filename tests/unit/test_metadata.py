from __future__ import annotations

from waam_rag.ingestion.enrichment import MetadataEnricher
from waam_rag.schemas import ChunkRecord


def test_metadata_enricher_detects_domain_terms() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_file="paper.pdf",
        title="Paper",
        authors=["Alice Smith"],
        year=2022,
        venue="WAAM Journal",
        start_page=2,
        end_page=2,
        page_numbers=[2],
        section="Results and Discussion",
        subsection="Mitigation Guidance",
        text=(
            "Porosity in WAAM aluminum was caused by unstable shielding gas and high heat input. "
            "The results show that stable argon flow and lower current mitigate pore formation."
        ),
        token_count=32,
    )

    enriched = MetadataEnricher().enrich_chunk(chunk)

    assert "porosity" in enriched.defect_terms
    assert "waam" in enriched.process_types
    assert "aluminum" in enriched.materials
    assert "shielding_gas" in enriched.process_parameters
    assert "heat_input" in enriched.process_parameters
    assert "mitigation" in enriched.evidence_types
