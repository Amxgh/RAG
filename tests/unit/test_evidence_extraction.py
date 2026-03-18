from __future__ import annotations

from waam_rag.retrieval.extraction import EvidenceExtractor
from waam_rag.schemas import ChunkRecord, QueryBundle


def test_structured_evidence_extraction_returns_strategy_and_citation() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_file="paper.pdf",
        title="Porosity Mitigation",
        authors=["Alice Smith", "Bob Lee"],
        year=2022,
        venue="WAAM Journal",
        start_page=4,
        end_page=5,
        page_numbers=[4, 5],
        section="Results and Discussion",
        subsection="Mitigation Guidance",
        text=(
            "Higher current increased porosity through excessive heat input. "
            "Optimizing pulse frequency and reducing current reduced pore formation in the experiments."
        ),
        token_count=24,
        defect_terms=["porosity"],
        process_parameters=["current", "pulse_frequency"],
        process_types=["waam"],
        materials=["aluminum"],
        evidence_types=["mitigation", "result", "mechanism"],
        evidence_directness_score=0.82,
        review_or_experimental="experimental",
    )
    query_bundle = QueryBundle(
        mode="combined",
        dense_query="porosity mitigation",
        lexical_query="porosity mitigation current pulse frequency waam",
        query_summary="test",
        defect_name="porosity",
        expanded_terms=["porosity", "pore formation"],
        process_fragments=["current", "pulse frequency"],
        parameter_categories=["current", "pulse_frequency"],
        parameter_values={"current_A": 180, "pulse_frequency_hz": 150},
        materials=["aluminum"],
        process_types=["waam"],
        subqueries={"defect_mitigation": "porosity mitigation current pulse frequency waam"},
    )

    evidence = EvidenceExtractor().extract(chunk, "[Smith et al., 2022, pp. 4-5]", query_bundle=query_bundle)

    assert evidence.strategy is not None
    assert "current" in evidence.parameters
    assert evidence.parameter_effects
    assert evidence.citation == "[Smith et al., 2022, pp. 4-5]"
    assert evidence.evidence_type in {"mitigation", "result"}
