from __future__ import annotations

import pytest

from waam_rag.retrieval.extraction import EvidenceExtractor
from waam_rag.schemas import ChunkRecord, QueryBundle


@pytest.mark.parametrize(
    ("text", "parameter", "expected"),
    [
        ("Higher current increased porosity during WAAM deposition.", "current", "increase_risk_when_high"),
        ("Too low current caused lack of fusion in the deposited wall.", "current", "increase_risk_when_low"),
        (
            "Too high or too low travel speed increased porosity through unstable heat input.",
            "travel_speed",
            "non_monotonic",
        ),
        ("An optimal voltage range minimized porosity in the experiments.", "voltage", "optimal_window"),
    ],
)
def test_parameter_directionality_categories(text: str, parameter: str, expected: str) -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_file="paper.pdf",
        title="Paper",
        authors=["Alice Smith"],
        year=2023,
        venue="WAAM Journal",
        start_page=2,
        end_page=2,
        page_numbers=[2],
        section="Results",
        subsection=None,
        text=text,
        token_count=14,
        defect_terms=["porosity"],
        process_parameters=[parameter],
        evidence_types=["mechanism", "result"],
    )
    bundle = QueryBundle(
        mode="combined",
        dense_query="porosity parameter relationship",
        lexical_query="porosity parameter relationship",
        query_summary="test",
        defect_name="porosity",
        expanded_terms=["porosity"],
        process_fragments=[parameter.replace("_", " ")],
        parameter_categories=[parameter],
        parameter_values={},
        materials=[],
        process_types=["waam"],
        subqueries={"defect_parameter_relationship": "porosity parameter relationship"},
    )

    evidence = EvidenceExtractor().extract(chunk, "[Smith, 2023, p. 2]", query_bundle=bundle)

    assert evidence.parameter_effects
    assert evidence.parameter_effects[0].directionality == expected
