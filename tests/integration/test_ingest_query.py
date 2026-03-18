from __future__ import annotations

from pathlib import Path

from waam_rag.schemas import QueryRequest


def test_ingest_and_query_round_trip(rag_service, workspace_tmp_path: Path) -> None:
    pdf_path = workspace_tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake document")

    ingest_response = rag_service.ingest_paths([pdf_path])
    query_response = rag_service.query(
        QueryRequest(
            defect_name="porosity",
            question="What literature-backed mitigation strategies reduce porosity in WAAM?",
            top_k=3,
        )
    )
    context_response = rag_service.retrieve_context(
        QueryRequest(
            defect_name="porosity",
            question="What literature-backed mitigation strategies reduce porosity in WAAM?",
            top_k=2,
        )
    )

    assert ingest_response.documents_ingested == 1
    assert ingest_response.chunks_created >= 1
    assert query_response.results
    assert query_response.extracted_evidence
    assert query_response.reasoning_hints
    assert query_response.diagnostics.candidate_chunks_considered >= 1
    assert "porosity" in query_response.results[0].metadata.get("defect_terms", [])
    assert context_response.evidence
    assert context_response.extracted_evidence
    assert context_response.reasoning_hints
    assert context_response.evidence[0].citation.startswith("[")
