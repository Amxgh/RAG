from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
testclient_module = pytest.importorskip("fastapi.testclient")

import waam_rag.api.app as api_module
from waam_rag.config import Settings
from waam_rag.indexing.embeddings import HashingEmbedder
from waam_rag.schemas import PageText, ParsedDocument
from waam_rag.services.rag_service import RAGService
from waam_rag.utils.files import sha1_file, stable_doc_id


class FakeParser:
    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        pages = [
            PageText(
                page_number=1,
                text=(
                    "A Study on WAAM Defect Control\n"
                    "Smith, Lee\n\n"
                    "Results and Discussion\n"
                    "Porosity in WAAM aluminum builds is reduced by stable argon shielding gas."
                ),
            )
        ]
        return ParsedDocument(
            doc_id=stable_doc_id(path),
            checksum=sha1_file(path),
            source_file=str(path.resolve()),
            file_name=path.name,
            title="A Study on WAAM Defect Control",
            authors=["Alice Smith", "Brian Lee"],
            year=2022,
            venue="Journal of WAAM Studies",
            page_count=1,
            raw_metadata={},
            pages=pages,
        )


def test_api_ingest_and_query(monkeypatch, workspace_tmp_path: Path) -> None:
    pdf_path = workspace_tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake document")
    settings = Settings(
        storage_dir=workspace_tmp_path / "data",
        uploads_dir=workspace_tmp_path / "uploads",
        embedding_backend="hashing",
        reranker_enabled=False,
        chunk_size_tokens=50,
        chunk_overlap_tokens=10,
    )
    fake_service = RAGService(settings, parser=FakeParser(), embedder=HashingEmbedder())

    monkeypatch.setattr(api_module, "RAGService", lambda _settings: fake_service)
    app = api_module.create_app(settings)
    client = testclient_module.TestClient(app)

    ingest_response = client.post("/ingest", json={"folder_path": str(workspace_tmp_path), "force": False})
    query_response = client.post(
        "/query",
        json={
            "defect_name": "porosity",
            "question": "What literature-backed mitigation strategies reduce porosity in WAAM?",
            "top_k": 2,
        },
    )

    assert ingest_response.status_code == 200
    assert query_response.status_code == 200
    body = query_response.json()
    assert body["results"]
    assert body["extracted_evidence"]
    assert body["reasoning_hints"]
    assert body["diagnostics"]["candidate_chunks_considered"] >= 1
    assert body["results"][0]["citation"].startswith("[")
