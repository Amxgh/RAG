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

QUERY_PAYLOAD = {
    "defect_name": "porosity",
    "question": "What literature-backed mitigation strategies reduce porosity in WAAM?",
    "top_k": 2,
}


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


def _build_test_client(monkeypatch, workspace_tmp_path: Path, **settings_overrides):
    pdf_path = workspace_tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake document")
    settings = Settings(
        storage_dir=workspace_tmp_path / "data",
        uploads_dir=workspace_tmp_path / "uploads",
        embedding_backend="hashing",
        reranker_enabled=False,
        chunk_size_tokens=50,
        chunk_overlap_tokens=10,
        **settings_overrides,
    )
    fake_service = RAGService(settings, parser=FakeParser(), embedder=HashingEmbedder())

    monkeypatch.setattr(api_module, "RAGService", lambda _settings: fake_service)
    app = api_module.create_app(settings)
    client = testclient_module.TestClient(app)
    ingest_response = client.post("/ingest", json={"folder_path": str(workspace_tmp_path), "force": False})

    assert ingest_response.status_code == 200
    return client


def test_api_ingest_and_query(monkeypatch, workspace_tmp_path: Path) -> None:
    client = _build_test_client(monkeypatch, workspace_tmp_path, forward_results_to_open_webui=False)
    query_response = client.post("/query", json=QUERY_PAYLOAD)

    assert query_response.status_code == 200
    body = query_response.json()
    assert body["results"]
    assert body["extracted_evidence"]
    assert body["reasoning_hints"]
    assert body["diagnostics"]["candidate_chunks_considered"] >= 1
    assert body["results"][0]["citation"].startswith("[")


def test_query_requires_request_api_key_when_forwarding_enabled(monkeypatch, workspace_tmp_path: Path) -> None:
    client = _build_test_client(monkeypatch, workspace_tmp_path, forward_results_to_open_webui=True)

    response = client.post("/query", json=QUERY_PAYLOAD)

    assert response.status_code == 400
    assert api_module.OPEN_WEBUI_API_KEY_HEADER in response.json()["detail"]


def test_query_forwards_request_api_key_to_open_webui(monkeypatch, workspace_tmp_path: Path) -> None:
    forwarded_request: dict[str, str] = {}

    async def fake_send_result_to_open_webui(**kwargs) -> None:
        forwarded_request.update(kwargs)

    monkeypatch.setattr(api_module, "_send_result_to_open_webui", fake_send_result_to_open_webui)
    client = _build_test_client(monkeypatch, workspace_tmp_path, forward_results_to_open_webui=True)

    response = client.post(
        "/query",
        json=QUERY_PAYLOAD,
        headers={api_module.OPEN_WEBUI_API_KEY_HEADER: "request-scoped-key"},
    )

    assert response.status_code == 200
    assert forwarded_request["api_key"] == "request-scoped-key"
    assert forwarded_request["title"] == "WAAM Query - porosity"
