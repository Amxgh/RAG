from __future__ import annotations

import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
                    "Abstract\n"
                    "Porosity in WAAM aluminum builds is linked to shielding gas instability.\n\n"
                    "1 Introduction\n"
                    "Wire arc additive manufacturing suffers from porosity and spatter when process control is poor."
                ),
            ),
            PageText(
                page_number=2,
                text=(
                    "2 Results and Discussion\n"
                    "Increasing argon shielding quality and reducing excessive heat input reduced porosity.\n\n"
                    "2.1 Mitigation Guidance\n"
                    "The experiments indicate that lower travel speed variation and stable wire feed speed minimize pore formation."
                ),
            ),
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
            page_count=len(pages),
            raw_metadata={},
            pages=pages,
        )


@pytest.fixture
def workspace_tmp_path() -> Path:
    base = PROJECT_ROOT / ".tmp_test_runs"
    base.mkdir(exist_ok=True)
    path = base / uuid4().hex
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def settings(workspace_tmp_path: Path) -> Settings:
    return Settings(
        storage_dir=workspace_tmp_path / "data",
        uploads_dir=workspace_tmp_path / "uploads",
        log_json=False,
        embedding_backend="hashing",
        reranker_enabled=False,
        bm25_enabled=True,
        chunk_size_tokens=50,
        chunk_overlap_tokens=10,
        top_k=5,
        dense_top_k=8,
        sparse_top_k=8,
        retrieve_candidates=8,
    )


@pytest.fixture
def fake_parser() -> FakeParser:
    return FakeParser()


@pytest.fixture
def rag_service(settings: Settings, fake_parser: FakeParser) -> RAGService:
    return RAGService(
        settings,
        parser=fake_parser,
        embedder=HashingEmbedder(),
    )
