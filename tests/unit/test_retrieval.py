from __future__ import annotations

import numpy as np

from waam_rag.config import Settings
from waam_rag.indexing.bm25 import BM25Index
from waam_rag.indexing.embeddings import HashingEmbedder
from waam_rag.indexing.repository import DocumentRepository
from waam_rag.retrieval.query_builder import QueryBuilder
from waam_rag.retrieval.service import HybridRetriever
from waam_rag.schemas import ChunkRecord, ParsedDocument, QueryRequest


def test_hybrid_retrieval_prioritizes_porosity_mitigation(workspace_tmp_path) -> None:
    repository = DocumentRepository(workspace_tmp_path / "catalog.sqlite3")
    embedder = HashingEmbedder()
    bm25 = BM25Index()
    settings = Settings(
        storage_dir=workspace_tmp_path / "data",
        uploads_dir=workspace_tmp_path / "uploads",
        embedding_backend="hashing",
        reranker_enabled=False,
        chunk_size_tokens=80,
        top_k=3,
        dense_top_k=5,
        sparse_top_k=5,
        retrieve_candidates=5,
    )

    document = ParsedDocument(
        doc_id="doc-1",
        checksum="abc",
        source_file="paper.pdf",
        file_name="paper.pdf",
        title="WAAM mitigation",
        authors=["Alice Smith"],
        year=2023,
        venue="WAAM Journal",
        page_count=2,
        pages=[],
    )
    chunks = [
        ChunkRecord(
            chunk_id="doc-1:0000",
            doc_id="doc-1",
            source_file="paper.pdf",
            title="WAAM mitigation",
            authors=["Alice Smith"],
            year=2023,
            venue="WAAM Journal",
            start_page=1,
            end_page=1,
            page_numbers=[1],
            section="Results and Discussion",
            subsection=None,
            text="Porosity mitigation in WAAM aluminum was achieved by improving argon shielding gas stability and reducing heat input.",
            token_count=18,
            defect_terms=["porosity"],
            process_parameters=["shielding_gas", "heat_input"],
            materials=["aluminum"],
            process_types=["waam"],
            evidence_types=["mitigation", "result"],
            metadata={},
        ),
        ChunkRecord(
            chunk_id="doc-1:0001",
            doc_id="doc-1",
            source_file="paper.pdf",
            title="WAAM mitigation",
            authors=["Alice Smith"],
            year=2023,
            venue="WAAM Journal",
            start_page=2,
            end_page=2,
            page_numbers=[2],
            section="Methodology",
            subsection=None,
            text="Crack susceptibility was evaluated with tensile specimens and residual stress analysis.",
            token_count=11,
            defect_terms=["crack"],
            process_parameters=[],
            materials=[],
            process_types=["waam"],
            evidence_types=["result"],
            metadata={},
        ),
    ]
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks]).astype(np.float32)
    repository.upsert_document(document, chunks, embeddings)
    bm25.rebuild(chunks)

    retriever = HybridRetriever(
        settings=settings,
        repository=repository,
        embedder=embedder,
        bm25_index=bm25,
        query_builder=QueryBuilder(settings),
    )

    _, results, _ = retriever.retrieve(
        QueryRequest(
            defect_name="porosity",
            question="What mitigation strategies reduce porosity in WAAM aluminum builds?",
            top_k=2,
        )
    )

    assert results
    assert results[0].chunk.chunk_id == "doc-1:0000"
