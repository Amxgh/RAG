from __future__ import annotations

import numpy as np

from waam_rag.config import Settings
from waam_rag.indexing.bm25 import BM25Index
from waam_rag.indexing.embeddings import HashingEmbedder
from waam_rag.indexing.repository import DocumentRepository
from waam_rag.retrieval.query_builder import QueryBuilder
from waam_rag.retrieval.service import HybridRetriever
from waam_rag.schemas import ChunkRecord, ParsedDocument, QueryRequest


def test_answerability_reranking_prefers_mitigation_over_background_and_references(workspace_tmp_path) -> None:
    repository = DocumentRepository(workspace_tmp_path / "catalog.sqlite3")
    embedder = HashingEmbedder()
    bm25 = BM25Index()
    settings = Settings(
        storage_dir=workspace_tmp_path / "data",
        uploads_dir=workspace_tmp_path / "uploads",
        embedding_backend="hashing",
        reranker_enabled=False,
        exclude_reference_chunks=True,
        reference_detection_enabled=True,
        top_k=3,
        dense_top_k=6,
        sparse_top_k=6,
        retrieve_candidates=6,
    )
    document = ParsedDocument(
        doc_id="doc-1",
        checksum="abc",
        source_file="paper.pdf",
        file_name="paper.pdf",
        title="WAAM evidence",
        authors=["Alice Smith"],
        year=2023,
        venue="WAAM Journal",
        page_count=3,
        pages=[],
    )
    chunks = [
        ChunkRecord(
            chunk_id="doc-1:0000",
            doc_id="doc-1",
            source_file="paper.pdf",
            title="WAAM evidence",
            authors=["Alice Smith"],
            year=2023,
            venue="WAAM Journal",
            start_page=1,
            end_page=1,
            page_numbers=[1],
            section="Introduction",
            subsection=None,
            text="WAAM has attracted broad attention in recent years and is widely used in manufacturing research.",
            token_count=16,
            defect_terms=["porosity"],
            process_types=["waam"],
            generic_background_score=0.7,
            metadata={},
        ),
        ChunkRecord(
            chunk_id="doc-1:0001",
            doc_id="doc-1",
            source_file="paper.pdf",
            title="WAAM evidence",
            authors=["Alice Smith"],
            year=2023,
            venue="WAAM Journal",
            start_page=2,
            end_page=2,
            page_numbers=[2],
            section="Results and Discussion",
            subsection=None,
            text="Reducing heat input and stabilizing argon shielding reduced porosity in WAAM aluminum during the experiments.",
            token_count=16,
            defect_terms=["porosity"],
            process_parameters=["heat_input", "shielding_gas"],
            materials=["aluminum"],
            process_types=["waam"],
            evidence_types=["mitigation", "result"],
            evidence_directness_score=0.86,
            review_or_experimental="experimental",
            metadata={},
        ),
        ChunkRecord(
            chunk_id="doc-1:0002",
            doc_id="doc-1",
            source_file="paper.pdf",
            title="WAAM evidence",
            authors=["Alice Smith"],
            year=2023,
            venue="WAAM Journal",
            start_page=3,
            end_page=3,
            page_numbers=[3],
            section="Results and Discussion",
            subsection=None,
            text="Smith et al. (2020); Lee et al. (2021); doi 10.1000/xyz; https://doi.org/10.1000/xyz",
            token_count=12,
            defect_terms=["porosity"],
            process_types=["waam"],
            reference_contamination_score=0.95,
            is_reference_heavy=True,
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

    bundle, results, diagnostics = retriever.retrieve(
        QueryRequest(
            defect_name="porosity",
            question="What mitigation strategies reduce porosity in WAAM aluminum?",
            top_k=3,
        )
    )

    assert bundle.subqueries
    assert diagnostics.reference_heavy_chunks_removed == 1
    assert results[0].chunk.chunk_id == "doc-1:0001"
