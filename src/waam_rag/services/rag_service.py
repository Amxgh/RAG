"""End-to-end RAG service orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from waam_rag.citations import format_citation, format_short_citation
from waam_rag.config import Settings
from waam_rag.indexing.bm25 import BM25Index
from waam_rag.indexing.embeddings import Embedder, build_embedder
from waam_rag.indexing.repository import DocumentRepository
from waam_rag.ingestion.chunking import ScientificChunker
from waam_rag.ingestion.cleaning import TextCleaner
from waam_rag.ingestion.enrichment import MetadataEnricher
from waam_rag.ingestion.parser import ResearchPaperParser
from waam_rag.ingestion.structure import StructureExtractor
from waam_rag.retrieval.query_builder import QueryBuilder
from waam_rag.retrieval.reranker import Reranker, build_reranker
from waam_rag.retrieval.service import HybridRetriever
from waam_rag.schemas import (
    ContextPackEntry,
    ContextPackResponse,
    DocumentRecord,
    IngestFileResult,
    IngestResponse,
    ParsedDocument,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from waam_rag.utils.files import sha1_file

LOGGER = logging.getLogger(__name__)


class RAGService:
    """High-level service that owns ingestion, indexing, and retrieval."""

    def __init__(
        self,
        settings: Settings,
        *,
        parser: ResearchPaperParser | None = None,
        cleaner: TextCleaner | None = None,
        structure_extractor: StructureExtractor | None = None,
        chunker: ScientificChunker | None = None,
        enricher: MetadataEnricher | None = None,
        repository: DocumentRepository | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.parser = parser or ResearchPaperParser(enable_ocr_fallback=settings.enable_ocr_fallback)
        self.cleaner = cleaner or TextCleaner()
        self.structure_extractor = structure_extractor or StructureExtractor(
            exclude_references=settings.exclude_references
        )
        self.chunker = chunker or ScientificChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            generate_chunk_summary=settings.generate_chunk_summary,
        )
        self.enricher = enricher or MetadataEnricher()
        self.repository = repository or DocumentRepository(settings.catalog_path)
        self.embedder = embedder or build_embedder(settings)
        self.bm25_index = BM25Index()
        self.retriever = HybridRetriever(
            settings=settings,
            repository=self.repository,
            embedder=self.embedder,
            bm25_index=self.bm25_index,
            query_builder=QueryBuilder(settings),
            reranker=reranker or build_reranker(settings),
        )
        self.retriever.refresh()

    def ingest_folder(self, folder_path: str | Path, force: bool = False) -> IngestResponse:
        path = Path(folder_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Folder does not exist: {path}")
        pdf_paths = sorted(candidate for candidate in path.glob("*.pdf") if candidate.is_file())
        if not pdf_paths:
            raise ValueError(f"No PDF files found in {path}")
        return self.ingest_paths(pdf_paths, force=force)

    def ingest_paths(self, paths: Iterable[str | Path], force: bool = False) -> IngestResponse:
        results: list[IngestFileResult] = []
        chunks_created = 0
        ingested_count = 0
        skipped_count = 0

        for raw_path in paths:
            path = Path(raw_path)
            source_file = str(path.resolve())
            LOGGER.info("ingest_start", extra={"file": source_file})
            try:
                existing = self.repository.get_document_by_source(source_file)
                checksum = sha1_file(path)
                if existing and existing.checksum == checksum and not force:
                    skipped_count += 1
                    results.append(
                        IngestFileResult(
                            source_file=source_file,
                            doc_id=existing.doc_id,
                            status="skipped",
                            message="Checksum unchanged; skipping incremental ingest.",
                        )
                    )
                    continue

                parsed = self.parser.parse(path)
                prepared_document, chunks = self._prepare_document(parsed)
                embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks]).astype(np.float32)
                self.repository.upsert_document(prepared_document, chunks, embeddings)
                status = "updated" if existing else "ingested"
                ingested_count += 1
                chunks_created += len(chunks)
                results.append(
                    IngestFileResult(
                        source_file=source_file,
                        doc_id=prepared_document.doc_id,
                        status=status,
                        chunk_count=len(chunks),
                    )
                )
                LOGGER.info(
                    "ingest_complete",
                    extra={
                        "file": source_file,
                        "doc_id": prepared_document.doc_id,
                        "chunks": len(chunks),
                        "status": status,
                    },
                )
            except Exception as exc:
                results.append(
                    IngestFileResult(
                        source_file=source_file,
                        status="failed",
                        message=str(exc),
                    )
                )
                LOGGER.exception("ingest_failed", extra={"file": source_file, "error": str(exc)})

        self.retriever.refresh()
        return IngestResponse(
            documents_processed=len(results),
            documents_ingested=ingested_count,
            documents_skipped=skipped_count,
            chunks_created=chunks_created,
            results=results,
        )

    def reindex(self, folder_path: str | Path | None = None, force: bool = True) -> dict[str, int]:
        if folder_path:
            response = self.ingest_folder(folder_path, force=force)
            return {
                "documents_reindexed": response.documents_ingested,
                "chunks_reembedded": response.chunks_created,
            }

        chunks = [chunk for chunk, _ in self.repository.get_chunks(include_embeddings=False)]
        if not chunks:
            return {"documents_reindexed": 0, "chunks_reembedded": 0}
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks]).astype(np.float32)
        self.repository.reembed_chunks(
            {chunk.chunk_id: embedding for chunk, embedding in zip(chunks, embeddings, strict=True)}
        )
        self.retriever.refresh()
        return {
            "documents_reindexed": len({chunk.doc_id for chunk in chunks}),
            "chunks_reembedded": len(chunks),
        }

    def list_documents(self) -> list[DocumentRecord]:
        return self.repository.list_documents()

    def get_document(self, doc_id: str) -> dict[str, object]:
        document = self.repository.get_document(doc_id)
        if document is None:
            raise KeyError(doc_id)
        chunks = self.repository.get_document_chunks(doc_id)
        return {
            "document": document,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "pages": chunk.page_numbers,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "token_count": chunk.token_count,
                    "defect_terms": chunk.defect_terms,
                    "materials": chunk.materials,
                    "process_types": chunk.process_types,
                }
                for chunk in chunks
            ],
        }

    def query(self, request: QueryRequest) -> QueryResponse:
        query_summary, candidates, timings = self.retriever.retrieve(request)
        results = [
            RetrievedChunk(
                rank=index,
                chunk_id=candidate.chunk.chunk_id,
                score=round(candidate.score, 4),
                source_file=candidate.chunk.source_file,
                title=candidate.chunk.title,
                authors=candidate.chunk.authors,
                year=candidate.chunk.year,
                pages=candidate.chunk.page_numbers,
                section=candidate.chunk.section,
                subsection=candidate.chunk.subsection,
                citation=format_citation(candidate.chunk, self.settings.citation_style),
                text=candidate.chunk.text,
                metadata={
                    **candidate.chunk.metadata,
                    "dense_score": candidate.dense_score,
                    "sparse_score": candidate.sparse_score,
                    "rerank_score": candidate.rerank_score,
                },
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
        return QueryResponse(query_summary=query_summary, results=results, timings_ms=timings)

    def retrieve_context(self, request: QueryRequest) -> ContextPackResponse:
        query_response = self.query(request)
        chunk_lookup = {chunk.chunk_id: chunk for chunk, _ in self.repository.get_chunks(include_embeddings=False)}
        evidence: list[ContextPackEntry] = []
        context_blocks: list[str] = []

        for result in query_response.results:
            chunk = chunk_lookup[result.chunk_id]
            short_citation = format_short_citation(chunk, self.settings.citation_style)
            evidence.append(
                ContextPackEntry(
                    rank=result.rank,
                    chunk_id=result.chunk_id,
                    citation=result.citation,
                    short_citation=short_citation,
                    source_file=result.source_file,
                    title=result.title,
                    year=result.year,
                    authors=result.authors,
                    pages=result.pages,
                    section=result.section,
                    subsection=result.subsection,
                    text=result.text,
                    metadata=result.metadata,
                )
            )
            context_blocks.append(f"{result.rank}. {result.citation}\n{result.text}")

        return ContextPackResponse(
            query_summary=query_response.query_summary,
            citation_style=self.settings.citation_style,
            evidence=evidence,
            context_text="\n\n".join(context_blocks),
            timings_ms=query_response.timings_ms,
        )

    def health(self) -> dict[str, object]:
        documents, chunks = self.repository.counts()
        return {
            "status": "ok",
            "documents": documents,
            "chunks": chunks,
            "storage_dir": str(self.settings.storage_dir.resolve()),
            "vector_backend": self.settings.vector_backend,
        }

    def _prepare_document(self, parsed_document: ParsedDocument) -> tuple[ParsedDocument, list]:
        cleaned_pages = self.cleaner.clean_pages(parsed_document.pages)
        structured_blocks = self.structure_extractor.extract_blocks(cleaned_pages)
        prepared_document = parsed_document.model_copy(update={"pages": cleaned_pages})
        chunks = self.chunker.chunk_document(prepared_document, structured_blocks)
        if not chunks:
            raise ValueError(f"No retrievable chunks were created for {parsed_document.file_name}")
        if self.settings.metadata_enrichment_enabled:
            chunks = self.enricher.enrich_chunks(chunks)

        prepared_document.raw_metadata["enrichment"] = {
            "defect_terms": sorted({term for chunk in chunks for term in chunk.defect_terms}),
            "materials": sorted({term for chunk in chunks for term in chunk.materials}),
            "process_types": sorted({term for chunk in chunks for term in chunk.process_types}),
        }
        return prepared_document, chunks
