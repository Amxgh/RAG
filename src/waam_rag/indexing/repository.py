"""SQLite-backed local catalog and vector storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from waam_rag.schemas import ChunkRecord, DocumentRecord, ParsedDocument, QueryFilters


class DocumentRepository:
    """Persist documents, chunks, and dense embeddings locally."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    source_file TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    title TEXT,
                    authors_json TEXT NOT NULL,
                    year INTEGER,
                    venue TEXT,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL,
                    raw_metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    title TEXT,
                    authors_json TEXT NOT NULL,
                    year INTEGER,
                    venue TEXT,
                    start_page INTEGER NOT NULL,
                    end_page INTEGER NOT NULL,
                    page_numbers_json TEXT NOT NULL,
                    section TEXT,
                    subsection TEXT,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    summary TEXT,
                    defect_terms_json TEXT NOT NULL,
                    process_parameters_json TEXT NOT NULL,
                    parameter_mentions_json TEXT NOT NULL,
                    materials_json TEXT NOT NULL,
                    process_types_json TEXT NOT NULL,
                    evidence_types_json TEXT NOT NULL,
                    reference_contamination_score REAL NOT NULL DEFAULT 0.0,
                    is_reference_heavy INTEGER NOT NULL DEFAULT 0,
                    generic_background_score REAL NOT NULL DEFAULT 0.0,
                    evidence_directness_score REAL NOT NULL DEFAULT 0.0,
                    review_or_experimental TEXT,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year)")
            self._migrate_chunks_table(connection)

    def get_document_by_source(self, source_file: str) -> DocumentRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE source_file = ?",
                (source_file,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY COALESCE(year, 0) DESC, title ASC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def get_document_chunks(self, doc_id: str) -> list[ChunkRecord]:
        return [chunk for chunk, _ in self.get_chunks(include_embeddings=False) if chunk.doc_id == doc_id]

    def get_chunks(
        self,
        filters: QueryFilters | None = None,
        *,
        include_embeddings: bool = True,
    ) -> list[tuple[ChunkRecord, np.ndarray | None]]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM chunks").fetchall()

        results: list[tuple[ChunkRecord, np.ndarray | None]] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            if not self._matches_filters(chunk, filters):
                continue
            embedding = self._blob_to_vector(row["embedding"], row["embedding_dim"]) if include_embeddings else None
            results.append((chunk, embedding))
        return results

    def upsert_document(
        self,
        document: ParsedDocument,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")

        ingested_at = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                "SELECT doc_id FROM documents WHERE source_file = ?",
                (document.source_file,),
            ).fetchone()
            if existing:
                connection.execute("DELETE FROM chunks WHERE doc_id = ?", (existing["doc_id"],))
                connection.execute("DELETE FROM documents WHERE doc_id = ?", (existing["doc_id"],))

            connection.execute(
                """
                INSERT INTO documents (
                    doc_id, checksum, source_file, file_name, title, authors_json, year,
                    venue, page_count, chunk_count, ingested_at, raw_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,
                    document.checksum,
                    document.source_file,
                    document.file_name,
                    document.title,
                    json.dumps(document.authors),
                    document.year,
                    document.venue,
                    document.page_count,
                    len(chunks),
                    ingested_at,
                    json.dumps(document.raw_metadata),
                ),
            )

            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, source_file, title, authors_json, year, venue, start_page, end_page,
                    page_numbers_json, section, subsection, text, token_count, summary, defect_terms_json,
                    process_parameters_json, parameter_mentions_json, materials_json, process_types_json,
                    evidence_types_json, reference_contamination_score, is_reference_heavy,
                    generic_background_score, evidence_directness_score, review_or_experimental,
                    metadata_json, embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.source_file,
                        chunk.title,
                        json.dumps(chunk.authors),
                        chunk.year,
                        chunk.venue,
                        chunk.start_page,
                        chunk.end_page,
                        json.dumps(chunk.page_numbers),
                        chunk.section,
                        chunk.subsection,
                        chunk.text,
                        chunk.token_count,
                        chunk.summary,
                        json.dumps(chunk.defect_terms),
                        json.dumps(chunk.process_parameters),
                        json.dumps(chunk.parameter_mentions),
                        json.dumps(chunk.materials),
                        json.dumps(chunk.process_types),
                        json.dumps(chunk.evidence_types),
                        chunk.reference_contamination_score,
                        int(chunk.is_reference_heavy),
                        chunk.generic_background_score,
                        chunk.evidence_directness_score,
                        chunk.review_or_experimental,
                        json.dumps(chunk.metadata),
                        self._vector_to_blob(embedding),
                        int(embedding.shape[0]),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )

    def reembed_chunks(self, embeddings_by_chunk_id: dict[str, np.ndarray]) -> None:
        with closing(self._connect()) as connection, connection:
            for chunk_id, embedding in embeddings_by_chunk_id.items():
                connection.execute(
                    "UPDATE chunks SET embedding = ?, embedding_dim = ? WHERE chunk_id = ?",
                    (self._vector_to_blob(embedding), int(embedding.shape[0]), chunk_id),
                )

    def counts(self) -> tuple[int, int]:
        with closing(self._connect()) as connection:
            doc_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(doc_count), int(chunk_count)

    def _matches_filters(self, chunk: ChunkRecord, filters: QueryFilters | None) -> bool:
        if not filters:
            return True
        if filters.year_min is not None and (chunk.year is None or chunk.year < filters.year_min):
            return False
        if filters.year_max is not None and (chunk.year is None or chunk.year > filters.year_max):
            return False
        if filters.source_files and chunk.source_file not in filters.source_files:
            return False
        if filters.process_types and not set(filters.process_types).intersection(chunk.process_types):
            return False
        if filters.materials and not set(filters.materials).intersection(chunk.materials):
            return False
        if filters.defect_terms and not set(filters.defect_terms).intersection(chunk.defect_terms):
            return False
        return True

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            checksum=row["checksum"],
            source_file=row["source_file"],
            file_name=row["file_name"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            venue=row["venue"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            raw_metadata=json.loads(row["raw_metadata_json"]),
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source_file=row["source_file"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            venue=row["venue"],
            start_page=row["start_page"],
            end_page=row["end_page"],
            page_numbers=json.loads(row["page_numbers_json"]),
            section=row["section"],
            subsection=row["subsection"],
            text=row["text"],
            token_count=row["token_count"],
            summary=row["summary"],
            defect_terms=json.loads(row["defect_terms_json"]),
            process_parameters=json.loads(row["process_parameters_json"]),
            parameter_mentions=json.loads(row["parameter_mentions_json"]),
            materials=json.loads(row["materials_json"]),
            process_types=json.loads(row["process_types_json"]),
            evidence_types=json.loads(row["evidence_types_json"]),
            reference_contamination_score=row["reference_contamination_score"] or 0.0,
            is_reference_heavy=bool(row["is_reference_heavy"]),
            generic_background_score=row["generic_background_score"] or 0.0,
            evidence_directness_score=row["evidence_directness_score"] or 0.0,
            review_or_experimental=row["review_or_experimental"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _vector_to_blob(self, vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    def _blob_to_vector(self, blob: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32, count=dim)

    def _migrate_chunks_table(self, connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        migrations = {
            "reference_contamination_score": "ALTER TABLE chunks ADD COLUMN reference_contamination_score REAL NOT NULL DEFAULT 0.0",
            "is_reference_heavy": "ALTER TABLE chunks ADD COLUMN is_reference_heavy INTEGER NOT NULL DEFAULT 0",
            "generic_background_score": "ALTER TABLE chunks ADD COLUMN generic_background_score REAL NOT NULL DEFAULT 0.0",
            "evidence_directness_score": "ALTER TABLE chunks ADD COLUMN evidence_directness_score REAL NOT NULL DEFAULT 0.0",
            "review_or_experimental": "ALTER TABLE chunks ADD COLUMN review_or_experimental TEXT",
        }
        for column_name, statement in migrations.items():
            if column_name not in existing_columns:
                connection.execute(statement)
