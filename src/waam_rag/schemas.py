"""Pydantic schemas shared across the RAG pipeline and API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int
    text: str


class ParsedDocument(BaseModel):
    doc_id: str
    checksum: str
    source_file: str
    file_name: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    page_count: int = 0
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    pages: list[PageText] = Field(default_factory=list)


class StructuredBlock(BaseModel):
    text: str
    page_number: int
    section: str | None = None
    subsection: str | None = None
    is_heading: bool = False
    excluded: bool = False


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    start_page: int
    end_page: int
    page_numbers: list[int] = Field(default_factory=list)
    section: str | None = None
    subsection: str | None = None
    text: str
    token_count: int
    summary: str | None = None
    defect_terms: list[str] = Field(default_factory=list)
    process_parameters: list[str] = Field(default_factory=list)
    parameter_mentions: dict[str, list[str]] = Field(default_factory=dict)
    materials: list[str] = Field(default_factory=list)
    process_types: list[str] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    doc_id: str
    checksum: str
    source_file: str
    file_name: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    ingested_at: datetime
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessParameters(BaseModel):
    current: float | None = None
    voltage: float | None = None
    wire_feed_speed: float | None = None
    travel_speed: float | None = None
    torch_angle: float | None = None
    shielding_gas: str | None = None
    heat_input: float | None = None
    interpass_temperature: float | None = None
    layer_height: float | None = None
    arc_length: float | None = None
    deposition_rate: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_text_fragments(self) -> list[str]:
        fragments: list[str] = []
        for field_name, value in self.model_dump(exclude_none=True).items():
            if field_name == "extra":
                continue
            fragments.append(f"{field_name.replace('_', ' ')} {value}")
        for key, value in self.extra.items():
            fragments.append(f"{key.replace('_', ' ')} {value}")
        return fragments


class QueryFilters(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    process_types: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)
    defect_terms: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    defect_name: str | None = None
    question: str | None = None
    process_parameters: ProcessParameters | None = None
    filters: QueryFilters | None = None
    top_k: int | None = None
    enable_rerank: bool | None = None
    query_expansion: bool | None = None


class QueryBundle(BaseModel):
    mode: Literal["defect_only", "defect_with_process", "free_text", "combined"]
    dense_query: str
    lexical_query: str
    query_summary: str
    defect_name: str | None = None
    expanded_terms: list[str] = Field(default_factory=list)
    process_fragments: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    process_types: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    score: float
    source_file: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    pages: list[int] = Field(default_factory=list)
    section: str | None = None
    subsection: str | None = None
    citation: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query_summary: str
    results: list[RetrievedChunk]
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ContextPackEntry(BaseModel):
    rank: int
    chunk_id: str
    citation: str
    short_citation: str
    source_file: str
    title: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    section: str | None = None
    subsection: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackResponse(BaseModel):
    query_summary: str
    citation_style: str
    evidence: list[ContextPackEntry]
    context_text: str
    timings_ms: dict[str, float] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    folder_path: Path | None = None
    force: bool = False


class IngestFileResult(BaseModel):
    source_file: str
    doc_id: str | None = None
    status: Literal["ingested", "updated", "skipped", "failed"]
    chunk_count: int = 0
    message: str | None = None


class IngestResponse(BaseModel):
    documents_processed: int
    documents_ingested: int
    documents_skipped: int
    chunks_created: int
    results: list[IngestFileResult]


class ReindexRequest(BaseModel):
    folder_path: Path | None = None
    force: bool = True


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    storage_dir: str
    vector_backend: str | None = None
