"""Pydantic schemas shared across the RAG pipeline and API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    reference_contamination_score: float = 0.0
    is_reference_heavy: bool = False
    generic_background_score: float = 0.0
    evidence_directness_score: float = 0.0
    review_or_experimental: str | None = None
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
    model_config = ConfigDict(extra="allow")

    current: float | None = None
    voltage: float | None = None
    wire_feed_speed: float | None = None
    travel_speed: float | None = None
    torch_angle: float | None = None
    shielding_gas: str | None = None
    heat_input: float | None = None
    pulse_frequency: float | None = None
    interpass_temperature: float | None = None
    layer_height: float | None = None
    arc_length: float | None = None
    deposition_rate: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        aliases = {
            "current_A": "current",
            "voltage_V": "voltage",
            "wire_feed_speed_m_min": "wire_feed_speed",
            "travel_speed_mm_s": "travel_speed",
            "pulse_frequency_hz": "pulse_frequency",
            "heat_input_kj_mm": "heat_input",
            "interpass_temperature_C": "interpass_temperature",
            "layer_height_mm": "layer_height",
            "arc_length_mm": "arc_length",
            "deposition_rate_kg_h": "deposition_rate",
        }
        normalized = dict(value)
        extra = dict(normalized.get("extra", {}))

        for alias, canonical in aliases.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized.pop(alias)
            elif alias in normalized:
                normalized.pop(alias)

        known_fields = set(cls.model_fields)
        for key in list(normalized.keys()):
            if key not in known_fields:
                extra[key] = normalized.pop(key)
        normalized["extra"] = extra
        return normalized

    def to_text_fragments(self) -> list[str]:
        fragments: list[str] = []
        for field_name, value in self.model_dump(exclude_none=True).items():
            if field_name == "extra":
                continue
            fragments.append(f"{field_name.replace('_', ' ')} {value}")
        for key, value in self.extra.items():
            fragments.append(f"{key.replace('_', ' ')} {value}")
        return fragments

    def parameter_categories(self) -> list[str]:
        categories: list[str] = []
        for field_name, value in self.model_dump(exclude_none=True).items():
            if field_name == "extra":
                continue
            categories.append(field_name)
        for key in self.extra:
            categories.append(key)
        return categories

    def normalized_values(self) -> dict[str, Any]:
        units = {
            "current": "A",
            "voltage": "V",
            "wire_feed_speed": "m_min",
            "travel_speed": "mm_s",
            "pulse_frequency": "Hz",
            "heat_input": "kj_mm",
            "interpass_temperature": "C",
            "layer_height": "mm",
            "arc_length": "mm",
            "deposition_rate": "kg_h",
        }
        normalized: dict[str, Any] = {}
        for field_name, value in self.model_dump(exclude_none=True).items():
            if field_name == "extra":
                continue
            suffix = units.get(field_name)
            key = f"{field_name}_{suffix}" if suffix else field_name
            normalized[key] = value
        for key, value in self.extra.items():
            normalized[key] = value
        return normalized

    def categorical_terms(self) -> list[str]:
        terms: list[str] = []
        if self.shielding_gas:
            terms.append(self.shielding_gas)
        return terms


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
    parameter_categories: list[str] = Field(default_factory=list)
    parameter_values: dict[str, Any] = Field(default_factory=dict)
    materials: list[str] = Field(default_factory=list)
    process_types: list[str] = Field(default_factory=list)
    subqueries: dict[str, str] = Field(default_factory=dict)


class ParameterEffect(BaseModel):
    parameter: str
    relationship_text: str
    directionality: Literal[
        "increase_risk_when_high",
        "increase_risk_when_low",
        "non_monotonic",
        "optimal_window",
        "unclear",
    ]
    effect_on_defect: str | None = None
    recommended_reasoning_hint: str | None = None


class ExtractedEvidence(BaseModel):
    chunk_id: str
    paper: str | None = None
    citation: str
    defects: list[str] = Field(default_factory=list)
    strategy: str | None = None
    mechanism: str | None = None
    parameters: list[str] = Field(default_factory=list)
    parameter_effects: list[ParameterEffect] = Field(default_factory=list)
    evidence_type: str | None = None
    material: str | None = None
    process: str | None = None
    experimental_outcome: str | None = None
    recommendation: str | None = None
    directness: Literal["high", "medium", "low"] = "low"
    confidence: float = 0.0
    review_or_experimental: Literal["review", "experimental", "mixed", "unclear"] = "unclear"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceTheme(BaseModel):
    theme: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supporting_citations: list[str] = Field(default_factory=list)


class RetrievalDiagnostics(BaseModel):
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    candidate_chunks_considered: int = 0
    reference_heavy_chunks_removed: int = 0
    subqueries_used: list[str] = Field(default_factory=list)


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
    extracted_evidence: list[ExtractedEvidence] = Field(default_factory=list)
    evidence_summary: list[EvidenceTheme] = Field(default_factory=list)
    reasoning_hints: list[str] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
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
    extracted_evidence: list[ExtractedEvidence] = Field(default_factory=list)
    evidence_summary: list[EvidenceTheme] = Field(default_factory=list)
    reasoning_hints: list[str] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
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
