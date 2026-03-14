"""Configuration loading for the WAAM RAG service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    """Application configuration loaded from YAML and environment variables."""

    app_name: str = "WAAM Research RAG"
    environment: str = "local"
    storage_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    log_level: str = "INFO"
    log_json: bool = True
    vector_backend: str = "sqlite_local"

    embedding_backend: str = "hashing"
    embedding_model: str = "intfloat/e5-base-v2"
    embedding_batch_size: int = 16

    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 120
    top_k: int = 8
    retrieve_candidates: int = 24
    dense_top_k: int = 24
    sparse_top_k: int = 24
    fusion_rrf_k: int = 60

    bm25_enabled: bool = True
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 12

    exclude_references: bool = True
    enable_ocr_fallback: bool = False
    metadata_enrichment_enabled: bool = True
    generate_chunk_summary: bool = False
    query_expansion_enabled: bool = True
    citation_style: str = "author_year_pages"

    default_process_terms: list[str] = Field(
        default_factory=lambda: [
            "WAAM",
            "wire arc additive manufacturing",
            "arc additive manufacturing",
            "welding",
            "process monitoring",
            "quality control",
        ]
    )
    section_boosts: dict[str, float] = Field(
        default_factory=lambda: {
            "results": 0.08,
            "discussion": 0.08,
            "conclusions": 0.05,
            "recommendations": 0.09,
            "abstract": -0.03,
            "introduction": -0.02,
        }
    )

    @field_validator("storage_dir", "uploads_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return value if isinstance(value, Path) else Path(str(value))

    @property
    def catalog_path(self) -> Path:
        return self.storage_dir / "catalog.sqlite3"

    @property
    def temp_upload_dir(self) -> Path:
        return self.uploads_dir / "incoming"


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from optional YAML, then apply environment overrides."""

    yaml_path = _resolve_config_path(config_path)
    yaml_payload = _read_yaml(yaml_path) if yaml_path else {}
    env_payload = _read_env_overrides()

    merged = dict(yaml_payload)
    merged.update(env_payload)
    settings = Settings(**merged)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path:
        path = Path(config_path)
        return path if path.exists() else None

    env_value = os.getenv("WAAM_RAG_CONFIG_PATH")
    if env_value:
        path = Path(env_value)
        return path if path.exists() else None

    default_path = Path("config/settings.yaml")
    return default_path if default_path.exists() else None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return loaded


def _read_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    prefix = "WAAM_RAG_"
    for key, raw_value in os.environ.items():
        if not key.startswith(prefix) or key == "WAAM_RAG_CONFIG_PATH":
            continue
        field_name = key[len(prefix) :].lower()
        overrides[field_name] = _parse_env_value(raw_value)
    return overrides


def _parse_env_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in value and not value.strip().startswith("{") and not value.strip().startswith("["):
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
