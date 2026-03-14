"""FastAPI entrypoint for the WAAM RAG service."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from waam_rag.config import Settings, load_settings
from waam_rag.logging_utils import configure_logging
from waam_rag.schemas import (
    ContextPackResponse,
    HealthResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
    ReindexRequest,
)
from waam_rag.services.rag_service import RAGService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI app."""

    app_settings = settings or load_settings()
    configure_logging(app_settings)
    service = RAGService(app_settings)

    app = FastAPI(title=app_settings.app_name, version="0.1.0")
    app.state.settings = app_settings
    app.state.service = service

    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def handle_key_error(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Document not found: {exc.args[0]}"})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(**service.health())

    @app.get("/documents")
    async def list_documents() -> list[dict]:
        return [document.model_dump(mode="json") for document in service.list_documents()]

    @app.get("/documents/{doc_id}")
    async def get_document(doc_id: str) -> dict:
        return _serialize(service.get_document(doc_id))

    @app.post("/ingest")
    async def ingest(request: Request) -> dict:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            form = await request.form()
            uploads = form.getlist("files")
            if not uploads:
                raise HTTPException(status_code=400, detail="Provide one or more PDF files in the files field.")
            force = _parse_bool(form.get("force", False))
            saved_paths = []
            for upload in uploads:
                filename = getattr(upload, "filename", "")
                if not filename.lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail=f"Unsupported upload type: {filename}")
                saved_paths.append(_persist_upload(upload, app_settings.temp_upload_dir))
            return service.ingest_paths(saved_paths, force=force).model_dump(mode="json")

        payload = IngestRequest.model_validate(await request.json())
        if not payload.folder_path:
            raise HTTPException(status_code=400, detail="Provide folder_path or upload files.")
        return service.ingest_folder(payload.folder_path, force=payload.force).model_dump(mode="json")

    @app.post("/reindex")
    async def reindex(payload: ReindexRequest) -> dict[str, int]:
        return service.reindex(payload.folder_path, force=payload.force)

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest) -> QueryResponse:
        return service.query(payload)

    @app.post("/retrieve-context", response_model=ContextPackResponse)
    async def retrieve_context(payload: QueryRequest) -> ContextPackResponse:
        return service.retrieve_context(payload)

    return app


app = create_app()


def _persist_upload(upload, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4().hex}_{Path(upload.filename).name}"
    destination = target_dir / safe_name
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _serialize(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value
