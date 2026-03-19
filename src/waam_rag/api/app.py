"""FastAPI entrypoint for the WAAM RAG service."""

from __future__ import annotations
import time
from uuid import uuid4
import httpx
from fastapi import BackgroundTasks
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
    async def query(payload: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
        result = service.query(payload)

        if app_settings.forward_results_to_open_webui and app_settings.open_webui_api_key:
            background_tasks.add_task(
                _send_result_to_open_webui,
                base_url=app_settings.open_webui_base_url,
                api_key=app_settings.open_webui_api_key,
                model=app_settings.open_webui_model,
                title=f"WAAM Query - {payload.defect_name or 'General'}",
                user_prompt=payload.model_dump_json(indent=2),
                result_text=result.model_dump_json(indent=2),
            )

        return result

    @app.post("/retrieve-context", response_model=ContextPackResponse)
    async def retrieve_context(
            payload: QueryRequest,
            background_tasks: BackgroundTasks,
    ) -> ContextPackResponse:
        result = service.retrieve_context(payload)
        print(app_settings.forward_results_to_open_webui)
        print(app_settings.open_webui_api_key)
        if app_settings.forward_results_to_open_webui and app_settings.open_webui_api_key:
            print("Attempting to post to the LLM")
            background_tasks.add_task(
                _send_result_to_open_webui,
                base_url=app_settings.open_webui_base_url,
                api_key=app_settings.open_webui_api_key,
                model=app_settings.open_webui_model,
                title=f"WAAM Context - {payload.defect_name or 'General'}",
                user_prompt=payload.model_dump_json(indent=2),
                result_text=result.model_dump_json(indent=2),
            )

        return result

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


# async def _send_result_to_open_webui(
#         *,
#         base_url: str,
#         api_key: str,
#         model: str,
#         title: str,
#         user_prompt: str,
#         result_text: str,
# ) -> None:
#     """
#     Create a new chat in Open WebUI and send the WAAM API result there.
#     """
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }
#
#     timestamp = int(time.time() * 1000)
#     user_msg_id = str(uuid4())
#     assistant_msg_id = str(uuid4())
#
#     # What will appear in the new Open WebUI chat
#     combined_prompt = (
#         f"Original user/API request:\n{user_prompt}\n\n"
#         f"Result from WAAM API:\n{result_text}\n\n"
#         "Please continue from this result."
#     )
#
#     create_chat_payload = {
#         "chat": {
#             "title": title,
#             "models": [model],
#             "messages": [
#                 {
#                     "id": user_msg_id,
#                     "role": "user",
#                     "content": combined_prompt,
#                     "timestamp": timestamp,
#                     "models": [model],
#                 },
#                 {
#                     "id": assistant_msg_id,
#                     "role": "assistant",
#                     "content": "",
#                     "parentId": user_msg_id,
#                     "timestamp": timestamp + 1,
#                     "models": [model],
#                     "modelName": model,
#                     "modelIdx": 0,
#                 },
#             ],
#             "history": {
#                 "current_id": assistant_msg_id,
#                 "messages": {
#                     user_msg_id: {
#                         "id": user_msg_id,
#                         "role": "user",
#                         "content": combined_prompt,
#                         "timestamp": timestamp,
#                         "models": [model],
#                     },
#                     assistant_msg_id: {
#                         "id": assistant_msg_id,
#                         "role": "assistant",
#                         "content": "",
#                         "parentId": user_msg_id,
#                         "timestamp": timestamp + 1,
#                         "models": [model],
#                         "modelName": model,
#                         "modelIdx": 0,
#                     },
#                 },
#             },
#         }
#     }
#
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         # Step 1: create chat
#         create_resp = await client.post(
#             f"{base_url}/api/v1/chats/new",
#             headers=headers,
#             json=create_chat_payload,
#         )
#         create_resp.raise_for_status()
#         create_data = create_resp.json()
#
#         chat_id = create_data["id"]
#
#
#         if not chat_id:
#             raise RuntimeError(
#                 f"Could not find chat id in Open WebUI response. Response was: {create_data}"
#             )
#
#         completion_payload = {
#             "model": model,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": combined_prompt,
#                 }
#             ],
#             "stream": False,
#         }
#         print("Open WebUI model being used:", model)
#         completion_resp = await client.post(
#             f"{base_url}/api/chat/completions",
#             headers=headers,
#             json=completion_payload,
#         )
#         print("completion status:", completion_resp.status_code)
#         print("completion body:", completion_resp.text)
#         completion_resp.raise_for_status()
#
#         completion_data = completion_resp.json()
#         assistant_text = (
#             completion_data.get("choices", [{}])[0]
#             .get("message", {})
#             .get("content", "")
#         )
#
#         if not assistant_text:
#             print("No assistant response content found")
#             return
#
#         completed_resp = await client.post(
#             f"{base_url}/api/chat/completed",
#             headers=headers,
#             json={
#                 "chat_id": chat_id,
#                 "id": assistant_msg_id,
#                 "model": model,
#                 "session_id": str(uuid4()),
#                 "message": {
#                     "id": assistant_msg_id,
#                     "role": "assistant",
#                     "content": assistant_text,
#                     "parentId": user_msg_id,
#                     "timestamp": timestamp + 1,
#                     "models": [model],
#                     "modelName": model,
#                     "modelIdx": 0,
#                 },
#             },
#         )
#         print("completed status:", completed_resp.status_code)
#         print("completed body:", completed_resp.text)
#
#
#         # Optional: mark completed
#         await client.post(
#             f"{base_url}/api/chat/completed",
#             headers=headers,
#             json={
#                 "chat_id": chat_id,
#                 "id": assistant_msg_id,
#                 "model": model,
#                 "session_id": str(uuid4()),
#             },
#         )
async def _send_result_to_open_webui(
    *,
    base_url: str,
    api_key: str,
    model: str,
    title: str,
    user_prompt: str,
    result_text: str,
) -> None:
    import time
    from uuid import uuid4
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    seed_user_content = (
        f"Original user/API request:\n{user_prompt}\n\n"
        f"Result from WAAM API:\n{result_text}\n\n"
        "Please continue from this result and suggest changes to the input "
        "parameters to fix my situation. GIVE NEW VALUES FOR INPUT PARAMETERS "
        "TO FIX THE ISSUES IDENTIFIED."
        "Do not force suggestions, only make suggestions that are absolutely necessary. Each suggestion MUST be backed by citations."
    )

    timestamp = int(time.time() * 1000)
    user_msg_id = str(uuid4())
    assistant_msg_id = str(uuid4())
    session_id = str(uuid4())

    create_chat_payload = {
        "chat": {
            "title": title,
            "models": [model],
            "messages": [
                {
                    "id": user_msg_id,
                    "role": "user",
                    "content": seed_user_content,
                    "timestamp": timestamp,
                    "models": [model],
                },
                {
                    "id": assistant_msg_id,
                    "role": "assistant",
                    "content": "",
                    "parentId": user_msg_id,
                    "timestamp": timestamp + 1,
                    "modelName": model,
                    "modelIdx": 0,
                },
            ],
            "history": {
                "current_id": assistant_msg_id,
                "messages": {
                    user_msg_id: {
                        "id": user_msg_id,
                        "role": "user",
                        "content": seed_user_content,
                        "timestamp": timestamp,
                        "models": [model],
                    },
                    assistant_msg_id: {
                        "id": assistant_msg_id,
                        "role": "assistant",
                        "content": "",
                        "parentId": user_msg_id,
                        "timestamp": timestamp + 1,
                        "modelName": model,
                        "modelIdx": 0,
                    },
                },
            },
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1) create chat
        create_resp = await client.post(
            f"{base_url}/api/v1/chats/new",
            headers=headers,
            json=create_chat_payload,
        )
        create_resp.raise_for_status()
        create_data = create_resp.json()
        chat_id = create_data["id"]

        # 2) update chat state explicitly (matches documented flow)
        update_payload = create_chat_payload.copy()
        update_payload["chat"]["id"] = chat_id

        update_resp = await client.post(
            f"{base_url}/api/v1/chats/{chat_id}",
            headers=headers,
            json=update_payload,
        )
        update_resp.raise_for_status()

        # 3) trigger completion tied to this chat + assistant message
        completion_resp = await client.post(
            f"{base_url}/api/chat/completions",
            headers=headers,
            json={
                "chat_id": chat_id,
                "id": assistant_msg_id,
                "session_id": session_id,
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": seed_user_content,
                    }
                ],
                "stream": False,
                "background_tasks": {
                    "title_generation": True,
                    "tags_generation": False,
                    "follow_up_generation": False,
                },
                "features": {
                    "code_interpreter": False,
                    "web_search": False,
                    "image_generation": False,
                    "memory": False,
                },
            },
        )
        completion_resp.raise_for_status()

        # 4) mark completed
        completed_resp = await client.post(
            f"{base_url}/api/chat/completed",
            headers=headers,
            json={
                "chat_id": chat_id,
                "id": assistant_msg_id,
                "session_id": session_id,
                "model": model,
            },
        )
        completed_resp.raise_for_status()