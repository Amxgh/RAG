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

    # seed_user_content = (
    #     f"Original user/API request:\n{user_prompt}\n\n"
    #     f"Result from WAAM API:\n{result_text}\n\n"
    #     "Please continue from this result and suggest changes to the input "
    #     "parameters to fix my situation. GIVE NEW VALUES FOR INPUT PARAMETERS "
    #     "TO FIX THE ISSUES IDENTIFIED."
    #     "Do not force suggestions, only make suggestions that are absolutely necessary. Each suggestion MUST be backed by citations."
    # )
    seed_user_content = f"""
    You are a technical WAAM process optimization assistant.

    Your task is to analyze the user's original request together with the retrieved RAG evidence from literature, and produce a highly technical, citation-grounded recommendation for adjusting WAAM process parameters to mitigate the identified defect.

    Original user/API request:
    {user_prompt}

    Retrieved literature evidence from WAAM API:
    {result_text}

    Instructions:
    1. Focus specifically on defect mitigation for the defect identified in the original request.
    2. Use ONLY the retrieved literature evidence provided above as your evidence base. Do not invent citations, do not use external knowledge, and do not make unsupported claims.
    3. Your response must be highly technical and written for an engineering/research audience.
    4. You must evaluate the current input parameters and determine whether each one should be:
       - kept unchanged,
       - increased,
       - decreased,
       - or replaced.
    5. For every parameter that should change, you MUST provide:
       - the original value,
       - the recommended new value or recommended operating range,
       - the direction of change,
       - a technical justification,
       - and one or more supporting citations from the retrieved evidence.
    6. If the literature does NOT support a change for a parameter, explicitly say that no justified change can be made from the provided evidence.
    7. Do not force recommendations. Only recommend parameter changes that are directly supported by the retrieved literature and are necessary for mitigating the identified defect.
    8. Prefer concrete numeric recommendations whenever supported by the evidence. If the literature only supports trends or ranges, state the safest technically justified range and explain the uncertainty.
    9. When proposing new values, ensure they are physically and operationally plausible for WAAM and consistent with the retrieved evidence.
    10. Pay close attention to interactions between parameters such as:
        - current vs heat input,
        - voltage vs arc stability,
        - travel speed vs heat input per unit length,
        - wire feed speed vs deposition behavior,
        - shielding gas vs porosity formation.
    11. If multiple literature sources disagree, state the disagreement explicitly and give the most conservative evidence-backed recommendation.
    12. Your goal is not to summarize papers. Your goal is to generate actionable parameter updates for the user's exact parameter set.

    Required output format:

    ## Defect
    State the defect being mitigated.

    ## Current Parameter Set
    List the current input parameters exactly as given.

    ## Evidence-Based Parameter Recommendations
    For each parameter in the input:
    ### <parameter_name>
    - Current value: <value>
    - Recommendation: <keep unchanged / increase / decrease / replace>
    - New value or range: <explicit number or range, if justified>
    - Technical rationale: <technical explanation tied to defect physics and WAAM behavior>
    - Evidence: <citation(s) from retrieved literature>

    ## Additional Process Notes
    List any non-parameter observations that are strongly supported by the evidence and materially affect defect mitigation.

    ## Final Recommended Parameter Set
    Return the full updated parameter set in valid JSON. 
    Rules:
    - Include every original parameter.
    - Preserve parameters unchanged if there is insufficient evidence to modify them.
    - Only output technically justified changes.

    ## Confidence and Limits
    Briefly state:
    - which recommendations are strongly supported,
    - which are weakly supported,
    - and what evidence gaps prevent stronger optimization.

    Important constraints:
    - Every nontrivial recommendation must be backed by citations from the retrieved evidence.
    - Do not output vague advice such as "optimize parameters" or "adjust as needed."
    - Do not omit numeric values when the evidence supports a numeric update.
    - Do not fabricate exact numbers if the evidence only supports directional adjustment.
    - Keep the analysis technical, explicit, and defect-focused.
    """

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