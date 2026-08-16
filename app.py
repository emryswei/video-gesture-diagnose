from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

from video_analysis.analyzer import VideoAnalyzer
from video_analysis.jobs import AnalysisInProgress, JobManager
from video_analysis.models import AnalysisJob, ErrorResponse, JobCreated, ModelOption
from video_analysis.sop import PROJECT_ROOT, load_sop


ALLOWED_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MODEL_CATALOG = (
    ModelOption(
        id="qwen3-vl:4b-instruct",
        label="Qwen3-VL 4B",
        description="Better accuracy · slower",
    ),
    ModelOption(
        id="qwen3-vl:2b-instruct",
        label="Qwen3-VL 2B",
        description="Faster · lower accuracy",
    ),
)
ALLOWED_MODEL_IDS = {model.id for model in MODEL_CATALOG}


class APIError(Exception):
    def __init__(self, status_code: int, error: str, message: str) -> None:
        self.status_code = status_code
        self.error = error
        self.message = message


async def installed_model_ids() -> set[str]:
    base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    api_key = os.getenv("MODEL_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            data = response.json().get("data", [])
        return {
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise APIError(
            503,
            "model_service_unavailable",
            "The local model service is unavailable. Start Ollama and try again.",
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = JobManager(VideoAnalyzer())
    yield


app = FastAPI(title="SOP Video Analysis Demo", version="0.1.0", lifespan=lifespan)


@app.exception_handler(APIError)
async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
    payload = ErrorResponse(error=exc.error, message=exc.message)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=list[ModelOption])
async def get_models() -> list[ModelOption]:
    installed = await installed_model_ids()
    default_model = os.getenv("MODEL_NAME", "qwen3-vl:4b-instruct")
    return [
        model.model_copy(update={"is_default": model.id == default_model})
        for model in MODEL_CATALOG
        if model.id in installed
    ]


@app.get("/api/sops/{sop_id}")
def get_sop(sop_id: str):
    try:
        return load_sop(sop_id)
    except ValueError as exc:
        raise APIError(404, "sop_not_found", str(exc)) from exc


@app.post(
    "/api/analyses",
    status_code=202,
    response_model=JobCreated,
    responses={409: {"model": ErrorResponse}},
)
async def create_analysis(
    video: UploadFile = File(...),
    sop_id: str = Form("hk_chp_handrub"),
    model_name: str = Form(os.getenv("MODEL_NAME", "qwen3-vl:4b-instruct")),
):
    jobs: JobManager = app.state.jobs
    if await jobs.has_active_job():
        raise APIError(
            409,
            "analysis_in_progress",
            "Another analysis is already in progress.",
        )

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise APIError(
            415,
            "unsupported_video_type",
            "Upload an MP4, MOV, WebM, MKV, or AVI video.",
        )
    try:
        sop = load_sop(sop_id)
    except ValueError as exc:
        raise APIError(404, "sop_not_found", str(exc)) from exc
    if model_name not in ALLOWED_MODEL_IDS:
        raise APIError(400, "unsupported_model", "Select a supported vision model.")
    if model_name not in await installed_model_ids():
        raise APIError(
            400,
            "model_not_installed",
            "The selected model is not installed in the local model service.",
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as output:
            temp_path = Path(output.name)
            total = 0
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise APIError(413, "video_too_large", "The uploaded video is too large.")
                output.write(chunk)
        if total == 0:
            raise APIError(400, "empty_video", "The uploaded video is empty.")
        try:
            job = await jobs.submit(temp_path, sop, model_name)
        except AnalysisInProgress as exc:
            raise APIError(
                409,
                "analysis_in_progress",
                "Another analysis is already in progress.",
            ) from exc
        temp_path = None
        return JobCreated(
            analysis_id=job.analysis_id,
            model_name=job.model_name,
            status=job.status,
            progress_percent=job.progress_percent,
            progress_stage=job.progress_stage,
            progress_message=job.progress_message,
            progress_is_estimated=job.progress_is_estimated,
            expires_at=job.expires_at,
            cache_hit=job.cache_hit,
        )
    finally:
        await video.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisJob)
async def get_analysis(analysis_id: str) -> AnalysisJob:
    jobs: JobManager = app.state.jobs
    job = await jobs.get(analysis_id)
    if job is None:
        raise APIError(
            410,
            "analysis_expired",
            "Analysis expired; please run it again.",
        )
    return job
