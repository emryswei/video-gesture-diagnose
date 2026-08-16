from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .models import (
    AnalysisJob,
    AnalysisResult,
    JobStatus,
    ProgressStage,
    SOPDefinition,
)
from .video import VideoValidationError
from .vlm import ModelServiceError


class Analyzer(Protocol):
    def analyze(
        self,
        video_path: Path,
        sop: SOPDefinition,
        progress_callback=None,
        model_name: str | None = None,
    ) -> AnalysisResult: ...


class AnalysisInProgress(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        analyzer: Analyzer,
        retention_minutes: int = 30,
        model_estimated_seconds: float | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._retention = timedelta(minutes=retention_minutes)
        self._jobs: dict[str, AnalysisJob] = {}
        self._active_id: str | None = None
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        configured_estimate = model_estimated_seconds or float(
            os.getenv("MODEL_ESTIMATED_SECONDS", "300")
        )
        self._model_estimated_seconds = max(configured_estimate, 1)

    async def has_active_job(self) -> bool:
        async with self._lock:
            return self._active_job_exists()

    async def submit(
        self,
        video_path: Path,
        sop: SOPDefinition,
        model_name: str | None = None,
    ) -> AnalysisJob:
        selected_model = model_name or os.getenv("MODEL_NAME", "qwen3-vl:4b-instruct")
        async with self._lock:
            self._purge_expired()
            if self._active_job_exists():
                raise AnalysisInProgress
            analysis_id = uuid4().hex
            job = AnalysisJob(
                analysis_id=analysis_id,
                model_name=selected_model,
                status=JobStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                progress_percent=5,
                progress_stage=ProgressStage.UPLOADED,
                progress_message="Video uploaded; waiting to start analysis.",
            )
            self._jobs[analysis_id] = job
            self._active_id = analysis_id
            task = asyncio.create_task(
                self._run(analysis_id, video_path, sop, selected_model)
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return job.model_copy(deep=True)

    async def get(self, analysis_id: str) -> AnalysisJob | None:
        async with self._lock:
            self._purge_expired()
            job = self._jobs.get(analysis_id)
            return job.model_copy(deep=True) if job else None

    def _active_job_exists(self) -> bool:
        if self._active_id is None:
            return False
        job = self._jobs.get(self._active_id)
        return job is not None and job.status in {JobStatus.QUEUED, JobStatus.ANALYZING}

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.expires_at is not None and job.expires_at <= now
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    async def _run(
        self,
        analysis_id: str,
        video_path: Path,
        sop: SOPDefinition,
        model_name: str,
    ) -> None:
        job = self._jobs[analysis_id]
        job.status = JobStatus.ANALYZING
        loop = asyncio.get_running_loop()

        def report_progress(
            percent: int,
            stage: str,
            message: str,
            is_estimated: bool,
        ) -> None:
            loop.call_soon_threadsafe(
                self._apply_progress,
                analysis_id,
                percent,
                stage,
                message,
                is_estimated,
            )

        analysis_task = asyncio.create_task(
            asyncio.to_thread(
                self._analyzer.analyze,
                video_path,
                sop,
                progress_callback=report_progress,
                model_name=model_name,
            )
        )
        estimate_task = asyncio.create_task(
            self._advance_estimated_progress(analysis_id, analysis_task)
        )
        terminal_status = JobStatus.FAILED
        try:
            job.result = await analysis_task
            self._apply_progress(
                analysis_id,
                100,
                ProgressStage.COMPLETE,
                "Analysis complete.",
                False,
            )
            terminal_status = JobStatus.SUCCEEDED
        except (VideoValidationError, ModelServiceError) as exc:
            job.error = str(exc)
            self._mark_failed(job)
        except Exception:
            job.error = "Analysis failed. Please verify the video and model service, then try again."
            self._mark_failed(job)
        finally:
            estimate_task.cancel()
            with suppress(asyncio.CancelledError):
                await estimate_task
            video_path.unlink(missing_ok=True)
            job.expires_at = datetime.now(timezone.utc) + self._retention
            job.status = terminal_status
            async with self._lock:
                if self._active_id == analysis_id:
                    self._active_id = None

    def _apply_progress(
        self,
        analysis_id: str,
        percent: int,
        stage: str | ProgressStage,
        message: str,
        is_estimated: bool,
    ) -> None:
        job = self._jobs.get(analysis_id)
        if job is None or job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            return
        job.progress_percent = max(job.progress_percent, min(percent, 100))
        job.progress_stage = ProgressStage(stage)
        job.progress_message = message
        job.progress_is_estimated = is_estimated

    async def _advance_estimated_progress(
        self,
        analysis_id: str,
        analysis_task: asyncio.Task[AnalysisResult],
    ) -> None:
        model_started_at: float | None = None
        while not analysis_task.done():
            job = self._jobs.get(analysis_id)
            if job is None:
                return
            if job.progress_stage == ProgressStage.MODEL_ANALYSIS:
                if model_started_at is None:
                    model_started_at = time.monotonic()
                elapsed = time.monotonic() - model_started_at
                estimated = 25 + int(65 * elapsed / self._model_estimated_seconds)
                self._apply_progress(
                    analysis_id,
                    min(estimated, 90),
                    ProgressStage.MODEL_ANALYSIS,
                    "AI is checking the 7 procedure steps.",
                    True,
                )
            await asyncio.sleep(1)

    @staticmethod
    def _mark_failed(job: AnalysisJob) -> None:
        job.progress_stage = ProgressStage.FAILED
        job.progress_message = "Analysis failed."
        job.progress_is_estimated = False
