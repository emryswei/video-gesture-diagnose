import asyncio
import time
from pathlib import Path

from video_analysis.analyzer import finalize_result
from video_analysis.jobs import AnalysisInProgress, JobManager
from video_analysis.models import JobStatus, ModelAnalysis, ModelStepResult, StepStatus
from video_analysis.sop import load_sop


class SlowFakeAnalyzer:
    def __init__(self):
        self.model_name = None
        self.calls = 0

    def analyze(self, _video_path: Path, sop, progress_callback=None, model_name=None):
        self.calls += 1
        self.model_name = model_name
        if progress_callback:
            progress_callback(10, "preparing_video", "Preparing video.", False)
            progress_callback(25, "model_analysis", "Analyzing.", True)
        time.sleep(0.08)
        if progress_callback:
            progress_callback(95, "preparing_report", "Preparing report.", False)
        model = ModelAnalysis(
            procedure_start_sec=1,
            procedure_end_sec=25,
            summary="All actions are visible.",
            steps=[
                ModelStepResult(
                    step_id=step.id,
                    status=StepStatus.PASSED,
                    confidence=0.9,
                    start_sec=1,
                    end_sec=24,
                    observation="Visible.",
                )
                for step in sop.steps
            ],
        )
        return finalize_result(model, sop, source_duration=28)


def test_only_one_job_can_be_active_and_temp_file_is_deleted(tmp_path):
    async def scenario():
        sop = load_sop()
        analyzer = SlowFakeAnalyzer()
        manager = JobManager(analyzer)
        first_video = tmp_path / "first.mp4"
        first_video.write_bytes(b"temporary")
        second_video = tmp_path / "second.mp4"
        second_video.write_bytes(b"temporary")

        first = await manager.submit(first_video, sop, "qwen3-vl:2b-instruct")
        assert first.status == JobStatus.QUEUED
        assert first.progress_percent == 5
        assert first.model_name == "qwen3-vl:2b-instruct"

        try:
            await manager.submit(second_video, sop, "qwen3-vl:4b-instruct")
            raise AssertionError("The second active job should have been rejected")
        except AnalysisInProgress:
            pass

        for _ in range(50):
            current = await manager.get(first.analysis_id)
            if current and current.status == JobStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)

        current = await manager.get(first.analysis_id)
        assert current is not None
        assert current.status == JobStatus.SUCCEEDED
        assert current.progress_percent == 100
        assert current.progress_stage.value == "complete"
        assert current.progress_is_estimated is False
        assert current.model_name == "qwen3-vl:2b-instruct"
        assert analyzer.model_name == "qwen3-vl:2b-instruct"
        assert current.expires_at is not None
        assert not first_video.exists()
        second_video.unlink()

    asyncio.run(scenario())


def test_identical_video_uses_cached_result(tmp_path):
    async def scenario():
        sop = load_sop()
        analyzer = SlowFakeAnalyzer()
        manager = JobManager(analyzer)

        async def submit_and_wait(path: Path):
            job = await manager.submit(path, sop, "qwen3-vl:2b-instruct")
            for _ in range(50):
                current = await manager.get(job.analysis_id)
                if current and current.status == JobStatus.SUCCEEDED:
                    return current
                await asyncio.sleep(0.01)
            raise AssertionError("Analysis did not complete")

        first_video = tmp_path / "first.mp4"
        first_video.write_bytes(b"same video content")
        first = await submit_and_wait(first_video)

        second_video = tmp_path / "second.mp4"
        second_video.write_bytes(b"same video content")
        second = await submit_and_wait(second_video)

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert second.progress_message == "Analysis loaded from cache."
        assert analyzer.calls == 1
        assert not second_video.exists()

    asyncio.run(scenario())
