from video_analysis.models import StepStatus
from video_analysis.sop import load_sop
from video_analysis.video import SampledFrame
from video_analysis.vlm import OpenAICompatibleVLM


def test_segment_timestamps_are_limited_to_the_supplied_window(monkeypatch):
    vlm = OpenAICompatibleVLM()
    monkeypatch.setattr(
        vlm,
        "_request",
        lambda *_args, **_kwargs: {
            "step_id": load_sop().steps[0].id,
            "status": "passed",
            "confidence": 0.9,
            "start_sec": 1,
            "end_sec": 99,
            "observation": "The action is visible.",
        },
    )
    frames = [
        SampledFrame(timestamp_sec=10, jpeg_bytes=b"jpeg"),
        SampledFrame(timestamp_sec=12, jpeg_bytes=b"jpeg"),
    ]

    result = vlm.analyze_segment(frames, load_sop())

    assert result.status == StepStatus.PASSED
    assert result.start_sec == 10
    assert result.end_sec == 12


def test_passed_segment_without_timestamps_becomes_uncertain(monkeypatch):
    vlm = OpenAICompatibleVLM()
    monkeypatch.setattr(
        vlm,
        "_request",
        lambda *_args, **_kwargs: {
            "step_id": load_sop().steps[0].id,
            "status": "passed",
            "confidence": 0.9,
            "start_sec": None,
            "end_sec": None,
            "observation": "The action may be visible.",
        },
    )

    result = vlm.analyze_segment(
        [SampledFrame(timestamp_sec=10, jpeg_bytes=b"jpeg")],
        load_sop(),
    )

    assert result.status == StepStatus.UNCERTAIN
    assert result.start_sec is None
    assert result.end_sec is None
