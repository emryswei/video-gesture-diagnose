from pathlib import Path

import video_analysis.analyzer as analyzer_module
from video_analysis.analyzer import (
    VideoAnalyzer,
    apply_landmark_guard,
    backs_candidate_pairs,
    finalize_result,
    merge_segment_analyses,
    merge_segment_classifications,
    split_frame_segments,
)
from video_analysis.mediapipe_hands import HandFrameEvidence
from video_analysis.models import (
    ModelAnalysis,
    ModelSegmentAnalysis,
    ModelStepResult,
    StepStatus,
)
from video_analysis.sop import load_sop
from video_analysis.video import SampledFrame, SampledVideo


def model_result(status: StepStatus = StepStatus.PASSED) -> ModelAnalysis:
    sop = load_sop()
    return ModelAnalysis(
        procedure_start_sec=2,
        procedure_end_sec=26,
        summary="The required actions are visible.",
        steps=[
            ModelStepResult(
                step_id=step.id,
                status=status,
                confidence=0.9,
                start_sec=2 + index * 3,
                end_sec=4 + index * 3,
                observation="The action is visible.",
            )
            for index, step in enumerate(sop.steps)
        ],
    )


def test_all_steps_and_duration_must_pass():
    sop = load_sop()

    result = finalize_result(model_result(), sop, source_duration=30)

    assert result.overall_status == StepStatus.PASSED
    assert result.duration_compliant is True
    assert len(result.steps) == 7
    assert [step.step_order for step in result.steps] == list(range(1, 8))
    assert result.warnings == []
    assert "score" not in result.model_dump()


def test_missing_model_step_becomes_uncertain():
    sop = load_sop()
    incomplete = model_result().model_copy(
        update={"steps": model_result().steps[:-1]}
    )

    result = finalize_result(incomplete, sop, source_duration=30)

    assert result.overall_status == StepStatus.UNCERTAIN
    assert result.steps[-1].status == StepStatus.UNCERTAIN
    assert result.steps[-1].start_sec is None


def test_definite_failure_wins_over_uncertainty():
    sop = load_sop()
    output = model_result()
    output.steps[0].status = StepStatus.FAILED
    output.steps[1].status = StepStatus.UNCERTAIN

    result = finalize_result(output, sop, source_duration=30)

    assert result.overall_status == StepStatus.FAILED


def test_out_of_range_duration_fails():
    sop = load_sop()
    output = model_result().model_copy(
        update={"procedure_start_sec": 2, "procedure_end_sec": 15}
    )

    result = finalize_result(output, sop, source_duration=30)

    assert result.duration_compliant is False
    assert result.overall_status == StepStatus.FAILED


def test_passed_step_without_complete_timestamps_becomes_uncertain():
    sop = load_sop()
    output = model_result()
    output.steps[0].end_sec = None

    result = finalize_result(output, sop, source_duration=30)

    assert result.steps[0].status == StepStatus.UNCERTAIN
    assert result.steps[0].start_sec is None
    assert result.steps[0].end_sec is None
    assert result.overall_status == StepStatus.UNCERTAIN


def test_segment_results_merge_visible_evidence_across_the_video():
    sop = load_sop()
    first = ModelAnalysis(
        procedure_start_sec=5,
        procedure_end_sec=12,
        summary="First segment.",
        steps=[
            ModelStepResult(
                step_id=step.id,
                status=StepStatus.PASSED if step.order <= 2 else StepStatus.UNCERTAIN,
                confidence=0.9 if step.order <= 2 else 0.2,
                start_sec=5 if step.order <= 2 else None,
                end_sec=8 if step.order <= 2 else None,
                observation="Visible early." if step.order <= 2 else "Not visible here.",
            )
            for step in sop.steps
        ],
    )
    second = ModelAnalysis(
        procedure_start_sec=28,
        procedure_end_sec=39,
        summary="Second segment.",
        steps=[
            ModelStepResult(
                step_id=step.id,
                status=StepStatus.PASSED if step.order >= 6 else StepStatus.UNCERTAIN,
                confidence=0.92 if step.order >= 6 else 0.3,
                start_sec=31 if step.order >= 6 else None,
                end_sec=36 if step.order >= 6 else None,
                observation="Visible late." if step.order >= 6 else "Not visible here.",
            )
            for step in sop.steps
        ],
    )

    merged = merge_segment_analyses([first, second], sop)

    assert merged.procedure_start_sec == 5
    assert merged.procedure_end_sec == 39
    assert [step.status for step in merged.steps[:2]] == [StepStatus.PASSED] * 2
    assert [step.status for step in merged.steps[-2:]] == [StepStatus.PASSED] * 2
    assert merged.steps[3].status == StepStatus.UNCERTAIN


def test_overlapping_segments_preserve_the_end_of_the_video():
    frames = [
        SampledFrame(timestamp_sec=float(index), jpeg_bytes=b"jpeg")
        for index in range(16)
    ]

    segments = split_frame_segments(frames, target_size=3, overlap=1)

    assert [len(segment) for segment in segments] == [3] * 8
    assert [segment[0].timestamp_sec for segment in segments] == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
    ]
    assert segments[-1][-1].timestamp_sec == 15


def test_segment_classifications_are_aggregated_by_step_and_time():
    sop = load_sop()
    first_step = sop.steps[0].id
    last_step = sop.steps[-1].id
    results = [
        ModelSegmentAnalysis(
            step_id=first_step,
            status=StepStatus.PASSED,
            confidence=0.8,
            start_sec=5,
            end_sec=8,
            observation="First action visible.",
        ),
        ModelSegmentAnalysis(
            step_id=first_step,
            status=StepStatus.PASSED,
            confidence=0.9,
            start_sec=7,
            end_sec=10,
            observation="First action continues.",
        ),
        ModelSegmentAnalysis(
            step_id=None,
            status=StepStatus.UNCERTAIN,
            confidence=0.9,
            observation="Title card.",
        ),
        ModelSegmentAnalysis(
            step_id=last_step,
            status=StepStatus.PASSED,
            confidence=0.92,
            start_sec=31,
            end_sec=35,
            observation="Last action visible.",
        ),
    ]

    merged = merge_segment_classifications(results, sop)

    assert merged.procedure_start_sec == 5
    assert merged.procedure_end_sec == 35
    assert merged.steps[0].status == StepStatus.PASSED
    assert merged.steps[0].start_sec == 5
    assert merged.steps[0].end_sec == 10
    assert merged.steps[-1].status == StepStatus.PASSED
    assert merged.steps[3].status == StepStatus.UNCERTAIN


def test_landmark_guard_rejects_unconfirmed_backs_of_fingers():
    classification = ModelSegmentAnalysis(
        step_id="backs_of_fingers",
        status=StepStatus.PASSED,
        confidence=0.95,
        start_sec=22,
        end_sec=25,
        observation="Model chose bent fingers.",
    )
    evidence = [
        HandFrameEvidence(22, 1, 0.85, 0.62, 0.9),
        HandFrameEvidence(24, 1, 0.82, 0.58, 0.9),
        HandFrameEvidence(25, 1, 0.64, 0.25, 0.9),
    ]

    guarded = apply_landmark_guard(classification, evidence)

    assert guarded.step_id is None
    assert guarded.status == StepStatus.UNCERTAIN
    assert guarded.start_sec is None


def test_landmark_guard_recovers_interlaced_pair_from_confused_palm_label():
    classification = ModelSegmentAnalysis(
        step_id="palm_to_palm",
        status=StepStatus.PASSED,
        confidence=0.95,
        start_sec=22,
        end_sec=24,
        observation="Model saw facing palms.",
    )
    evidence = [
        HandFrameEvidence(22, 1, 0.85, 0.62, 0.9),
        HandFrameEvidence(24, 1, 0.82, 0.58, 0.9),
    ]

    guarded = apply_landmark_guard(
        classification,
        evidence,
        allow_interlaced_relabel=True,
    )

    assert guarded.step_id == "palms_fingers_interlaced"
    assert guarded.status == StepStatus.PASSED
    assert guarded.confidence == 0.85


def test_backs_candidate_pairs_require_two_consecutive_compact_frames():
    frames = [SampledFrame(index, b"jpeg") for index in range(4)]
    evidence = {
        0: HandFrameEvidence(0, 1, 0.80, 0.60, 0.9),
        1: HandFrameEvidence(1, 1, 0.64, 0.25, 0.9),
        2: HandFrameEvidence(2, 1, 0.55, 0.24, 0.9),
        3: HandFrameEvidence(3, 1, 0.62, 0.45, 0.9),
    }

    pairs = backs_candidate_pairs(frames, evidence)

    assert [[frame.timestamp_sec for frame in pair] for pair in pairs] == [[1, 2]]


def test_video_analyzer_sends_overlapping_windows_to_segment_classifier(monkeypatch):
    sop = load_sop()
    sampled = SampledVideo(
        duration_sec=24,
        frames=[
            SampledFrame(timestamp_sec=float(index), jpeg_bytes=b"jpeg")
            for index in range(16)
        ],
    )
    monkeypatch.setattr(analyzer_module, "sample_video", lambda *_args, **_kwargs: sampled)
    monkeypatch.setenv("MODEL_FRAMES_PER_SEGMENT", "3")

    class RecordingVLM:
        model_name = "qwen3-vl:4b-instruct"

        def __init__(self):
            self.calls = []

        def analyze_segment(self, frames, definition):
            self.calls.append(frames)
            return ModelSegmentAnalysis(
                step_id=None,
                status=StepStatus.UNCERTAIN,
                confidence=0.2,
                observation="Insufficient evidence in this segment.",
            )

    vlm = RecordingVLM()
    analyzer = VideoAnalyzer(vlm=vlm)

    analyzer.analyze(Path("video.mp4"), sop)

    assert [len(call) for call in vlm.calls] == [3] * 8
    assert [call[0].timestamp_sec for call in vlm.calls] == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
    ]
