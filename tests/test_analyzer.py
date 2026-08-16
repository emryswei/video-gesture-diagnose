from pathlib import Path

import video_analysis.analyzer as analyzer_module
from video_analysis.analyzer import (
    VideoAnalyzer,
    apply_landmark_guard,
    backs_candidate_pairs,
    finalize_result,
    merge_segment_analyses,
    merge_segment_classifications,
    open_hand_candidate_windows,
    palm_candidate_windows,
    refine_timeline_ranges,
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
from video_analysis.video import SampledFrame, SampledVideo, VisualBoundary


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


def test_merge_trims_thumb_start_after_confirmed_backs_of_fingers():
    sop = load_sop()
    results = [
        ModelSegmentAnalysis(
            step_id="backs_of_fingers",
            status=StepStatus.PASSED,
            confidence=0.95,
            start_sec=25.492,
            end_sec=27.037,
            observation="Bent knuckles rub the opposite palm.",
        ),
        ModelSegmentAnalysis(
            step_id="thumbs",
            status=StepStatus.PASSED,
            confidence=0.95,
            start_sec=25.492,
            end_sec=31.672,
            observation="The thumb is clasped and rotated.",
        ),
        ModelSegmentAnalysis(
            step_id="thumbs",
            status=StepStatus.PASSED,
            confidence=0.95,
            start_sec=30.127,
            end_sec=34.762,
            observation="Thumb rubbing continues.",
        ),
    ]

    merged = merge_segment_classifications(results, sop)
    by_id = {step.step_id: step for step in merged.steps}

    assert by_id["backs_of_fingers"].end_sec == 27.037
    assert by_id["thumbs"].start_sec == 27.037


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


def test_landmark_guard_keeps_full_palm_candidate_range():
    classification = ModelSegmentAnalysis(
        step_id="palm_to_palm",
        status=StepStatus.PASSED,
        confidence=0.95,
        start_sec=12.1,
        end_sec=13.6,
        observation="Palms rub together.",
    )
    evidence = [
        HandFrameEvidence(12.1, 1, 0.86, 0.32, 0.9),
        HandFrameEvidence(13.6, 1, 0.88, 0.30, 0.9),
        HandFrameEvidence(14.9, 1, 0.84, 0.34, 0.9),
    ]

    guarded = apply_landmark_guard(
        classification,
        evidence,
        allow_palm_range=True,
    )

    assert guarded.start_sec == 12.1
    assert guarded.end_sec == 14.9


def test_landmark_guard_changes_open_fingers_from_backs_to_dorsum():
    classification = ModelSegmentAnalysis(
        step_id="backs_of_fingers",
        status=StepStatus.PASSED,
        confidence=0.95,
        start_sec=16.2,
        end_sec=20.9,
        observation="Model confused the hand back with folded knuckles.",
    )
    evidence = [
        HandFrameEvidence(16.2, 1, 0.78, 0.62, 0.9),
        HandFrameEvidence(17.8, 1, 0.73, 0.48, 0.9),
        HandFrameEvidence(20.9, 1, 0.77, 0.59, 0.9),
    ]

    guarded = apply_landmark_guard(
        classification,
        evidence,
        allow_dorsum_relabel=True,
    )

    assert guarded.step_id == "palm_over_dorsum"
    assert guarded.start_sec == 16.2
    assert guarded.end_sec == 20.9


def test_landmark_guard_changes_confused_interlaced_to_dorsum():
    classification = ModelSegmentAnalysis(
        step_id="palms_fingers_interlaced",
        status=StepStatus.PASSED,
        confidence=0.9,
        start_sec=16.2,
        end_sec=20.9,
        observation="Model confused two similar open-hand actions.",
    )
    evidence = [
        HandFrameEvidence(16.2, 1, 0.78, 0.62, 0.9),
        HandFrameEvidence(17.8, 1, 0.73, 0.48, 0.9),
        HandFrameEvidence(20.9, 1, 0.77, 0.59, 0.9),
    ]

    guarded = apply_landmark_guard(
        classification,
        evidence,
        allow_dorsum_relabel=True,
    )

    assert guarded.step_id == "palm_over_dorsum"
    assert guarded.status == StepStatus.PASSED


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


def test_open_hand_window_keeps_tracking_across_one_missing_detection():
    frames = [SampledFrame(index, b"jpeg") for index in range(6)]
    evidence = {
        0: HandFrameEvidence(0, 1, 0.60, 0.30, 0.9),
        1: HandFrameEvidence(1, 2, 0.78, 0.62, 0.9),
        2: HandFrameEvidence(2, 1, 0.73, 0.48, 0.9),
        3: HandFrameEvidence(3, 0),
        4: HandFrameEvidence(4, 1, 0.77, 0.59, 0.9),
        5: HandFrameEvidence(5, 1, 0.63, 0.25, 0.9),
    }

    windows = open_hand_candidate_windows(frames, evidence)

    assert [[frame.timestamp_sec for frame in window] for window in windows] == [
        [1, 2, 4]
    ]


def test_palm_candidate_uses_full_continuous_extended_compact_run():
    frames = [SampledFrame(index * 0.25, b"jpeg") for index in range(16)]
    evidence = {
        round(frame.timestamp_sec, 3): HandFrameEvidence(
            frame.timestamp_sec,
            1,
            0.86 if 2 <= index <= 14 else 0.60,
            0.32 if 2 <= index <= 14 else 0.55,
            0.9,
        )
        for index, frame in enumerate(frames)
    }

    windows = palm_candidate_windows(frames, evidence)

    assert len(windows) == 1
    assert [frame.timestamp_sec for frame in windows[0]] == [0.5, 2.0, 3.5]


def test_timeline_ranges_use_visual_changes_between_action_centers():
    analysis = ModelAnalysis(
        procedure_start_sec=10,
        procedure_end_sec=44,
        summary="Detected actions.",
        steps=[
            ModelStepResult(
                step_id="palm_to_palm",
                status=StepStatus.PASSED,
                confidence=0.95,
                start_sec=12.11,
                end_sec=14.857,
                observation="Palms rub together.",
            ),
            ModelStepResult(
                step_id="palm_over_dorsum",
                status=StepStatus.PASSED,
                confidence=0.95,
                start_sec=16.223,
                end_sec=20.857,
                observation="Palm rubs dorsum.",
            ),
            ModelStepResult(
                step_id="palms_fingers_interlaced",
                status=StepStatus.PASSED,
                confidence=0.90,
                start_sec=22.402,
                end_sec=23.947,
                observation="Fingers interlace.",
            ),
        ],
    )
    boundaries = [
        VisualBoundary(15.356, 0.12),
        VisualBoundary(16.105, 0.10),
        VisualBoundary(21.848, 0.11),
        VisualBoundary(22.098, 0.09),
        VisualBoundary(25.344, 0.15),
    ]

    refined = refine_timeline_ranges(analysis, boundaries, source_duration=49.44)
    by_id = {step.step_id: step for step in refined.steps}

    assert by_id["palm_to_palm"].end_sec == 15.356
    assert by_id["palm_over_dorsum"].start_sec == 15.356
    assert by_id["palm_over_dorsum"].end_sec == 21.848
    assert by_id["palms_fingers_interlaced"].start_sec == 21.848


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


def test_video_analyzer_uses_timeline_evidence_for_palm_candidate(monkeypatch):
    sop = load_sop()
    primary = SampledVideo(
        duration_sec=24,
        frames=[SampledFrame(float(index), b"jpeg") for index in range(16)],
    )
    timeline = SampledVideo(
        duration_sec=24,
        frames=[SampledFrame(index * 0.25, b"jpeg") for index in range(16)],
    )

    def fake_sample_video(_path, *, max_frames, **_kwargs):
        return timeline if max_frames == 240 else primary

    monkeypatch.setattr(analyzer_module, "sample_video", fake_sample_video)
    monkeypatch.setattr(analyzer_module, "visual_change_boundaries", lambda _frames: [])
    monkeypatch.setenv("TIMELINE_MAX_FRAMES", "240")

    class FakeHandAnalyzer:
        def __init__(self):
            self.call_count = 0

        def analyze(self, frames):
            self.call_count += 1
            is_timeline = self.call_count == 1
            return [
                HandFrameEvidence(
                    frame.timestamp_sec,
                    1,
                    0.86 if is_timeline and 2 <= index <= 14 else 0.55,
                    0.32 if is_timeline and 2 <= index <= 14 else 0.60,
                    0.9,
                )
                for index, frame in enumerate(frames)
            ]

    class RecordingVLM:
        model_name = "qwen3-vl:2b-instruct"

        def set_frame_evidence(self, _evidence):
            pass

        def analyze_segment(self, frames, _definition):
            timestamps = [frame.timestamp_sec for frame in frames]
            if timestamps == [0.5, 2.0, 3.5]:
                return ModelSegmentAnalysis(
                    step_id="palm_to_palm",
                    status=StepStatus.PASSED,
                    confidence=0.95,
                    start_sec=0.5,
                    end_sec=2.0,
                    observation="Palms rub together.",
                )
            return ModelSegmentAnalysis(
                step_id=None,
                status=StepStatus.UNCERTAIN,
                confidence=0.2,
                observation="Insufficient evidence in this segment.",
            )

    analyzer = VideoAnalyzer(vlm=RecordingVLM(), hand_analyzer=FakeHandAnalyzer())

    result = analyzer.analyze(Path("video.mp4"), sop)
    palm = next(step for step in result.steps if step.step_id == "palm_to_palm")

    assert palm.status == StepStatus.PASSED
    assert palm.start_sec == 0.5
    assert palm.end_sec == 3.5
