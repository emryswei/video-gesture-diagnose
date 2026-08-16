from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .models import (
    AnalysisResult,
    ModelAnalysis,
    ModelSegmentAnalysis,
    ModelStepResult,
    SOPDefinition,
    StepResult,
    StepStatus,
)
from .mediapipe_hands import (
    HandFrameEvidence,
    MediaPipeAnalysisError,
    MediaPipeHandAnalyzer,
    is_bent_and_compact,
    is_extended_and_wide,
)
from .video import SampledFrame, sample_video
from .vlm import OpenAICompatibleVLM


ProgressCallback = Callable[[int, str, str, bool], None]


def apply_landmark_guard(
    result: ModelSegmentAnalysis,
    evidence: list[HandFrameEvidence],
    *,
    allow_interlaced_relabel: bool = False,
) -> ModelSegmentAnalysis:
    if (
        result.step_id == "backs_of_fingers"
        and sum(is_bent_and_compact(item) for item in evidence) < 2
    ):
        return result.model_copy(
            update={
                "step_id": None,
                "status": StepStatus.UNCERTAIN,
                "start_sec": None,
                "end_sec": None,
                "observation": (
                    "MediaPipe landmarks did not confirm bent, compact fingers across "
                    "two frames, so the backs-of-fingers classification needs review."
                ),
            }
        )
    if (
        allow_interlaced_relabel
        and len(evidence) == 2
        and all(is_extended_and_wide(item) for item in evidence)
        and result.step_id
        in {"palm_to_palm", "palms_fingers_interlaced", "backs_of_fingers"}
        and result.status == StepStatus.PASSED
    ):
        return result.model_copy(
            update={
                "step_id": "palms_fingers_interlaced",
                "confidence": min(result.confidence, 0.85),
                "observation": (
                    f"{result.observation} MediaPipe confirmed extended fingers with "
                    "wide fingertip spacing across both frames."
                ),
            }
        )
    return result


def interlaced_candidate_pairs(
    frames: list[SampledFrame],
    evidence_by_timestamp: dict[float, HandFrameEvidence],
) -> list[list[SampledFrame]]:
    pairs: list[list[SampledFrame]] = []
    for index in range(len(frames) - 1):
        pair = frames[index : index + 2]
        evidence = [
            evidence_by_timestamp.get(round(frame.timestamp_sec, 3)) for frame in pair
        ]
        if all(item is not None and is_extended_and_wide(item) for item in evidence):
            pairs.append(pair)
    return pairs


def backs_candidate_pairs(
    frames: list[SampledFrame],
    evidence_by_timestamp: dict[float, HandFrameEvidence],
) -> list[list[SampledFrame]]:
    pairs: list[list[SampledFrame]] = []
    for index in range(len(frames) - 1):
        pair = frames[index : index + 2]
        evidence = [
            evidence_by_timestamp.get(round(frame.timestamp_sec, 3)) for frame in pair
        ]
        if all(item is not None and is_bent_and_compact(item) for item in evidence):
            pairs.append(pair)
    return pairs


def split_frame_segments(
    frames: list[SampledFrame],
    target_size: int,
    overlap: int = 1,
) -> list[list[SampledFrame]]:
    if len(frames) < 2:
        raise ValueError("Segmented analysis requires at least two frames")
    if target_size < 2 or overlap < 0 or overlap >= target_size:
        raise ValueError("Segment size and overlap are invalid")
    if len(frames) <= target_size:
        return [frames]

    stride = target_size - overlap
    starts = list(range(0, len(frames) - target_size + 1, stride))
    final_start = len(frames) - target_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [frames[index : index + target_size] for index in starts]


def normalize_model_result(result: ModelAnalysis, sop: SOPDefinition) -> ModelAnalysis:
    expected_ids = {step.id for step in sop.steps}
    unique: dict[str, ModelStepResult] = {}
    for item in result.steps:
        if item.step_id in expected_ids and item.step_id not in unique:
            unique[item.step_id] = item

    ordered: list[ModelStepResult] = []
    for definition in sorted(sop.steps, key=lambda item: item.order):
        ordered.append(
            unique.get(
                definition.id,
                ModelStepResult(
                    step_id=definition.id,
                    status=StepStatus.UNCERTAIN,
                    confidence=0,
                    observation="The model did not return a result for this step.",
                ),
            )
        )
    return result.model_copy(update={"steps": ordered})


def merge_segment_analyses(
    results: list[ModelAnalysis],
    sop: SOPDefinition,
) -> ModelAnalysis:
    if not results:
        raise ValueError("At least one segment analysis is required")

    candidates: dict[str, list[ModelStepResult]] = {
        step.id: [] for step in sop.steps
    }
    procedure_starts: list[float] = []
    procedure_ends: list[float] = []
    for result in results:
        normalized = normalize_model_result(result, sop)
        if normalized.procedure_start_sec is not None:
            procedure_starts.append(normalized.procedure_start_sec)
        if normalized.procedure_end_sec is not None:
            procedure_ends.append(normalized.procedure_end_sec)
        for item in normalized.steps:
            candidates[item.step_id].append(item)

    status_priority = {
        StepStatus.PASSED: 2,
        StepStatus.FAILED: 1,
        StepStatus.UNCERTAIN: 0,
    }
    merged_steps: list[ModelStepResult] = []
    for definition in sorted(sop.steps, key=lambda item: item.order):
        options = candidates[definition.id]
        visible = [
            item
            for item in options
            if item.start_sec is not None and item.end_sec is not None
        ]
        pool = visible or options
        merged_steps.append(
            max(
                pool,
                key=lambda item: (status_priority[item.status], item.confidence),
            )
        )

    passed = sum(item.status == StepStatus.PASSED for item in merged_steps)
    failed = sum(item.status == StepStatus.FAILED for item in merged_steps)
    uncertain = len(merged_steps) - passed - failed
    return ModelAnalysis(
        procedure_start_sec=min(procedure_starts) if procedure_starts else None,
        procedure_end_sec=max(procedure_ends) if procedure_ends else None,
        summary=(
            f"Segmented visual analysis found visible evidence for {passed} of "
            f"{len(merged_steps)} steps; {failed} failed and {uncertain} need review."
        ),
        steps=merged_steps,
    )


def _contiguous_groups(
    candidates: list[ModelSegmentAnalysis],
) -> list[list[ModelSegmentAnalysis]]:
    ordered = sorted(
        (
            item
            for item in candidates
            if item.start_sec is not None and item.end_sec is not None
        ),
        key=lambda item: item.start_sec or 0,
    )
    groups: list[list[ModelSegmentAnalysis]] = []
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        previous_end = max(candidate.end_sec or 0 for candidate in groups[-1])
        previous_duration = max(
            (candidate.end_sec or 0) - (candidate.start_sec or 0)
            for candidate in groups[-1]
        )
        allowed_gap = max(0.75, previous_duration)
        if (item.start_sec or 0) <= previous_end + allowed_gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def merge_segment_classifications(
    results: list[ModelSegmentAnalysis],
    sop: SOPDefinition,
) -> ModelAnalysis:
    expected_ids = {step.id for step in sop.steps}
    candidates: dict[str, list[ModelSegmentAnalysis]] = {
        step.id: [] for step in sop.steps
    }
    for item in results:
        if item.step_id in expected_ids:
            candidates[item.step_id].append(item)

    merged_steps: list[ModelStepResult] = []
    for definition in sorted(sop.steps, key=lambda item: item.order):
        groups = _contiguous_groups(candidates[definition.id])
        if not groups:
            merged_steps.append(
                ModelStepResult(
                    step_id=definition.id,
                    status=StepStatus.UNCERTAIN,
                    confidence=0,
                    observation="No video window contained reliable evidence for this step.",
                )
            )
            continue

        best_group = max(
            groups,
            key=lambda group: (
                sum(item.confidence for item in group),
                max(item.end_sec or 0 for item in group)
                - min(item.start_sec or 0 for item in group),
            ),
        )
        passed = [item for item in best_group if item.status == StepStatus.PASSED]
        failed = [item for item in best_group if item.status == StepStatus.FAILED]
        if passed:
            status = StepStatus.PASSED
            evidence = max(passed, key=lambda item: item.confidence)
        elif failed:
            status = StepStatus.FAILED
            evidence = max(failed, key=lambda item: item.confidence)
        else:
            status = StepStatus.UNCERTAIN
            evidence = max(best_group, key=lambda item: item.confidence)
        merged_steps.append(
            ModelStepResult(
                step_id=definition.id,
                status=status,
                confidence=round(
                    sum(item.confidence for item in best_group) / len(best_group), 3
                ),
                start_sec=min(item.start_sec or 0 for item in best_group),
                end_sec=max(item.end_sec or 0 for item in best_group),
                observation=(
                    f"{evidence.observation} Confirmed across "
                    f"{len(best_group)} overlapping window(s)."
                ),
            )
        )

    visible_steps = [
        item
        for item in merged_steps
        if item.start_sec is not None and item.end_sec is not None
    ]
    passed_count = sum(item.status == StepStatus.PASSED for item in merged_steps)
    failed_count = sum(item.status == StepStatus.FAILED for item in merged_steps)
    uncertain_count = len(merged_steps) - passed_count - failed_count
    return ModelAnalysis(
        procedure_start_sec=(
            min(item.start_sec or 0 for item in visible_steps) if visible_steps else None
        ),
        procedure_end_sec=(
            max(item.end_sec or 0 for item in visible_steps) if visible_steps else None
        ),
        summary=(
            f"Overlapping-window analysis found {passed_count} passed, "
            f"{failed_count} failed, and {uncertain_count} uncertain steps."
        ),
        steps=merged_steps,
    )


def finalize_result(
    model_result: ModelAnalysis,
    sop: SOPDefinition,
    source_duration: float,
) -> AnalysisResult:
    normalized = normalize_model_result(model_result, sop)
    definitions = {step.id: step for step in sop.steps}
    steps: list[StepResult] = []
    for item in normalized.steps:
        definition = definitions[item.step_id]
        start_sec = item.start_sec
        end_sec = item.end_sec
        status = item.status
        observation = item.observation
        if start_sec is None or end_sec is None:
            start_sec = None
            end_sec = None
            if status == StepStatus.PASSED:
                status = StepStatus.UNCERTAIN
                observation = (
                    f"{observation} A complete timestamp range was not provided, "
                    "so the step cannot be confirmed."
                )
        else:
            start_sec = min(start_sec, source_duration)
            end_sec = min(end_sec, source_duration)
        steps.append(
            StepResult(
                step_id=item.step_id,
                status=status,
                confidence=item.confidence,
                start_sec=start_sec,
                end_sec=end_sec,
                observation=observation,
                step_order=definition.order,
                step_label=definition.label,
            )
        )

    warnings: list[str] = []
    start = normalized.procedure_start_sec
    end = normalized.procedure_end_sec
    procedure_duration: float | None = None
    duration_compliant: bool | None = None
    if start is not None and end is not None and end >= start:
        start = min(start, source_duration)
        end = min(end, source_duration)
        procedure_duration = round(end - start, 2)
        duration_compliant = (
            sop.duration_min_seconds
            <= procedure_duration
            <= sop.duration_max_seconds
        )
        if not duration_compliant:
            warnings.append(
                f"Procedure duration is outside the required "
                f"{sop.duration_min_seconds:g}-{sop.duration_max_seconds:g} seconds."
            )
    else:
        warnings.append("The procedure time range could not be established reliably.")

    required_steps = [
        result for result in steps if definitions[result.step_id].required
    ]
    has_failed = any(item.status == StepStatus.FAILED for item in required_steps)
    has_uncertain = any(item.status == StepStatus.UNCERTAIN for item in required_steps)
    all_passed = all(item.status == StepStatus.PASSED for item in required_steps)
    if all_passed and duration_compliant is True:
        overall = StepStatus.PASSED
    elif has_failed or duration_compliant is False:
        overall = StepStatus.FAILED
    elif has_uncertain or duration_compliant is None:
        overall = StepStatus.UNCERTAIN
    else:
        overall = StepStatus.FAILED

    return AnalysisResult(
        sop_id=sop.id,
        standard_version=sop.standard_version,
        definition_version=sop.definition_version,
        overall_status=overall,
        source_video_duration_sec=round(source_duration, 2),
        procedure_start_sec=start,
        procedure_end_sec=end,
        procedure_duration_sec=procedure_duration,
        duration_compliant=duration_compliant,
        summary=normalized.summary,
        steps=steps,
        warnings=warnings,
    )


class VideoAnalyzer:
    def __init__(
        self,
        vlm: OpenAICompatibleVLM | None = None,
        hand_analyzer: MediaPipeHandAnalyzer | None = None,
    ) -> None:
        self.vlm = vlm or OpenAICompatibleVLM()
        self.sample_fps = float(os.getenv("VIDEO_SAMPLE_FPS", "2"))
        self.max_frames = int(os.getenv("VIDEO_MAX_FRAMES", "96"))
        self.max_image_edge = int(os.getenv("VIDEO_MAX_IMAGE_EDGE", "720"))
        self.frames_per_segment = int(os.getenv("MODEL_FRAMES_PER_SEGMENT", "3"))
        self.segment_overlap = int(os.getenv("MODEL_SEGMENT_OVERLAP_FRAMES", "1"))
        mediapipe_enabled = os.getenv("MEDIAPIPE_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self.hand_analyzer = hand_analyzer
        if self.hand_analyzer is None and mediapipe_enabled:
            configured_path = Path(
                os.getenv("MEDIAPIPE_MODEL_PATH", "models/hand_landmarker.task")
            )
            if not configured_path.is_absolute():
                configured_path = Path(__file__).resolve().parents[1] / configured_path
            self.hand_analyzer = MediaPipeHandAnalyzer(configured_path)
        if not 2 <= self.frames_per_segment <= 4:
            raise ValueError("MODEL_FRAMES_PER_SEGMENT must be between 2 and 4")
        if not 0 <= self.segment_overlap < self.frames_per_segment:
            raise ValueError(
                "MODEL_SEGMENT_OVERLAP_FRAMES must be smaller than MODEL_FRAMES_PER_SEGMENT"
            )

    def analyze(
        self,
        video_path: Path,
        sop: SOPDefinition,
        progress_callback: ProgressCallback | None = None,
        model_name: str | None = None,
    ) -> AnalysisResult:
        if progress_callback:
            progress_callback(
                10,
                "preparing_video",
                "Validating the video and extracting representative frames.",
                False,
            )
        sampled = sample_video(
            video_path,
            sample_fps=self.sample_fps,
            max_frames=self.max_frames,
            max_image_edge=self.max_image_edge,
        )
        vlm = self.vlm
        if model_name and model_name != getattr(vlm, "model_name", None):
            vlm = OpenAICompatibleVLM(model_name=model_name)
        mediapipe_warning: str | None = None
        hand_evidence_by_timestamp: dict[float, HandFrameEvidence] = {}
        set_frame_evidence = getattr(vlm, "set_frame_evidence", None)
        if callable(set_frame_evidence):
            evidence_map: dict[float, str] = {}
            if self.hand_analyzer is not None:
                if progress_callback:
                    progress_callback(
                        22,
                        "preparing_video",
                        "Extracting MediaPipe hand landmarks.",
                        False,
                    )
                try:
                    hand_evidence = self.hand_analyzer.analyze(sampled.frames)
                    hand_evidence_by_timestamp = {
                        round(item.timestamp_sec, 3): item for item in hand_evidence
                    }
                    evidence_map = {
                        item.timestamp_sec: item.to_prompt_text()
                        for item in hand_evidence
                    }
                except MediaPipeAnalysisError as exc:
                    mediapipe_warning = str(exc)
            set_frame_evidence(evidence_map)
        if progress_callback:
            progress_callback(
                23,
                "preparing_video",
                f"{len(sampled.frames)} representative frames and hand landmarks prepared.",
                False,
            )
            progress_callback(
                25,
                "model_analysis",
                "AI is checking the 7 procedure steps.",
                True,
            )
        segments = split_frame_segments(
            sampled.frames,
            self.frames_per_segment,
            self.segment_overlap,
        )
        segment_results: list[ModelSegmentAnalysis] = []
        legacy_results: list[ModelAnalysis] = []
        classify_segment = getattr(vlm, "analyze_segment", None)
        for index, segment in enumerate(segments, start=1):
            if progress_callback:
                progress_callback(
                    25 + int(65 * (index - 1) / len(segments)),
                    "model_analysis",
                    f"AI is checking video segment {index} of {len(segments)}.",
                    True,
                )
            if callable(classify_segment):
                classification = classify_segment(segment, sop)
                if hand_evidence_by_timestamp:
                    segment_evidence = [
                        hand_evidence_by_timestamp[round(frame.timestamp_sec, 3)]
                        for frame in segment
                    ]
                    classification = apply_landmark_guard(
                        classification,
                        segment_evidence,
                    )
                segment_results.append(classification)
            else:
                legacy_results.append(vlm.analyze(segment, sop))
        if callable(classify_segment) and hand_evidence_by_timestamp:
            candidate_groups = [
                (
                    "interlaced-finger",
                    interlaced_candidate_pairs(
                        sampled.frames,
                        hand_evidence_by_timestamp,
                    ),
                    True,
                ),
                (
                    "backs-of-fingers",
                    backs_candidate_pairs(
                        sampled.frames,
                        hand_evidence_by_timestamp,
                    ),
                    False,
                ),
            ]
            candidate_count = sum(len(pairs) for _, pairs, _ in candidate_groups)
            candidate_index = 0
            for label, pairs, allow_interlaced_relabel in candidate_groups:
                for pair in pairs:
                    candidate_index += 1
                    if progress_callback:
                        progress_callback(
                            90 + int(
                                4 * (candidate_index - 1) / max(candidate_count, 1)
                            ),
                            "model_analysis",
                            (
                                f"AI is checking {label} evidence "
                                f"{candidate_index} of {candidate_count}."
                            ),
                            True,
                        )
                    classification = classify_segment(pair, sop)
                    pair_evidence = [
                        hand_evidence_by_timestamp[round(frame.timestamp_sec, 3)]
                        for frame in pair
                    ]
                    segment_results.append(
                        apply_landmark_guard(
                            classification,
                            pair_evidence,
                            allow_interlaced_relabel=allow_interlaced_relabel,
                        )
                    )
        model_result = (
            merge_segment_classifications(segment_results, sop)
            if segment_results
            else merge_segment_analyses(legacy_results, sop)
        )
        if progress_callback:
            progress_callback(
                95,
                "preparing_report",
                "Preparing the evidence report.",
                False,
            )
        result = finalize_result(model_result, sop, sampled.duration_sec)
        if mediapipe_warning:
            result = result.model_copy(
                update={"warnings": [*result.warnings, mediapipe_warning]}
            )
        return result
