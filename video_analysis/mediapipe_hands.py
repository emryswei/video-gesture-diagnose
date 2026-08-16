from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from .video import SampledFrame


FINGER_JOINTS = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


class MediaPipeAnalysisError(RuntimeError):
    """Raised when optional landmark extraction cannot run."""


@dataclass(frozen=True)
class HandFrameEvidence:
    timestamp_sec: float
    detected_hands: int
    mean_extension_score: float | None = None
    mean_tip_spread_ratio: float | None = None
    mean_handedness_confidence: float | None = None

    @property
    def finger_shape(self) -> str:
        if self.mean_extension_score is None:
            return "not_available"
        if self.mean_extension_score >= 0.72:
            return "mostly_extended"
        if self.mean_extension_score <= 0.45:
            return "mostly_bent"
        return "mixed"

    def to_prompt_text(self) -> str:
        if not self.detected_hands:
            return "MediaPipe support: no reliable hand landmarks detected."
        extension = (
            f"{self.mean_extension_score:.2f}"
            if self.mean_extension_score is not None
            else "n/a"
        )
        spread = (
            f"{self.mean_tip_spread_ratio:.2f}"
            if self.mean_tip_spread_ratio is not None
            else "n/a"
        )
        return (
            f"MediaPipe support: hands={self.detected_hands}; "
            f"finger_shape={self.finger_shape}; extension={extension}; "
            f"tip_spread={spread}. Treat landmarks as supporting evidence only."
        )


def is_extended_and_wide(evidence: HandFrameEvidence) -> bool:
    return (
        evidence.mean_extension_score is not None
        and evidence.mean_tip_spread_ratio is not None
        and evidence.mean_extension_score >= 0.78
        and evidence.mean_tip_spread_ratio >= 0.52
    )


def is_bent_and_compact(evidence: HandFrameEvidence) -> bool:
    return (
        evidence.mean_extension_score is not None
        and evidence.mean_tip_spread_ratio is not None
        and evidence.mean_extension_score <= 0.68
        and evidence.mean_tip_spread_ratio <= 0.38
    )


def _distance(first, second) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def _joint_angle(first, middle, last) -> float:
    left = (first.x - middle.x, first.y - middle.y, first.z - middle.z)
    right = (last.x - middle.x, last.y - middle.y, last.z - middle.z)
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator <= 1e-9:
        return 0.0
    cosine = max(
        -1.0,
        min(1.0, sum(a * b for a, b in zip(left, right, strict=True)) / denominator),
    )
    return math.degrees(math.acos(cosine))


def finger_extension_score(landmarks: Sequence[object]) -> float:
    scores: list[float] = []
    for mcp, pip, dip, tip in FINGER_JOINTS:
        pip_angle = _joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        dip_angle = _joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])
        mean_angle = (pip_angle + dip_angle) / 2
        scores.append(max(0.0, min(1.0, (mean_angle - 70) / 100)))
    return sum(scores) / len(scores)


def fingertip_spread_ratio(landmarks: Sequence[object]) -> float | None:
    palm_width = _distance(landmarks[5], landmarks[17])
    if palm_width <= 1e-6:
        return None
    adjacent_tip_distances = [
        _distance(landmarks[first], landmarks[second])
        for first, second in ((8, 12), (12, 16), (16, 20))
    ]
    return sum(adjacent_tip_distances) / len(adjacent_tip_distances) / palm_width


class MediaPipeHandAnalyzer:
    def __init__(
        self,
        model_path: Path,
        *,
        min_detection_confidence: float = 0.35,
        min_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
    ) -> None:
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def analyze(self, frames: list[SampledFrame]) -> list[HandFrameEvidence]:
        if not self.model_path.is_file():
            raise MediaPipeAnalysisError(
                "MediaPipe model is missing. Run scripts/setup_mediapipe.ps1."
            )
        try:
            import mediapipe as mp
            import numpy as np

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.model_path.resolve())
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            evidence: list[HandFrameEvidence] = []
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                for frame in frames:
                    rgb = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
                    image_data = np.ascontiguousarray(np.asarray(rgb))
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image_data,
                    )
                    result = landmarker.detect_for_video(
                        mp_image,
                        int(round(frame.timestamp_sec * 1000)),
                    )
                    evidence.append(_summarize_result(frame.timestamp_sec, result))
            return evidence
        except MediaPipeAnalysisError:
            raise
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            raise MediaPipeAnalysisError(
                "MediaPipe hand landmark extraction failed."
            ) from exc


def _summarize_result(timestamp_sec: float, result) -> HandFrameEvidence:
    hand_count = len(result.hand_landmarks)
    if not hand_count:
        return HandFrameEvidence(timestamp_sec=timestamp_sec, detected_hands=0)

    extension_scores: list[float] = []
    spread_ratios: list[float] = []
    handedness_scores: list[float] = []
    for index, normalized_landmarks in enumerate(result.hand_landmarks):
        metric_landmarks = (
            result.hand_world_landmarks[index]
            if index < len(result.hand_world_landmarks)
            else normalized_landmarks
        )
        extension_scores.append(finger_extension_score(metric_landmarks))
        spread = fingertip_spread_ratio(normalized_landmarks)
        if spread is not None:
            spread_ratios.append(spread)
        if index < len(result.handedness) and result.handedness[index]:
            handedness_scores.append(result.handedness[index][0].score)

    return HandFrameEvidence(
        timestamp_sec=timestamp_sec,
        detected_hands=hand_count,
        mean_extension_score=sum(extension_scores) / len(extension_scores),
        mean_tip_spread_ratio=(
            sum(spread_ratios) / len(spread_ratios) if spread_ratios else None
        ),
        mean_handedness_confidence=(
            sum(handedness_scores) / len(handedness_scores)
            if handedness_scores
            else None
        ),
    )
