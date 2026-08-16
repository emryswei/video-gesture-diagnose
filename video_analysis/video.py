from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import av
from av.error import FFmpegError
from PIL import Image, ImageChops, ImageStat


MIN_VIDEO_SECONDS = 15.0
MAX_VIDEO_SECONDS = 60.0


class VideoValidationError(ValueError):
    """A safe validation error that can be shown to the user."""


@dataclass(frozen=True)
class SampledFrame:
    timestamp_sec: float
    jpeg_bytes: bytes


@dataclass(frozen=True)
class SampledVideo:
    duration_sec: float
    frames: list[SampledFrame]


@dataclass(frozen=True)
class VisualBoundary:
    timestamp_sec: float
    score: float


def visual_change_boundaries(frames: list[SampledFrame]) -> list[VisualBoundary]:
    if len(frames) < 2:
        return []
    boundaries: list[VisualBoundary] = []
    previous = Image.open(io.BytesIO(frames[0].jpeg_bytes)).convert("RGB")
    for frame in frames[1:]:
        current = Image.open(io.BytesIO(frame.jpeg_bytes)).convert("RGB")
        difference = ImageChops.difference(current, previous)
        score = sum(ImageStat.Stat(difference).mean) / (3 * 255)
        boundaries.append(VisualBoundary(frame.timestamp_sec, score))
        previous = current
    return boundaries


def _duration_seconds(container: av.container.InputContainer, stream: av.video.VideoStream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration = float(container.duration / av.time_base)
    else:
        raise VideoValidationError("The video duration could not be determined.")
    if not math.isfinite(duration) or duration <= 0:
        raise VideoValidationError("The video has an invalid duration.")
    return duration


def _encode_frame(frame: av.VideoFrame, max_image_edge: int) -> bytes:
    image = frame.to_image().convert("RGB")
    image.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    return output.getvalue()


def sample_video(
    path: Path,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 96,
    max_image_edge: int = 720,
) -> SampledVideo:
    if sample_fps <= 0 or max_frames <= 0 or max_image_edge <= 0:
        raise ValueError("Video sampling settings must be positive")

    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise VideoValidationError("The uploaded file does not contain a video stream.")

            duration = _duration_seconds(container, stream)
            if duration < MIN_VIDEO_SECONDS or duration > MAX_VIDEO_SECONDS:
                raise VideoValidationError(
                    f"Video duration must be between {MIN_VIDEO_SECONDS:g} and "
                    f"{MAX_VIDEO_SECONDS:g} seconds."
                )

            target_count = min(max_frames, max(1, math.ceil(duration * sample_fps)))
            target_times = [
                (index + 0.5) * duration / target_count for index in range(target_count)
            ]
            sampled: list[SampledFrame] = []
            target_index = 0
            last_jpeg: bytes | None = None

            for frame in container.decode(stream):
                timestamp = float(frame.time) if frame.time is not None else None
                if timestamp is None:
                    continue
                if target_index >= len(target_times):
                    break
                if timestamp < target_times[target_index]:
                    continue

                last_jpeg = _encode_frame(frame, max_image_edge)
                while (
                    target_index < len(target_times)
                    and target_times[target_index] <= timestamp
                ):
                    sampled.append(
                        SampledFrame(
                            timestamp_sec=round(target_times[target_index], 3),
                            jpeg_bytes=last_jpeg,
                        )
                    )
                    target_index += 1

            if last_jpeg is not None:
                while target_index < len(target_times):
                    sampled.append(
                        SampledFrame(
                            timestamp_sec=round(target_times[target_index], 3),
                            jpeg_bytes=last_jpeg,
                        )
                    )
                    target_index += 1

            if not sampled:
                raise VideoValidationError("No readable frames were found in the video.")
            return SampledVideo(duration_sec=duration, frames=sampled)
    except VideoValidationError:
        raise
    except (FFmpegError, OSError, ValueError) as exc:
        raise VideoValidationError(
            "The video could not be decoded. Please upload a valid MP4, MOV, WebM, MKV, or AVI file."
        ) from exc
