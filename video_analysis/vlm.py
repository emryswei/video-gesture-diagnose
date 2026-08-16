from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import httpx

from .models import (
    ModelAnalysis,
    ModelSegmentAnalysis,
    SOPDefinition,
    StepStatus,
)
from .video import SampledFrame


class ModelServiceError(RuntimeError):
    """A safe model-service error that can be shown to the user."""


SYSTEM_PROMPT = """You are a strict visual auditor for a standard operating procedure.
Judge only visible evidence in the supplied timestamped frames. Never infer hidden or
off-camera actions. Use uncertain when sampling, blur, or occlusion prevents a reliable
decision. For bilateral steps, passed requires visible evidence for both sides. A clearly
one-sided or visibly incorrect action is failed. Return JSON only, with no markdown."""


def build_prompt(sop: SOPDefinition) -> str:
    steps = [
        {
            "step_id": step.id,
            "instruction": step.instruction,
            "requires_both_sides": step.requires_both_sides,
            "pose_signature": step.pose_signature.model_dump(exclude_none=True),
        }
        for step in sorted(sop.steps, key=lambda item: item.order)
    ]
    return f"""Audit the frames against this SOP definition:
{json.dumps({'duration_seconds': [sop.duration_min_seconds, sop.duration_max_seconds], 'steps': steps}, indent=2)}

V1 checks whether each action clearly appears; do not judge repetition count or per-step duration.
The supplied frames are one chronological segment of a larger video. Judge each step only
against this segment. Mark a step passed only when it is visibly present here. Mark it failed
only when a visibly attempted action is clearly incorrect; absence from this segment is
uncertain, not failed. Return exactly one item for every configured step_id. Segment start/end
values must refer only to visibly procedural frames in this segment, or be null.

Required JSON shape:
{{
  "procedure_start_sec": 0.0,
  "procedure_end_sec": 0.0,
  "summary": "brief evidence-based conclusion in English",
  "steps": [
    {{
      "step_id": "configured id",
      "status": "passed|failed|uncertain",
      "confidence": 0.0,
      "start_sec": 0.0,
      "end_sec": 0.0,
      "observation": "visible evidence or the specific reason evidence is insufficient"
    }}
  ]
}}"""


def build_segment_prompt(sop: SOPDefinition) -> str:
    steps = [
        {
            "step_id": step.id,
            "label": step.label,
            "instruction": step.instruction,
            "requires_both_sides": step.requires_both_sides,
            "pose_signature": step.pose_signature.model_dump(exclude_none=True),
        }
        for step in sorted(sop.steps, key=lambda item: item.order)
    ]
    return f"""Classify one short chronological video window against this SOP:
{json.dumps({'steps': steps}, indent=2)}

Choose at most one dominant configured step_id. Compare all configured actions before
choosing; similar hand gestures must remain separate. Use null when the window contains
an intro, outro, product-only shot, unrelated action, or no clearly identifiable SOP action.
Do not report several SOP steps from the same window.

Treat each pose_signature as required visual evidence, especially distinguishing_cue.
Do not classify straight interlaced fingers as backs of fingers; backs_of_fingers requires
visibly bent or folded fingers with their backs or knuckles contacting the opposite palm.
When MediaPipe support is supplied, high finger extension with wide tip spread contradicts
backs_of_fingers; lower extension with compact tip spread supports bent fingers. MediaPipe
may detect only one hand during overlap, so use it to check finger geometry, not step identity.

For a selected step, passed means the characteristic action is clearly visible. Failed is
only for a clearly attempted but visibly incorrect action. Use uncertain when the action,
side, or motion cannot be distinguished. start_sec and end_sec must be within the supplied
frame timestamps and cover only visible evidence. For a null step_id, status must be
uncertain and timestamps must be null.

Required JSON shape:
{{
  "step_id": "one configured id or null",
  "status": "passed|failed|uncertain",
  "confidence": 0.0,
  "start_sec": 0.0,
  "end_sec": 0.0,
  "observation": "brief visible evidence in English"
}}"""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ModelServiceError("The model returned an invalid response. Please try again.")
        cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelServiceError("The model returned an invalid response. Please try again.") from exc
    if not isinstance(value, dict):
        raise ModelServiceError("The model returned an invalid response. Please try again.")
    return value


class OpenAICompatibleVLM:
    def __init__(self, model_name: str | None = None) -> None:
        self.base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
        self.api_key = os.getenv("MODEL_API_KEY", "")
        self.model_name = model_name or os.getenv("MODEL_NAME", "qwen3-vl:4b-instruct")
        self.timeout_seconds = float(os.getenv("MODEL_TIMEOUT_SECONDS", "600"))
        self.segment_max_tokens = int(os.getenv("MODEL_SEGMENT_MAX_TOKENS", "220"))
        if self.segment_max_tokens <= 0:
            raise ValueError("MODEL_SEGMENT_MAX_TOKENS must be greater than zero")
        self._frame_evidence: dict[float, str] = {}

    def set_frame_evidence(self, evidence: dict[float, str]) -> None:
        self._frame_evidence = {
            round(timestamp, 3): text for timestamp, text in evidence.items()
        }

    def _request(
        self,
        frames: list[SampledFrame],
        prompt: str,
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, frame in enumerate(frames, start=1):
            evidence_text = self._frame_evidence.get(round(frame.timestamp_sec, 3))
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Frame {index}, source timestamp {frame.timestamp_sec:.3f} seconds"
                        f"{f'; {evidence_text}' if evidence_text else ''}:"
                    ),
                }
            )
            encoded = base64.b64encode(frame.jpeg_bytes).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            model_text = body["choices"][0]["message"]["content"]
            if isinstance(model_text, list):
                model_text = "".join(
                    item.get("text", "") for item in model_text if isinstance(item, dict)
                )
            if not isinstance(model_text, str):
                raise KeyError("missing model content")
            return _extract_json(model_text)
        except ModelServiceError:
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelServiceError(
                "The model service is unavailable or returned an invalid response. Please try again."
            ) from exc

    def analyze(self, frames: list[SampledFrame], sop: SOPDefinition) -> ModelAnalysis:
        return ModelAnalysis.model_validate(
            self._request(frames, build_prompt(sop), max_tokens=1800)
        )

    def analyze_segment(
        self,
        frames: list[SampledFrame],
        sop: SOPDefinition,
    ) -> ModelSegmentAnalysis:
        result = ModelSegmentAnalysis.model_validate(
            self._request(
                frames,
                build_segment_prompt(sop),
                max_tokens=self.segment_max_tokens,
            )
        )
        valid_ids = {step.id for step in sop.steps}
        if result.step_id not in valid_ids:
            return result.model_copy(
                update={
                    "step_id": None,
                    "status": StepStatus.UNCERTAIN,
                    "start_sec": None,
                    "end_sec": None,
                }
            )
        if result.start_sec is None or result.end_sec is None:
            return result.model_copy(
                update={
                    "status": StepStatus.UNCERTAIN,
                    "start_sec": None,
                    "end_sec": None,
                    "observation": (
                        f"{result.observation} The segment did not include a complete "
                        "evidence timestamp range."
                    ),
                }
            )
        window_start = min(frame.timestamp_sec for frame in frames)
        window_end = max(frame.timestamp_sec for frame in frames)
        return result.model_copy(
            update={
                "start_sec": max(window_start, min(result.start_sec, window_end)),
                "end_sec": max(window_start, min(result.end_sec, window_end)),
            }
        )
