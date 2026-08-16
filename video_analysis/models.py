from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StepStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class JobStatus(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProgressStage(str, Enum):
    UPLOADED = "uploaded"
    PREPARING_VIDEO = "preparing_video"
    MODEL_ANALYSIS = "model_analysis"
    PREPARING_REPORT = "preparing_report"
    COMPLETE = "complete"
    FAILED = "failed"


class ModelOption(BaseModel):
    id: str
    label: str
    description: str
    is_default: bool = False


class PoseSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hand_orientation: str
    fingers_interlaced: bool | None = None
    fingers_bent: bool | None = None
    motion_pattern: str
    symmetry_required: bool = False
    distinguishing_cue: str


class StepDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    order: int = Field(gt=0)
    label: str
    instruction: str
    required: bool = True
    requires_both_sides: bool = False
    pose_signature: PoseSignature


class SOPDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    standard_version: str
    definition_version: int = Field(gt=0)
    title: str
    source_url: str
    duration_min_seconds: float = Field(ge=0)
    duration_max_seconds: float | None = Field(default=None, gt=0)
    steps: list[StepDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_definition(self) -> "SOPDefinition":
        if (
            self.duration_max_seconds is not None
            and self.duration_max_seconds < self.duration_min_seconds
        ):
            raise ValueError("duration_max_seconds must be >= duration_min_seconds")
        if len({step.id for step in self.steps}) != len(self.steps):
            raise ValueError("step ids must be unique")
        if len({step.order for step in self.steps}) != len(self.steps):
            raise ValueError("step order values must be unique")
        return self


class ModelStepResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: str
    status: StepStatus
    confidence: float = Field(ge=0, le=1)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)
    observation: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_time_range(self) -> "ModelStepResult":
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec < self.start_sec
        ):
            raise ValueError("end_sec must be >= start_sec")
        return self


class ModelAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    procedure_start_sec: float | None = Field(default=None, ge=0)
    procedure_end_sec: float | None = Field(default=None, ge=0)
    summary: str = Field(min_length=1, max_length=1500)
    steps: list[ModelStepResult]


class ModelSegmentAnalysis(BaseModel):
    """The single dominant SOP action visible in one short video window."""

    model_config = ConfigDict(extra="ignore")

    step_id: str | None = None
    status: StepStatus
    confidence: float = Field(ge=0, le=1)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)
    observation: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_time_range(self) -> "ModelSegmentAnalysis":
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec < self.start_sec
        ):
            raise ValueError("end_sec must be >= start_sec")
        return self


class StepResult(ModelStepResult):
    step_order: int = Field(gt=0)
    step_label: str


class AnalysisResult(BaseModel):
    sop_id: str
    standard_version: str
    definition_version: int
    overall_status: StepStatus
    source_video_duration_sec: float
    procedure_start_sec: float | None
    procedure_end_sec: float | None
    procedure_duration_sec: float | None
    duration_compliant: bool | None
    summary: str
    steps: list[StepResult]
    warnings: list[str] = Field(default_factory=list)


class AnalysisJob(BaseModel):
    analysis_id: str
    model_name: str
    status: JobStatus
    created_at: datetime
    progress_percent: int = Field(default=0, ge=0, le=100)
    progress_stage: ProgressStage = ProgressStage.UPLOADED
    progress_message: str = "Video uploaded."
    progress_is_estimated: bool = False
    expires_at: datetime | None = None
    cache_hit: bool = False
    result: AnalysisResult | None = None
    error: str | None = None


class JobCreated(BaseModel):
    analysis_id: str
    model_name: str
    status: JobStatus
    progress_percent: int = Field(default=0, ge=0, le=100)
    progress_stage: ProgressStage = ProgressStage.UPLOADED
    progress_message: str = "Video uploaded."
    progress_is_estimated: bool = False
    expires_at: datetime | None = None
    cache_hit: bool = False


class ErrorResponse(BaseModel):
    error: str
    message: str
