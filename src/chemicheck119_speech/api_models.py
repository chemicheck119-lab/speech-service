"""Strict public models for the bounded Speech API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat


SCHEMA_VERSION = "chemicheck119-speech-api-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(StrictModel):
    code: str
    message: str
    retryable: bool


class ErrorResponse(StrictModel):
    schema_version: Literal["chemicheck119-speech-api-v1"] = SCHEMA_VERSION
    request_id: str
    error: ErrorDetail


class QualitySignals(StrictModel):
    avg_log_probability: float
    no_speech_probability: float
    compression_ratio: NonNegativeFloat
    calibrated_correctness_probability: Literal[False]


class TranscriptSegmentResponse(StrictModel):
    start_seconds: NonNegativeFloat
    end_seconds: NonNegativeFloat
    text: str
    quality_signals: QualitySignals


class TranscriptResponse(StrictModel):
    text: str
    segments: list[TranscriptSegmentResponse]
    audio_seconds: PositiveFloat
    voiced_seconds: NonNegativeFloat


class AudioInputResponse(StrictModel):
    media_type: Literal["audio/wav"]
    channels: Literal[1, 2]
    sample_width_bits: Literal[16]
    sample_rate_hz: int
    duration_seconds: PositiveFloat
    audio_retained: Literal[False]


class RuntimeResponse(StrictModel):
    implementation: Literal["faster-whisper"]
    package_version: Literal["1.2.1"]
    service_version: str
    model: str
    requested_device: str
    requested_compute_type: str
    actual_device: str
    actual_compute_type: str
    initialization_fallback: str | None
    hotwords_used: Literal[False]
    processing_seconds: NonNegativeFloat
    real_time_factor: NonNegativeFloat


class SafetyBoundaryResponse(StrictModel):
    uncertainty_preserved: Literal[True]
    quality_signals_are_calibrated_probabilities: Literal[False]
    chemical_identification_performed: Literal[False]
    cas_confirmation_performed: Literal[False]
    risk_assessment_performed: Literal[False]
    decision_support_only: Literal[True]


class TranscriptionResponse(StrictModel):
    schema_version: Literal["chemicheck119-speech-api-v1"] = SCHEMA_VERSION
    request_id: str
    status: Literal["TRANSCRIBED", "ABSTAINED_NO_TRANSCRIPT"]
    abstained: bool
    transcript: TranscriptResponse
    input: AudioInputResponse
    runtime: RuntimeResponse
    safety_boundary: SafetyBoundaryResponse


class LiveResponse(StrictModel):
    schema_version: Literal["chemicheck119-speech-api-v1"] = SCHEMA_VERSION
    request_id: str
    status: Literal["LIVE"] = "LIVE"


class ReadyResponse(StrictModel):
    schema_version: Literal["chemicheck119-speech-api-v1"] = SCHEMA_VERSION
    request_id: str
    status: Literal["READY"] = "READY"
    max_audio_bytes: int
    max_audio_seconds: PositiveFloat


class NotReadyResponse(StrictModel):
    schema_version: Literal["chemicheck119-speech-api-v1"] = SCHEMA_VERSION
    request_id: str
    status: Literal["NOT_READY"] = "NOT_READY"
    reason: Literal["MODEL_LOAD_FAILED"] = "MODEL_LOAD_FAILED"
    error_type: str | None


__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "LiveResponse",
    "NotReadyResponse",
    "ReadyResponse",
    "SCHEMA_VERSION",
    "TranscriptionResponse",
]
