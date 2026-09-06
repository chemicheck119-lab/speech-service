"""Bounded HTTP boundary for uncertainty-preserving speech transcription."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import hmac
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Annotated, Any, AsyncIterator
from uuid import uuid4
import wave

from fastapi import FastAPI, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from chemicheck119_speech import __version__
from chemicheck119_speech.api_models import (
    ErrorDetail,
    ErrorResponse,
    LiveResponse,
    NotReadyResponse,
    ReadyResponse,
    SCHEMA_VERSION,
    TranscriptionResponse,
)
from chemicheck119_speech.runtime import (
    FasterWhisperTranscriber,
    Transcript,
    Transcriber,
)


REQUEST_ID_HEADER = "X-Request-Id"
API_KEY_HEADER = "X-API-Key"
API_KEY_SCHEME = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
ALLOWED_CONTENT_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})
MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_AUDIO_SECONDS = 60.0
MAX_QUEUE_WAIT_SECONDS = 1.0
MAX_SEGMENTS = 2_000
MAX_TRANSCRIPT_CHARACTERS = 20_000
MAX_SEGMENT_CHARACTERS = 2_000
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LOGGER = logging.getLogger("chemicheck119_speech.api")
KNOWN_ROUTES = frozenset({"/health/live", "/health/ready", "/api/v1/transcriptions"})


class SpeechApiError(RuntimeError):
    """Client-safe error without audio or transcript content."""

    def __init__(
        self, code: str, message: str, *, status_code: int, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _request_id(request: Request) -> str:
    return str(request.state.request_id)


def _route_label(path: str) -> str:
    return path if path in KNOWN_ROUTES else "<unmatched>"


def _error_response(request_id: str, error: SpeechApiError) -> JSONResponse:
    payload = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
        ),
    )
    return JSONResponse(
        status_code=error.status_code,
        content=payload.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


def _authenticate(request: Request, supplied: str | None) -> None:
    if request.app.state.allow_anonymous:
        return
    expected = request.app.state.api_key
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        raise SpeechApiError(
            "UNAUTHORIZED",
            "유효한 Speech API Key가 필요합니다.",
            status_code=401,
        )


async def _read_audio(request: Request, *, max_audio_bytes: int) -> bytes:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise SpeechApiError(
            "UNSUPPORTED_AUDIO_TYPE",
            "PCM WAV 음성만 지원합니다.",
            status_code=415,
        )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise SpeechApiError(
                "INVALID_CONTENT_LENGTH",
                "Content-Length가 올바르지 않습니다.",
                status_code=400,
            ) from error
        if declared_length <= 0 or declared_length > max_audio_bytes:
            raise SpeechApiError(
                "AUDIO_SIZE_OUT_OF_RANGE",
                "음성 크기가 허용 범위를 벗어났습니다.",
                status_code=413,
            )

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_audio_bytes:
            raise SpeechApiError(
                "AUDIO_SIZE_OUT_OF_RANGE",
                "음성 크기가 허용 범위를 벗어났습니다.",
                status_code=413,
            )
    if not body:
        raise SpeechApiError(
            "EMPTY_AUDIO",
            "음성 본문이 비어 있습니다.",
            status_code=400,
        )
    return bytes(body)


def _validate_wav(content: bytes, *, max_audio_seconds: float) -> dict[str, Any]:
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
    except (EOFError, wave.Error) as error:
        raise SpeechApiError(
            "INVALID_WAV",
            "유효한 PCM WAV 파일이 아닙니다.",
            status_code=422,
        ) from error

    if (
        channels not in {1, 2}
        or sample_width != 2
        or not 8_000 <= sample_rate <= 48_000
        or frame_count <= 0
        or compression != "NONE"
    ):
        raise SpeechApiError(
            "UNSUPPORTED_WAV_FORMAT",
            "8~48kHz, mono/stereo, 16-bit PCM WAV만 지원합니다.",
            status_code=422,
        )
    duration = frame_count / sample_rate
    if not math.isfinite(duration) or duration <= 0 or duration > max_audio_seconds:
        raise SpeechApiError(
            "AUDIO_DURATION_OUT_OF_RANGE",
            "음성 길이가 허용 범위를 벗어났습니다.",
            status_code=422,
        )
    return {
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "sample_rate_hz": sample_rate,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SpeechApiError(
            "MODEL_OUTPUT_INVALID",
            f"STT 출력의 {label} 값이 올바르지 않습니다.",
            status_code=502,
        )
    return float(value)


def _response_payload(
    *,
    request_id: str,
    transcript: Transcript,
    wav: dict[str, Any],
    elapsed_seconds: float,
    transcriber: Transcriber,
) -> TranscriptionResponse:
    text = transcript.text.strip()
    if len(text) > MAX_TRANSCRIPT_CHARACTERS or len(transcript.segments) > MAX_SEGMENTS:
        raise SpeechApiError(
            "MODEL_OUTPUT_INVALID",
            "STT 출력이 응답 상한을 초과했습니다.",
            status_code=502,
        )
    audio_seconds = _finite_number(transcript.audio_seconds, "audio_seconds")
    voiced_seconds = _finite_number(transcript.voiced_seconds, "voiced_seconds")
    if (
        audio_seconds <= 0
        or abs(audio_seconds - float(wav["duration_seconds"])) > 0.25
        or voiced_seconds < 0
        or voiced_seconds > audio_seconds + 0.1
    ):
        raise SpeechApiError(
            "MODEL_OUTPUT_INVALID",
            "STT 출력의 음성 길이가 올바르지 않습니다.",
            status_code=502,
        )

    segments: list[dict[str, Any]] = []
    previous_end = 0.0
    for segment in transcript.segments:
        start = _finite_number(segment.start_seconds, "segment.start_seconds")
        end = _finite_number(segment.end_seconds, "segment.end_seconds")
        segment_text = segment.text.strip()
        if (
            start < 0
            or end < start
            or start + 0.01 < previous_end
            or end > audio_seconds + 0.1
            or len(segment_text) > MAX_SEGMENT_CHARACTERS
        ):
            raise SpeechApiError(
                "MODEL_OUTPUT_INVALID",
                "STT 구간 출력이 올바르지 않습니다.",
                status_code=502,
            )
        previous_end = end
        avg_log_probability = _finite_number(
            segment.avg_log_probability, "avg_log_probability"
        )
        no_speech_probability = _finite_number(
            segment.no_speech_probability, "no_speech_probability"
        )
        compression_ratio = _finite_number(
            segment.compression_ratio, "compression_ratio"
        )
        if not 0 <= no_speech_probability <= 1 or compression_ratio < 0:
            raise SpeechApiError(
                "MODEL_OUTPUT_INVALID",
                "STT 품질 신호가 허용 범위를 벗어났습니다.",
                status_code=502,
            )
        segments.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "text": segment_text,
                "quality_signals": {
                    "avg_log_probability": avg_log_probability,
                    "no_speech_probability": no_speech_probability,
                    "compression_ratio": compression_ratio,
                    "calibrated_correctness_probability": False,
                },
            }
        )

    abstained = not text
    model_name = Path(str(getattr(transcriber, "model", "injected-transcriber"))).name
    requested_device = str(getattr(transcriber, "requested_device", "unknown"))
    requested_compute_type = str(
        getattr(transcriber, "requested_compute_type", "unknown")
    )
    actual_device = str(getattr(transcriber, "actual_device", "unknown"))
    actual_compute_type = str(getattr(transcriber, "actual_compute_type", "unknown"))
    return TranscriptionResponse.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "status": "ABSTAINED_NO_TRANSCRIPT" if abstained else "TRANSCRIBED",
            "abstained": abstained,
            "transcript": {
                "text": text,
                "segments": segments,
                "audio_seconds": audio_seconds,
                "voiced_seconds": voiced_seconds,
            },
            "input": {
                "media_type": "audio/wav",
                "channels": wav["channels"],
                "sample_width_bits": wav["sample_width_bits"],
                "sample_rate_hz": wav["sample_rate_hz"],
                "duration_seconds": wav["duration_seconds"],
                "audio_retained": False,
            },
            "runtime": {
                "implementation": "faster-whisper",
                "package_version": "1.2.1",
                "service_version": __version__,
                "model": model_name,
                "requested_device": requested_device,
                "requested_compute_type": requested_compute_type,
                "actual_device": actual_device,
                "actual_compute_type": actual_compute_type,
                "initialization_fallback": getattr(
                    transcriber, "initialization_fallback", None
                ),
                "hotwords_used": False,
                "processing_seconds": elapsed_seconds,
                "real_time_factor": elapsed_seconds / max(audio_seconds, 0.001),
            },
            "safety_boundary": {
                "uncertainty_preserved": True,
                "quality_signals_are_calibrated_probabilities": False,
                "chemical_identification_performed": False,
                "cas_confirmation_performed": False,
                "risk_assessment_performed": False,
                "decision_support_only": True,
            },
        }
    )


def _load_transcriber_from_env() -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber(
        model=os.getenv("CHEMICHECK119_SPEECH_MODEL", "small"),
        device=os.getenv("CHEMICHECK119_SPEECH_DEVICE", "cpu"),
        compute_type=os.getenv("CHEMICHECK119_SPEECH_COMPUTE_TYPE", "int8"),
        cpu_threads=int(os.getenv("CHEMICHECK119_SPEECH_CPU_THREADS", "4")),
        download_root=os.getenv("CHEMICHECK119_SPEECH_DOWNLOAD_ROOT"),
        local_files_only=_env_flag("CHEMICHECK119_SPEECH_LOCAL_FILES_ONLY", True),
    )


def create_app(
    *,
    transcriber: Transcriber | None = None,
    api_key: str | None = None,
    allow_anonymous: bool | None = None,
    max_audio_bytes: int = MAX_AUDIO_BYTES,
    max_audio_seconds: float = MAX_AUDIO_SECONDS,
    max_concurrent_transcriptions: int = 1,
    max_queue_wait_seconds: float = MAX_QUEUE_WAIT_SECONDS,
) -> FastAPI:
    if max_audio_bytes <= 0 or max_audio_seconds <= 0 or max_queue_wait_seconds <= 0:
        raise ValueError("audio bounds must be positive")
    if not 1 <= max_concurrent_transcriptions <= 4:
        raise ValueError("max_concurrent_transcriptions must be in [1, 4]")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.startup_error = None
        if transcriber is not None:
            application.state.transcriber = transcriber
        else:
            try:
                application.state.transcriber = await run_in_threadpool(
                    _load_transcriber_from_env
                )
            except Exception as error:
                application.state.transcriber = None
                application.state.startup_error = type(error).__name__
                LOGGER.error(
                    json.dumps(
                        {
                            "event": "speech_runtime_startup_failed",
                            "error_type": type(error).__name__,
                        }
                    )
                )
        yield

    application = FastAPI(
        title="케미체크119 Speech API",
        version=__version__,
        description=(
            "제한된 WAV 음성을 전사하고 모델 고유 품질 신호를 보존합니다. "
            "CAS 확정·화학 위험 판단·CAMEO 실행은 수행하지 않습니다."
        ),
        lifespan=lifespan,
    )
    application.state.transcriber = transcriber
    application.state.startup_error = None
    application.state.api_key = (
        api_key if api_key is not None else os.getenv("CHEMICHECK119_SPEECH_API_KEY")
    )
    application.state.allow_anonymous = (
        allow_anonymous
        if allow_anonymous is not None
        else _env_flag("CHEMICHECK119_SPEECH_ALLOW_ANONYMOUS", False)
    )
    application.state.max_audio_bytes = max_audio_bytes
    application.state.max_audio_seconds = max_audio_seconds
    application.state.transcription_semaphore = asyncio.Semaphore(
        max_concurrent_transcriptions
    )
    application.state.max_queue_wait_seconds = max_queue_wait_seconds

    @application.middleware("http")
    async def request_boundary(request: Request, call_next: Any) -> JSONResponse:
        supplied_request_id = request.headers.get(REQUEST_ID_HEADER)
        if supplied_request_id and not REQUEST_ID_PATTERN.fullmatch(
            supplied_request_id
        ):
            error = SpeechApiError(
                "INVALID_REQUEST_ID",
                "X-Request-Id 형식이 올바르지 않습니다.",
                status_code=400,
            )
            return _error_response(f"REQ-{uuid4().hex.upper()}", error)
        request_id = supplied_request_id or f"REQ-{uuid4().hex.upper()}"
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        LOGGER.info(
            json.dumps(
                {
                    "event": "speech_http_request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "route": _route_label(request.url.path),
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        )
        return response

    @application.exception_handler(SpeechApiError)
    async def speech_api_error_handler(
        request: Request, error: SpeechApiError
    ) -> JSONResponse:
        return _error_response(_request_id(request), error)

    @application.get("/health/live", response_model=LiveResponse)
    async def live(request: Request) -> LiveResponse:
        return LiveResponse(request_id=_request_id(request))

    @application.get(
        "/health/ready",
        response_model=ReadyResponse,
        responses={503: {"model": NotReadyResponse}},
    )
    async def ready(request: Request) -> JSONResponse:
        if request.app.state.transcriber is None:
            payload = NotReadyResponse(
                request_id=_request_id(request),
                error_type=request.app.state.startup_error,
            )
            return JSONResponse(
                status_code=503,
                content=payload.model_dump(mode="json"),
            )
        payload = ReadyResponse(
            request_id=_request_id(request),
            max_audio_bytes=request.app.state.max_audio_bytes,
            max_audio_seconds=request.app.state.max_audio_seconds,
        )
        return JSONResponse(content=payload.model_dump(mode="json"))

    @application.post(
        "/api/v1/transcriptions",
        response_model=TranscriptionResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def transcribe_audio(
        request: Request,
        supplied_api_key: Annotated[str | None, Security(API_KEY_SCHEME)] = None,
    ) -> TranscriptionResponse:
        _authenticate(request, supplied_api_key)
        active_transcriber = request.app.state.transcriber
        if active_transcriber is None:
            raise SpeechApiError(
                "MODEL_NOT_READY",
                "STT 모델을 사용할 수 없습니다.",
                status_code=503,
                retryable=True,
            )
        semaphore = request.app.state.transcription_semaphore
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=request.app.state.max_queue_wait_seconds,
            )
        except TimeoutError as error:
            raise SpeechApiError(
                "TRANSCRIBER_BUSY",
                "STT 처리 용량이 가득 찼습니다. 잠시 후 다시 시도하세요.",
                status_code=429,
                retryable=True,
            ) from error
        try:
            content = await _read_audio(
                request, max_audio_bytes=request.app.state.max_audio_bytes
            )
            wav = _validate_wav(
                content, max_audio_seconds=request.app.state.max_audio_seconds
            )
            with tempfile.TemporaryDirectory(
                prefix="chemicheck119-speech-"
            ) as directory:
                audio_path = Path(directory) / "request.wav"
                descriptor = os.open(
                    audio_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as destination:
                    destination.write(content)
                started = time.perf_counter()
                try:
                    transcript = await run_in_threadpool(
                        active_transcriber.transcribe, audio_path, None
                    )
                except Exception as error:
                    LOGGER.error(
                        json.dumps(
                            {
                                "event": "speech_transcription_failed",
                                "request_id": _request_id(request),
                                "error_type": type(error).__name__,
                            }
                        )
                    )
                    raise SpeechApiError(
                        "TRANSCRIPTION_FAILED",
                        "음성 전사에 실패했습니다.",
                        status_code=503,
                        retryable=True,
                    ) from error
                elapsed = time.perf_counter() - started
                try:
                    return _response_payload(
                        request_id=_request_id(request),
                        transcript=transcript,
                        wav=wav,
                        elapsed_seconds=elapsed,
                        transcriber=active_transcriber,
                    )
                except SpeechApiError:
                    raise
                except Exception as error:
                    LOGGER.error(
                        json.dumps(
                            {
                                "event": "speech_model_output_invalid",
                                "request_id": _request_id(request),
                                "error_type": type(error).__name__,
                            }
                        )
                    )
                    raise SpeechApiError(
                        "MODEL_OUTPUT_INVALID",
                        "STT 출력 형식이 올바르지 않습니다.",
                        status_code=502,
                    ) from error
        finally:
            semaphore.release()

    return application


app = create_app()


def run() -> None:
    """Run the bounded Speech API with access logs disabled."""

    import uvicorn

    host = os.getenv("CHEMICHECK119_SPEECH_API_HOST", "127.0.0.1")
    port = int(os.getenv("CHEMICHECK119_SPEECH_API_PORT", "8080"))
    if host not in LOCAL_HOSTS and (
        not os.getenv("CHEMICHECK119_SPEECH_API_KEY")
        or _env_flag("CHEMICHECK119_SPEECH_ALLOW_ANONYMOUS", False)
    ):
        raise RuntimeError("로컬호스트 외 Speech API는 API Key가 필요합니다.")
    uvicorn.run(
        "chemicheck119_speech.api:app",
        host=host,
        port=port,
        workers=1,
        access_log=False,
    )


__all__ = ["SCHEMA_VERSION", "SpeechApiError", "app", "create_app", "run"]
