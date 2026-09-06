from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import struct
import threading
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient
import httpx

from chemicheck119_speech.api import create_app
from chemicheck119_speech.runtime import Transcript, TranscriptSegment


def wav_bytes(*, seconds: float = 0.1, sample_rate: int = 16_000) -> bytes:
    frames = int(seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(struct.pack("<h", 0) * frames)
    return buffer.getvalue()


class FakeTranscriber:
    model = "fixture-model"
    requested_device = "cpu"
    requested_compute_type = "int8"
    actual_device = "cpu"
    actual_compute_type = "int8"
    initialization_fallback = None

    def __init__(self, *, text: str = "아세톤 누출 의심", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0
        self.path_seen: Path | None = None

    def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript:
        self.calls += 1
        self.path_seen = audio_path
        if self.fail:
            raise RuntimeError("sensitive transcript must not be logged")
        self.assert_path_is_private(audio_path)
        segments = (
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=0.1,
                text=self.text,
                avg_log_probability=-0.42,
                no_speech_probability=0.08,
                compression_ratio=1.15,
            ),
        )
        return Transcript(
            text=self.text,
            segments=segments if self.text else (),
            audio_seconds=0.1,
            voiced_seconds=0.1 if self.text else 0.0,
        )

    def assert_path_is_private(self, audio_path: Path) -> None:
        self_path_mode = audio_path.stat().st_mode & 0o777
        if self_path_mode != 0o600:
            raise AssertionError(f"unexpected temp mode: {self_path_mode:o}")


class SpeechApiTest(unittest.TestCase):
    def test_transcribes_bounded_wav_and_preserves_uncertainty(self) -> None:
        transcriber = FakeTranscriber()
        app = create_app(transcriber=transcriber, allow_anonymous=True)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={
                    "Content-Type": "audio/wav",
                    "X-Request-Id": "REQ-E2E-0001",
                },
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("REQ-E2E-0001", payload["request_id"])
        self.assertEqual("REQ-E2E-0001", response.headers["X-Request-Id"])
        self.assertEqual("TRANSCRIBED", payload["status"])
        self.assertFalse(payload["abstained"])
        self.assertEqual("아세톤 누출 의심", payload["transcript"]["text"])
        quality = payload["transcript"]["segments"][0]["quality_signals"]
        self.assertEqual(-0.42, quality["avg_log_probability"])
        self.assertFalse(quality["calibrated_correctness_probability"])
        boundary = payload["safety_boundary"]
        self.assertFalse(boundary["chemical_identification_performed"])
        self.assertFalse(boundary["cas_confirmation_performed"])
        self.assertFalse(boundary["risk_assessment_performed"])
        self.assertFalse(payload["runtime"]["hotwords_used"])
        self.assertFalse(payload["input"]["audio_retained"])
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        self.assertNotIn("cas_number", serialized)
        self.assertNotIn("hazard_level", serialized)
        self.assertNotIn("compatibility_result", serialized)
        self.assertEqual(1, transcriber.calls)
        self.assertIsNotNone(transcriber.path_seen)
        self.assertFalse(transcriber.path_seen.exists())

    def test_empty_transcript_is_explicit_abstention(self) -> None:
        transcriber = FakeTranscriber(text="")
        app = create_app(transcriber=transcriber, allow_anonymous=True)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={"Content-Type": "audio/wav"},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("ABSTAINED_NO_TRANSCRIPT", response.json()["status"])
        self.assertTrue(response.json()["abstained"])

    def test_rejects_unsupported_media_before_transcription(self) -> None:
        transcriber = FakeTranscriber()
        app = create_app(transcriber=transcriber, allow_anonymous=True)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcriptions",
                content=b"not audio",
                headers={"Content-Type": "audio/mpeg"},
            )

        self.assertEqual(415, response.status_code)
        self.assertEqual("UNSUPPORTED_AUDIO_TYPE", response.json()["error"]["code"])
        self.assertEqual(0, transcriber.calls)

    def test_rejects_size_duration_and_invalid_wav_before_transcription(self) -> None:
        transcriber = FakeTranscriber()
        app = create_app(
            transcriber=transcriber,
            allow_anonymous=True,
            max_audio_bytes=100,
            max_audio_seconds=0.05,
        )
        with TestClient(app) as client:
            oversized = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={"Content-Type": "audio/wav"},
            )
        self.assertEqual(413, oversized.status_code)
        self.assertEqual("AUDIO_SIZE_OUT_OF_RANGE", oversized.json()["error"]["code"])

        duration_app = create_app(
            transcriber=transcriber,
            allow_anonymous=True,
            max_audio_bytes=10_000,
            max_audio_seconds=0.05,
        )
        with TestClient(duration_app) as client:
            too_long = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={"Content-Type": "audio/wav"},
            )
            invalid = client.post(
                "/api/v1/transcriptions",
                content=b"not-a-wav",
                headers={"Content-Type": "audio/wav"},
            )
        self.assertEqual(422, too_long.status_code)
        self.assertEqual(
            "AUDIO_DURATION_OUT_OF_RANGE", too_long.json()["error"]["code"]
        )
        self.assertEqual(422, invalid.status_code)
        self.assertEqual("INVALID_WAV", invalid.json()["error"]["code"])
        self.assertEqual(0, transcriber.calls)

    def test_requires_api_key_when_anonymous_access_is_disabled(self) -> None:
        transcriber = FakeTranscriber()
        app = create_app(
            transcriber=transcriber,
            api_key="test-speech-api-key",
            allow_anonymous=False,
        )
        with TestClient(app) as client:
            unauthorized = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={"Content-Type": "audio/wav"},
            )
            authorized = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={
                    "Content-Type": "audio/wav",
                    "X-API-Key": "test-speech-api-key",
                },
            )

        self.assertEqual(401, unauthorized.status_code)
        self.assertEqual("UNAUTHORIZED", unauthorized.json()["error"]["code"])
        self.assertEqual(200, authorized.status_code)
        self.assertEqual(1, transcriber.calls)

    def test_failure_hides_internal_detail_and_removes_temp_audio(self) -> None:
        transcriber = FakeTranscriber(fail=True)
        app = create_app(transcriber=transcriber, allow_anonymous=True)
        with self.assertLogs("chemicheck119_speech.api", level="ERROR") as logs:
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/transcriptions",
                    content=wav_bytes(),
                    headers={"Content-Type": "audio/wav"},
                )

        self.assertEqual(503, response.status_code)
        payload = response.json()
        self.assertEqual("TRANSCRIPTION_FAILED", payload["error"]["code"])
        self.assertTrue(payload["error"]["retryable"])
        serialized_logs = "\n".join(logs.output)
        self.assertIn("RuntimeError", serialized_logs)
        self.assertNotIn("sensitive transcript", serialized_logs)
        self.assertFalse(transcriber.path_seen.exists())

    def test_invalid_model_output_fails_closed(self) -> None:
        class InvalidOutputTranscriber(FakeTranscriber):
            def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript:
                result = super().transcribe(audio_path, hotwords)
                return Transcript(
                    text=result.text,
                    segments=result.segments,
                    audio_seconds=10.0,
                    voiced_seconds=result.voiced_seconds,
                )

        transcriber = InvalidOutputTranscriber()
        app = create_app(transcriber=transcriber, allow_anonymous=True)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcriptions",
                content=wav_bytes(),
                headers={"Content-Type": "audio/wav"},
            )

        self.assertEqual(502, response.status_code)
        self.assertEqual("MODEL_OUTPUT_INVALID", response.json()["error"]["code"])
        self.assertFalse(transcriber.path_seen.exists())

    def test_concurrent_request_fails_fast_when_transcriber_is_busy(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingTranscriber(FakeTranscriber):
            def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript:
                started.set()
                if not release.wait(timeout=1):
                    raise RuntimeError("test release timeout")
                return super().transcribe(audio_path, hotwords)

        transcriber = BlockingTranscriber()
        app = create_app(
            transcriber=transcriber,
            allow_anonymous=True,
            max_queue_wait_seconds=0.01,
        )

        async def exercise() -> tuple[httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                first_task = asyncio.create_task(
                    client.post(
                        "/api/v1/transcriptions",
                        content=wav_bytes(),
                        headers={"Content-Type": "audio/wav"},
                    )
                )
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(started.is_set())
                second = await client.post(
                    "/api/v1/transcriptions",
                    content=wav_bytes(),
                    headers={"Content-Type": "audio/wav"},
                )
                release.set()
                first = await first_task
                return first, second

        first, second = asyncio.run(exercise())
        self.assertEqual(200, first.status_code)
        self.assertEqual(429, second.status_code)
        self.assertEqual("TRANSCRIBER_BUSY", second.json()["error"]["code"])
        self.assertTrue(second.json()["error"]["retryable"])

    def test_readiness_reports_injected_runtime_and_failed_startup(self) -> None:
        ready_app = create_app(transcriber=FakeTranscriber(), allow_anonymous=True)
        with TestClient(ready_app) as client:
            ready = client.get("/health/ready")
        self.assertEqual(200, ready.status_code)
        self.assertEqual("READY", ready.json()["status"])

        with patch(
            "chemicheck119_speech.api._load_transcriber_from_env",
            side_effect=RuntimeError("model path unavailable"),
        ):
            failed_app = create_app(allow_anonymous=True)
            with TestClient(failed_app) as client:
                not_ready = client.get("/health/ready")
        self.assertEqual(503, not_ready.status_code)
        self.assertEqual("MODEL_LOAD_FAILED", not_ready.json()["reason"])
        self.assertEqual("RuntimeError", not_ready.json()["error_type"])

    def test_invalid_request_id_is_replaced_in_safe_error_response(self) -> None:
        app = create_app(transcriber=FakeTranscriber(), allow_anonymous=True)
        with TestClient(app) as client:
            response = client.get(
                "/health/live", headers={"X-Request-Id": "invalid request id"}
            )
        self.assertEqual(400, response.status_code)
        self.assertEqual("INVALID_REQUEST_ID", response.json()["error"]["code"])
        self.assertRegex(response.json()["request_id"], r"^REQ-[0-9A-F]{32}$")

    def test_unmatched_path_and_local_model_path_are_not_exposed(self) -> None:
        transcriber = FakeTranscriber()
        transcriber.model = "/private/models/faster-whisper-small"
        app = create_app(transcriber=transcriber, allow_anonymous=True)
        with self.assertLogs("chemicheck119_speech.api", level="INFO") as logs:
            with TestClient(app) as client:
                unknown = client.get("/private/incident-secret")
                transcription = client.post(
                    "/api/v1/transcriptions",
                    content=wav_bytes(),
                    headers={"Content-Type": "audio/wav"},
                )

        self.assertEqual(404, unknown.status_code)
        serialized_logs = "\n".join(logs.output)
        self.assertIn("<unmatched>", serialized_logs)
        self.assertNotIn("incident-secret", serialized_logs)
        self.assertEqual(
            "faster-whisper-small", transcription.json()["runtime"]["model"]
        )
        self.assertNotIn("/private/models", json.dumps(transcription.json()))


if __name__ == "__main__":
    unittest.main()
