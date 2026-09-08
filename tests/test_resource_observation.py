from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient

from chemicheck119_speech.api import (
    APPLICATION_LOG_HANDLER_MARKER,
    _configure_application_logging,
    create_app,
)
from chemicheck119_speech.resource_observation import capture_resource_snapshot
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

    def __init__(self, *, text: str = "아세톤 누출 의심") -> None:
        self.text = text

    def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript:
        return Transcript(
            text=self.text,
            segments=(
                TranscriptSegment(
                    start_seconds=0.0,
                    end_seconds=0.1,
                    text=self.text,
                    avg_log_probability=-0.42,
                    no_speech_probability=0.08,
                    compression_ratio=1.15,
                ),
            ),
            audio_seconds=0.1,
            voiced_seconds=0.1,
        )


class ResourceObservationTest(unittest.TestCase):
    def test_reads_bounded_cgroup_v2_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.current").write_text("123456\n", encoding="ascii")
            (root / "memory.peak").write_text("234567\n", encoding="ascii")
            (root / "memory.max").write_text("8589934592\n", encoding="ascii")
            statm = root / "statm"
            statm.write_text("1000 250 0 0 0 0 0\n", encoding="ascii")
            snapshot = capture_resource_snapshot(
                cgroup_root=root, proc_statm_path=statm
            )

        self.assertTrue(snapshot["resource_observation_available"])
        self.assertEqual("v2", snapshot["cgroup_version"])
        self.assertEqual(123456, snapshot["cgroup_memory_current_bytes"])
        self.assertEqual(234567, snapshot["cgroup_memory_peak_bytes"])
        self.assertEqual(8589934592, snapshot["cgroup_memory_limit_bytes"])
        self.assertGreater(snapshot["process_current_rss_bytes"], 0)
        self.assertGreater(snapshot["process_max_rss_bytes"], 0)

    def test_rejects_unbounded_or_non_numeric_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.current").write_text("secret-value\n", encoding="ascii")
            (root / "memory.peak").write_text("9" * 40, encoding="ascii")
            (root / "memory.max").write_text("max\n", encoding="ascii")
            snapshot = capture_resource_snapshot(
                cgroup_root=root, proc_statm_path=root / "missing-statm"
            )

        self.assertEqual("v2", snapshot["cgroup_version"])
        self.assertIsNone(snapshot["cgroup_memory_current_bytes"])
        self.assertIsNone(snapshot["cgroup_memory_peak_bytes"])
        self.assertIsNone(snapshot["cgroup_memory_limit_bytes"])
        self.assertIsNone(snapshot["process_current_rss_bytes"])
        self.assertGreater(snapshot["process_max_rss_bytes"], 0)

    def test_logs_numeric_resource_sample_without_transcript(self) -> None:
        snapshot = {
            "resource_observation_available": True,
            "cgroup_version": "v2",
            "cgroup_memory_current_bytes": 123,
            "cgroup_memory_peak_bytes": 456,
            "cgroup_memory_limit_bytes": 789,
            "process_current_rss_bytes": 100,
            "process_max_rss_bytes": 200,
        }
        app = create_app(
            transcriber=FakeTranscriber(text="민감한 아세톤 전사문"),
            allow_anonymous=True,
        )
        with patch(
            "chemicheck119_speech.api.capture_resource_snapshot",
            return_value=snapshot,
        ):
            with self.assertLogs("chemicheck119_speech.api", level="INFO") as logs:
                with TestClient(app) as client:
                    response = client.post(
                        "/api/v1/transcriptions",
                        content=wav_bytes(),
                        headers={
                            "Content-Type": "audio/wav",
                            "X-Request-Id": "REQ-RESOURCE-TEST-1",
                        },
                    )

        self.assertEqual(200, response.status_code)
        serialized = "\n".join(logs.output)
        events = [json.loads(record.getMessage()) for record in logs.records]
        self.assertIn("speech_resource_sample", serialized)
        self.assertIn("cgroup_memory_peak_bytes", serialized)
        self.assertNotIn("민감한", serialized)
        self.assertNotIn("아세톤", serialized)
        forbidden_keys = {
            "transcript",
            "text",
            "segments",
            "audio_bytes",
            "api_key",
            "cas_number",
            "risk_assessment",
        }
        for event in events:
            self.assertTrue(forbidden_keys.isdisjoint(event))

    def test_observation_failure_does_not_fail_transcription(self) -> None:
        app = create_app(transcriber=FakeTranscriber(), allow_anonymous=True)
        with patch(
            "chemicheck119_speech.api.capture_resource_snapshot",
            side_effect=RuntimeError("sensitive internal path"),
        ):
            with self.assertLogs("chemicheck119_speech.api", level="INFO") as logs:
                with TestClient(app) as client:
                    response = client.post(
                        "/api/v1/transcriptions",
                        content=wav_bytes(),
                        headers={"Content-Type": "audio/wav"},
                    )

        self.assertEqual(200, response.status_code)
        serialized = "\n".join(logs.output)
        self.assertIn('"resource_observation_available": false', serialized)
        self.assertIn('"resource_observation_error_type": "RuntimeError"', serialized)
        self.assertNotIn("sensitive internal path", serialized)

    def test_application_logger_configuration_is_idempotent(self) -> None:
        logger = logging.getLogger("chemicheck119-test-resource-logger")
        previous_handlers = list(logger.handlers)
        previous_level = logger.level
        previous_propagate = logger.propagate
        try:
            logger.handlers = []
            _configure_application_logging(logger)
            _configure_application_logging(logger)
            marked = [
                handler
                for handler in logger.handlers
                if getattr(handler, APPLICATION_LOG_HANDLER_MARKER, False)
            ]
            self.assertEqual(1, len(marked))
            self.assertEqual(logging.INFO, logger.level)
            self.assertFalse(logger.propagate)
        finally:
            for handler in logger.handlers:
                if handler not in previous_handlers:
                    handler.close()
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate


if __name__ == "__main__":
    unittest.main()
