from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chemicheck119_speech.model_provenance import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
)
from chemicheck119_speech.runtime import FasterWhisperTranscriber


class FakeWhisperModel:
    calls: list[dict[str, object]] = []

    def __init__(self, model: str, **options: object) -> None:
        self.model = model
        self.options = options

    def transcribe(self, audio_path: str, **options: object):
        self.calls.append(options)
        segment = SimpleNamespace(
            start=1.25,
            end=2.75,
            text=" 가스 누출 ",
            avg_logprob=-0.42,
            no_speech_prob=0.08,
            compression_ratio=1.15,
        )
        info = SimpleNamespace(duration=3.0, duration_after_vad=1.5)
        return iter([segment]), info


class RuntimeTest(unittest.TestCase):
    def test_binds_verified_model_provenance_before_inference(self) -> None:
        with TemporaryDirectory() as directory:
            model_directory = Path(directory) / "model"
            model_directory.mkdir()
            model_path = model_directory / "model.bin"
            model_path.write_bytes(b"runtime-model")
            model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
            manifest_path = model_directory / MANIFEST_FILENAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "repository": "Systran/faster-whisper-small",
                        "revision": "5" * 40,
                        "model_bin_sha256": model_sha256,
                    }
                ),
                encoding="utf-8",
            )
            module = SimpleNamespace(WhisperModel=FakeWhisperModel)
            with patch.dict(sys.modules, {"faster_whisper": module}):
                transcriber = FasterWhisperTranscriber(
                    model=str(model_directory),
                    provenance_manifest=str(manifest_path),
                )

        self.assertTrue(transcriber.model_artifact_verified)
        self.assertEqual("5" * 40, transcriber.model_revision)
        self.assertEqual(model_sha256, transcriber.model_bin_sha256)

    def test_maps_faster_whisper_timestamps_and_quality_signals(self) -> None:
        module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            transcriber = FasterWhisperTranscriber()
            result = transcriber.transcribe(Path("fixture.wav"), "가스")
        self.assertEqual("가스 누출", result.text)
        self.assertEqual(1.25, result.segments[0].start_seconds)
        self.assertEqual(2.75, result.segments[0].end_seconds)
        self.assertEqual(-0.42, result.segments[0].avg_log_probability)
        self.assertEqual(0.08, result.segments[0].no_speech_probability)
        self.assertEqual(1.15, result.segments[0].compression_ratio)
        self.assertEqual(3.0, result.audio_seconds)
        self.assertEqual(1.5, result.voiced_seconds)

    def test_falls_back_to_cpu_when_cuda_initialization_fails(self) -> None:
        calls: list[tuple[str, str]] = []

        class FailingCudaModel(FakeWhisperModel):
            def __init__(self, model: str, **options: object) -> None:
                calls.append((str(options["device"]), str(options["compute_type"])))
                if options["device"] == "cuda":
                    raise RuntimeError("fixture CUDA failure")
                super().__init__(model, **options)

        module = SimpleNamespace(WhisperModel=FailingCudaModel)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            transcriber = FasterWhisperTranscriber(
                device="cuda", compute_type="float16"
            )
        self.assertEqual([("cuda", "float16"), ("cpu", "int8")], calls)
        self.assertEqual("cpu", transcriber.actual_device)
        self.assertEqual("RuntimeError", transcriber.initialization_fallback)


if __name__ == "__main__":
    unittest.main()
