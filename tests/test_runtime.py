from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
