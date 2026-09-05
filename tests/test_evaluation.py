from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_speech.evaluation import evaluate_archives, load_hotwords
from chemicheck119_speech.runtime import Transcript, TranscriptSegment


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 800)
    return output.getvalue()


class FakeTranscriber:
    def transcribe(self, audio_path: Path, hotwords: str | None) -> Transcript:
        text = "가스 누출" if hotwords else "가자 누출"
        segment = TranscriptSegment(0.0, 0.1, text, -0.2, 0.01, 1.0)
        return Transcript(text, (segment,), 0.1, 0.1)


class EvaluationTest(unittest.TestCase):
    def test_load_hotwords_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.txt"
            path.write_text("가스\n# comment\n가스\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_hotwords(path)

    def test_runs_balanced_paired_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio.zip"
            label_path = root / "labels.zip"
            with zipfile.ZipFile(audio_path, "w") as archive:
                for name in ("a", "b"):
                    archive.writestr(f"nested/{name}.wav", wav_bytes())
            with zipfile.ZipFile(label_path, "w") as archive:
                for name in ("a", "b"):
                    label = {
                        "recordId": f"record-{name}",
                        "utterances": [
                            {
                                "startAt": 0,
                                "endAt": 100,
                                "text": "가스 누출",
                            }
                        ],
                    }
                    archive.writestr(
                        f"nested/{name}.json",
                        json.dumps(label, ensure_ascii=False),
                    )
            summary, rows = evaluate_archives(
                audio_archive=audio_path,
                label_archive=label_path,
                transcriber=FakeTranscriber(),
                terms=["가스"],
                model="fixture",
                device="cpu",
                compute_type="int8",
                expected_records=None,
                generated_at="2026-09-05T00:00:00Z",
            )
            self.assertEqual(4, len(rows))
            self.assertEqual(2, summary["dataset"]["record_count"])
            self.assertEqual("evaluation", summary["usage_role"])
            self.assertGreater(
                summary["variants"]["baseline"]["cer"],
                summary["variants"]["hotwords"]["cer"],
            )
            self.assertLess(summary["paired_comparison"]["estimate"], 0)
            self.assertNotIn("record-a", json.dumps(rows, ensure_ascii=False))

    def test_marks_limited_run_as_development_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio.zip"
            label_path = root / "labels.zip"
            with zipfile.ZipFile(audio_path, "w") as audio_archive:
                audio_archive.writestr("a.wav", wav_bytes())
            label = {
                "recordId": "record-a",
                "utterances": [{"startAt": 0, "endAt": 100, "text": "가스"}],
            }
            with zipfile.ZipFile(label_path, "w") as label_archive:
                label_archive.writestr(
                    "a.json", json.dumps(label, ensure_ascii=False)
                )
            summary, _ = evaluate_archives(
                audio_archive=audio_path,
                label_archive=label_path,
                transcriber=FakeTranscriber(),
                terms=["가스"],
                model="fixture",
                device="cpu",
                compute_type="int8",
                limit=1,
            )
            self.assertEqual("development", summary["usage_role"])
            self.assertEqual(
                "speech_aihub119_gwangju_fire_smoke_1",
                summary["experiment_id"],
            )


if __name__ == "__main__":
    unittest.main()
