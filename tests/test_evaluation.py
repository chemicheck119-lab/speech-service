from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_speech.evaluation import (
    _read_bounded,
    evaluate_archives,
    load_hotwords,
)
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
    def test_can_run_a_baseline_only_cross_region_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.zip"
            labels = root / "labels.zip"
            with zipfile.ZipFile(audio, "w") as archive:
                archive.writestr("record-a.wav", wav_bytes())
            label = {
                "recordId": "seoul-record-a",
                "utterances": [
                    {"startAt": 0, "endAt": 100, "text": "가스 누출"}
                ],
            }
            with zipfile.ZipFile(labels, "w") as archive:
                archive.writestr(
                    "record-a.json", json.dumps(label, ensure_ascii=False)
                )
            summary, rows = evaluate_archives(
                audio_archive=audio,
                label_archive=labels,
                transcriber=FakeTranscriber(),
                terms=["가스"],
                model="small",
                device="cpu",
                compute_type="int8",
                dataset_provenance={
                    "dataset_id": "aihub_71768_seoul_fire",
                    "dataset_version": "v1",
                    "evaluation_id": "speech_aihub119_seoul_fire_validation_1",
                    "record_count": 1,
                },
                variants=("baseline",),
            )
            self.assertEqual(["baseline"], summary["runtime"]["variants"])
            self.assertNotIn("paired_comparison", summary)
            self.assertEqual(1, len(rows))

    def test_rejects_archive_member_above_expanded_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "payload.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("payload.json", b"0123456789")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "expanded size"):
                    _read_bounded(archive, "payload.json", 5)

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
                dataset_provenance={
                    "dataset_id": "fixture",
                    "dataset_version": "1",
                    "evaluation_id": "fixture-evaluation",
                    "record_count": 2,
                    "manifest_sha256": "a" * 64,
                    "archive_sha256": {},
                },
                generated_at="2026-09-05T00:00:00Z",
            )
            self.assertEqual(4, len(rows))
            self.assertEqual(2, summary["dataset"]["record_count"])
            self.assertEqual("evaluation", summary["usage_role"])
            self.assertEqual("fixture-evaluation", summary["experiment_id"])
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

    def test_preserves_registered_development_scope_for_full_dev_run(self) -> None:
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
                dataset_provenance={
                    "dataset_id": "fixture-dev",
                    "dataset_version": "1",
                    "evaluation_id": "fixture-dev-wind-1",
                    "record_count": 1,
                    "usage_role": "development",
                    "evidence_scope": "procedural wind; not field-radio validation",
                    "split": "Training internal dev",
                    "condition": "wind_snr0",
                },
                variants=("baseline",),
            )

            self.assertEqual("development", summary["usage_role"])
            self.assertEqual("fixture-dev-wind-1", summary["experiment_id"])
            self.assertEqual(
                "procedural wind; not field-radio validation",
                summary["evidence_scope"],
            )
            self.assertEqual("Training internal dev", summary["dataset"]["split"])
            self.assertEqual("wind_snr0", summary["dataset"]["condition"])


if __name__ == "__main__":
    unittest.main()
