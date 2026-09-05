import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile

from chemicheck119_speech.provenance import sha256_file
from chemicheck119_speech.robustness import (
    REGISTERED_VARIANTS,
    aggregate_robustness,
    archive_audio_seconds,
    load_simulation_run_summary,
    validate_local_file_size,
    validate_simulation_manifest,
)


def wav_bytes(seconds: float = 0.25, sample_rate: int = 8000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x01" * round(seconds * sample_rate))
    return output.getvalue()


def aggregate_summary(
    *, cer: float, wer: float, recall: float, precision: float, false_insertion: int
) -> dict[str, object]:
    f1 = 2 * recall * precision / (recall + precision)
    return {
        "cer": cer,
        "wer": wer,
        "failed_record_count": 0,
        "priority_term_presence": {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "false_insertion": false_insertion,
        },
    }


class RobustnessTest(unittest.TestCase):
    def test_loads_bounded_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-summary.json"
            digest = "a" * 64
            summary = {
                "profile_id": "radio-sim-v1",
                "variant_count": len(REGISTERED_VARIANTS),
                "source_manifest_sha256": digest,
                "source_audio_sha256": digest,
                "source_labels_sha256": digest,
                "priority_terms_sha256": digest,
                "seed": 119,
                "selected": {"total": 2},
                "manifests": [
                    {
                        "variant": variant,
                        "manifest": f"manifests/{variant}.json",
                        "manifest_sha256": digest,
                    }
                    for variant in sorted(REGISTERED_VARIANTS)
                ],
            }
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(summary, load_simulation_run_summary(path))

            summary["manifests"].pop(0)
            summary["variant_count"] = len(REGISTERED_VARIANTS) - 1
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest count"):
                load_simulation_run_summary(path)

    def test_validates_manifest_binding_and_field_radio_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "b" * 64
            run_summary = {
                "source_manifest_sha256": digest,
                "source_audio_sha256": digest,
                "source_labels_sha256": digest,
                "priority_terms_sha256": digest,
                "selected": {"total": 4},
            }
            manifest = {
                "classification": "derived",
                "usage_role": "evaluation",
                "preprocessing": {
                    "parameters": {
                        "profile_id": "radio-sim-v1",
                        "variant": {"id": "wind_snr10"},
                        "source_manifest_sha256": digest,
                        "source_audio_sha256": digest,
                        "source_labels_sha256": digest,
                        "priority_terms_sha256": digest,
                    }
                },
                "evidence_scope": "simulated distortion; not field-radio validation",
                "evaluation": {"id": "fixture", "record_count": 4},
                "artifacts": [
                    {
                        "path": "gs://private/run/audio/wind_snr10.zip",
                        "sha256": "c" * 64,
                    },
                    {
                        "path": "gs://private/run/labels/sampled-labels.zip",
                        "sha256": "d" * 64,
                    },
                ],
            }
            path = root / "wind_snr10.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _, audio, labels = validate_simulation_manifest(
                path,
                run_summary=run_summary,
                variant="wind_snr10",
                expected_sha256=sha256_file(path),
            )
            self.assertEqual("gs://private/run/audio/wind_snr10.zip", audio)
            self.assertEqual("gs://private/run/labels/sampled-labels.zip", labels)

            manifest["evidence_scope"] = "field validation"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "field-radio limitation"):
                validate_simulation_manifest(
                    path,
                    run_summary=run_summary,
                    variant="wind_snr10",
                    expected_sha256=sha256_file(path),
                )

    def test_counts_archive_duration_before_model_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "audio.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("a.wav", wav_bytes(0.25))
                archive.writestr("b.wav", wav_bytes(0.5))
            count, seconds = archive_audio_seconds(archive_path)
            self.assertEqual(2, count)
            self.assertAlmostEqual(0.75, seconds, places=3)

    def test_rejects_file_larger_than_materialization_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.bin"
            path.write_bytes(b"1234")
            self.assertEqual(4, validate_local_file_size(path, 4, "fixture"))
            with self.assertRaisesRegex(ValueError, "bounded range"):
                validate_local_file_size(path, 3, "fixture")

    def test_aggregates_paired_degradation_without_exposing_record_keys(self) -> None:
        summaries = {
            "clean": aggregate_summary(
                cer=0.1, wer=0.2, recall=0.8, precision=1.0, false_insertion=0
            ),
            "wind_snr10": aggregate_summary(
                cer=0.3, wer=0.5, recall=0.5, precision=0.5, false_insertion=2
            ),
        }
        clean_rows = [
            {"record_key": "a", "reference": "염산 누출", "hypothesis": "염산 누출"},
            {"record_key": "b", "reference": "창고 화재", "hypothesis": "창고 화재"},
        ]
        wind_rows = [
            {"record_key": "b", "reference": "창고 화재", "hypothesis": "창고"},
            {"record_key": "a", "reference": "염산 누출", "hypothesis": "연산 누출"},
        ]
        result = aggregate_robustness(
            summaries, {"clean": clean_rows, "wind_snr10": wind_rows}
        )
        paired = result["paired_vs_clean"]["wind_snr10"]
        self.assertAlmostEqual(0.2, paired["cer_delta"])
        self.assertAlmostEqual(0.3, paired["wer_delta"])
        self.assertEqual(2, paired["false_insertion_delta"])
        self.assertEqual(64, len(result["record_key_set_sha256"]))
        self.assertNotIn("record_keys", result)

    def test_rejects_unpaired_records(self) -> None:
        summary = aggregate_summary(
            cer=0.1, wer=0.2, recall=0.8, precision=1.0, false_insertion=0
        )
        with self.assertRaisesRegex(ValueError, "record keys"):
            aggregate_robustness(
                {"clean": summary, "wind": summary},
                {
                    "clean": [
                        {"record_key": "a", "reference": "같음", "hypothesis": "같음"}
                    ],
                    "wind": [
                        {"record_key": "b", "reference": "같음", "hypothesis": "같음"}
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()
