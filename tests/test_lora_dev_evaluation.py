from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chemicheck119_speech.lora_dev_evaluation import (
    EXPECTED_CONDITION,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_EVIDENCE_SCOPE,
    EXPECTED_RECORDS,
    _secure_write_results,
    validate_dev_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG = ROOT / "config" / "whisper_lora_execution_v1.json"
EXPERIMENT_CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"
WIND_RUNNER = ROOT / "scripts" / "run_whisper_lora_wind_dev_once.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LoraDevelopmentEvaluationTest(unittest.TestCase):
    def test_runner_pins_two_arms_and_clean_repository(self) -> None:
        source = WIND_RUNNER.read_text(encoding="utf-8")
        self.assertIn("status --porcelain", source)
        self.assertIn("ls-remote origin refs/heads/main", source)
        self.assertEqual(2, source.count('run_arm "'))
        self.assertIn("B_same_conversion_base_control", source)
        self.assertIn("C_lora_merged_candidate", source)
        self.assertIn("chmod 600", source)
        self.assertIn("--clean-report", source)

    def test_binds_registered_wind_development_artifacts_and_model_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            conversion = root / "conversion"
            artifacts.mkdir()
            conversion.mkdir()
            manifest = {
                "dataset_id": EXPECTED_DATASET_ID,
                "dataset_version": EXPECTED_DATASET_VERSION,
                "usage_role": "development",
                "classification": "derived",
                "evidence_scope": EXPECTED_EVIDENCE_SCOPE,
                "split": {
                    "name": "Training internal dev",
                    "parameters": {
                        "partition": "dev",
                        "condition": EXPECTED_CONDITION,
                        "used_for_tuning": True,
                    },
                },
                "inventory": {"paired_count": EXPECTED_RECORDS},
            }
            manifest_path = artifacts / "dev-wind_snr0.manifest.json"
            audio_path = artifacts / "dev-wind_snr0.zip"
            labels_path = artifacts / "dev-labels.zip"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audio_path.write_bytes(b"audio")
            labels_path.write_bytes(b"labels")
            snapshots = [
                {"file": path.name, "sha256": _sha256(path)}
                for path in (manifest_path, audio_path, labels_path)
            ]
            arm = "C_lora_merged_candidate"
            model_path = conversion / "C"
            model_path.mkdir()
            (model_path / "model.bin").write_bytes(b"model")
            conversion_report = conversion / "conversion-report.json"
            conversion_report.write_text("{}", encoding="utf-8")
            conversion_payload = {
                "arms": {
                    arm: {
                        "path": "C",
                        "source_model": "openai/whisper-small",
                        "source_revision": "a" * 40,
                    }
                }
            }
            preflight = {
                "artifact_snapshots": snapshots,
                "execution_config_sha256": "1" * 64,
                "experiment_config_sha256": "2" * 64,
                "run_summary_sha256": "3" * 64,
            }

            with (
                patch(
                    "chemicheck119_speech.lora_dev_evaluation.validate_lora_data_preflight",
                    return_value=preflight,
                ),
                patch(
                    "chemicheck119_speech.lora_dev_evaluation.validate_conversion_output",
                    return_value=conversion_payload,
                ),
            ):
                result = validate_dev_inputs(
                    execution_config_path=EXECUTION_CONFIG,
                    experiment_config_path=EXPERIMENT_CONFIG,
                    artifact_root=artifacts,
                    conversion_report_path=conversion_report,
                    arm=arm,
                )

            self.assertEqual(model_path, result["model_path"])
            self.assertEqual(
                EXPECTED_RECORDS, result["dataset_provenance"]["record_count"]
            )
            self.assertTrue(result["dataset_provenance"]["used_for_tuning"])
            self.assertEqual("development", result["dataset_provenance"]["usage_role"])

    def test_private_outputs_are_new_and_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run" / "B"
            _secure_write_results(output, {"status": "fixture"}, [{"row": 1}])
            self.assertEqual(0o700, output.stat().st_mode & 0o777)
            self.assertEqual(0o600, (output / "summary.json").stat().st_mode & 0o777)
            self.assertEqual(
                0o600, (output / "records.private.jsonl").stat().st_mode & 0o777
            )
            with self.assertRaises(FileExistsError):
                _secure_write_results(output, {}, [])


if __name__ == "__main__":
    unittest.main()
