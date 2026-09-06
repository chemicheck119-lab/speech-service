from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.lora_conversion import validate_training_run


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG = ROOT / "config" / "whisper_lora_execution_v1.json"
EXPERIMENT_CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_fixture(root: Path) -> Path:
    training = root / "training"
    adapter = training / "adapter"
    processor = training / "processor"
    adapter.mkdir(parents=True)
    processor.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    (processor / "tokenizer.json").write_text("{}", encoding="utf-8")
    for path in training.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    snapshots = []
    for path in sorted(
        candidate for candidate in training.rglob("*") if candidate.is_file()
    ):
        snapshots.append(
            {
                "path": path.relative_to(training).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    report = {
        "schema_version": "1.0.0",
        "protocol_id": "whisper-small-lora-training-v1",
        "status": "trained_unvalidated",
        "fact_status": "부분 구현 또는 개발용 데모",
        "automatic_adoption_allowed": False,
        "execution_config_sha256": hashlib.sha256(
            EXECUTION_CONFIG.read_bytes()
        ).hexdigest(),
        "experiment_config_sha256": hashlib.sha256(
            EXPERIMENT_CONFIG.read_bytes()
        ).hexdigest(),
        "cost_guard": {
            "authorized_speech_revision": "a" * 40,
            "quoted_total_krw_with_contingency": 0,
        },
        "output_artifacts": snapshots,
    }
    report_path = training / "training-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_path.chmod(0o600)
    return training


class LoraConversionValidationTest(unittest.TestCase):
    def test_accepts_exact_zero_cost_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            training = _training_fixture(Path(directory))
            result = validate_training_run(
                training_dir=training,
                execution_config_path=EXECUTION_CONFIG,
                experiment_config_path=EXPERIMENT_CONFIG,
            )
        self.assertEqual("trained_unvalidated", result["report"]["status"])

    def test_rejects_adapter_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            training = _training_fixture(Path(directory))
            (training / "adapter" / "adapter_model.safetensors").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(ValueError, "drifted"):
                validate_training_run(
                    training_dir=training,
                    execution_config_path=EXECUTION_CONFIG,
                    experiment_config_path=EXPERIMENT_CONFIG,
                )

    def test_rejects_symlinked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            training = _training_fixture(Path(directory))
            target = training / "processor" / "tokenizer.json"
            target.unlink()
            target.symlink_to(training / "adapter" / "adapter_config.json")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                validate_training_run(
                    training_dir=training,
                    execution_config_path=EXECUTION_CONFIG,
                    experiment_config_path=EXPERIMENT_CONFIG,
                )


if __name__ == "__main__":
    unittest.main()
