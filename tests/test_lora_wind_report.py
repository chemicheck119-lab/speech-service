from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.evaluation import _aggregate, load_hotwords
from chemicheck119_speech.lora_dev_evaluation import (
    DEV_EVALUATION_PROTOCOL_ID,
    EXPECTED_CONDITION,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_EVIDENCE_SCOPE,
    EXPECTED_RECORDS,
)
from chemicheck119_speech.lora_wind_report import ARMS, build_wind_report


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"
PRIORITY_TERMS = ROOT / "config" / "domain_hotwords.txt"
REVISION = "5" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conversion_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    conversion = root / "conversion"
    model_paths = {
        "B_same_conversion_base_control": conversion / "B",
        "C_lora_merged_candidate": conversion / "C",
    }
    for path in model_paths.values():
        path.mkdir(parents=True)
        (path / "model.bin").write_bytes(b"model-" + path.name.encode())
        (path / "config.json").write_text("{}", encoding="utf-8")
    snapshots = []
    for path in sorted(
        candidate for candidate in conversion.rglob("*") if candidate.is_file()
    ):
        snapshots.append(
            {
                "path": path.relative_to(conversion).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    report = {
        "schema_version": "1.0.0",
        "protocol_id": "whisper-small-lora-abc-conversion-v1",
        "status": "converted_unvalidated",
        "fact_status": "부분 구현 또는 개발용 데모",
        "converter_revision": "a" * 40,
        "automatic_adoption_allowed": False,
        "arms": {
            "A_operational_baseline": {
                "model": "Systran/faster-whisper-small",
                "revision": REVISION,
                "artifact_created": False,
            },
            "B_same_conversion_base_control": {
                "source_model": "openai/whisper-small",
                "source_revision": "b" * 40,
                "path": "B",
            },
            "C_lora_merged_candidate": {
                "source_model": "openai/whisper-small",
                "source_revision": "b" * 40,
                "path": "C",
            },
        },
        "output_artifacts": snapshots,
    }
    report_path = conversion / "conversion-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, model_paths


def _clean_report_fixture(root: Path, conversion_report: Path) -> Path:
    path = root / "clean-report.json"
    payload = {
        "protocol_id": "whisper-small-lora-abc-locked-evaluation-v1",
        "decision": "continue_wind_and_downstream_gates",
        "automatic_adoption_allowed": False,
        "provenance": {"conversion_report_sha256": _sha256(conversion_report)},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evaluation_fixture(
    root: Path,
    arm: str,
    model_path: Path,
    hypothesis: str,
    conversion_report: Path,
) -> tuple[Path, Path]:
    terms = load_hotwords(PRIORITY_TERMS)
    rows = [
        {
            "record_key": f"{index:016x}",
            "variant": "baseline",
            "status": "completed",
            "error_type": None,
            "reference": "연기 가스 누출",
            "hypothesis": hypothesis,
            "segments": [],
            "audio_seconds": 10.0,
            "voiced_seconds": 5.0,
            "processing_seconds": 1.0,
        }
        for index in range(EXPECTED_RECORDS)
    ]
    aggregate, _ = _aggregate(rows, terms)
    summary = {
        "schema_version": "1.0.0",
        "protocol_id": DEV_EVALUATION_PROTOCOL_ID,
        "experiment_id": (
            f"speech_aihub119_gwangju_lora_dev_{EXPECTED_CONDITION}_{EXPECTED_RECORDS}"
        ),
        "usage_role": "development",
        "evidence_scope": EXPECTED_EVIDENCE_SCOPE,
        "fact_status": "부분 구현 또는 개발용 데모",
        "model_arm": arm,
        "automatic_adoption_allowed": False,
        "input_bindings": {
            "conversion_report_sha256": _sha256(conversion_report),
            "data_preflight": {
                "execution_config": "4" * 64,
                "experiment_config": "5" * 64,
                "run_summary": "6" * 64,
            },
        },
        "dataset": {
            "dataset_id": EXPECTED_DATASET_ID,
            "dataset_version": EXPECTED_DATASET_VERSION,
            "evaluation_id": (
                f"speech_aihub119_gwangju_lora_dev_{EXPECTED_CONDITION}_{EXPECTED_RECORDS}"
            ),
            "record_count": EXPECTED_RECORDS,
            "expected_record_count": EXPECTED_RECORDS,
            "manifest_sha256": "1" * 64,
            "archive_sha256": {"audio": "2" * 64, "labels": "3" * 64},
            "split": "Training internal dev",
            "condition": EXPECTED_CONDITION,
            "used_for_tuning": True,
        },
        "runtime": {
            "implementation": "faster-whisper",
            "version": "1.2.1",
            "model": str(model_path),
            "requested_device": "cpu",
            "device": "cpu",
            "compute_type": "int8",
            "initialization_fallback": None,
            "language": "ko (configured, not detected)",
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "variants": ["baseline"],
        },
        "variants": {"baseline": aggregate},
        "priority_terms": terms,
    }
    output = root / arm
    output.mkdir()
    summary_path = output / "summary.json"
    records_path = output / "records.private.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    records_path.chmod(0o600)
    return summary_path, records_path


class LoraWindReportTest(unittest.TestCase):
    def _build(self, root: Path, candidate_hypothesis: str) -> dict[str, object]:
        conversion, models = _conversion_fixture(root)
        clean = _clean_report_fixture(root, conversion)
        inputs = {
            ARMS[0]: _evaluation_fixture(
                root,
                ARMS[0],
                models[ARMS[0]],
                "가스 누출",
                conversion,
            ),
            ARMS[1]: _evaluation_fixture(
                root,
                ARMS[1],
                models[ARMS[1]],
                candidate_hypothesis,
                conversion,
            ),
        }
        return build_wind_report(
            conversion_report_path=conversion,
            clean_report_path=clean,
            experiment_config_path=EXPERIMENT_CONFIG,
            priority_terms_path=PRIORITY_TERMS,
            summaries={arm: inputs[arm][0] for arm in ARMS},
            records={arm: inputs[arm][1] for arm in ARMS},
            output_path=root / "report" / "wind-report.json",
            evaluator_revision="c" * 40,
        )

    def test_continues_only_when_candidate_passes_all_wind_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._build(root, "연기 가스 누출")
            output = root / "report" / "wind-report.json"

            self.assertEqual("continue_downstream_safety_gate", report["decision"])
            self.assertTrue(all(report["checks"].values()))
            self.assertFalse(report["automatic_adoption_allowed"])
            self.assertEqual(0o600, output.stat().st_mode & 0o777)

    def test_rejects_candidate_when_priority_false_insertion_increases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._build(Path(directory), "연기 가스 폭발")

            self.assertFalse(report["checks"]["false_insertion_nonincrease"])
            self.assertEqual(
                "reject_candidate_keep_operational_baseline", report["decision"]
            )


if __name__ == "__main__":
    unittest.main()
