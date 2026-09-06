from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.evaluation import _aggregate, load_hotwords
from chemicheck119_speech.lora_abc_report import build_abc_report


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"
PRIORITY_TERMS = ROOT / "config" / "domain_hotwords.txt"
REVISION = "5" * 40
ARMS = (
    "A_operational_baseline",
    "B_same_conversion_base_control",
    "C_lora_merged_candidate",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conversion_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    conversion = root / "conversion"
    model_paths = {
        "A_operational_baseline": root / REVISION,
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


def _evaluation_fixture(
    root: Path, arm: str, model_path: Path, hypothesis: str = "가스 화재"
) -> tuple[Path, Path]:
    terms = load_hotwords(PRIORITY_TERMS)
    rows = [
        {
            "record_key": f"{index:016x}",
            "variant": "baseline",
            "status": "completed",
            "error_type": None,
            "reference": "가스 화재",
            "hypothesis": hypothesis,
            "segments": [],
            "audio_seconds": 10.0,
            "voiced_seconds": 5.0,
            "processing_seconds": 1.0,
        }
        for index in range(77)
    ]
    aggregate, _ = _aggregate(rows, terms)
    summary = {
        "schema_version": "1.0.0",
        "experiment_id": "speech_aihub119_gwangju_fire_validation_77",
        "usage_role": "evaluation",
        "evidence_scope": "AIHub 119 emergency-call proxy; not field-radio validation",
        "dataset": {
            "dataset_id": "aihub_71768_gwangju_fire",
            "dataset_version": "dataset-71768_downloaded-2026-09-05",
            "evaluation_id": "speech_aihub119_gwangju_fire_validation_77",
            "record_count": 77,
            "manifest_sha256": "1" * 64,
            "archive_sha256": {"audio": "2" * 64, "labels": "3" * 64},
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
    directory = root / arm
    directory.mkdir()
    summary_path = directory / "summary.json"
    records_path = directory / "records.private.jsonl"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    records_path.chmod(0o600)
    return summary_path, records_path


class LoraAbcReportTest(unittest.TestCase):
    def test_accepts_paired_clean_runs_without_adopting_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversion, models = _conversion_fixture(root)
            inputs = {
                arm: _evaluation_fixture(root, arm, models[arm]) for arm in ARMS
            }
            output = root / "report" / "abc-report.json"
            report = build_abc_report(
                conversion_report_path=conversion,
                experiment_config_path=EXPERIMENT_CONFIG,
                priority_terms_path=PRIORITY_TERMS,
                summaries={arm: inputs[arm][0] for arm in ARMS},
                records={arm: inputs[arm][1] for arm in ARMS},
                output_path=output,
            )
            self.assertEqual(
                "continue_wind_and_downstream_gates", report["decision"]
            )
            self.assertFalse(report["automatic_adoption_allowed"])
            self.assertEqual(0o600, output.stat().st_mode & 0o777)

    def test_rejects_summary_that_does_not_match_private_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversion, models = _conversion_fixture(root)
            inputs = {
                arm: _evaluation_fixture(root, arm, models[arm]) for arm in ARMS
            }
            summary_path = inputs["C_lora_merged_candidate"][0]
            summary = json.loads(summary_path.read_text())
            summary["variants"]["baseline"]["cer"] = 0.5
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "error rates"):
                build_abc_report(
                    conversion_report_path=conversion,
                    experiment_config_path=EXPERIMENT_CONFIG,
                    priority_terms_path=PRIORITY_TERMS,
                    summaries={arm: inputs[arm][0] for arm in ARMS},
                    records={arm: inputs[arm][1] for arm in ARMS},
                    output_path=root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
