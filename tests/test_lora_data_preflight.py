from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

from chemicheck119_speech import lora_data_preflight
from chemicheck119_speech.lora_data_preflight import (
    main,
    validate_lora_data_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG = ROOT / "config" / "whisper_lora_execution_v1.json"
EXPERIMENT_CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"
GCS_PREFIX = (
    "gs://chemi-check-ml-data-181872008704/derived/aihub/71768/"
    "gwangju-fire/lora-v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _membership_digest(record_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(record_ids):
        encoded = record_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _wav_bytes(seconds: float = 1.0) -> bytes:
    output = io.BytesIO()
    sample_rate = 8000
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        frames = bytearray()
        for frame in range(round(seconds * sample_rate)):
            value = round(3000 * math.sin(2 * math.pi * 180 * frame / sample_rate))
            frames.extend(int(value).to_bytes(2, "little", signed=True))
        audio.writeframes(bytes(frames))
    return output.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    path.chmod(0o600)


def _fixture(
    root: Path,
    *,
    overlap: bool = False,
    over_limit: bool = False,
) -> tuple[dict[str, Path], str]:
    artifacts = root / "artifacts"
    artifacts.mkdir(mode=0o700)
    ledger = artifacts / "provenance.private.jsonl"
    ledger.write_text("{}\n", encoding="utf-8")
    ledger.chmod(0o600)

    partitions = {
        "train": ["train-0", "train-1", "train-2", "train-3"],
        "dev": ["train-0" if overlap else "dev-0", "dev-1"],
    }
    inventory: dict[str, dict[str, object]] = {}
    manifest_entries: list[dict[str, object]] = []
    for partition, record_ids in partitions.items():
        labels: dict[str, bytes] = {}
        audio: dict[str, bytes] = {}
        duration_ms = 13000 if over_limit and partition == "train" else 900
        for index, record_id in enumerate(record_ids):
            stem = f"{partition}-record-{index}"
            document = {
                "recordId": record_id,
                "utterances": [
                    {
                        "id": f"utterance-{index}",
                        "startAt": 0,
                        "endAt": duration_ms,
                        "text": "연기 테스트 문장",
                    }
                ],
            }
            labels[f"{stem}.json"] = json.dumps(
                document, ensure_ascii=False
            ).encode("utf-8")
            audio[f"{stem}.wav"] = _wav_bytes(14.0 if duration_ms > 1000 else 1.0)
        label_path = artifacts / f"{partition}-labels.zip"
        _write_zip(label_path, labels)
        inventory[partition] = {
            "record_ids": record_ids,
            "record_count": len(record_ids),
            "utterance_count": len(record_ids),
            "membership_sha256": _membership_digest(record_ids),
        }
        for condition in ("clean", "wind_snr0"):
            audio_path = artifacts / f"{partition}-{condition}.zip"
            _write_zip(audio_path, audio)
            manifest = {
                "schema_version": "1.0.0",
                "dataset_id": (
                    f"aihub_71768_gwangju_fire_lora_{partition}_{condition}"
                ),
                "classification": "derived",
                "usage_role": "training" if partition == "train" else "development",
                "split": {
                    "parameters": {
                        "protocol_id": "whisper-lora-clean-wind-snr0-v1",
                        "partition": partition,
                        "condition": condition,
                        "membership_sha256": inventory[partition][
                            "membership_sha256"
                        ],
                        "clean_and_derived_share_partition": True,
                        "used_for_tuning": True,
                    }
                },
                "artifacts": [
                    {
                        "path": f"{GCS_PREFIX}/{audio_path.name}",
                        "sha256": _sha256(audio_path),
                        "bytes": audio_path.stat().st_size,
                        "access": "private",
                    },
                    {
                        "path": f"{GCS_PREFIX}/{label_path.name}",
                        "sha256": _sha256(label_path),
                        "bytes": label_path.stat().st_size,
                        "access": "private",
                    },
                    {
                        "path": f"{GCS_PREFIX}/{ledger.name}",
                        "sha256": _sha256(ledger),
                        "bytes": ledger.stat().st_size,
                        "access": "private",
                    },
                ],
                "inventory": {
                    "paired_count": len(record_ids),
                    "utterance_count": len(record_ids),
                },
                "evidence_scope": (
                    "AIHub emergency-call Training derivative with procedural wind; "
                    "not field-radio validation"
                ),
            }
            manifest_path = artifacts / f"{partition}-{condition}.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            manifest_path.chmod(0o600)
            manifest_entries.append(
                {
                    "partition": partition,
                    "condition": condition,
                    "manifest": manifest_path.name,
                    "manifest_sha256": _sha256(manifest_path),
                    "audio_sha256": _sha256(audio_path),
                    "labels_sha256": _sha256(label_path),
                    "record_count": len(record_ids),
                    "utterance_count": len(record_ids),
                }
            )

    summary = {
        "schema_version": "1.0.0",
        "protocol_id": "whisper-lora-clean-wind-snr0-v1",
        "status": "completed",
        "fact_status": "구현 완료",
        "source_manifest_sha256": lora_data_preflight.EXPECTED_SOURCE_MANIFEST_SHA256,
        "split_manifest_sha256": lora_data_preflight.EXPECTED_SPLIT_MANIFEST_SHA256,
        "priority_terms_sha256": lora_data_preflight.EXPECTED_PRIORITY_TERMS_SHA256,
        "private_ledger_sha256": _sha256(ledger),
        "manifests": manifest_entries,
        "privacy": {
            "git_commit_allowed": False,
            "private_storage_required": True,
            "console_contains_record_ids_or_transcripts": False,
        },
        "automatic_training_allowed": False,
    }
    summary_path = artifacts / "run-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    summary_path.chmod(0o600)

    execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
    execution["derived_data"]["run_summary_sha256"] = _sha256(summary_path)
    execution_path = root / "execution.json"
    execution_path.write_text(
        json.dumps(execution, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "execution_config_path": execution_path,
        "experiment_config_path": EXPERIMENT_CONFIG,
        "artifact_root": artifacts,
    }, _sha256(execution_path)


class LoraDataPreflightTest(unittest.TestCase):
    def test_validates_private_artifacts_without_exposing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, config_sha256 = _fixture(Path(directory))
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                config_sha256,
            ):
                report = validate_lora_data_preflight(
                    **inputs,
                    generated_at="2026-09-06T00:00:00Z",
                )
        self.assertEqual("limited", report["status"])
        self.assertEqual("구현 완료", report["fact_status"])
        self.assertEqual("설계 완료·구현 전", report["training_fact_status"])
        self.assertEqual(4, report["partitions"]["train"]["record_count"])
        self.assertEqual(
            4,
            sum(
                arm["record_count"]
                for arm in report["training_condition_assignment"].values()
            ),
        )
        self.assertIs(report["automatic_training_allowed"], False)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("연기 테스트 문장", serialized)
        self.assertNotIn("train-0", serialized)

    def test_rejects_archive_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, config_sha256 = _fixture(Path(directory))
            archive = inputs["artifact_root"] / "train-clean.zip"
            archive.write_bytes(archive.read_bytes() + b"changed")
            archive.chmod(0o600)
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                config_sha256,
            ), self.assertRaisesRegex(ValueError, "immutable manifest"):
                validate_lora_data_preflight(**inputs)

    def test_rejects_train_dev_overlap_and_long_utterance(self) -> None:
        for options, message in (
            ({"overlap": True}, "overlap"),
            ({"over_limit": True}, "audio limit"),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                inputs, config_sha256 = _fixture(Path(directory), **options)
                with patch.object(
                    lora_data_preflight,
                    "REGISTERED_EXECUTION_CONFIG_SHA256",
                    config_sha256,
                ), self.assertRaisesRegex(ValueError, message):
                    validate_lora_data_preflight(**inputs)

    def test_cli_refuses_to_overwrite_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, config_sha256 = _fixture(root)
            output = root / "report.json"
            output.write_text("existing", encoding="utf-8")
            arguments = [
                "--execution-config",
                str(inputs["execution_config_path"]),
                "--experiment-config",
                str(inputs["experiment_config_path"]),
                "--artifact-root",
                str(inputs["artifact_root"]),
                "--output",
                str(output),
            ]
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                config_sha256,
            ), self.assertRaisesRegex(FileExistsError, "overwrite"):
                main(arguments)
            self.assertEqual("existing", output.read_text(encoding="utf-8"))

    def test_repository_execution_config_is_registered(self) -> None:
        self.assertEqual(
            lora_data_preflight.REGISTERED_EXECUTION_CONFIG_SHA256,
            _sha256(EXECUTION_CONFIG),
        )

    def test_rejects_generated_at_without_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, config_sha256 = _fixture(Path(directory))
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                config_sha256,
            ), self.assertRaisesRegex(ValueError, "generated_at"):
                validate_lora_data_preflight(
                    **inputs,
                    generated_at="2026-09-06T00:00:00",
                )


if __name__ == "__main__":
    unittest.main()
