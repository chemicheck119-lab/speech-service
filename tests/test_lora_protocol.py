from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chemicheck119_speech import lora_protocol
from chemicheck119_speech.lora_protocol import (
    load_experiment_config,
    main,
    validate_lora_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"


def _registered_fixture(inputs: dict[str, Path]):
    return patch.multiple(
        lora_protocol,
        REGISTERED_CONFIG_SHA256=hashlib.sha256(
            inputs["config_path"].read_bytes()
        ).hexdigest(),
        REGISTERED_SPLIT_MANIFEST_SHA256=hashlib.sha256(
            inputs["split_manifest_path"].read_bytes()
        ).hexdigest(),
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    audio = tmp_path / "TS_광주_화재.zip"
    labels = tmp_path / "TL_광주_화재.zip"
    terms = tmp_path / "speech_priority_terms_v1.txt"
    audio.write_bytes(b"bounded audio fixture")
    labels.write_bytes(b"bounded label fixture")
    terms.write_bytes((ROOT / "config" / "domain_hotwords.txt").read_bytes())
    split = {
        "schema_version": "1.0.0",
        "dataset_id": "aihub_71768_gwangju_fire",
        "dataset_version": "dataset-71768_downloaded-2026-09-05",
        "usage_role": "training",
        "split": {
            "parameters": {
                "protocol_id": "whisper-lora-gwangju-train-dev-v1",
                "group_key": "recordId",
                "clean_and_derived_share_partition": True,
            },
            "seed": 119,
        },
        "integrity_report": {
            "split_integrity": {
                "entities": {
                    "speaker": {
                        "status": "not_applicable",
                        "reason": "no stable speaker ID",
                    },
                    "source": {"status": "passed", "overlap_count": 0},
                    "event": {
                        "status": "not_evaluated",
                        "reason": "no stable event ID",
                    },
                }
            }
        },
        "artifacts": [
            {"path": f"gs://private/{audio.name}", "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()},
            {"path": f"gs://private/{labels.name}", "sha256": hashlib.sha256(labels.read_bytes()).hexdigest()},
        ],
        "provenance": {
            "generator_source_sha256": "a" * 64,
            "priority_terms_sha256": hashlib.sha256(terms.read_bytes()).hexdigest(),
            "contains_record_ids": False,
            "contains_transcripts": False,
            "contains_addresses": False,
        },
        "inventory": {
            "train": {"record_count": 527},
            "dev": {
                "record_count": 132,
                "utterance_count": 4750,
                "priority_term_support": {
                    "연기": {"record_support": 74, "utterance_support": 261}
                },
            },
        },
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["data"]["split_manifest_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return {
        "config_path": config_path,
        "split_manifest_path": split_path,
        "audio_archive": audio,
        "label_archive": labels,
        "priority_terms_path": terms,
    }


class LoraProtocolTest(unittest.TestCase):
    def test_repository_experiment_config_is_valid(self) -> None:
        config, content = load_experiment_config(CONFIG)
        self.assertIs(config["cost_guard"]["automatic_training_allowed"], False)
        self.assertIs(config["adoption"]["production_default_allowed"], False)
        self.assertTrue(hashlib.sha256(content).hexdigest())

    def test_preflight_passes_without_authorizing_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = _fixture(Path(directory))
            with _registered_fixture(inputs):
                report = validate_lora_preflight(
                    **inputs,
                    generated_at="2026-09-06T00:00:00Z",
                )
        self.assertEqual("limited", report["status"])
        self.assertIs(report["automatic_training_allowed"], False)
        self.assertEqual(74, report["dataset"]["dev_smoke_record_support"])
        self.assertEqual("not_evaluated", report["dataset"]["speaker_overlap_status"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("transcript", serialized)
        self.assertNotIn("recordId", serialized)

    def test_preflight_rejects_archive_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = _fixture(Path(directory))
            inputs["audio_archive"].write_bytes(b"changed")
            with _registered_fixture(inputs), self.assertRaisesRegex(
                ValueError, "audio archive SHA-256"
            ):
                validate_lora_preflight(**inputs)

    def test_priority_terms_digest_is_read_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = _fixture(Path(directory))
            original = lora_protocol.sha256_file
            calls: list[Path] = []

            def recording_digest(path: Path) -> str:
                calls.append(path)
                return original(path)

            with _registered_fixture(inputs), patch.object(
                lora_protocol, "sha256_file", side_effect=recording_digest
            ):
                report = validate_lora_preflight(**inputs)
            self.assertEqual(
                1,
                calls.count(inputs["priority_terms_path"]),
            )
            self.assertEqual(
                hashlib.sha256(inputs["priority_terms_path"].read_bytes()).hexdigest(),
                report["priority_terms_sha256"],
            )

    def test_config_rejects_training_auto_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(CONFIG.read_text(encoding="utf-8"))
            payload["cost_guard"]["automatic_training_allowed"] = True
            path = Path(directory) / "unsafe.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(lora_protocol, "REGISTERED_CONFIG_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "cost guard"):
                    load_experiment_config(path)

    def test_config_rejects_unpinned_identity_and_evidence_scope(self) -> None:
        for field, value in (
            ("dataset_id", ""),
            ("dataset_version", None),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                payload = json.loads(CONFIG.read_text(encoding="utf-8"))
                payload["data"][field] = value
                path = Path(directory) / "unsafe.json"
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with patch.object(lora_protocol, "REGISTERED_CONFIG_SHA256", digest):
                    with self.assertRaisesRegex(ValueError, "split protocol"):
                        load_experiment_config(path)

        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(CONFIG.read_text(encoding="utf-8"))
            payload["evidence_scope"] = "실제 현장 무전 검증 완료"
            path = Path(directory) / "unsafe.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(lora_protocol, "REGISTERED_CONFIG_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "incorrectly stated"):
                    load_experiment_config(path)

    def test_rejects_modified_config_and_registered_input_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = json.loads(CONFIG.read_text(encoding="utf-8"))
            payload["purpose"] = "변조된 목적"
            path = root / "changed-config.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registered artifact"):
                load_experiment_config(path)

            payload = json.loads(CONFIG.read_text(encoding="utf-8"))
            payload["data"]["split_manifest_sha256"] = "b" * 64
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(lora_protocol, "REGISTERED_CONFIG_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "not pre-registered"):
                    load_experiment_config(path)

    def test_cli_uses_atomic_no_clobber_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _fixture(root)
            output = root / "report.json"
            output.write_text("existing", encoding="utf-8")
            arguments = [
                "--config",
                str(inputs["config_path"]),
                "--split-manifest",
                str(inputs["split_manifest_path"]),
                "--audio-archive",
                str(inputs["audio_archive"]),
                "--label-archive",
                str(inputs["label_archive"]),
                "--priority-terms",
                str(inputs["priority_terms_path"]),
                "--output",
                str(output),
            ]
            with _registered_fixture(inputs), self.assertRaisesRegex(
                FileExistsError, "overwrite"
            ):
                main(arguments)
            self.assertEqual("existing", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
