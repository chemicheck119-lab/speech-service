"""Fail-closed preflight for the bounded Whisper LoRA experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .provenance import sha256_file


PROTOCOL_ID = "whisper-small-lora-gwangju-v1"
SPLIT_PROTOCOL_ID = "whisper-lora-gwangju-train-dev-v1"
DATASET_ID = "aihub_71768_gwangju_fire"
DATASET_VERSION = "dataset-71768_downloaded-2026-09-05"
EVIDENCE_SCOPE = "AIHub 신고전화와 절차적 모의 통신 왜곡; 실제 현장 무전 검증 아님"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_FACT_STATUS = "설계 완료·구현 전"


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _read_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    content = path.read_bytes()
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    return _object(payload, name), content


def load_experiment_config(path: Path) -> tuple[dict[str, object], bytes]:
    config, content = _read_json(path, "experiment config")
    if (
        config.get("schema_version") != "1.0.0"
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("fact_status") != ALLOWED_FACT_STATUS
        or config.get("evidence_scope") != EVIDENCE_SCOPE
    ):
        raise ValueError("unsupported or incorrectly stated LoRA protocol")
    data = _object(config.get("data"), "config.data")
    models = _object(config.get("models"), "config.models")
    lora = _object(config.get("lora"), "config.lora")
    training = _object(config.get("training"), "config.training")
    evaluation = _object(config.get("evaluation"), "config.evaluation")
    cost = _object(config.get("cost_guard"), "config.cost_guard")
    adoption = _object(config.get("adoption"), "config.adoption")

    if (
        data.get("dataset_id") != DATASET_ID
        or data.get("dataset_version") != DATASET_VERSION
        or data.get("split_protocol_id") != SPLIT_PROTOCOL_ID
    ):
        raise ValueError("config split protocol is not pinned")
    for field in ("split_manifest_sha256", "priority_terms_sha256"):
        _sha256(data.get(field), f"config.data.{field}")
    if data.get("train_records") != 527 or data.get("dev_records") != 132:
        raise ValueError("config record counts are not the pre-registered split")
    if data.get("dev_smoke_record_support") != 74:
        raise ValueError("config smoke support is not the pre-registered split")
    if data.get("train_partitions_forbidden") != [
        "Gwangju Validation",
        "Seoul Validation",
        "Incheon Validation",
    ]:
        raise ValueError("held-out partitions are not explicitly forbidden for training")
    assignment = _object(
        data.get("train_condition_assignment"),
        "config.data.train_condition_assignment",
    )
    if (
        assignment.get("unit") != "recordId group"
        or assignment.get("clean_fraction") != 0.6
        or assignment.get("wind_snr0_fraction") != 0.4
        or assignment.get("seed") != 9119
        or assignment.get("each_utterance_seen_once") is not True
    ):
        raise ValueError("training condition assignment is not pinned")

    operational = _object(models.get("operational_baseline"), "operational model")
    base = _object(models.get("transformers_base"), "transformers base")
    if (
        operational.get("id") != "Systran/faster-whisper-small"
        or operational.get("revision") != "536b0662742c02347bc0e980a01041f333bce120"
        or base.get("id") != "openai/whisper-small"
        or base.get("revision") != "973afd24965f72e36ca33b3055d56a652f456b4d"
        or models.get("comparison_arms")
        != [
            "A_operational_baseline",
            "B_same_conversion_base_control",
            "C_lora_merged_candidate",
        ]
    ):
        raise ValueError("model IDs or revisions are not pinned")
    if lora != {
        "task_type": "SEQ_2_SEQ_LM",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],
    }:
        raise ValueError("LoRA parameters are not the pre-registered configuration")
    expected_training = {
        "epochs": 1,
        "learning_rate": 0.0001,
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 16,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_gradient_norm": 1.0,
        "seed": 119,
        "fp16": True,
        "gradient_checkpointing": True,
        "language": "ko",
        "task": "transcribe",
        "max_audio_seconds": 12.0,
        "max_label_tokens": 160,
        "intermediate_dev_selection": False,
        "retry_count": 0,
    }
    if training != expected_training:
        raise ValueError("training parameters are not the pre-registered configuration")
    expected_evaluation = {
        "bootstrap_samples": 2000,
        "bootstrap_seed": 119,
        "clean_cer_max_absolute_regression": 0.01,
        "clean_wer_max_absolute_regression": 0.015,
        "wind_cer_or_wer_min_relative_improvement": 0.05,
        "wind_improved_metric_ci95_upper_max": 0.0,
        "wind_smoke_recall_min_absolute_improvement": 0.1,
        "wind_priority_f1_min_absolute_improvement": 0.03,
        "false_insertion_max_increase": 0,
        "operational_to_conversion_control_cer_max_absolute_regression": 0.005,
        "operational_to_conversion_control_wer_max_absolute_regression": 0.005,
        "max_real_time_factor": 0.5,
        "downstream_silver_top3_max_regression": 0.0,
        "wrong_single_cas_promotion_max": 0,
        "preconfirmation_rule_execution_max": 0,
    }
    if evaluation != expected_evaluation:
        raise ValueError("evaluation thresholds are not fully pre-registered")
    if (
        cost.get("gcp_region") != "asia-northeast3"
        or cost.get("gpu_type") != "nvidia-tesla-t4"
        or cost.get("gpu_count") != 1
        or cost.get("machine_type") != "n1-standard-4"
        or cost.get("max_runtime_hours") != 3
        or cost.get("max_instances") != 1
        or cost.get("retry_count") != 0
        or cost.get("experiment_hard_cap_krw") != 20000
        or cost.get("total_development_server_cap_krw") != 70000
        or cost.get("automatic_training_allowed") is not False
    ):
        raise ValueError("cost guard is not the pre-registered bounded configuration")
    if (
        adoption.get("production_default_allowed") is not False
        or adoption.get("field_safety_claim_allowed") is not False
        or adoption.get("maximum_without_untouched_region")
        != "조건부 채택·development preview 전용"
    ):
        raise ValueError("adoption claim boundary must fail closed")
    return config, content


def validate_lora_preflight(
    *,
    config_path: Path,
    split_manifest_path: Path,
    audio_archive: Path,
    label_archive: Path,
    priority_terms_path: Path,
    generated_at: str | None = None,
) -> dict[str, object]:
    config, config_bytes = load_experiment_config(config_path)
    split_manifest, split_bytes = _read_json(split_manifest_path, "split manifest")
    data = _object(config["data"], "config.data")
    observed_split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if observed_split_sha256 != data["split_manifest_sha256"]:
        raise ValueError("split manifest SHA-256 does not match the experiment config")
    if (
        split_manifest.get("dataset_id") != DATASET_ID
        or split_manifest.get("dataset_version") != DATASET_VERSION
        or split_manifest.get("usage_role") != "training"
    ):
        raise ValueError("split manifest dataset contract does not match")
    split = _object(split_manifest.get("split"), "split manifest split")
    parameters = _object(split.get("parameters"), "split manifest parameters")
    if (
        parameters.get("protocol_id") != SPLIT_PROTOCOL_ID
        or parameters.get("group_key") != "recordId"
        or parameters.get("clean_and_derived_share_partition") is not True
        or split.get("seed") != 119
    ):
        raise ValueError("split manifest partition contract does not match")
    provenance = _object(split_manifest.get("provenance"), "split provenance")
    for field in ("generator_source_sha256", "priority_terms_sha256"):
        _sha256(provenance.get(field), f"split provenance {field}")
    if (
        provenance.get("priority_terms_sha256") != data["priority_terms_sha256"]
        or provenance.get("contains_record_ids") is not False
        or provenance.get("contains_transcripts") is not False
        or provenance.get("contains_addresses") is not False
    ):
        raise ValueError("split manifest provenance or privacy contract does not match")
    if sha256_file(priority_terms_path) != data["priority_terms_sha256"]:
        raise ValueError("priority terms SHA-256 does not match")

    inventory = _object(split_manifest.get("inventory"), "split inventory")
    train = _object(inventory.get("train"), "split inventory train")
    dev = _object(inventory.get("dev"), "split inventory dev")
    dev_terms = _object(dev.get("priority_term_support"), "dev priority terms")
    smoke_support = _object(dev_terms.get("연기"), "dev smoke support")
    if (
        train.get("record_count") != data["train_records"]
        or dev.get("record_count") != data["dev_records"]
        or dev.get("utterance_count") != data["dev_utterances"]
        or smoke_support.get("record_support") != data["dev_smoke_record_support"]
        or smoke_support.get("utterance_support")
        != data["dev_smoke_utterance_support"]
    ):
        raise ValueError("split inventory is not the pre-registered support profile")
    integrity = _object(split_manifest.get("integrity_report"), "split integrity")
    split_integrity = _object(integrity.get("split_integrity"), "split integrity")
    entities = _object(split_integrity.get("entities"), "split integrity entities")
    source = _object(entities.get("source"), "split source integrity")
    event = _object(entities.get("event"), "split event integrity")
    if source.get("status") != "passed" or source.get("overlap_count") != 0:
        raise ValueError("train/dev source overlap gate did not pass")
    if event.get("status") != "not_evaluated":
        raise ValueError("missing event IDs must remain explicitly not evaluated")

    artifacts = split_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("split manifest artifacts must be an array")
    expected_by_name: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        digest = artifact.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            expected_by_name[PurePosixPath(path).name] = digest
    observed_archives = {
        "audio": sha256_file(audio_archive),
        "labels": sha256_file(label_archive),
    }
    for role, path, digest in (
        ("audio", audio_archive, observed_archives["audio"]),
        ("labels", label_archive, observed_archives["labels"]),
    ):
        if expected_by_name.get(path.name) != digest:
            raise ValueError(f"{role} archive SHA-256 does not match the split manifest")

    return {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "fact_status": "설계 완료·구현 전",
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "split_manifest_sha256": observed_split_sha256,
        "dataset": {
            "id": split_manifest["dataset_id"],
            "version": split_manifest["dataset_version"],
            "train_records": train["record_count"],
            "dev_records": dev["record_count"],
            "dev_utterances": dev["utterance_count"],
            "dev_smoke_record_support": smoke_support["record_support"],
            "dev_smoke_utterance_support": smoke_support["utterance_support"],
            "event_overlap_status": "not_evaluated",
        },
        "archive_sha256": observed_archives,
        "priority_terms_sha256": sha256_file(priority_terms_path),
        "automatic_training_allowed": False,
        "next_gate": "reviewed training harness, immutable derived archives, and current cost quote",
        "claim_scope": "preflight only; no LoRA performance or field-radio claim",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--label-archive", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    report = validate_lora_preflight(
        config_path=args.config,
        split_manifest_path=args.split_manifest,
        audio_archive=args.audio_archive,
        label_archive=args.label_archive,
        priority_terms_path=args.priority_terms,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite preflight report: {args.output}"
        ) from error
    print(
        json.dumps(
            {
                "status": report["status"],
                "automatic_training_allowed": report["automatic_training_allowed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
