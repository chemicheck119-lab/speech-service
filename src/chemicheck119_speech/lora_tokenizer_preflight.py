"""Check restricted LoRA labels with the pinned Whisper tokenizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import zipfile

from .lora_data_preflight import (
    MAX_LABEL_MEMBER_BYTES,
    _object,
    _read_member,
    _safe_members,
    _timestamp,
    validate_lora_data_preflight,
)
from .lora_protocol import load_experiment_config


EXPECTED_TRANSFORMERS_VERSION = "4.57.6"


def _load_pinned_tokenizer(experiment_config: dict[str, object]):
    if distribution_version("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("transformers runtime does not match the execution config")
    try:
        from transformers import WhisperTokenizerFast
    except ImportError as error:
        raise RuntimeError("pinned transformers tokenizer is unavailable") from error
    models = _object(experiment_config.get("models"), "experiment models")
    base = _object(models.get("transformers_base"), "transformers base")
    return WhisperTokenizerFast.from_pretrained(
        str(base["id"]),
        revision=str(base["revision"]),
        language="ko",
        task="transcribe",
    )


def _token_count(tokenizer: object, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    if not isinstance(encoded, dict):
        try:
            encoded = dict(encoded)
        except (TypeError, ValueError) as error:
            raise RuntimeError("tokenizer returned an unsupported result") from error
    token_ids = encoded.get("input_ids")
    if not isinstance(token_ids, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in token_ids
    ):
        raise RuntimeError("tokenizer did not return a single integer token sequence")
    return len(token_ids)


def validate_lora_tokenizer_preflight(
    *,
    execution_config_path: Path,
    experiment_config_path: Path,
    artifact_root: Path,
    tokenizer: object | None = None,
    tokenizer_version: str | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Return aggregate token statistics without retaining label text."""

    data_report = validate_lora_data_preflight(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
        generated_at=generated_at,
    )
    experiment_config, _ = load_experiment_config(experiment_config_path)
    selected_tokenizer = tokenizer or _load_pinned_tokenizer(experiment_config)
    observed_version = tokenizer_version or EXPECTED_TRANSFORMERS_VERSION
    if observed_version != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("tokenizer version does not match the registered runtime")
    training = _object(experiment_config.get("training"), "experiment training")
    maximum_tokens = int(training["max_label_tokens"])

    partitions: dict[str, dict[str, int]] = {}
    total_utterances = 0
    over_limit_count = 0
    observed_maximum = 0
    for partition in ("train", "dev"):
        label_path = artifact_root / f"{partition}-labels.zip"
        with zipfile.ZipFile(label_path) as labels:
            members = _safe_members(labels, ".json")
            partition_utterances = 0
            partition_over_limit = 0
            partition_maximum = 0
            for stem in sorted(members):
                content = _read_member(
                    labels,
                    members[stem],
                    MAX_LABEL_MEMBER_BYTES,
                )
                try:
                    document = json.loads(content.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("label archive contains invalid JSON") from error
                document = _object(document, "label record")
                utterances = document.get("utterances")
                if not isinstance(utterances, list):
                    raise ValueError("label utterances must be an array")
                for utterance_value in utterances:
                    utterance = _object(utterance_value, "utterance")
                    text = utterance.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("utterance text must be non-empty")
                    count = _token_count(selected_tokenizer, text)
                    partition_utterances += 1
                    partition_maximum = max(partition_maximum, count)
                    if count > maximum_tokens:
                        partition_over_limit += 1
            partitions[partition] = {
                "utterance_count": partition_utterances,
                "max_label_tokens": partition_maximum,
                "over_limit_count": partition_over_limit,
            }
            total_utterances += partition_utterances
            over_limit_count += partition_over_limit
            observed_maximum = max(observed_maximum, partition_maximum)

    expected_total = sum(
        int(data_report["partitions"][partition]["utterance_count"])
        for partition in ("train", "dev")
    )
    if total_utterances != expected_total:
        raise ValueError("tokenized label count does not match data preflight")
    status = "limited" if over_limit_count == 0 else "rejected"
    models = _object(experiment_config.get("models"), "experiment models")
    base = _object(models.get("transformers_base"), "transformers base")
    created = _timestamp(
        generated_at or datetime.now(timezone.utc).isoformat(), "generated_at"
    )
    return {
        "schema_version": "1.0.0",
        "protocol_id": "whisper-small-lora-tokenizer-preflight-v1",
        "status": status,
        "fact_status": "구현 완료",
        "training_fact_status": "설계 완료·구현 전",
        "generated_at": created,
        "data_preflight": {
            "status": data_report["status"],
            "execution_config_sha256": data_report["execution_config_sha256"],
            "run_summary_sha256": data_report["run_summary_sha256"],
        },
        "tokenizer": {
            "model_id": base["id"],
            "revision": base["revision"],
            "transformers_version": observed_version,
            "class": type(selected_tokenizer).__name__,
        },
        "registered_max_label_tokens": maximum_tokens,
        "observed_max_label_tokens": observed_maximum,
        "over_limit_count": over_limit_count,
        "partitions": partitions,
        "privacy": {
            "contains_record_ids": False,
            "contains_transcripts": False,
            "contains_addresses": False,
        },
        "known_limitations": {
            "speaker_overlap": "not_evaluated",
            "event_overlap": "not_evaluated",
            "untouched_region_test": "not_available",
            "field_radio_validation": "not_available",
        },
        "automatic_training_allowed": False,
        "training_eligible_by_token_limit": over_limit_count == 0,
        "next_gate": (
            "reviewed local MPS trainer and current zero-cost attestation"
            if over_limit_count == 0
            else "revise the pre-registered segment contract before any training"
        ),
        "claim_scope": "token-length readiness only; no LoRA performance or field-radio claim",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    report = validate_lora_tokenizer_preflight(
        execution_config_path=args.execution_config,
        experiment_config_path=args.experiment_config,
        artifact_root=args.artifact_root,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite LoRA tokenizer preflight: {args.output}"
        ) from error
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "observed_max_label_tokens": report["observed_max_label_tokens"],
                "over_limit_count": report["over_limit_count"],
                "automatic_training_allowed": report["automatic_training_allowed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
