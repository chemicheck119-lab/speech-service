"""Fail-closed aggregate report for locked Whisper LoRA A/B/C evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Mapping

from .evaluation import load_hotwords
from .lora_conversion import validate_conversion_output
from .lora_data_preflight import _object
from .lora_protocol import load_experiment_config
from .metrics import RecordMetric, paired_bootstrap_error_delta, score_record
from .metrics import term_presence_counts


ABC_REPORT_PROTOCOL_ID = "whisper-small-lora-abc-locked-evaluation-v1"
EXPECTED_EXPERIMENT_ID = "speech_aihub119_gwangju_fire_validation_77"
EXPECTED_RECORDS = 77
MAX_SUMMARY_BYTES = 4 * 1024 * 1024
MAX_RECORDS_BYTES = 128 * 1024 * 1024
MAX_RECORD_LINE_BYTES = 2 * 1024 * 1024
RECORD_KEY_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, maximum: int, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > maximum:
        raise ValueError(f"{name} exceeds the bounded size")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _object(payload, name)


def _read_private_rows(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("records must be a regular non-symlink file")
    file_stat = path.stat()
    if (
        file_stat.st_size <= 0
        or file_stat.st_size > MAX_RECORDS_BYTES
        or stat.S_IMODE(file_stat.st_mode) & 0o077
    ):
        raise ValueError("records size or permissions violate the private boundary")
    rows: list[dict[str, object]] = []
    with path.open("rb") as source:
        for line in source:
            if len(line) > MAX_RECORD_LINE_BYTES:
                raise ValueError("record line exceeds the bounded size")
            rows.append(_object(json.loads(line.decode("utf-8")), "evaluation row"))
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError("locked evaluation must contain exactly 77 rows")
    return rows


def _dataset_fingerprint(summary: Mapping[str, object]) -> dict[str, object]:
    dataset = _object(summary.get("dataset"), "evaluation dataset")
    archive = _object(dataset.get("archive_sha256"), "dataset archive hashes")
    return {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("dataset_version"),
        "evaluation_id": dataset.get("evaluation_id"),
        "record_count": dataset.get("record_count"),
        "manifest_sha256": dataset.get("manifest_sha256"),
        "audio_sha256": archive.get("audio"),
        "labels_sha256": archive.get("labels"),
    }


def _validate_summary(
    summary: dict[str, object], terms: list[str]
) -> dict[str, object]:
    runtime = _object(summary.get("runtime"), "evaluation runtime")
    variants = _object(summary.get("variants"), "evaluation variants")
    if (
        summary.get("schema_version") != "1.0.0"
        or summary.get("experiment_id") != EXPECTED_EXPERIMENT_ID
        or summary.get("usage_role") != "evaluation"
        or summary.get("evidence_scope")
        != "AIHub 119 emergency-call proxy; not field-radio validation"
        or summary.get("priority_terms") != terms
        or set(variants) != {"baseline"}
    ):
        raise ValueError("summary is not the locked baseline-only evaluation")
    if (
        runtime.get("implementation") != "faster-whisper"
        or runtime.get("version") != "1.2.1"
        or runtime.get("requested_device") != "cpu"
        or runtime.get("device") != "cpu"
        or runtime.get("compute_type") != "int8"
        or runtime.get("initialization_fallback") is not None
        or runtime.get("language") != "ko (configured, not detected)"
        or runtime.get("beam_size") != 5
        or runtime.get("temperature") != 0.0
        or runtime.get("vad_filter") is not True
        or runtime.get("condition_on_previous_text") is not False
        or runtime.get("variants") != ["baseline"]
    ):
        raise ValueError("runtime settings differ across the locked comparison")
    fingerprint = _dataset_fingerprint(summary)
    if (
        fingerprint["dataset_id"] != "aihub_71768_gwangju_fire"
        or fingerprint["dataset_version"]
        != "dataset-71768_downloaded-2026-09-05"
        or fingerprint["evaluation_id"] != EXPECTED_EXPERIMENT_ID
        or fingerprint["record_count"] != EXPECTED_RECORDS
        or any(
            not isinstance(fingerprint[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", str(fingerprint[field])) is None
            for field in ("manifest_sha256", "audio_sha256", "labels_sha256")
        )
    ):
        raise ValueError("dataset fingerprint is not the locked 77-record input")
    return fingerprint


def _index_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        key = row.get("record_key")
        if (
            not isinstance(key, str)
            or RECORD_KEY_PATTERN.fullmatch(key) is None
            or key in indexed
            or row.get("variant") != "baseline"
            or row.get("status") not in {"completed", "failed"}
            or not isinstance(row.get("reference"), str)
            or not isinstance(row.get("hypothesis"), str)
            or not isinstance(row.get("audio_seconds"), (int, float))
        ):
            raise ValueError("private evaluation rows violate the locked schema")
        indexed[key] = row
    return indexed


def _recompute(
    rows: list[dict[str, object]], terms: list[str]
) -> tuple[dict[str, object], list[RecordMetric]]:
    metrics = [
        score_record(str(row["reference"]), str(row["hypothesis"]))
        for row in rows
    ]
    character_edits = sum(item.character_edits for item in metrics)
    reference_characters = sum(item.reference_characters for item in metrics)
    word_edits = sum(item.word_edits for item in metrics)
    reference_words = sum(item.reference_words for item in metrics)
    presence = term_presence_counts(
        [str(row["reference"]) for row in rows],
        [str(row["hypothesis"]) for row in rows],
        terms,
    )
    audio_seconds = sum(float(row["audio_seconds"]) for row in rows)
    processing_seconds = sum(float(row["processing_seconds"]) for row in rows)
    return (
        {
            "record_count": len(rows),
            "failed_record_count": sum(row["status"] != "completed" for row in rows),
            "cer": character_edits / max(1, reference_characters),
            "wer": word_edits / max(1, reference_words),
            "character_edits": character_edits,
            "reference_characters": reference_characters,
            "word_edits": word_edits,
            "reference_words": reference_words,
            "audio_seconds": audio_seconds,
            "processing_seconds": processing_seconds,
            "real_time_factor": processing_seconds / max(0.001, audio_seconds),
            "priority_term_presence": presence,
        },
        metrics,
    )


def _assert_summary_matches_rows(
    summary: dict[str, object], recomputed: dict[str, object]
) -> None:
    baseline = _object(
        _object(summary["variants"], "summary variants").get("baseline"),
        "summary baseline",
    )
    for field in (
        "record_count",
        "failed_record_count",
        "character_edits",
        "reference_characters",
        "word_edits",
        "reference_words",
    ):
        if baseline.get(field) != recomputed[field]:
            raise ValueError("summary does not match private record metrics")
    for field in (
        "cer",
        "wer",
        "audio_seconds",
        "processing_seconds",
        "real_time_factor",
    ):
        if not math.isclose(
            float(baseline.get(field, math.nan)),
            float(recomputed[field]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("summary does not match private record error rates")
    observed_presence = _object(
        baseline.get("priority_term_presence"), "summary priority terms"
    )
    expected_presence = _object(
        recomputed["priority_term_presence"], "recomputed priority terms"
    )
    for field in (
        "true_positive",
        "false_negative",
        "false_insertion",
        "recall",
        "precision",
        "f1",
    ):
        observed = observed_presence.get(field)
        expected = expected_presence.get(field)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("summary priority metrics differ from private rows")
        elif observed != expected:
            raise ValueError("summary priority counts differ from private rows")


def _validate_model_binding(
    *,
    arm: str,
    summary: dict[str, object],
    conversion_report: dict[str, object],
    conversion_root: Path,
) -> dict[str, object]:
    runtime = _object(summary["runtime"], "evaluation runtime")
    model_value = runtime.get("model")
    if not isinstance(model_value, str):
        raise ValueError("evaluation model path is missing")
    model_path = Path(model_value)
    arms = _object(conversion_report["arms"], "conversion arms")
    if arm == "A_operational_baseline":
        identity = _object(arms[arm], "operational arm")
        revision = str(identity["revision"])
        if model_path.name != revision or not (model_path / "model.bin").is_file():
            raise ValueError("operational model is not the pinned local snapshot")
        return {
            "model": identity["model"],
            "revision": revision,
            "model_bin_sha256": _sha256(model_path / "model.bin"),
        }
    identity = _object(arms[arm], f"conversion arm {arm}")
    expected = (conversion_root / str(identity["path"])).resolve()
    if model_path.resolve() != expected:
        raise ValueError("evaluation model does not match its conversion arm")
    return {
        "source_model": identity["source_model"],
        "source_revision": identity["source_revision"],
        "conversion_path": identity["path"],
    }


def build_abc_report(
    *,
    conversion_report_path: Path,
    experiment_config_path: Path,
    priority_terms_path: Path,
    summaries: Mapping[str, Path],
    records: Mapping[str, Path],
    output_path: Path,
    evaluator_revision: str,
) -> dict[str, object]:
    """Validate three locked runs and decide only the clean evaluation gate."""

    arms = (
        "A_operational_baseline",
        "B_same_conversion_base_control",
        "C_lora_merged_candidate",
    )
    if set(summaries) != set(arms) or set(records) != set(arms):
        raise ValueError("all and only A/B/C inputs are required")
    if re.fullmatch(r"[0-9a-f]{40}", evaluator_revision) is None:
        raise ValueError("evaluator revision must be a full Git commit")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("refusing to overwrite A/B/C report")
    conversion = validate_conversion_output(conversion_report_path)
    experiment, experiment_bytes = load_experiment_config(experiment_config_path)
    if priority_terms_path.is_symlink() or not priority_terms_path.is_file():
        raise ValueError("priority terms must be a regular non-symlink file")
    terms_bytes = priority_terms_path.read_bytes()
    if hashlib.sha256(terms_bytes).hexdigest() != experiment["data"][
        "priority_terms_sha256"
    ]:
        raise ValueError("priority terms differ from the experiment registration")
    terms = load_hotwords(priority_terms_path)

    loaded_summaries: dict[str, dict[str, object]] = {}
    indexed_rows: dict[str, dict[str, dict[str, object]]] = {}
    aggregate: dict[str, dict[str, object]] = {}
    metrics: dict[str, list[RecordMetric]] = {}
    fingerprints: dict[str, dict[str, object]] = {}
    model_provenance: dict[str, dict[str, object]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for arm in arms:
        summary = _read_json(
            summaries[arm], maximum=MAX_SUMMARY_BYTES, name=f"{arm} summary"
        )
        rows = _read_private_rows(records[arm])
        fingerprint = _validate_summary(summary, terms)
        ordered = sorted(rows, key=lambda row: str(row["record_key"]))
        recomputed, arm_metrics = _recompute(ordered, terms)
        _assert_summary_matches_rows(summary, recomputed)
        loaded_summaries[arm] = summary
        indexed_rows[arm] = _index_rows(ordered)
        aggregate[arm] = recomputed
        metrics[arm] = arm_metrics
        fingerprints[arm] = fingerprint
        model_provenance[arm] = _validate_model_binding(
            arm=arm,
            summary=summary,
            conversion_report=conversion,
            conversion_root=conversion_report_path.parent,
        )
        input_hashes[arm] = {
            "summary_sha256": _sha256(summaries[arm]),
            "records_sha256": _sha256(records[arm]),
        }
    if not all(value == fingerprints[arms[0]] for value in fingerprints.values()):
        raise ValueError("A/B/C dataset fingerprints differ")
    expected_keys = set(indexed_rows[arms[0]])
    for arm in arms[1:]:
        if set(indexed_rows[arm]) != expected_keys:
            raise ValueError("A/B/C record pairing differs")
        for key in expected_keys:
            left = indexed_rows[arms[0]][key]
            right = indexed_rows[arm][key]
            if (
                left["reference"] != right["reference"]
                or float(left["audio_seconds"]) != float(right["audio_seconds"])
            ):
                raise ValueError("A/B/C paired references or audio durations differ")

    evaluation = _object(experiment["evaluation"], "evaluation thresholds")
    a, b, c = arms
    ab_cer = float(aggregate[b]["cer"]) - float(aggregate[a]["cer"])
    ab_wer = float(aggregate[b]["wer"]) - float(aggregate[a]["wer"])
    bc_cer = float(aggregate[c]["cer"]) - float(aggregate[b]["cer"])
    bc_wer = float(aggregate[c]["wer"]) - float(aggregate[b]["wer"])
    b_terms = _object(aggregate[b]["priority_term_presence"], "B priority terms")
    c_terms = _object(aggregate[c]["priority_term_presence"], "C priority terms")
    checks = {
        "all_records_completed": all(
            value["failed_record_count"] == 0 for value in aggregate.values()
        ),
        "conversion_control_cer_drift": abs(ab_cer)
        <= float(evaluation["operational_to_conversion_control_cer_max_absolute_regression"]),
        "conversion_control_wer_drift": abs(ab_wer)
        <= float(evaluation["operational_to_conversion_control_wer_max_absolute_regression"]),
        "candidate_clean_cer_regression": bc_cer
        <= float(evaluation["clean_cer_max_absolute_regression"]),
        "candidate_clean_wer_regression": bc_wer
        <= float(evaluation["clean_wer_max_absolute_regression"]),
        "candidate_false_insertion_nonincrease": int(c_terms["false_insertion"])
        <= int(b_terms["false_insertion"]),
        "all_arms_rtf_within_service_limit": all(
            float(value["real_time_factor"])
            <= float(evaluation["max_real_time_factor"])
            for value in aggregate.values()
        ),
    }
    if not checks["all_records_completed"]:
        decision = "reject_candidate_keep_operational_baseline"
    elif not (
        checks["conversion_control_cer_drift"]
        and checks["conversion_control_wer_drift"]
    ):
        decision = "comparison_invalid_keep_operational_baseline"
    elif all(checks.values()):
        decision = "continue_wind_and_downstream_gates"
    else:
        decision = "reject_candidate_keep_operational_baseline"

    report = {
        "schema_version": "1.0.0",
        "protocol_id": ABC_REPORT_PROTOCOL_ID,
        "status": "evaluated",
        "fact_status": "부분 구현 또는 개발용 데모",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_scope": (
            "AIHub Gwangju emergency-call Validation; not field-radio or field-safety "
            "validation"
        ),
        "dataset": fingerprints[a],
        "provenance": {
            "evaluator_revision": evaluator_revision,
            "conversion_report_sha256": _sha256(conversion_report_path),
            "experiment_config_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
            "priority_terms_sha256": hashlib.sha256(terms_bytes).hexdigest(),
            "inputs": input_hashes,
            "models": model_provenance,
        },
        "aggregate": aggregate,
        "effects": {
            "B_minus_A_conversion_control": {
                "cer_delta": ab_cer,
                "wer_delta": ab_wer,
                "cer_bootstrap": paired_bootstrap_error_delta(
                    metrics[a], metrics[b], metric="cer"
                ),
                "wer_bootstrap": paired_bootstrap_error_delta(
                    metrics[a], metrics[b], metric="wer"
                ),
            },
            "C_minus_B_lora_candidate": {
                "cer_delta": bc_cer,
                "wer_delta": bc_wer,
                "false_insertion_delta": int(c_terms["false_insertion"])
                - int(b_terms["false_insertion"]),
                "cer_bootstrap": paired_bootstrap_error_delta(
                    metrics[b], metrics[c], metric="cer"
                ),
                "wer_bootstrap": paired_bootstrap_error_delta(
                    metrics[b], metrics[c], metric="wer"
                ),
            },
        },
        "checks": checks,
        "decision": decision,
        "automatic_adoption_allowed": False,
        "remaining_gates": [
            "wind_snr0 paired evaluation",
            "priority-term improvement",
            "downstream Parser and Resolver regression",
            "wrong single-CAS promotion equals zero",
            "preconfirmation Rule Engine execution equals zero",
        ],
        "claim_scope": (
            "locked clean proxy evaluation only; no field-radio, field-safety, production "
            "adoption, or deployment claim"
        ),
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.parent.chmod(0o700)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    for arm in ("a", "b", "c"):
        parser.add_argument(f"--{arm}-summary", type=Path, required=True)
        parser.add_argument(f"--{arm}-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-revision", required=True)
    args = parser.parse_args(argv)
    arm_names = (
        "A_operational_baseline",
        "B_same_conversion_base_control",
        "C_lora_merged_candidate",
    )
    report = build_abc_report(
        conversion_report_path=args.conversion_report,
        experiment_config_path=args.experiment_config,
        priority_terms_path=args.priority_terms,
        summaries=dict(zip(arm_names, (args.a_summary, args.b_summary, args.c_summary))),
        records=dict(zip(arm_names, (args.a_records, args.b_records, args.c_records))),
        output_path=args.output,
        evaluator_revision=args.evaluator_revision,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "fact_status": report["fact_status"],
                "decision": report["decision"],
                "automatic_adoption_allowed": report["automatic_adoption_allowed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
