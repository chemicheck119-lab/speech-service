"""Fail-closed B/C comparison for the registered LoRA wind development set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

from .evaluation import load_hotwords
from .lora_abc_report import (
    _assert_summary_matches_rows,
    _index_rows,
    _read_json,
    _read_private_rows,
    _recompute,
    _sha256,
    _validate_model_binding,
)
from .lora_conversion import validate_conversion_output
from .lora_data_preflight import _object
from .lora_dev_evaluation import (
    DEV_EVALUATION_PROTOCOL_ID,
    EXPECTED_CONDITION,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_VERSION,
    EXPECTED_EVIDENCE_SCOPE,
    EXPECTED_RECORDS,
)
from .lora_protocol import load_experiment_config
from .metrics import RecordMetric, paired_bootstrap_error_delta, term_presence_counts


WIND_REPORT_PROTOCOL_ID = "whisper-small-lora-wind-dev-evaluation-v1"
EXPECTED_EXPERIMENT_ID = (
    f"speech_aihub119_gwangju_lora_dev_{EXPECTED_CONDITION}_{EXPECTED_RECORDS}"
)
ARMS = (
    "B_same_conversion_base_control",
    "C_lora_merged_candidate",
)


def _validate_summary(
    summary: dict[str, object],
    *,
    arm: str,
    terms: list[str],
    conversion_report_sha256: str,
) -> dict[str, object]:
    runtime = _object(summary.get("runtime"), "development runtime")
    variants = _object(summary.get("variants"), "development variants")
    dataset = _object(summary.get("dataset"), "development dataset")
    archive = _object(dataset.get("archive_sha256"), "development archive hashes")
    bindings = _object(summary.get("input_bindings"), "development input bindings")
    preflight = _object(bindings.get("data_preflight"), "development data preflight")
    if (
        summary.get("schema_version") != "1.0.0"
        or summary.get("protocol_id") != DEV_EVALUATION_PROTOCOL_ID
        or summary.get("experiment_id") != EXPECTED_EXPERIMENT_ID
        or summary.get("usage_role") != "development"
        or summary.get("evidence_scope") != EXPECTED_EVIDENCE_SCOPE
        or summary.get("fact_status") != "부분 구현 또는 개발용 데모"
        or summary.get("model_arm") != arm
        or summary.get("automatic_adoption_allowed") is not False
        or summary.get("priority_terms") != terms
        or set(variants) != {"baseline"}
    ):
        raise ValueError("summary is not the registered LoRA wind development run")
    if (
        dataset.get("dataset_id") != EXPECTED_DATASET_ID
        or dataset.get("dataset_version") != EXPECTED_DATASET_VERSION
        or dataset.get("record_count") != EXPECTED_RECORDS
        or dataset.get("expected_record_count") != EXPECTED_RECORDS
        or dataset.get("split") != "Training internal dev"
        or dataset.get("condition") != EXPECTED_CONDITION
        or dataset.get("used_for_tuning") is not True
        or not isinstance(dataset.get("manifest_sha256"), str)
        or any(
            not isinstance(archive.get(name), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(archive.get(name))) is None
            for name in ("audio", "labels")
        )
    ):
        raise ValueError(
            "summary dataset is not the registered 132-record wind dev set"
        )
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
        raise ValueError("wind development runtime settings differ")
    if (
        bindings.get("conversion_report_sha256") != conversion_report_sha256
        or set(preflight) != {"execution_config", "experiment_config", "run_summary"}
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in preflight.values()
        )
    ):
        raise ValueError("wind development input bindings differ")
    return {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_version": dataset.get("dataset_version"),
        "evaluation_id": dataset.get("evaluation_id"),
        "record_count": dataset.get("record_count"),
        "manifest_sha256": dataset.get("manifest_sha256"),
        "audio_sha256": archive.get("audio"),
        "labels_sha256": archive.get("labels"),
        "membership_role": "development_used_for_tuning",
        "condition": EXPECTED_CONDITION,
        "data_preflight": dict(preflight),
    }


def _safe_relative_improvement(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return (baseline - candidate) / baseline


def _required_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def build_wind_report(
    *,
    conversion_report_path: Path,
    clean_report_path: Path,
    experiment_config_path: Path,
    priority_terms_path: Path,
    summaries: Mapping[str, Path],
    records: Mapping[str, Path],
    output_path: Path,
    evaluator_revision: str,
) -> dict[str, object]:
    if set(summaries) != set(ARMS) or set(records) != set(ARMS):
        raise ValueError("both and only B/C wind inputs are required")
    if re.fullmatch(r"[0-9a-f]{40}", evaluator_revision) is None:
        raise ValueError("evaluator revision must be a full Git commit")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("refusing to overwrite wind evaluation report")
    conversion = validate_conversion_output(conversion_report_path)
    conversion_report_sha256 = _sha256(conversion_report_path)
    clean_report = _read_json(
        clean_report_path, maximum=4 * 1024 * 1024, name="clean A/B/C report"
    )
    clean_provenance = _object(clean_report.get("provenance"), "clean provenance")
    if (
        clean_report.get("protocol_id") != "whisper-small-lora-abc-locked-evaluation-v1"
        or clean_report.get("decision") != "continue_wind_and_downstream_gates"
        or clean_report.get("automatic_adoption_allowed") is not False
        or clean_provenance.get("conversion_report_sha256") != conversion_report_sha256
    ):
        raise ValueError("clean A/B/C gate does not authorize wind evaluation")
    experiment, experiment_bytes = load_experiment_config(experiment_config_path)
    if priority_terms_path.is_symlink() or not priority_terms_path.is_file():
        raise ValueError("priority terms must be a regular non-symlink file")
    terms_bytes = priority_terms_path.read_bytes()
    if (
        hashlib.sha256(terms_bytes).hexdigest()
        != experiment["data"]["priority_terms_sha256"]
    ):
        raise ValueError("priority terms differ from the experiment registration")
    terms = load_hotwords(priority_terms_path)

    aggregates: dict[str, dict[str, object]] = {}
    metrics: dict[str, list[RecordMetric]] = {}
    fingerprints: dict[str, dict[str, object]] = {}
    indexed: dict[str, dict[str, dict[str, object]]] = {}
    model_provenance: dict[str, dict[str, object]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        summary = _read_json(
            summaries[arm], maximum=4 * 1024 * 1024, name=f"{arm} wind summary"
        )
        rows = _read_private_rows(records[arm], expected_records=EXPECTED_RECORDS)
        ordered = sorted(rows, key=lambda row: str(row["record_key"]))
        recomputed, arm_metrics = _recompute(ordered, terms)
        _assert_summary_matches_rows(summary, recomputed)
        aggregates[arm] = recomputed
        metrics[arm] = arm_metrics
        fingerprints[arm] = _validate_summary(
            summary,
            arm=arm,
            terms=terms,
            conversion_report_sha256=conversion_report_sha256,
        )
        indexed[arm] = _index_rows(ordered)
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
    if fingerprints[ARMS[0]] != fingerprints[ARMS[1]]:
        raise ValueError("B/C wind dataset fingerprints differ")
    if set(indexed[ARMS[0]]) != set(indexed[ARMS[1]]):
        raise ValueError("B/C wind record pairing differs")
    for key, left in indexed[ARMS[0]].items():
        right = indexed[ARMS[1]][key]
        if left["reference"] != right["reference"] or float(
            left["audio_seconds"]
        ) != float(right["audio_seconds"]):
            raise ValueError("B/C wind references or audio durations differ")

    b, c = ARMS
    b_aggregate = aggregates[b]
    c_aggregate = aggregates[c]
    cer_relative = _safe_relative_improvement(
        float(b_aggregate["cer"]), float(c_aggregate["cer"])
    )
    wer_relative = _safe_relative_improvement(
        float(b_aggregate["wer"]), float(c_aggregate["wer"])
    )
    evaluation = _object(experiment["evaluation"], "evaluation thresholds")
    bootstrap_samples = int(evaluation["bootstrap_samples"])
    bootstrap_seed = int(evaluation["bootstrap_seed"])
    cer_bootstrap = paired_bootstrap_error_delta(
        metrics[b],
        metrics[c],
        metric="cer",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    wer_bootstrap = paired_bootstrap_error_delta(
        metrics[b],
        metrics[c],
        metric="wer",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    b_terms = _object(b_aggregate["priority_term_presence"], "B priority terms")
    c_terms = _object(c_aggregate["priority_term_presence"], "C priority terms")
    b_rows = list(indexed[b].values())
    c_rows = list(indexed[c].values())
    smoke_b = term_presence_counts(
        [str(row["reference"]) for row in b_rows],
        [str(row["hypothesis"]) for row in b_rows],
        ["연기"],
    )
    smoke_c = term_presence_counts(
        [str(row["reference"]) for row in c_rows],
        [str(row["hypothesis"]) for row in c_rows],
        ["연기"],
    )
    minimum_relative = float(evaluation["wind_cer_or_wer_min_relative_improvement"])
    maximum_ci = float(evaluation["wind_improved_metric_ci95_upper_max"])
    cer_improved = (
        cer_relative is not None
        and cer_relative >= minimum_relative
        and float(cer_bootstrap["ci95_high"]) <= maximum_ci
    )
    wer_improved = (
        wer_relative is not None
        and wer_relative >= minimum_relative
        and float(wer_bootstrap["ci95_high"]) <= maximum_ci
    )
    smoke_recall_delta = _required_number(
        smoke_c["recall"], "candidate smoke recall"
    ) - _required_number(smoke_b["recall"], "control smoke recall")
    priority_f1_delta = _required_number(
        c_terms["f1"], "candidate priority F1"
    ) - _required_number(b_terms["f1"], "control priority F1")
    false_insertion_delta = int(c_terms["false_insertion"]) - int(
        b_terms["false_insertion"]
    )
    checks = {
        "clean_gate_passed": True,
        "all_wind_records_completed": all(
            aggregate["failed_record_count"] == 0 for aggregate in aggregates.values()
        ),
        "wind_cer_or_wer_improved_with_ci": cer_improved or wer_improved,
        "wind_smoke_recall_improved": smoke_recall_delta
        >= float(evaluation["wind_smoke_recall_min_absolute_improvement"]),
        "wind_priority_f1_improved": priority_f1_delta
        >= float(evaluation["wind_priority_f1_min_absolute_improvement"]),
        "false_insertion_nonincrease": false_insertion_delta
        <= int(evaluation["false_insertion_max_increase"]),
        "both_arms_rtf_within_service_limit": all(
            float(aggregate["real_time_factor"])
            <= float(evaluation["max_real_time_factor"])
            for aggregate in aggregates.values()
        ),
    }
    decision = (
        "continue_downstream_safety_gate"
        if all(checks.values())
        else "reject_candidate_keep_operational_baseline"
    )
    report = {
        "schema_version": "1.0.0",
        "protocol_id": WIND_REPORT_PROTOCOL_ID,
        "status": "evaluated",
        "fact_status": "부분 구현 또는 개발용 데모",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_scope": EXPECTED_EVIDENCE_SCOPE,
        "dataset": fingerprints[b],
        "provenance": {
            "evaluator_revision": evaluator_revision,
            "conversion_report_sha256": conversion_report_sha256,
            "clean_report_sha256": _sha256(clean_report_path),
            "experiment_config_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
            "priority_terms_sha256": hashlib.sha256(terms_bytes).hexdigest(),
            "inputs": input_hashes,
            "models": model_provenance,
        },
        "aggregate": aggregates,
        "effects": {
            "C_minus_B_lora_candidate": {
                "cer_delta": float(c_aggregate["cer"]) - float(b_aggregate["cer"]),
                "wer_delta": float(c_aggregate["wer"]) - float(b_aggregate["wer"]),
                "cer_relative_improvement": cer_relative,
                "wer_relative_improvement": wer_relative,
                "cer_bootstrap": cer_bootstrap,
                "wer_bootstrap": wer_bootstrap,
                "smoke_recall_delta": smoke_recall_delta,
                "priority_f1_delta": priority_f1_delta,
                "false_insertion_delta": false_insertion_delta,
            }
        },
        "checks": checks,
        "decision": decision,
        "automatic_adoption_allowed": False,
        "remaining_gates": [
            "downstream Parser and Resolver regression",
            "wrong single-CAS promotion equals zero",
            "preconfirmation Rule Engine execution equals zero",
            "untouched-region evaluation is unavailable",
        ],
        "claim_scope": (
            "Training internal dev model-selection evidence only; no locked test, "
            "field-radio, field-safety, production adoption, or deployment claim"
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
    parser.add_argument("--clean-report", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--b-summary", type=Path, required=True)
    parser.add_argument("--b-records", type=Path, required=True)
    parser.add_argument("--c-summary", type=Path, required=True)
    parser.add_argument("--c-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-revision", required=True)
    args = parser.parse_args(argv)
    report = build_wind_report(
        conversion_report_path=args.conversion_report,
        clean_report_path=args.clean_report,
        experiment_config_path=args.experiment_config,
        priority_terms_path=args.priority_terms,
        summaries=dict(zip(ARMS, (args.b_summary, args.c_summary))),
        records=dict(zip(ARMS, (args.b_records, args.c_records))),
        output_path=args.output,
        evaluator_revision=args.evaluator_revision,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "automatic_adoption_allowed": False,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
