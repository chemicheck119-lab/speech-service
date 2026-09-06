"""서울·인천 radio-sim 결과를 결합해 Whisper LoRA 실행 Gate를 판정한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .cross_region_report import CONTAINER_DIGEST_PATTERN, GIT_COMMIT_PATTERN
from .robustness import MAX_RECORDS_PER_VARIANT, PROFILE_ID, REGISTERED_VARIANTS
from .runtime_provenance import _gcloud_describer


SCHEMA_VERSION = "speech-radio-sim-cross-region-gate-v1"
EXPECTED_REGIONS = ("incheon", "seoul")
EXPECTED_JOBS = {
    "incheon": "chemicheck119-speech-radio-sim-incheon-cpu",
    "seoul": "chemicheck119-speech-radio-sim-seoul-cpu",
}
SUMMARY_SCHEMA_VERSION = "1.0.0"
DOWNSTREAM_SCHEMA_VERSION = "stt-radio-sim-downstream-silver-eval-v1"
MAX_SUMMARY_BYTES = 16 * 1024 * 1024
MAX_DOWNSTREAM_BYTES = 32 * 1024 * 1024
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
LORA_AGGREGATE_DENOMINATOR_MINIMUM = 20
LORA_TERM_DENOMINATOR_MINIMUM = 5
LORA_RECALL_MAXIMUM_EXCLUSIVE = 0.8
SAFETY_METRICS = (
    "candidate_promotion_violation_count",
    "rule_execution_before_confirmation_count",
    "two_cas_gate_violation_count",
    "unconfirmed_risk_output_violation_count",
)
RUNTIME_FIELDS = (
    "implementation",
    "version",
    "model",
    "device",
    "compute_type",
    "language",
    "beam_size",
    "temperature",
    "vad_filter",
    "condition_on_previous_text",
    "variants",
)


class RadioSimGateError(RuntimeError):
    """원문 전사나 민감한 Cloud Run annotation을 노출하지 않는 Gate 오류."""


DescribeExecution = Callable[[str], Mapping[str, Any]]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise RadioSimGateError(f"{label} 크기가 허용 범위를 벗어났습니다.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RadioSimGateError(f"{label} JSON이 잘못됐습니다.") from error
    if not isinstance(payload, dict):
        raise RadioSimGateError(f"{label} root는 객체여야 합니다.")
    return payload


def _runtime_fingerprint(runtime: object) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise RadioSimGateError("STT runtime 정보가 없습니다.")
    fingerprint = {field: runtime.get(field) for field in RUNTIME_FIELDS}
    if (
        fingerprint["implementation"] != "faster-whisper"
        or fingerprint["model"] != "small"
        or fingerprint["device"] != "cpu"
        or fingerprint["compute_type"] != "int8"
        or fingerprint["beam_size"] != 5
        or fingerprint["temperature"] != 0.0
        or fingerprint["vad_filter"] is not True
        or fingerprint["condition_on_previous_text"] is not False
        or fingerprint["variants"] != ["baseline"]
    ):
        raise RadioSimGateError("사전 고정한 faster-whisper 기준선과 다릅니다.")
    return fingerprint


def _load_summary(path: Path) -> dict[str, Any]:
    summary = _load_json(path, maximum_bytes=MAX_SUMMARY_BYTES, label="STT summary")
    simulation = summary.get("simulation_run")
    variants = summary.get("variants")
    record_count = summary.get("record_count")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("usage_role") != "evaluation"
        or "not field-radio" not in str(summary.get("evidence_scope") or "")
        or not isinstance(simulation, Mapping)
        or simulation.get("profile_id") != PROFILE_ID
        or simulation.get("variant_count") != len(REGISTERED_VARIANTS)
        or type(record_count) is not int
        or record_count <= 0
        or record_count > MAX_RECORDS_PER_VARIANT
        or not isinstance(variants, Mapping)
        or set(variants) != set(REGISTERED_VARIANTS)
    ):
        raise RadioSimGateError("STT summary가 radio-sim-v1 평가 계약과 다릅니다.")
    for field in (
        "run_summary_sha256",
        "source_manifest_sha256",
        "priority_terms_sha256",
    ):
        if not DIGEST_PATTERN.fullmatch(str(simulation.get(field) or "")):
            raise RadioSimGateError(f"STT summary의 {field}가 잘못됐습니다.")
    selected = simulation.get("selected")
    if not isinstance(selected, Mapping) or selected.get("total") != record_count:
        raise RadioSimGateError("STT summary 표본 수가 simulation manifest와 다릅니다.")
    for condition, metrics in variants.items():
        terms = metrics.get("priority_term_presence") if isinstance(metrics, Mapping) else None
        if (
            not isinstance(metrics, Mapping)
            or metrics.get("record_count") != record_count
            or type(metrics.get("failed_record_count")) is not int
            or not isinstance(terms, Mapping)
            or type(terms.get("true_positive")) is not int
            or type(terms.get("false_negative")) is not int
            or type(terms.get("false_insertion")) is not int
        ):
            raise RadioSimGateError(f"조건별 STT 지표 계약이 잘못됐습니다: {condition}")
    _runtime_fingerprint(summary.get("runtime"))
    return summary


def _load_downstream(
    path: Path, *, summary: Mapping[str, Any], summary_sha256: str
) -> dict[str, Any]:
    report = _load_json(
        path, maximum_bytes=MAX_DOWNSTREAM_BYTES, label="downstream report"
    )
    artifacts = report.get("input_artifacts")
    dataset = report.get("dataset")
    metrics = report.get("metrics")
    evaluator = report.get("evaluation_runtime")
    speech_artifact = report.get("speech_evaluator_artifact")
    by_condition = metrics.get("by_condition") if isinstance(metrics, Mapping) else None
    simulation = summary["simulation_run"]
    if (
        report.get("schema_version") != DOWNSTREAM_SCHEMA_VERSION
        or "현장 무전" not in str(report.get("evidence_scope") or "")
        or not isinstance(artifacts, Mapping)
        or artifacts.get("speech_summary_sha256") != summary_sha256
        or artifacts.get("priority_terms_sha256")
        != simulation["priority_terms_sha256"]
        or not isinstance(dataset, Mapping)
        or dataset.get("profile_id") != PROFILE_ID
        or dataset.get("source_manifest_sha256")
        != simulation["source_manifest_sha256"]
        or dataset.get("record_count_per_condition") != summary["record_count"]
        or dataset.get("condition_count") != len(REGISTERED_VARIANTS)
        or dataset.get("derived_data") is not True
        or not isinstance(metrics, Mapping)
        or metrics.get("condition_count") != len(REGISTERED_VARIANTS)
        or metrics.get("record_count_per_condition") != summary["record_count"]
        or not isinstance(by_condition, Mapping)
        or set(by_condition) != set(REGISTERED_VARIANTS)
        or metrics.get("cas_ground_truth_available") is not False
        or metrics.get("is_cas_accuracy_evaluation") is not False
        or metrics.get("wrong_single_cas_promotion_ground_truth_count") is not None
        or not isinstance(evaluator, Mapping)
        or not GIT_COMMIT_PATTERN.fullmatch(str(evaluator.get("git_commit") or ""))
        or not isinstance(speech_artifact, Mapping)
        or not CONTAINER_DIGEST_PATTERN.fullmatch(
            str(speech_artifact.get("container_image_digest") or "")
        )
    ):
        raise RadioSimGateError("downstream report가 STT summary와 결합되지 않습니다.")
    if _runtime_fingerprint(report.get("stt_runtime")) != _runtime_fingerprint(
        summary.get("runtime")
    ):
        raise RadioSimGateError("STT와 downstream runtime이 다릅니다.")
    for condition, condition_result in by_condition.items():
        if not isinstance(condition_result, Mapping) or not isinstance(
            condition_result.get("priority_term_by_term"), list
        ):
            raise RadioSimGateError(f"조건별 downstream 계약이 잘못됐습니다: {condition}")
        for row in condition_result["priority_term_by_term"]:
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("term"), str)
                or type(row.get("reference_positive_count")) is not int
                or type(row.get("false_negative")) is not int
                or type(row.get("false_insertion")) is not int
                or not (
                    row.get("recall") is None
                    or isinstance(row.get("recall"), (int, float))
                )
            ):
                raise RadioSimGateError(
                    f"우선용어 downstream 지표가 잘못됐습니다: {condition}"
                )
    return report


def _completed_successfully(snapshot: Mapping[str, Any]) -> bool:
    status = snapshot.get("status")
    conditions = status.get("conditions") if isinstance(status, Mapping) else None
    return isinstance(conditions, list) and any(
        isinstance(item, Mapping)
        and item.get("type") == "Completed"
        and item.get("status") == "True"
        for item in conditions
    )


def _execution_evidence(
    *,
    region: str,
    execution_name: str,
    snapshot: Mapping[str, Any],
    summary_sha256: str,
) -> dict[str, Any]:
    metadata = snapshot.get("metadata")
    spec = snapshot.get("spec")
    status = snapshot.get("status")
    labels = metadata.get("labels") if isinstance(metadata, Mapping) else None
    template = spec.get("template") if isinstance(spec, Mapping) else None
    template_spec = template.get("spec") if isinstance(template, Mapping) else None
    containers = template_spec.get("containers") if isinstance(template_spec, Mapping) else None
    image = containers[0].get("image") if isinstance(containers, list) and containers else None
    digest = image.rpartition("@")[2] if isinstance(image, str) else ""
    start_time = status.get("startTime") if isinstance(status, Mapping) else None
    completion_time = status.get("completionTime") if isinstance(status, Mapping) else None
    if (
        not RESOURCE_NAME_PATTERN.fullmatch(execution_name)
        or not isinstance(metadata, Mapping)
        or metadata.get("name") != execution_name
        or not isinstance(labels, Mapping)
        or labels.get("run.googleapis.com/job") != EXPECTED_JOBS[region]
        or not CONTAINER_DIGEST_PATTERN.fullmatch(digest)
        or not isinstance(start_time, str)
        or not isinstance(completion_time, str)
        or not _completed_successfully(snapshot)
    ):
        raise RadioSimGateError(f"완료된 {region} radio-sim execution이 아닙니다.")
    return {
        "execution_name": execution_name,
        "job_name": EXPECTED_JOBS[region],
        "container_image_digest": digest,
        "start_time": start_time,
        "completion_time": completion_time,
        "completion_succeeded": True,
        "summary_sha256": summary_sha256,
    }


def _gate_passed(report: Mapping[str, Any]) -> bool:
    metrics = report["metrics"]
    gate_names = (
        "evaluation_integrity_gate",
        "analysis_coverage_gate",
        "safety_contract_gate",
        "downstream_evaluation_gate",
    )
    gates_passed = all(
        isinstance(metrics.get(name), Mapping)
        and metrics[name].get("passed") is True
        for name in gate_names
    )
    totals = metrics.get("safety_violation_totals")
    return gates_passed and isinstance(totals, Mapping) and all(
        totals.get(name) == 0 for name in SAFETY_METRICS
    )


def _safety_totals_zero(report: Mapping[str, Any]) -> bool:
    totals = report["metrics"].get("safety_violation_totals")
    return isinstance(totals, Mapping) and all(
        totals.get(name) == 0 for name in SAFETY_METRICS
    )


def _specific_signals(
    summary: Mapping[str, Any], report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for condition in sorted(REGISTERED_VARIANTS - {"clean"}):
        aggregate = summary["variants"][condition]["priority_term_presence"]
        denominator = int(aggregate["true_positive"]) + int(
            aggregate["false_negative"]
        )
        recall = aggregate.get("recall")
        if (
            denominator < LORA_AGGREGATE_DENOMINATOR_MINIMUM
            or not isinstance(recall, (int, float))
            or float(recall) >= LORA_RECALL_MAXIMUM_EXCLUSIVE
        ):
            continue
        for row in report["metrics"]["by_condition"][condition][
            "priority_term_by_term"
        ]:
            term_recall = row.get("recall")
            if (
                row["reference_positive_count"] >= LORA_TERM_DENOMINATOR_MINIMUM
                and isinstance(term_recall, (int, float))
                and float(term_recall) < LORA_RECALL_MAXIMUM_EXCLUSIVE
            ):
                signals.append(
                    {
                        "condition": condition,
                        "public_term": row["term"],
                        "aggregate_denominator": denominator,
                        "aggregate_recall": recall,
                        "term_denominator": row["reference_positive_count"],
                        "term_recall": term_recall,
                        "term_false_insertion": row["false_insertion"],
                    }
                )
    return signals


def _model_api_fingerprint(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime = report.get("model_api_runtime")
    if not isinstance(runtime, Mapping):
        raise RadioSimGateError("Model API runtime 근거가 없습니다.")
    result = {
        "service_git_commit": runtime.get("service_git_commit"),
        "runtime_manifest_sha256": runtime.get("runtime_manifest_sha256"),
        "api_schema": runtime.get("api_schema"),
    }
    if (
        not GIT_COMMIT_PATTERN.fullmatch(str(result["service_git_commit"] or ""))
        or not DIGEST_PATTERN.fullmatch(str(result["runtime_manifest_sha256"] or ""))
        or result["api_schema"] != "chemiguard119-api-v1"
    ):
        raise RadioSimGateError("Model API runtime 식별자가 잘못됐습니다.")
    return result


def build_radio_sim_gate(
    *,
    summary_paths: Mapping[str, Path],
    downstream_paths: Mapping[str, Path],
    execution_names: Mapping[str, str],
    describe_execution: DescribeExecution,
    evaluator_git_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    expected = set(EXPECTED_REGIONS)
    if (
        set(summary_paths) != expected
        or set(downstream_paths) != expected
        or set(execution_names) != expected
    ):
        raise RadioSimGateError("인천·서울 입력이 모두 필요합니다.")
    if not GIT_COMMIT_PATTERN.fullmatch(evaluator_git_commit):
        raise RadioSimGateError("evaluator Git commit은 40자리 SHA여야 합니다.")

    summaries: dict[str, dict[str, Any]] = {}
    downstream: dict[str, dict[str, Any]] = {}
    execution_evidence: dict[str, dict[str, Any]] = {}
    signals: dict[str, list[dict[str, Any]]] = {}
    for region in EXPECTED_REGIONS:
        summary_path = Path(summary_paths[region])
        summary_sha256 = sha256_file(summary_path)
        summaries[region] = _load_summary(summary_path)
        downstream[region] = _load_downstream(
            Path(downstream_paths[region]),
            summary=summaries[region],
            summary_sha256=summary_sha256,
        )
        execution_evidence[region] = _execution_evidence(
            region=region,
            execution_name=execution_names[region],
            snapshot=describe_execution(execution_names[region]),
            summary_sha256=summary_sha256,
        )
        expected_digest = downstream[region]["speech_evaluator_artifact"][
            "container_image_digest"
        ]
        if expected_digest != execution_evidence[region]["container_image_digest"]:
            raise RadioSimGateError(
                f"{region} downstream report와 execution image가 다릅니다."
            )
        signals[region] = _specific_signals(summaries[region], downstream[region])

    runtime_fingerprints = {
        region: _runtime_fingerprint(summaries[region]["runtime"])
        for region in EXPECTED_REGIONS
    }
    model_api_fingerprints = {
        region: _model_api_fingerprint(downstream[region])
        for region in EXPECTED_REGIONS
    }
    evaluation_commits = {
        region: downstream[region].get("evaluation_runtime", {}).get("git_commit")
        for region in EXPECTED_REGIONS
    }
    priority_hashes = {
        region: summaries[region]["simulation_run"]["priority_terms_sha256"]
        for region in EXPECTED_REGIONS
    }
    source_hashes = {
        region: summaries[region]["simulation_run"]["source_manifest_sha256"]
        for region in EXPECTED_REGIONS
    }
    comparability_checks = {
        "same_stt_runtime": len(
            {json.dumps(value, sort_keys=True) for value in runtime_fingerprints.values()}
        )
        == 1,
        "same_speech_container_image": len(
            {item["container_image_digest"] for item in execution_evidence.values()}
        )
        == 1,
        "same_priority_term_set": len(set(priority_hashes.values())) == 1,
        "different_source_manifests": len(set(source_hashes.values()))
        == len(EXPECTED_REGIONS),
        "same_downstream_evaluator_commit": len(set(evaluation_commits.values())) == 1
        and all(GIT_COMMIT_PATTERN.fullmatch(str(value or "")) for value in evaluation_commits.values()),
        "same_model_api_runtime": len(
            {
                json.dumps(value, sort_keys=True)
                for value in model_api_fingerprints.values()
            }
        )
        == 1,
    }
    region_gate = {region: _gate_passed(downstream[region]) for region in EXPECTED_REGIONS}
    region_safety = {
        region: _safety_totals_zero(downstream[region])
        for region in EXPECTED_REGIONS
    }
    signal_maps = {
        region: {
            (item["condition"], item["public_term"]): item
            for item in signals[region]
        }
        for region in EXPECTED_REGIONS
    }
    repeated_keys = set(signal_maps["incheon"]) & set(signal_maps["seoul"])
    repeated_signals = [
        {
            "condition": condition,
            "public_term": term,
            "regions": {
                region: signal_maps[region][(condition, term)]
                for region in EXPECTED_REGIONS
            },
        }
        for condition, term in sorted(repeated_keys)
    ]

    comparable = all(comparability_checks.values())
    downstream_safe = all(region_gate.values())
    if not comparable:
        decision = "DO_NOT_RUN_LORA_INCOMPARABLE_EVALUATIONS"
        reason = "동일 runtime·용어 목록·서로 다른 지역 source 결합 조건을 충족하지 못했습니다."
    elif not downstream_safe:
        decision = "DO_NOT_RUN_LORA_SAFETY_OR_INTEGRITY_GATE_FAILED"
        reason = "downstream 무결성 또는 확인 전 안전 계약 Gate가 실패했습니다."
    elif repeated_signals:
        decision = "PROCEED_TO_BOUNDED_LORA_EXPERIMENT"
        reason = (
            "두 지역의 동일 모의 왜곡과 동일 공개 용어에서 사전 등록한 반복 누락 신호가 "
            "확인됐습니다. 광주 Training 내부 train/dev로 제한한 LoRA 대조 실험만 허용합니다."
        )
    else:
        decision = "DO_NOT_RUN_LORA_NO_REPEATED_SPECIFIC_ERROR"
        reason = "두 지역에서 반복되는 동일 조건·동일 공개 용어 누락 신호가 없습니다."

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fact_status": "구현 완료",
        "evaluation_name": "서울·인천 모의 통신 왜곡 Whisper LoRA 실행 Gate",
        "evidence_scope": (
            "AIHub 신고접수 전화의 절차적 radio-sim-v1 파생 평가; "
            "실제 현장 무전·CAS 정답·현장 안전성 검증 아님"
        ),
        "input_artifacts": {
            region: {
                **execution_evidence[region],
                "downstream_report_sha256": sha256_file(
                    Path(downstream_paths[region])
                ),
                "source_manifest_sha256": source_hashes[region],
            }
            for region in EXPECTED_REGIONS
        },
        "evaluation_runtime": {
            "repository": "chemicheck119-lab/speech-service",
            "git_commit": evaluator_git_commit,
        },
        "thresholds": {
            "aggregate_priority_term_denominator_minimum": LORA_AGGREGATE_DENOMINATOR_MINIMUM,
            "specific_term_denominator_minimum_per_region": LORA_TERM_DENOMINATOR_MINIMUM,
            "recall_maximum_exclusive": LORA_RECALL_MAXIMUM_EXCLUSIVE,
            "same_condition_and_public_term_required_across_regions": True,
        },
        "comparability_gate": {
            "passed": comparable,
            "checks": comparability_checks,
        },
        "downstream_gate": {
            "passed": downstream_safe,
            "regions": region_gate,
            "safety_violation_totals_zero_by_region": region_safety,
            "all_safety_violation_totals_zero": all(region_safety.values()),
        },
        "region_signals": signals,
        "repeated_specific_signals": repeated_signals,
        "whisper_lora_gate": {
            "decision": decision,
            "reason": reason,
            "training_data_allowed": (
                "AIHub 광주 화재 Training 내부 train/dev만"
                if decision == "PROCEED_TO_BOUNDED_LORA_EXPERIMENT"
                else None
            ),
            "seoul_incheon_may_not_be_used_for_training_or_tuning": True,
            "gpu_execution_started": False,
        },
        "claims_not_allowed": [
            "실제 현장 무전 인식 성능",
            "화학물질 CAS 정답 정확도",
            "모의 왜곡 조건을 독립 현장 표본으로 해석",
            "LoRA의 성능 개선 또는 채택을 실험 전에 주장",
            "실제 화학사고 안전성 또는 전국 일반화",
        ],
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--gcp-region", required=True)
    for region in EXPECTED_REGIONS:
        parser.add_argument(f"--{region}-summary", type=Path, required=True)
        parser.add_argument(f"--{region}-downstream", type=Path, required=True)
        parser.add_argument(f"--{region}-execution", required=True)
    parser.add_argument("--evaluator-git-commit", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_radio_sim_gate(
        summary_paths={
            region: getattr(args, f"{region}_summary") for region in EXPECTED_REGIONS
        },
        downstream_paths={
            region: getattr(args, f"{region}_downstream")
            for region in EXPECTED_REGIONS
        },
        execution_names={
            region: getattr(args, f"{region}_execution")
            for region in EXPECTED_REGIONS
        },
        describe_execution=_gcloud_describer(args.project, args.gcp_region),
        evaluator_git_commit=args.evaluator_git_commit,
        generated_at=args.generated_at,
    )
    _write_exclusive(args.output, report)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "decision": report["whisper_lora_gate"]["decision"],
                "repeated_signal_count": len(report["repeated_specific_signals"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
