"""Build a provenance-bound comparison for fixed cross-region STT summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "speech-cross-region-comparison-v2"
RUNTIME_PROVENANCE_SCHEMA_VERSION = "speech-cross-region-runtime-provenance-v1"
MAX_RUNTIME_PROVENANCE_BYTES = 64 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RESOURCE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,126}$")
EXPECTED_RUNTIME = {
    "implementation": "faster-whisper",
    "version": "1.2.1",
    "model": "small",
    "requested_device": "cpu",
    "device": "cpu",
    "compute_type": "int8",
    "language": "ko (configured, not detected)",
    "beam_size": 5,
    "temperature": 0.0,
    "vad_filter": True,
    "condition_on_previous_text": False,
}
EXPECTED_EVALUATIONS = {
    "gwangju": ("speech_aihub119_gwangju_fire_validation_77", 77),
    "incheon": ("speech_aihub_71768_incheon_fire_validation_129", 129),
    "seoul": ("speech_aihub_71768_seoul_fire_validation_965", 965),
}
CROSS_REGIONS = ("incheon", "seoul")
EXPECTED_JOBS = {
    "gwangju": "chemicheck119-speech-eval-cpu",
    "incheon": "chemicheck119-speech-cross-region-cpu",
    "seoul": "chemicheck119-speech-seoul-cpu",
}
RTF_MAXIMUM = 0.5
TERM_PRECISION_MINIMUM = 0.95
FALSE_INSERTION_RATE_MAXIMUM = 0.01
LORA_TERM_RECALL_SIGNAL = 0.8


class CrossRegionReportError(RuntimeError):
    """A comparison error that never includes transcript text."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aware_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CrossRegionReportError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CrossRegionReportError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise CrossRegionReportError(f"{label} timestamp must include a timezone")
    return parsed


def load_runtime_provenance(
    path: Path, *, summary_paths: dict[str, Path]
) -> dict[str, Any]:
    """비식별 Cloud Run execution snapshot을 summary hash와 결합 검증한다."""

    size = path.stat().st_size
    if size <= 0 or size > MAX_RUNTIME_PROVENANCE_BYTES:
        raise CrossRegionReportError(
            f"runtime provenance size is outside the allowed range: {size}"
        )
    provenance = json.loads(path.read_text(encoding="utf-8"))
    regions = provenance.get("regions") if isinstance(provenance, dict) else None
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA_VERSION
        or provenance.get("source") != "gcloud run jobs executions describe"
        or not isinstance(regions, dict)
        or set(regions) != set(EXPECTED_EVALUATIONS)
        or set(summary_paths) != set(EXPECTED_EVALUATIONS)
    ):
        raise CrossRegionReportError("runtime provenance contract is invalid")
    captured_at = _aware_timestamp(provenance.get("captured_at"), "capture")

    for region, expected_job in EXPECTED_JOBS.items():
        item = regions.get(region)
        if not isinstance(item, dict):
            raise CrossRegionReportError(f"{region} runtime provenance is missing")
        execution_name = item.get("execution_name")
        job_name = item.get("job_name")
        digest = item.get("container_image_digest")
        summary_digest = item.get("summary_sha256")
        if (
            not isinstance(execution_name, str)
            or not RESOURCE_NAME_PATTERN.fullmatch(execution_name)
            or not execution_name.startswith(f"{expected_job}-")
            or job_name != expected_job
            or not isinstance(digest, str)
            or not CONTAINER_DIGEST_PATTERN.fullmatch(digest)
            or not isinstance(summary_digest, str)
            or not SHA256_PATTERN.fullmatch(summary_digest)
            or summary_digest != sha256_file(summary_paths[region])
            or item.get("completion_succeeded") is not True
        ):
            raise CrossRegionReportError(
                f"{region} runtime provenance does not match the fixed execution"
            )
        start_time = _aware_timestamp(item.get("start_time"), f"{region} start")
        completion_time = _aware_timestamp(
            item.get("completion_time"), f"{region} completion"
        )
        if completion_time < start_time or captured_at < completion_time:
            raise CrossRegionReportError(
                f"{region} runtime provenance timestamps are inconsistent"
            )

    cross_digests = {
        regions[region]["container_image_digest"] for region in CROSS_REGIONS
    }
    if len(cross_digests) != 1:
        raise CrossRegionReportError(
            "incheon and seoul must use the same immutable container image"
        )
    return provenance


def _load_summary(path: Path, region: str) -> dict[str, Any]:
    expected_id, expected_records = EXPECTED_EVALUATIONS[region]
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema_version") != "1.0.0":
        raise CrossRegionReportError(f"{region} summary schema is unsupported")
    if summary.get("usage_role") != "evaluation":
        raise CrossRegionReportError(f"{region} summary is not a fixed evaluation")
    if "not field-radio" not in str(summary.get("evidence_scope") or ""):
        raise CrossRegionReportError(f"{region} field-radio limitation is missing")

    dataset = summary.get("dataset")
    runtime = summary.get("runtime")
    variants = summary.get("variants")
    if not isinstance(dataset, dict) or not isinstance(runtime, dict):
        raise CrossRegionReportError(f"{region} summary metadata is incomplete")
    expected_record_count = dataset.get("expected_record_count")
    legacy_gwangju_count_is_valid = (
        region == "gwangju" and expected_record_count is None
    )
    if (
        dataset.get("evaluation_id") != expected_id
        or dataset.get("record_count") != expected_records
        or (
            expected_record_count != expected_records
            and not legacy_gwangju_count_is_valid
        )
    ):
        raise CrossRegionReportError(f"{region} evaluation identity does not match")
    for field, expected in EXPECTED_RUNTIME.items():
        if runtime.get(field) != expected:
            raise CrossRegionReportError(
                f"{region} runtime differs from the frozen setting: {field}"
            )
    if not isinstance(variants, dict) or "baseline" not in variants:
        raise CrossRegionReportError(f"{region} baseline metrics are missing")
    if region in CROSS_REGIONS and (
        set(variants) != {"baseline"} or runtime.get("variants") != ["baseline"]
    ):
        raise CrossRegionReportError(
            f"{region} cross-region evaluation must be baseline-only"
        )
    baseline = variants["baseline"]
    if (
        not isinstance(baseline, dict)
        or baseline.get("record_count") != expected_records
        or not isinstance(summary.get("priority_terms"), list)
        or not summary["priority_terms"]
    ):
        raise CrossRegionReportError(f"{region} aggregate is incomplete")
    return summary


def _term_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    baseline = summary["variants"]["baseline"]
    terms = baseline["priority_term_presence"]
    record_count = int(baseline["record_count"])
    positive_opportunities = int(terms["true_positive"]) + int(terms["false_negative"])
    total_opportunities = record_count * len(summary["priority_terms"])
    negative_opportunities = total_opportunities - positive_opportunities
    false_insertions = int(terms["false_insertion"])
    false_insertion_rate = (
        false_insertions / negative_opportunities if negative_opportunities else None
    )
    return {
        **terms,
        "positive_opportunities": positive_opportunities,
        "negative_opportunities": negative_opportunities,
        "false_insertion_rate_on_negative_opportunities": false_insertion_rate,
    }


def _region_result(summary: dict[str, Any]) -> dict[str, Any]:
    baseline = summary["variants"]["baseline"]
    terms = _term_metrics(summary)
    precision = terms.get("precision")
    false_insertion_rate = terms["false_insertion_rate_on_negative_opportunities"]
    gates = {
        "complete_without_stt_failure": {
            "passed": baseline.get("failed_record_count") == 0,
            "actual": baseline.get("failed_record_count"),
            "target": 0,
        },
        "rtf": {
            "passed": float(baseline["real_time_factor"]) <= RTF_MAXIMUM,
            "actual": baseline["real_time_factor"],
            "target_maximum": RTF_MAXIMUM,
        },
        "priority_term_precision": {
            "passed": precision is not None
            and float(precision) >= TERM_PRECISION_MINIMUM,
            "actual": precision,
            "target_minimum": TERM_PRECISION_MINIMUM,
        },
        "false_insertion_rate": {
            "passed": false_insertion_rate is not None
            and float(false_insertion_rate) <= FALSE_INSERTION_RATE_MAXIMUM,
            "actual": false_insertion_rate,
            "target_maximum": FALSE_INSERTION_RATE_MAXIMUM,
        },
    }
    return {
        "record_count": baseline["record_count"],
        "audio_hours": float(baseline["audio_seconds"]) / 3600.0,
        "cer": baseline["cer"],
        "wer": baseline["wer"],
        "rtf": baseline["real_time_factor"],
        "failed_record_count": baseline["failed_record_count"],
        "priority_terms": terms,
        "gates": gates,
        "gate_passed": all(item["passed"] for item in gates.values()),
    }


def build_cross_region_report(
    *,
    gwangju_summary_path: Path,
    incheon_summary_path: Path,
    seoul_summary_path: Path,
    runtime_provenance_path: Path,
    evaluator_git_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    paths = {
        "gwangju": gwangju_summary_path,
        "incheon": incheon_summary_path,
        "seoul": seoul_summary_path,
    }
    if not GIT_COMMIT_PATTERN.fullmatch(evaluator_git_commit):
        raise CrossRegionReportError("evaluator Git commit must be a full SHA-1")
    summaries = {region: _load_summary(path, region) for region, path in paths.items()}
    runtime_provenance = load_runtime_provenance(
        runtime_provenance_path, summary_paths=paths
    )
    priority_terms = {
        region: tuple(summary["priority_terms"])
        for region, summary in summaries.items()
    }
    if len(set(priority_terms.values())) != 1:
        raise CrossRegionReportError("priority terms differ between regions")

    regions = {region: _region_result(summary) for region, summary in summaries.items()}
    baseline = regions["gwangju"]
    for region in CROSS_REGIONS:
        regions[region]["unpaired_delta_vs_observed_gwangju"] = {
            metric: float(regions[region][metric]) - float(baseline[metric])
            for metric in ("cer", "wer", "rtf")
        }

    cross_region_gate_passed = all(
        regions[region]["gate_passed"] for region in CROSS_REGIONS
    )
    low_recall_regions = [
        region
        for region in CROSS_REGIONS
        if regions[region]["priority_terms"].get("recall") is not None
        and float(regions[region]["priority_terms"]["recall"]) < LORA_TERM_RECALL_SIGNAL
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fact_status": "구현 완료",
        "evaluation_name": "고정 faster-whisper 교차지역 신고접수 전화 평가",
        "evidence_scope": (
            "AIHub 광주·인천·서울 화재 신고접수 전화의 오프라인 비교; "
            "현장 무전·화학시설 신고 모집단·전국 일반화 검증 아님"
        ),
        "input_artifacts": {
            region: {
                "summary_sha256": sha256_file(path),
                "evaluation_id": EXPECTED_EVALUATIONS[region][0],
                "cloud_run_execution": runtime_provenance["regions"][region][
                    "execution_name"
                ],
                "container_image_digest": runtime_provenance["regions"][region][
                    "container_image_digest"
                ],
            }
            for region, path in paths.items()
        },
        "runtime_provenance_sha256": sha256_file(runtime_provenance_path),
        "evaluation_runtime": {
            "repository": "chemicheck119-lab/speech-service",
            "git_commit": evaluator_git_commit,
        },
        "comparability_gate": {
            "passed": True,
            "summary_hashes_bound_to_executions": True,
            "incheon_seoul_same_container_image": True,
            "gwangju_is_observed_legacy_baseline": True,
            "gwangju_image_may_differ": True,
        },
        "schema_compatibility": {
            "gwangju_missing_expected_record_count": (
                summaries["gwangju"]["dataset"].get("expected_record_count") is None
            ),
            "gwangju_legacy_exception_basis": (
                "고정 evaluation ID와 aggregate record_count 77 일치; "
                "인천·서울에는 expected_record_count를 필수로 유지"
            ),
        },
        "frozen_runtime": EXPECTED_RUNTIME,
        "priority_terms_sha256": hashlib.sha256(
            ("\n".join(priority_terms["gwangju"]) + "\n").encode("utf-8")
        ).hexdigest(),
        "regions": regions,
        "cross_region_gate": {
            "passed": cross_region_gate_passed,
            "regions": list(CROSS_REGIONS),
            "decision": (
                "CONDITIONALLY_ACCEPT_BASELINE_FOR_CALL_PROXY_EVALUATION"
                if cross_region_gate_passed
                else "DO_NOT_ACCEPT_BASELINE_BEFORE_FAILURE_ANALYSIS"
            ),
            "cer_wer_are_not_standalone_acceptance_gates": True,
        },
        "whisper_lora_gate": {
            "decision": "HOLD_PENDING_DISTORTION_AND_ERROR_TAXONOMY",
            "low_term_recall_regions": low_recall_regions,
            "aggregate_repeat_signal": len(low_recall_regions) >= 2,
            "same_error_type_repeat_proven": False,
            "downstream_candidate_denominator_checked_here": False,
            "reason": (
                "집계 Recall만으로 반복 오류 유형이나 후단 후보 손실을 증명할 수 없으므로 "
                "모의 왜곡과 비공개 실패 분류 전에는 LoRA를 실행하지 않습니다."
            ),
        },
        "claims_not_allowed": [
            "실제 현장 무전 정확도",
            "화학물질 CAS 정답 정확도",
            "광주·인천·서울 차이를 동일 표본의 paired 변화로 해석",
            "실제 안전성 또는 전국 일반화",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gwangju-summary", type=Path, required=True)
    parser.add_argument("--incheon-summary", type=Path, required=True)
    parser.add_argument("--seoul-summary", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--evaluator-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    report = build_cross_region_report(
        gwangju_summary_path=args.gwangju_summary,
        incheon_summary_path=args.incheon_summary,
        seoul_summary_path=args.seoul_summary,
        runtime_provenance_path=args.runtime_provenance,
        evaluator_git_commit=args.evaluator_git_commit,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "decision": report["cross_region_gate"]["decision"],
                "lora_decision": report["whisper_lora_gate"]["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
