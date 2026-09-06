"""서울·인천 radio-sim Cloud Run execution과 STT summary를 결합한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .cross_region_report import CONTAINER_DIGEST_PATTERN, GIT_COMMIT_PATTERN
from .robustness import MAX_RECORDS_PER_VARIANT, PROFILE_ID, REGISTERED_VARIANTS
from .runtime_provenance import _gcloud_describer


SCHEMA_VERSION = "speech-radio-sim-runtime-provenance-v1"
EXPECTED_REGIONS = ("incheon", "seoul")
EXPECTED_JOBS = {
    "incheon": "chemicheck119-speech-radio-sim-incheon-cpu",
    "seoul": "chemicheck119-speech-radio-sim-seoul-cpu",
}
MAX_SUMMARY_BYTES = 16 * 1024 * 1024
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
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


class RadioSimProvenanceError(RuntimeError):
    """원문 전사나 민감한 Cloud Run annotation을 노출하지 않는 오류."""


DescribeExecution = Callable[[str], Mapping[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_fingerprint(runtime: object) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise RadioSimProvenanceError("STT runtime 정보가 없습니다.")
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
        raise RadioSimProvenanceError(
            "사전 고정한 faster-whisper 기준선과 다릅니다."
        )
    return fingerprint


def _load_summary(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_SUMMARY_BYTES:
        raise RadioSimProvenanceError("STT summary 크기가 허용 범위를 벗어났습니다.")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RadioSimProvenanceError("STT summary JSON이 잘못됐습니다.") from error
    simulation = summary.get("simulation_run") if isinstance(summary, dict) else None
    variants = summary.get("variants") if isinstance(summary, dict) else None
    record_count = summary.get("record_count") if isinstance(summary, dict) else None
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != "1.0.0"
        or summary.get("usage_role") != "evaluation"
        or "not field-radio" not in str(summary.get("evidence_scope") or "")
        or not isinstance(simulation, Mapping)
        or simulation.get("profile_id") != PROFILE_ID
        or simulation.get("variant_count") != len(REGISTERED_VARIANTS)
        or type(record_count) is not int
        or not 0 < record_count <= MAX_RECORDS_PER_VARIANT
        or not isinstance(variants, Mapping)
        or set(variants) != set(REGISTERED_VARIANTS)
    ):
        raise RadioSimProvenanceError(
            "STT summary가 radio-sim-v1 평가 계약과 다릅니다."
        )
    for field in (
        "run_summary_sha256",
        "source_manifest_sha256",
        "priority_terms_sha256",
    ):
        if not DIGEST_PATTERN.fullmatch(str(simulation.get(field) or "")):
            raise RadioSimProvenanceError(
                f"STT summary의 {field}가 잘못됐습니다."
            )
    selected = simulation.get("selected")
    if not isinstance(selected, Mapping) or selected.get("total") != record_count:
        raise RadioSimProvenanceError(
            "STT summary 표본 수가 simulation manifest와 다릅니다."
        )
    for condition, metrics in variants.items():
        if not isinstance(metrics, Mapping) or metrics.get("record_count") != record_count:
            raise RadioSimProvenanceError(
                f"조건별 STT record 수가 다릅니다: {condition}"
            )
    _runtime_fingerprint(summary.get("runtime"))
    return summary


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
    summary: Mapping[str, Any],
    summary_sha256: str,
) -> dict[str, Any]:
    metadata = snapshot.get("metadata")
    spec = snapshot.get("spec")
    status = snapshot.get("status")
    labels = metadata.get("labels") if isinstance(metadata, Mapping) else None
    template = spec.get("template") if isinstance(spec, Mapping) else None
    template_spec = template.get("spec") if isinstance(template, Mapping) else None
    containers = (
        template_spec.get("containers") if isinstance(template_spec, Mapping) else None
    )
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
        raise RadioSimProvenanceError(
            f"완료된 {region} radio-sim execution이 아닙니다."
        )
    simulation = summary["simulation_run"]
    return {
        "execution_name": execution_name,
        "job_name": EXPECTED_JOBS[region],
        "container_image_digest": digest,
        "start_time": start_time,
        "completion_time": completion_time,
        "completion_succeeded": True,
        "summary_sha256": summary_sha256,
        "source_manifest_sha256": simulation["source_manifest_sha256"],
        "run_summary_sha256": simulation["run_summary_sha256"],
        "priority_terms_sha256": simulation["priority_terms_sha256"],
        "record_count_per_condition": summary["record_count"],
    }


def capture_radio_sim_provenance(
    *,
    summary_paths: Mapping[str, Path],
    execution_names: Mapping[str, str],
    describe_execution: DescribeExecution,
    collector_git_commit: str,
    captured_at: str | None = None,
) -> dict[str, Any]:
    if set(summary_paths) != set(EXPECTED_REGIONS) or set(execution_names) != set(
        EXPECTED_REGIONS
    ):
        raise RadioSimProvenanceError("인천·서울 입력이 모두 필요합니다.")
    if not GIT_COMMIT_PATTERN.fullmatch(collector_git_commit):
        raise RadioSimProvenanceError("collector Git commit은 40자리 SHA여야 합니다.")
    summaries: dict[str, dict[str, Any]] = {}
    regions: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    for region in EXPECTED_REGIONS:
        path = Path(summary_paths[region])
        summaries[region] = _load_summary(path)
        runtimes[region] = _runtime_fingerprint(summaries[region]["runtime"])
        regions[region] = _execution_evidence(
            region=region,
            execution_name=execution_names[region],
            snapshot=describe_execution(execution_names[region]),
            summary=summaries[region],
            summary_sha256=sha256_file(path),
        )
        regions[region]["stt_runtime"] = runtimes[region]
    checks = {
        "same_stt_runtime": len(
            {json.dumps(value, sort_keys=True) for value in runtimes.values()}
        )
        == 1,
        "same_speech_container_image": len(
            {value["container_image_digest"] for value in regions.values()}
        )
        == 1,
        "same_priority_term_set": len(
            {value["priority_terms_sha256"] for value in regions.values()}
        )
        == 1,
        "different_source_manifests": len(
            {value["source_manifest_sha256"] for value in regions.values()}
        )
        == len(EXPECTED_REGIONS),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": captured_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fact_status": "구현 완료",
        "evidence_scope": (
            "AIHub 신고접수 전화의 절차적 radio-sim-v1 실행 provenance; "
            "실제 현장 무전·CAS 정답·현장 안전성 검증 아님"
        ),
        "source": "gcloud run jobs executions describe",
        "collector": {
            "repository": "chemicheck119-lab/speech-service",
            "git_commit": collector_git_commit,
        },
        "regions": regions,
        "comparability_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "final_lora_decision_made_here": False,
        },
        "handoff": {
            "consumer": "chemicheck119-lab/analysis-engine",
            "purpose": "downstream 보고서 결합 및 단일 Whisper LoRA Gate 판정",
        },
    }


def write_provenance(path: Path, payload: Mapping[str, Any]) -> None:
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
        parser.add_argument(f"--{region}-execution", required=True)
    parser.add_argument("--collector-git-commit", required=True)
    parser.add_argument("--captured-at")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = capture_radio_sim_provenance(
        summary_paths={
            region: getattr(args, f"{region}_summary") for region in EXPECTED_REGIONS
        },
        execution_names={
            region: getattr(args, f"{region}_execution")
            for region in EXPECTED_REGIONS
        },
        describe_execution=_gcloud_describer(args.project, args.gcp_region),
        collector_git_commit=args.collector_git_commit,
        captured_at=args.captured_at,
    )
    write_provenance(args.output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "comparability_gate_passed": payload["comparability_gate"]["passed"],
                "final_lora_decision_made_here": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
