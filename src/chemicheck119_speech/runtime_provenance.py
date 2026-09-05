"""Cloud Run execution에서 교차지역 평가용 비식별 provenance만 추출한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

from chemicheck119_speech.cross_region_report import (
    CONTAINER_DIGEST_PATTERN,
    EXPECTED_EVALUATIONS,
    EXPECTED_JOBS,
    RESOURCE_NAME_PATTERN,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    sha256_file,
    validate_runtime_provenance,
)


class RuntimeProvenanceCaptureError(RuntimeError):
    """원본 gcloud stderr나 annotation을 노출하지 않는 capture 오류."""


DescribeExecution = Callable[[str], Mapping[str, Any]]


def _gcloud_describer(project: str, region: str) -> DescribeExecution:
    if not RESOURCE_NAME_PATTERN.fullmatch(project) or not RESOURCE_NAME_PATTERN.fullmatch(
        region
    ):
        raise RuntimeProvenanceCaptureError("project 또는 region 형식이 잘못되었습니다.")

    def describe(execution_name: str) -> Mapping[str, Any]:
        try:
            completed = subprocess.run(  # noqa: S603
                [
                    "gcloud",
                    "run",
                    "jobs",
                    "executions",
                    "describe",
                    execution_name,
                    f"--project={project}",
                    f"--region={region}",
                    "--format=json",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeProvenanceCaptureError(
                f"execution 조회에 실패했습니다: {execution_name}"
            ) from error
        if completed.returncode != 0:
            raise RuntimeProvenanceCaptureError(
                f"execution 조회가 거부됐습니다: {execution_name}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeProvenanceCaptureError(
                f"execution 응답 JSON이 잘못됐습니다: {execution_name}"
            ) from error
        if not isinstance(payload, Mapping):
            raise RuntimeProvenanceCaptureError(
                f"execution 응답이 객체가 아닙니다: {execution_name}"
            )
        return payload

    return describe


def _completed_successfully(snapshot: Mapping[str, Any]) -> bool:
    status = snapshot.get("status")
    conditions = status.get("conditions") if isinstance(status, Mapping) else None
    return isinstance(conditions, list) and any(
        isinstance(item, Mapping)
        and item.get("type") == "Completed"
        and item.get("status") == "True"
        for item in conditions
    )


def _region_payload(
    region: str,
    execution_name: str,
    summary_path: Path,
    snapshot: Mapping[str, Any],
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
    digest = image.rpartition("@")[2] if isinstance(image, str) else None
    job_name = (
        labels.get("run.googleapis.com/job") if isinstance(labels, Mapping) else None
    )
    actual_name = metadata.get("name") if isinstance(metadata, Mapping) else None
    start_time = status.get("startTime") if isinstance(status, Mapping) else None
    completion_time = (
        status.get("completionTime") if isinstance(status, Mapping) else None
    )
    if (
        actual_name != execution_name
        or job_name != EXPECTED_JOBS[region]
        or not isinstance(digest, str)
        or not CONTAINER_DIGEST_PATTERN.fullmatch(digest)
        or not isinstance(start_time, str)
        or not isinstance(completion_time, str)
        or not _completed_successfully(snapshot)
    ):
        raise RuntimeProvenanceCaptureError(
            f"완료된 고정 execution 계약과 다릅니다: {region}"
        )
    return {
        "execution_name": execution_name,
        "job_name": job_name,
        "container_image_digest": digest,
        "start_time": start_time,
        "completion_time": completion_time,
        "completion_succeeded": True,
        "summary_sha256": sha256_file(summary_path),
    }


def capture_runtime_provenance(
    *,
    execution_names: Mapping[str, str],
    summary_paths: Mapping[str, Path],
    describe_execution: DescribeExecution,
    captured_at: str | None = None,
) -> dict[str, Any]:
    if set(execution_names) != set(EXPECTED_EVALUATIONS) or set(summary_paths) != set(
        EXPECTED_EVALUATIONS
    ):
        raise RuntimeProvenanceCaptureError("광주·인천·서울 입력이 모두 필요합니다.")
    regions: dict[str, dict[str, Any]] = {}
    for region in EXPECTED_EVALUATIONS:
        execution_name = execution_names[region]
        if not RESOURCE_NAME_PATTERN.fullmatch(execution_name):
            raise RuntimeProvenanceCaptureError(
                f"execution 이름 형식이 잘못됐습니다: {region}"
            )
        regions[region] = _region_payload(
            region,
            execution_name,
            Path(summary_paths[region]),
            describe_execution(execution_name),
        )
    payload = {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "captured_at": captured_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "gcloud run jobs executions describe",
        "regions": regions,
    }
    return validate_runtime_provenance(
        payload,
        summary_paths={region: Path(path) for region, path in summary_paths.items()},
    )


def write_runtime_provenance(path: Path, payload: Mapping[str, Any]) -> None:
    """기존 파일을 덮어쓰지 않고 소유자 전용 권한으로 결과를 기록한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    for evaluation_region in EXPECTED_EVALUATIONS:
        parser.add_argument(f"--{evaluation_region}-execution", required=True)
        parser.add_argument(
            f"--{evaluation_region}-summary", type=Path, required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    execution_names = {
        region: getattr(args, f"{region}_execution") for region in EXPECTED_EVALUATIONS
    }
    summary_paths = {
        region: getattr(args, f"{region}_summary") for region in EXPECTED_EVALUATIONS
    }
    payload = capture_runtime_provenance(
        execution_names=execution_names,
        summary_paths=summary_paths,
        describe_execution=_gcloud_describer(args.project, args.region),
    )
    write_runtime_provenance(args.output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "region_count": len(payload["regions"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
