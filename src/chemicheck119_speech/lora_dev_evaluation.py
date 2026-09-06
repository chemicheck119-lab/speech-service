"""Run a bounded B/C evaluation on the registered LoRA development wind set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .evaluation import evaluate_archives, load_hotwords
from .lora_conversion import validate_conversion_output
from .lora_data_preflight import _object, validate_lora_data_preflight
from .lora_protocol import load_experiment_config
from .runtime import FasterWhisperTranscriber, Transcriber


DEV_EVALUATION_PROTOCOL_ID = "whisper-small-lora-wind-dev-arm-v1"
EXPECTED_RECORDS = 132
EXPECTED_CONDITION = "wind_snr0"
EXPECTED_DATASET_ID = "aihub_71768_gwangju_fire"
EXPECTED_DATASET_VERSION = "dataset-71768_downloaded-2026-09-05"
EXPECTED_EVIDENCE_SCOPE = (
    "AIHub emergency-call Training derivative with procedural wind; "
    "not field-radio validation"
)
SUPPORTED_ARMS = {
    "B_same_conversion_base_control",
    "C_lora_merged_candidate",
}
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PRIORITY_TERMS_BYTES = 64 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_snapshot_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    values = report.get("artifact_snapshots")
    if not isinstance(values, list):
        raise ValueError("LoRA data preflight has no artifact snapshots")
    result: dict[str, dict[str, object]] = {}
    for value in values:
        item = _object(value, "artifact snapshot")
        name = item.get("file")
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("LoRA data preflight artifact names are invalid")
        result[name] = item
    return result


def _read_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("development manifest must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("development manifest exceeds the bounded size")
    content = path.read_bytes()
    payload = json.loads(content.decode("utf-8"))
    return _object(payload, "development manifest"), content


def _load_registered_terms(
    priority_terms_path: Path, experiment_config_path: Path
) -> list[str]:
    if priority_terms_path.is_symlink() or not priority_terms_path.is_file():
        raise ValueError("priority terms must be a regular non-symlink file")
    if (
        priority_terms_path.stat().st_size <= 0
        or priority_terms_path.stat().st_size > MAX_PRIORITY_TERMS_BYTES
    ):
        raise ValueError("priority terms exceed the bounded size")
    experiment, _ = load_experiment_config(experiment_config_path)
    content = priority_terms_path.read_bytes()
    if (
        hashlib.sha256(content).hexdigest()
        != experiment["data"]["priority_terms_sha256"]
    ):
        raise ValueError("priority terms differ from the experiment registration")
    return load_hotwords(priority_terms_path)


def validate_dev_inputs(
    *,
    execution_config_path: Path,
    experiment_config_path: Path,
    artifact_root: Path,
    conversion_report_path: Path,
    arm: str,
) -> dict[str, object]:
    """Bind one model arm to the immutable 132-record wind development set."""

    if arm not in SUPPORTED_ARMS:
        raise ValueError("development evaluation requires B or C conversion arm")
    data_report = validate_lora_data_preflight(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
    )
    snapshots = _artifact_snapshot_map(data_report)
    manifest_name = f"dev-{EXPECTED_CONDITION}.manifest.json"
    audio_name = f"dev-{EXPECTED_CONDITION}.zip"
    label_name = "dev-labels.zip"
    for name in (manifest_name, audio_name, label_name):
        if name not in snapshots:
            raise ValueError(f"registered development artifact is missing: {name}")
    manifest_path = artifact_root / manifest_name
    audio_path = artifact_root / audio_name
    label_path = artifact_root / label_name
    manifest, manifest_bytes = _read_manifest(manifest_path)
    split = _object(manifest.get("split"), "development split")
    parameters = _object(split.get("parameters"), "development split parameters")
    inventory = _object(manifest.get("inventory"), "development inventory")
    if (
        manifest.get("dataset_id") != EXPECTED_DATASET_ID
        or manifest.get("dataset_version") != EXPECTED_DATASET_VERSION
        or manifest.get("usage_role") != "development"
        or manifest.get("classification") != "derived"
        or manifest.get("evidence_scope") != EXPECTED_EVIDENCE_SCOPE
        or split.get("name") != "Training internal dev"
        or parameters.get("partition") != "dev"
        or parameters.get("condition") != EXPECTED_CONDITION
        or parameters.get("used_for_tuning") is not True
        or inventory.get("paired_count") != EXPECTED_RECORDS
    ):
        raise ValueError("manifest is not the registered wind development set")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    observed = {
        manifest_name: manifest_digest,
        audio_name: _sha256(audio_path),
        label_name: _sha256(label_path),
    }
    for name, digest in observed.items():
        if snapshots[name].get("sha256") != digest:
            raise ValueError("development artifact drifted after data preflight")

    conversion = validate_conversion_output(conversion_report_path)
    arms = _object(conversion.get("arms"), "conversion arms")
    arm_payload = _object(arms.get(arm), f"conversion arm {arm}")
    relative = arm_payload.get("path")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("conversion arm path is invalid")
    model_path = conversion_report_path.parent / relative
    if (
        model_path.is_symlink()
        or not model_path.is_dir()
        or not (model_path / "model.bin").is_file()
    ):
        raise ValueError("conversion model arm is incomplete")
    return {
        "arm": arm,
        "model_path": model_path,
        "audio_path": audio_path,
        "label_path": label_path,
        "dataset_provenance": {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_version": manifest.get("dataset_version"),
            "evaluation_id": (
                f"speech_aihub119_gwangju_lora_dev_{EXPECTED_CONDITION}_{EXPECTED_RECORDS}"
            ),
            "record_count": EXPECTED_RECORDS,
            "manifest_sha256": manifest_digest,
            "archive_sha256": {
                "audio": observed[audio_name],
                "labels": observed[label_name],
            },
            "usage_role": "development",
            "evidence_scope": EXPECTED_EVIDENCE_SCOPE,
            "split": "Training internal dev",
            "condition": EXPECTED_CONDITION,
            "used_for_tuning": True,
        },
        "data_preflight_sha256": {
            "execution_config": data_report["execution_config_sha256"],
            "experiment_config": data_report["experiment_config_sha256"],
            "run_summary": data_report["run_summary_sha256"],
        },
        "conversion_report_sha256": _sha256(conversion_report_path),
    }


def _secure_write_results(
    output_dir: Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite development evaluation output")
    output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_dir.parent.chmod(0o700)
    output_dir.mkdir(mode=0o700)
    summary_path = output_dir / "summary.json"
    records_path = output_dir / "records.private.jsonl"
    try:
        summary_descriptor = os.open(
            summary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(summary_descriptor, "w", encoding="utf-8") as destination:
            destination.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        records_descriptor = os.open(
            records_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(records_descriptor, "w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    except BaseException:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise


def run_lora_dev_arm(
    *,
    execution_config_path: Path,
    experiment_config_path: Path,
    artifact_root: Path,
    conversion_report_path: Path,
    priority_terms_path: Path,
    arm: str,
    output_dir: Path,
    transcriber_factory: Callable[..., Transcriber] = FasterWhisperTranscriber,
) -> dict[str, object]:
    inputs = validate_dev_inputs(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
        conversion_report_path=conversion_report_path,
        arm=arm,
    )
    terms = _load_registered_terms(priority_terms_path, experiment_config_path)
    model_path = Path(inputs["model_path"])
    transcriber = transcriber_factory(
        model=str(model_path),
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        local_files_only=True,
    )
    summary, rows = evaluate_archives(
        audio_archive=Path(inputs["audio_path"]),
        label_archive=Path(inputs["label_path"]),
        transcriber=transcriber,
        terms=terms,
        model=str(model_path),
        requested_device="cpu",
        device=getattr(transcriber, "actual_device", "cpu"),
        compute_type=getattr(transcriber, "actual_compute_type", "int8"),
        initialization_fallback=getattr(transcriber, "initialization_fallback", None),
        dataset_provenance=_object(
            inputs["dataset_provenance"], "development dataset provenance"
        ),
        expected_records=EXPECTED_RECORDS,
        variants=("baseline",),
    )
    summary["protocol_id"] = DEV_EVALUATION_PROTOCOL_ID
    summary["fact_status"] = "부분 구현 또는 개발용 데모"
    summary["model_arm"] = arm
    summary["input_bindings"] = {
        "conversion_report_sha256": inputs["conversion_report_sha256"],
        "data_preflight": inputs["data_preflight_sha256"],
    }
    summary["automatic_adoption_allowed"] = False
    summary["claim_scope"] = (
        "LoRA Training internal dev wind evaluation only; no locked test, "
        "field-radio, safety, or production claim"
    )
    _secure_write_results(output_dir, summary, rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(SUPPORTED_ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_lora_dev_arm(
        execution_config_path=args.execution_config,
        experiment_config_path=args.experiment_config,
        artifact_root=args.artifact_root,
        conversion_report_path=args.conversion_report,
        priority_terms_path=args.priority_terms,
        arm=args.arm,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "fact_status": report["fact_status"],
                "arm": report["model_arm"],
                "record_count": report["dataset"]["record_count"],
                "automatic_adoption_allowed": False,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
