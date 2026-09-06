"""Create comparable CTranslate2 control and LoRA candidate artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
import hashlib
import json
from pathlib import Path
import re
import shutil

from .lora_data_preflight import _load_execution_config, _object
from .lora_protocol import load_experiment_config


CONVERSION_PROTOCOL_ID = "whisper-small-lora-abc-conversion-v1"
MAX_REPORT_BYTES = 1024 * 1024
MAX_ARTIFACT_FILES = 128
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 8 * 1024 * 1024 * 1024
COPY_FILES = ["tokenizer.json", "preprocessor_config.json"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("training report must be a bounded regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training report must be an object")
    return payload


def _artifact_snapshot(root: Path, path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("training artifact must be a regular non-symlink file")
    relative = path.relative_to(root).as_posix()
    return {"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}


def validate_training_run(
    *,
    training_dir: Path,
    execution_config_path: Path,
    experiment_config_path: Path,
) -> dict[str, object]:
    """Verify that the adapter is exactly the aggregate report's recorded output."""

    if training_dir.is_symlink() or not training_dir.is_dir():
        raise ValueError("training output must be a non-symlink directory")
    report = _read_report(training_dir / "training-report.json")
    execution, execution_bytes = _load_execution_config(execution_config_path)
    experiment, experiment_bytes = load_experiment_config(experiment_config_path)
    if (
        report.get("schema_version") != "1.0.0"
        or report.get("protocol_id") != "whisper-small-lora-training-v1"
        or report.get("status") != "trained_unvalidated"
        or report.get("fact_status") != "부분 구현 또는 개발용 데모"
        or report.get("automatic_adoption_allowed") is not False
        or report.get("execution_config_sha256")
        != hashlib.sha256(execution_bytes).hexdigest()
        or report.get("experiment_config_sha256")
        != hashlib.sha256(experiment_bytes).hexdigest()
    ):
        raise ValueError("training report identity or claim boundary does not match")
    cost = report.get("cost_guard")
    if (
        not isinstance(cost, dict)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(cost.get("authorized_speech_revision", ""))
        )
        is None
        or cost.get("quoted_total_krw_with_contingency") != 0
    ):
        raise ValueError("training report does not bind the local zero-cost execution")
    recorded = report.get("output_artifacts")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("training report has no output artifact snapshots")
    expected: dict[str, dict[str, object]] = {}
    for item in recorded:
        if not isinstance(item, dict):
            raise ValueError("invalid output artifact snapshot")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in expected
        ):
            raise ValueError("unsafe or duplicate output artifact path")
        expected[relative] = item
    actual_paths = sorted(
        path
        for path in training_dir.rglob("*")
        if path.is_file() and path.name != "training-report.json"
    )
    if len(actual_paths) > MAX_ARTIFACT_FILES:
        raise ValueError("training output has too many artifact files")
    actual = {
        snapshot["path"]: snapshot
        for snapshot in (
            _artifact_snapshot(training_dir, path) for path in actual_paths
        )
    }
    if actual != expected:
        raise ValueError("training output artifacts drifted from the aggregate report")
    if sum(int(item["bytes"]) for item in actual.values()) > MAX_ARTIFACT_BYTES:
        raise ValueError("training output exceeds the registered size boundary")
    if not (training_dir / "adapter" / "adapter_config.json").is_file():
        raise ValueError("training output is missing adapter_config.json")
    if not any((training_dir / "adapter").glob("*.safetensors")):
        raise ValueError("training output is missing safetensors adapter weights")
    return {"report": report, "execution": execution, "experiment": experiment}


def _hash_tree(root: Path) -> list[dict[str, object]]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if len(paths) > MAX_ARTIFACT_FILES:
        raise ValueError("conversion output has too many files")
    snapshots = [_artifact_snapshot(root, path) for path in paths]
    if sum(int(item["bytes"]) for item in snapshots) > MAX_ARTIFACT_BYTES:
        raise ValueError("conversion output exceeds the registered size boundary")
    return snapshots


def _harden_tree(root: Path) -> None:
    root.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("conversion output contains an unexpected symlink")
        path.chmod(0o700 if path.is_dir() else 0o600)


def convert_lora_arms(
    *,
    training_dir: Path,
    execution_config_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    converter_revision: str,
) -> dict[str, object]:
    """Build B and C with the same converter while preserving A as an external arm."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite conversion output")
    if re.fullmatch(r"[0-9a-f]{40}", converter_revision) is None:
        raise ValueError("converter revision must be a full Git commit")
    validated = validate_training_run(
        training_dir=training_dir,
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
    )
    experiment = _object(validated["experiment"], "experiment config")
    models = _object(experiment["models"], "experiment models")
    operational = _object(models["operational_baseline"], "operational model")
    base = _object(models["transformers_base"], "transformers base model")
    output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_dir.parent.chmod(0o700)
    if shutil.disk_usage(output_dir.parent).free < MIN_FREE_BYTES:
        raise OSError("less than 8 GiB is available for bounded model conversion")
    stage = output_dir.parent / f".{output_dir.name}.stage"
    work = output_dir.parent / f".{output_dir.name}.work"
    if any(path.exists() or path.is_symlink() for path in (stage, work)):
        raise FileExistsError("refusing to reuse conversion staging paths")
    stage.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    try:
        import torch
        from ctranslate2.converters import TransformersConverter
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        control = stage / "B_same_conversion_base_control"
        candidate = stage / "C_lora_merged_candidate"
        TransformersConverter(
            str(base["id"]),
            revision=str(base["revision"]),
            copy_files=COPY_FILES,
            load_as_float16=True,
            low_cpu_mem_usage=True,
        ).convert(str(control), quantization="float16", force=False)

        merged_root = work / "merged-transformers"
        base_model = WhisperForConditionalGeneration.from_pretrained(
            str(base["id"]),
            revision=str(base["revision"]),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(
            base_model, str(training_dir / "adapter"), is_trainable=False
        )
        merged_model = peft_model.merge_and_unload(safe_merge=True)
        merged_model.save_pretrained(
            merged_root, safe_serialization=True, max_shard_size="1GB"
        )
        processor = WhisperProcessor.from_pretrained(
            str(training_dir / "processor"), local_files_only=True
        )
        processor.save_pretrained(merged_root)
        TransformersConverter(
            str(merged_root),
            copy_files=COPY_FILES,
            load_as_float16=True,
            low_cpu_mem_usage=True,
        ).convert(str(candidate), quantization="float16", force=False)
        for arm in (control, candidate):
            if not (arm / "model.bin").is_file() or not (arm / "config.json").is_file():
                raise RuntimeError("CTranslate2 conversion output is incomplete")

        snapshots = _hash_tree(stage)
        report = {
            "schema_version": "1.0.0",
            "protocol_id": CONVERSION_PROTOCOL_ID,
            "status": "converted_unvalidated",
            "fact_status": "부분 구현 또는 개발용 데모",
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source_training_report_sha256": _sha256(
                training_dir / "training-report.json"
            ),
            "converter_revision": converter_revision,
            "authorized_speech_revision": validated["report"]["cost_guard"][
                "authorized_speech_revision"
            ],
            "arms": {
                "A_operational_baseline": {
                    "model": operational["id"],
                    "revision": operational["revision"],
                    "artifact_created": False,
                },
                "B_same_conversion_base_control": {
                    "source_model": base["id"],
                    "source_revision": base["revision"],
                    "path": control.name,
                },
                "C_lora_merged_candidate": {
                    "source_model": base["id"],
                    "source_revision": base["revision"],
                    "adapter_report_sha256": _sha256(
                        training_dir / "training-report.json"
                    ),
                    "path": candidate.name,
                },
            },
            "conversion": {
                "ctranslate2": distribution_version("ctranslate2"),
                "transformers": distribution_version("transformers"),
                "peft": distribution_version("peft"),
                "quantization": "float16",
                "copy_files": COPY_FILES,
                "same_converter_options_for_b_and_c": True,
            },
            "output_artifacts": snapshots,
            "automatic_adoption_allowed": False,
            "next_gate": "A/B/C locked validation and downstream safety evaluation",
            "claim_scope": "conversion comparability only; no accuracy, safety, or adoption claim",
        }
        report_path = stage / "conversion-report.json"
        with report_path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        _harden_tree(stage)
        stage.rename(output_dir)
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--converter-revision", required=True)
    args = parser.parse_args(argv)
    report = convert_lora_arms(
        training_dir=args.training_dir,
        execution_config_path=args.execution_config,
        experiment_config_path=args.experiment_config,
        output_dir=args.output_dir,
        converter_revision=args.converter_revision,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "fact_status": report["fact_status"],
                "automatic_adoption_allowed": report["automatic_adoption_allowed"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
