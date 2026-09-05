"""Paired STT evaluation over a provenance-bound radio simulation run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import wave
import zipfile

from .evaluation import (
    MAX_AUDIO_MEMBER_BYTES,
    MAX_AUDIO_SECONDS,
    MAX_COMPRESSION_RATIO,
    _members,
    _validate_member_size,
    evaluate_archives,
    load_hotwords,
)
from .metrics import paired_bootstrap_cer_delta, score_record
from .provenance import sha256_file, validate_evaluation_manifest
from .runtime import FasterWhisperTranscriber
from .storage import materialize, upload_file


PROFILE_ID = "radio-sim-v1"
MAX_RECORDS_PER_VARIANT = 200
VARIANT_PATTERN = re.compile(r"^[a-z0-9_]+$")
REGISTERED_VARIANTS = frozenset(
    {
        "clean",
        "bandlimit_8khz",
        "mulaw_8khz",
        "siren_snr20",
        "siren_snr10",
        "siren_snr0",
        "vehicle_snr20",
        "vehicle_snr10",
        "vehicle_snr0",
        "wind_snr20",
        "wind_snr10",
        "wind_snr0",
        "start_cut_300ms",
        "end_cut_300ms",
        "hard_clip_minus12dbfs",
        "gain_minus18db",
        "dropout_3x120ms",
        "combined_radio_snr10",
    }
)
MAX_TOTAL_AUDIO_HOURS = 24.0
MAX_RUN_SUMMARY_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_AUDIO_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_LABEL_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024


def validate_local_file_size(path: Path, maximum_bytes: int, label: str) -> int:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{label} file size is outside the bounded range: {size}")
    return size


def _resolve(root: str, relative: str) -> str:
    if relative.startswith("gs://") or Path(relative).is_absolute():
        return relative
    pure = PurePosixPath(relative)
    if ".." in pure.parts:
        raise ValueError("simulation run contains a parent-path reference")
    if root.startswith("gs://"):
        return root.rstrip("/") + "/" + str(pure)
    return str(Path(root) / Path(*pure.parts))


def load_simulation_run_summary(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("profile_id") != PROFILE_ID:
        raise ValueError("unsupported or missing simulation profile")
    items = summary.get("manifests")
    if not isinstance(items, list) or len(items) != len(REGISTERED_VARIANTS):
        raise ValueError("simulation manifest count is outside the registered profile")
    if summary.get("variant_count") != len(items):
        raise ValueError("simulation variant count does not match manifest inventory")
    for field in (
        "source_manifest_sha256",
        "source_audio_sha256",
        "source_labels_sha256",
        "priority_terms_sha256",
    ):
        digest = summary.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"simulation run has an invalid {field}")
    if type(summary.get("seed")) is not int:
        raise ValueError("simulation run seed must be an integer")
    variants: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("simulation manifest item must be an object")
        variant = item.get("variant")
        relative = item.get("manifest")
        digest = item.get("manifest_sha256")
        if (
            not isinstance(variant, str)
            or not VARIANT_PATTERN.fullmatch(variant)
            or not isinstance(relative, str)
            or not relative.strip()
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("invalid simulation manifest descriptor")
        if variant in variants:
            raise ValueError(f"duplicate simulation variant: {variant}")
        variants.add(variant)
    if variants != REGISTERED_VARIANTS:
        missing = sorted(REGISTERED_VARIANTS - variants)
        unexpected = sorted(variants - REGISTERED_VARIANTS)
        raise ValueError(
            "simulation run does not match the registered variant set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    selected = summary.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("simulation run is missing selection metadata")
    selected_total = selected.get("total")
    if (
        type(selected_total) is not int
        or selected_total <= 0
        or selected_total > MAX_RECORDS_PER_VARIANT
    ):
        raise ValueError("selected record count is outside the bounded range")
    return summary


def _artifact_uri(
    manifest: dict[str, object], *, suffix: str, contains: str, root: str
) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("simulation manifest artifacts must be an array")
    matches = [
        item.get("path")
        for item in artifacts
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and str(item["path"]).endswith(suffix)
        and contains in str(item["path"])
    ]
    if len(matches) != 1:
        raise ValueError(f"simulation manifest must bind one {contains} artifact")
    return _resolve(root, str(matches[0]))


def validate_simulation_manifest(
    path: Path,
    *,
    run_summary: dict[str, object],
    variant: str,
    expected_sha256: str,
) -> tuple[dict[str, object], str, str]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"simulation manifest digest mismatch: {variant}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("simulation manifest root must be an object")
    preprocessing = manifest.get("preprocessing")
    parameters = (
        preprocessing.get("parameters") if isinstance(preprocessing, dict) else None
    )
    variant_spec = parameters.get("variant") if isinstance(parameters, dict) else None
    if (
        manifest.get("classification") != "derived"
        or manifest.get("usage_role") != "evaluation"
        or not isinstance(parameters, dict)
        or parameters.get("profile_id") != PROFILE_ID
        or not isinstance(variant_spec, dict)
        or variant_spec.get("id") != variant
    ):
        raise ValueError(f"manifest is not the declared radio simulation: {variant}")
    for field in (
        "source_manifest_sha256",
        "source_audio_sha256",
        "source_labels_sha256",
        "priority_terms_sha256",
    ):
        if parameters.get(field) != run_summary.get(field):
            raise ValueError(f"simulation provenance mismatch for {field}: {variant}")
    evidence_scope = manifest.get("evidence_scope")
    if not isinstance(evidence_scope, str) or "not field-radio" not in evidence_scope:
        raise ValueError("simulation manifest must preserve the field-radio limitation")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("simulation manifest is missing evaluation metadata")
    if evaluation.get("record_count") != run_summary["selected"]["total"]:
        raise ValueError("simulation record count does not match the selected sample")
    audio_uri = _artifact_uri(
        manifest, suffix=f"/{variant}.zip", contains="/audio/", root=""
    )
    labels_uri = _artifact_uri(
        manifest,
        suffix="/sampled-labels.zip",
        contains="/labels/",
        root="",
    )
    return manifest, audio_uri, labels_uri


def archive_audio_seconds(path: Path) -> tuple[int, float]:
    total_seconds = 0.0
    with zipfile.ZipFile(path) as archive:
        members = _members(archive, ".wav")
        if not members or len(members) > MAX_RECORDS_PER_VARIANT:
            raise ValueError("audio archive record count is outside the bounded range")
        for name in members.values():
            _validate_member_size(archive, name, MAX_AUDIO_MEMBER_BYTES)
            info = archive.getinfo(name)
            if info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
                raise ValueError(f"unsafe audio archive compression ratio: {name}")
            with archive.open(name) as source:
                content = source.read(MAX_AUDIO_MEMBER_BYTES + 1)
            if len(content) > MAX_AUDIO_MEMBER_BYTES:
                raise ValueError(f"audio member exceeded bounded read: {name}")
            try:
                with wave.open(io.BytesIO(content)) as audio:
                    sample_rate = audio.getframerate()
                    seconds = audio.getnframes() / sample_rate if sample_rate else 0.0
            except (EOFError, wave.Error) as error:
                raise ValueError(f"invalid WAV member: {name}") from error
            if seconds <= 0 or seconds > MAX_AUDIO_SECONDS:
                raise ValueError(f"unsafe audio duration: {name}")
            total_seconds += seconds
    return len(members), total_seconds


def _delta(value: object, baseline: object) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def aggregate_robustness(
    variant_summaries: dict[str, dict[str, object]],
    variant_rows: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    if "clean" not in variant_summaries or "clean" not in variant_rows:
        raise ValueError("clean control is required for paired aggregation")
    clean_rows = sorted(variant_rows["clean"], key=lambda row: str(row["record_key"]))
    clean_keys = [str(row["record_key"]) for row in clean_rows]
    clean_references = [str(row["reference"]) for row in clean_rows]
    clean_metrics = [
        score_record(str(row["reference"]), str(row["hypothesis"]))
        for row in clean_rows
    ]
    clean_summary = variant_summaries["clean"]
    clean_terms = clean_summary["priority_term_presence"]
    paired: dict[str, object] = {}
    for variant, summary in sorted(variant_summaries.items()):
        rows = sorted(variant_rows[variant], key=lambda row: str(row["record_key"]))
        if [str(row["record_key"]) for row in rows] != clean_keys:
            raise ValueError(f"paired record keys differ from clean: {variant}")
        if [str(row["reference"]) for row in rows] != clean_references:
            raise ValueError(f"paired references differ from clean: {variant}")
        metrics = [
            score_record(str(row["reference"]), str(row["hypothesis"]))
            for row in rows
        ]
        cer_bootstrap = paired_bootstrap_cer_delta(clean_metrics, metrics)
        cer_bootstrap["metric"] = f"{variant}_cer_minus_clean_cer"
        terms = summary["priority_term_presence"]
        paired[variant] = {
            "cer_delta": float(summary["cer"]) - float(clean_summary["cer"]),
            "wer_delta": float(summary["wer"]) - float(clean_summary["wer"]),
            "priority_term_recall_delta": _delta(
                terms.get("recall"), clean_terms.get("recall")
            ),
            "priority_term_precision_delta": _delta(
                terms.get("precision"), clean_terms.get("precision")
            ),
            "priority_term_f1_delta": _delta(terms.get("f1"), clean_terms.get("f1")),
            "false_insertion_delta": int(terms["false_insertion"])
            - int(clean_terms["false_insertion"]),
            "failed_record_delta": int(summary["failed_record_count"])
            - int(clean_summary["failed_record_count"]),
            "paired_bootstrap": cer_bootstrap,
        }
    key_digest = hashlib.sha256("\n".join(clean_keys).encode("ascii")).hexdigest()
    return {
        "record_count": len(clean_rows),
        "record_key_set_sha256": key_digest,
        "variants": variant_summaries,
        "paired_vs_clean": paired,
    }


def _write_results(
    output_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]
) -> tuple[Path, Path]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    records_path = output_dir / "records.private.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with records_path.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    records_path.chmod(0o600)
    return summary_path, records_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--simulation-root", required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gcs-output-prefix")
    parser.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "small"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--model-cache", default=os.environ.get("MODEL_CACHE"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-total-audio-hours", type=float, default=MAX_TOTAL_AUDIO_HOURS
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if (
        args.max_total_audio_hours <= 0
        or args.max_total_audio_hours > MAX_TOTAL_AUDIO_HOURS
    ):
        parser.error("--max-total-audio-hours must be in (0, 24]")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite output directory: {args.output_dir}"
        )
    terms = load_hotwords(args.priority_terms)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        run_summary_path = materialize(
            args.run_summary, temporary / "run-summary.json"
        )
        validate_local_file_size(
            run_summary_path, MAX_RUN_SUMMARY_BYTES, "simulation run summary"
        )
        run_summary = load_simulation_run_summary(run_summary_path)
        run_summary_sha256 = sha256_file(run_summary_path)
        if sha256_file(args.priority_terms) != run_summary.get("priority_terms_sha256"):
            raise ValueError("priority term file does not match the simulation run")

        prepared: list[dict[str, object]] = []
        label_cache: dict[str, Path] = {}
        total_audio_seconds = 0.0
        total_archive_bytes = 0
        ordered_items = sorted(
            run_summary["manifests"], key=lambda item: (item["variant"] != "clean", item["variant"])
        )
        for item in ordered_items:
            variant = str(item["variant"])
            manifest_source = _resolve(args.simulation_root, str(item["manifest"]))
            manifest_path = materialize(
                manifest_source, temporary / "manifests" / f"{variant}.json"
            )
            validate_local_file_size(
                manifest_path, MAX_MANIFEST_BYTES, f"{variant} manifest"
            )
            _, audio_source, labels_source = validate_simulation_manifest(
                manifest_path,
                run_summary=run_summary,
                variant=variant,
                expected_sha256=str(item["manifest_sha256"]),
            )
            audio_source = _resolve(args.simulation_root, audio_source)
            labels_source = _resolve(args.simulation_root, labels_source)
            audio_path = materialize(
                audio_source, temporary / "audio" / f"{variant}.zip"
            )
            total_archive_bytes += validate_local_file_size(
                audio_path, MAX_AUDIO_ARCHIVE_BYTES, f"{variant} audio archive"
            )
            if labels_source not in label_cache:
                label_cache[labels_source] = materialize(
                    labels_source,
                    temporary / "labels" / f"labels-{len(label_cache):02d}.zip",
                )
                total_archive_bytes += validate_local_file_size(
                    label_cache[labels_source],
                    MAX_LABEL_ARCHIVE_BYTES,
                    "sampled label archive",
                )
            if total_archive_bytes > MAX_TOTAL_ARCHIVE_BYTES:
                raise ValueError(
                    "simulation archives exceed the 4GiB local materialization bound"
                )
            labels_path = label_cache[labels_source]
            provenance = validate_evaluation_manifest(
                manifest_path, audio_path, labels_path
            )
            record_count, audio_seconds = archive_audio_seconds(audio_path)
            if record_count != provenance["record_count"]:
                raise ValueError(f"audio inventory mismatch: {variant}")
            total_audio_seconds += audio_seconds
            prepared.append(
                {
                    "variant": variant,
                    "audio": audio_path,
                    "labels": labels_path,
                    "provenance": provenance,
                    "manifest_sha256": item["manifest_sha256"],
                    "audio_seconds": audio_seconds,
                }
            )
        total_audio_hours = total_audio_seconds / 3600.0
        if total_audio_hours > args.max_total_audio_hours:
            raise ValueError(
                "simulation audio exceeds the predeclared compute bound: "
                f"{total_audio_hours:.3f}h > {args.max_total_audio_hours:.3f}h"
            )

        transcriber = FasterWhisperTranscriber(
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            download_root=args.model_cache,
            local_files_only=args.local_files_only,
        )
        variant_summaries: dict[str, dict[str, object]] = {}
        variant_rows: dict[str, list[dict[str, object]]] = {}
        private_rows: list[dict[str, object]] = []
        runtime: dict[str, object] | None = None
        for prepared_item in prepared:
            variant = str(prepared_item["variant"])

            def progress(completed: int, total: int, name: str = variant) -> None:
                print(json.dumps({"variant": name, "completed": completed, "total": total}))

            evaluation_summary, rows = evaluate_archives(
                audio_archive=prepared_item["audio"],
                label_archive=prepared_item["labels"],
                transcriber=transcriber,
                terms=terms,
                model=args.model,
                requested_device=args.device,
                device=transcriber.actual_device,
                compute_type=transcriber.actual_compute_type,
                initialization_fallback=transcriber.initialization_fallback,
                dataset_provenance=prepared_item["provenance"],
                limit=args.limit,
                progress=progress,
                generated_at=generated_at,
                variants=("baseline",),
            )
            aggregate = evaluation_summary["variants"]["baseline"]
            variant_summaries[variant] = aggregate
            variant_rows[variant] = rows
            runtime = runtime or evaluation_summary["runtime"]
            for row in rows:
                private_rows.append(
                    {**row, "inference_variant": row["variant"], "channel_variant": variant}
                )

    aggregated = aggregate_robustness(variant_summaries, variant_rows)
    usage_role = "development" if args.limit is not None else "evaluation"
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "experiment_id": (
            f"speech_{run_summary['profile_id']}_robustness_"
            f"{run_summary['source_manifest_sha256'][:12]}"
        ),
        "usage_role": usage_role,
        "generated_at": generated_at,
        "evidence_scope": "simulated communication distortion on AIHub calls; not field-radio validation",
        "simulation_run": {
            "profile_id": run_summary["profile_id"],
            "run_summary_sha256": run_summary_sha256,
            "source_manifest_sha256": run_summary["source_manifest_sha256"],
            "priority_terms_sha256": run_summary["priority_terms_sha256"],
            "seed": run_summary["seed"],
            "selected": run_summary["selected"],
            "variant_count": run_summary["variant_count"],
            "total_audio_hours": total_audio_hours,
            "max_total_audio_hours": args.max_total_audio_hours,
            "materialized_archive_bytes": total_archive_bytes,
            "max_materialized_archive_bytes": MAX_TOTAL_ARCHIVE_BYTES,
        },
        "runtime": runtime,
        **aggregated,
        "decision_note": (
            "STT robustness alone cannot authorize a CAS or Rule Engine execution; "
            "downstream Parser/Resolver and 2-CAS gate evaluation remains separate"
        ),
    }
    summary_path, records_path = _write_results(args.output_dir, summary, private_rows)
    uploaded: list[str] = []
    if args.gcs_output_prefix:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = args.gcs_output_prefix.rstrip("/") + "/" + timestamp
        for source in (summary_path, records_path):
            destination = f"{prefix}/{source.name}"
            upload_file(source, destination)
            uploaded.append(destination)
    print(
        json.dumps(
            {
                "status": "completed",
                "usage_role": usage_role,
                "record_count": aggregated["record_count"],
                "variant_count": run_summary["variant_count"],
                "total_audio_hours": total_audio_hours,
                "summary": str(summary_path),
                "uploaded": uploaded,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
