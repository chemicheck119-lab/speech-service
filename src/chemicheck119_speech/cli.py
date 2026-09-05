"""Command-line entry point for bounded AIHub A/B evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .evaluation import evaluate_archives, load_hotwords
from .provenance import validate_evaluation_manifest
from .runtime import FasterWhisperTranscriber
from .storage import materialize, upload_file


def _write_results(
    output_dir: Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    records_path = output_dir / "records.private.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with records_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    records_path.chmod(0o600)
    return summary_path, records_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-archive", required=True)
    parser.add_argument("--label-archive", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--hotwords-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--gcs-output-prefix")
    parser.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "small"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--model-cache", default=os.environ.get("MODEL_CACHE"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")

    terms = load_hotwords(args.hotwords_file)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        audio_archive = materialize(
            args.audio_archive, temporary / "validation-audio.zip"
        )
        label_archive = materialize(
            args.label_archive, temporary / "validation-labels.zip"
        )
        manifest_path = materialize(
            args.dataset_manifest, temporary / "evaluation-manifest.json"
        )
        dataset_provenance = validate_evaluation_manifest(
            manifest_path, audio_archive, label_archive
        )
        transcriber = FasterWhisperTranscriber(
            model=args.model,
            requested_device=args.device,
            device=transcriber.actual_device,
            compute_type=transcriber.actual_compute_type,
            initialization_fallback=transcriber.initialization_fallback,
            dataset_provenance=dataset_provenance,
            cpu_threads=args.cpu_threads,
            download_root=args.model_cache,
            local_files_only=args.local_files_only,
        )

        def progress(completed: int, total: int) -> None:
            print(json.dumps({"completed": completed, "total": total}))

        summary, rows = evaluate_archives(
            audio_archive=audio_archive,
            label_archive=label_archive,
            transcriber=transcriber,
            terms=terms,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            limit=args.limit,
            progress=progress,
        )
    summary_path, records_path = _write_results(args.output_dir, summary, rows)
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
                "record_count": summary["dataset"]["record_count"],
                "summary": str(summary_path),
                "uploaded": uploaded,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
