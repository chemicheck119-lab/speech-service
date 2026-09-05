"""Paired A/B evaluation over an AIHub WAV/JSON archive pair."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Callable
import zipfile

from .metrics import (
    RecordMetric,
    paired_bootstrap_cer_delta,
    score_record,
    term_presence_counts,
)
from .runtime import Transcript, Transcriber


EVALUATION_ID = "speech_aihub119_gwangju_fire_validation_77"
EXPECTED_EVALUATION_RECORDS = 77
VARIANTS = ("baseline", "hotwords")


def load_hotwords(path: Path) -> list[str]:
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    if not terms or len(terms) != len(set(terms)):
        raise ValueError("hotword file must contain unique non-empty terms")
    return terms


def _members(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in archive.namelist():
        if not name.lower().endswith(suffix):
            continue
        stem = PurePosixPath(name).stem
        if stem in result:
            raise ValueError(f"duplicate archive stem: {stem}")
        result[stem] = name
    return result


def _reference(label_bytes: bytes) -> tuple[str, str]:
    label = json.loads(label_bytes.decode("utf-8-sig"))
    record_id = label.get("recordId")
    utterances = label.get("utterances")
    if not isinstance(record_id, str) or not isinstance(utterances, list):
        raise ValueError("invalid AIHub label")
    ordered = sorted(utterances, key=lambda item: (item["startAt"], item["endAt"]))
    reference = " ".join(item["text"].strip() for item in ordered).strip()
    if not reference:
        raise ValueError("empty AIHub reference")
    private_key = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]
    return private_key, reference


def _segment_payload(transcript: Transcript) -> list[dict[str, float | str]]:
    return [asdict(segment) for segment in transcript.segments]


def _aggregate(
    rows: list[dict[str, object]], terms: list[str]
) -> tuple[dict[str, object], list[RecordMetric]]:
    metrics = [score_record(str(row["reference"]), str(row["hypothesis"])) for row in rows]
    character_edits = sum(metric.character_edits for metric in metrics)
    reference_characters = sum(metric.reference_characters for metric in metrics)
    word_edits = sum(metric.word_edits for metric in metrics)
    reference_words = sum(metric.reference_words for metric in metrics)
    audio_seconds = sum(float(row["audio_seconds"]) for row in rows)
    processing_seconds = sum(float(row["processing_seconds"]) for row in rows)
    result: dict[str, object] = {
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
        "priority_term_presence": term_presence_counts(
            [str(row["reference"]) for row in rows],
            [str(row["hypothesis"]) for row in rows],
            terms,
        ),
    }
    return result, metrics


def evaluate_archives(
    *,
    audio_archive: Path,
    label_archive: Path,
    transcriber: Transcriber,
    terms: list[str],
    model: str,
    device: str,
    compute_type: str,
    limit: int | None = None,
    expected_records: int | None = EXPECTED_EVALUATION_RECORDS,
    progress: Callable[[int, int], None] | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    with zipfile.ZipFile(audio_archive) as audio_zip, zipfile.ZipFile(
        label_archive
    ) as label_zip, tempfile.TemporaryDirectory() as directory:
        audio_members = _members(audio_zip, ".wav")
        label_members = _members(label_zip, ".json")
        stems = sorted(audio_members.keys() & label_members.keys())
        if set(audio_members) != set(label_members):
            raise ValueError("audio/label archive mismatch")
        available_record_count = len(stems)
        if limit is None and expected_records is not None:
            if available_record_count != expected_records:
                raise ValueError(
                    "fixed evaluation record count mismatch: "
                    f"expected={expected_records}, actual={available_record_count}"
                )
        if limit is not None:
            stems = stems[:limit]
        if not stems:
            raise ValueError("empty archive pair")

        rows: list[dict[str, object]] = []
        hotwords = " ".join(terms)
        for index, stem in enumerate(stems):
            record_key, reference = _reference(label_zip.read(label_members[stem]))
            audio_path = Path(directory) / f"record-{index:04d}.wav"
            audio_path.write_bytes(audio_zip.read(audio_members[stem]))
            order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
            for variant in order:
                started = time.perf_counter()
                try:
                    transcript = transcriber.transcribe(
                        audio_path, hotwords if variant == "hotwords" else None
                    )
                    status = "completed"
                    error_type = None
                except Exception as error:  # keep failed records in the denominator
                    transcript = Transcript("", (), 0.0, 0.0)
                    status = "failed"
                    error_type = type(error).__name__
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "record_key": record_key,
                        "variant": variant,
                        "status": status,
                        "error_type": error_type,
                        "reference": reference,
                        "hypothesis": transcript.text,
                        "segments": _segment_payload(transcript),
                        "audio_seconds": transcript.audio_seconds,
                        "voiced_seconds": transcript.voiced_seconds,
                        "processing_seconds": elapsed,
                    }
                )
            audio_path.unlink()
            if progress:
                progress(index + 1, len(stems))

    by_variant = {
        variant: sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: str(row["record_key"]),
        )
        for variant in VARIANTS
    }
    aggregates: dict[str, dict[str, object]] = {}
    record_metrics: dict[str, list[RecordMetric]] = {}
    for variant, variant_rows in by_variant.items():
        aggregates[variant], record_metrics[variant] = _aggregate(variant_rows, terms)
    is_fixed_evaluation = limit is None and (
        expected_records is None or len(stems) == expected_records
    )
    experiment_id = (
        EVALUATION_ID
        if is_fixed_evaluation
        else f"speech_aihub119_gwangju_fire_smoke_{len(stems)}"
    )
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "usage_role": "evaluation" if is_fixed_evaluation else "development",
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_scope": "AIHub 119 emergency-call proxy; not field-radio validation",
        "dataset": {
            "id": "aihub_71768_gwangju_fire",
            "split": "Validation",
            "record_count": len(stems),
        },
        "runtime": {
            "implementation": "faster-whisper",
            "version": "1.2.1",
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "language": "ko (configured, not detected)",
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": True,
            "condition_on_previous_text": False,
        },
        "variants": aggregates,
        "paired_comparison": paired_bootstrap_cer_delta(
            record_metrics["baseline"], record_metrics["hotwords"]
        ),
        "priority_terms": terms,
        "quality_signal_note": (
            "segment avg_log_probability is an uncalibrated decoding score, "
            "not a correctness probability"
        ),
    }
    return summary, rows
