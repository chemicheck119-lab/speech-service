"""Privacy-preserving aggregate failure analysis for STT evaluation records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable

from .evaluation import load_hotwords
from .metrics import normalize_text, score_record


SCHEMA_VERSION = "speech-failure-analysis-v1"
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_INPUT_ROWS = 5_000
RECORD_KEY_PATTERN = re.compile(r"^[0-9a-f]{16}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class FailureAnalysisError(RuntimeError):
    """An analysis error that never includes transcript text."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p50_nearest_rank": _nearest_rank(values, 0.5),
        "p90_nearest_rank": _nearest_rank(values, 0.9),
        "p95_nearest_rank": _nearest_rank(values, 0.95),
        "maximum": max(values) if values else None,
    }


def load_variant_records(
    path: Path, *, variant: str = "baseline"
) -> list[dict[str, Any]]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise FailureAnalysisError(
            f"private record file size is outside the bounded range: {size}"
        )
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number > MAX_INPUT_ROWS:
                raise FailureAnalysisError("private record row count exceeds the bound")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FailureAnalysisError(
                    f"private record JSONL is invalid: line={line_number}"
                ) from exc
            if not isinstance(row, dict) or row.get("variant") != variant:
                continue
            record_key = row.get("record_key")
            if (
                not isinstance(record_key, str)
                or not RECORD_KEY_PATTERN.fullmatch(record_key)
                or record_key in seen_keys
                or not isinstance(row.get("reference"), str)
                or not isinstance(row.get("hypothesis"), str)
                or not isinstance(row.get("segments"), list)
            ):
                raise FailureAnalysisError(
                    f"selected private record contract is invalid: line={line_number}"
                )
            seen_keys.add(record_key)
            rows.append(row)
    if not rows:
        raise FailureAnalysisError("no selected variant records were found")
    return rows


def load_bound_summary(
    path: Path,
    *,
    expected_records: int,
    priority_terms: list[str],
    variant: str = "baseline",
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    variants = summary.get("variants") if isinstance(summary, dict) else None
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != "1.0.0"
        or summary.get("usage_role") != "evaluation"
        or "not field-radio" not in str(summary.get("evidence_scope") or "")
        or not isinstance(variants, dict)
        or not isinstance(variants.get(variant), dict)
        or variants[variant].get("record_count") != expected_records
        or summary.get("priority_terms") != priority_terms
    ):
        raise FailureAnalysisError(
            "summary, private records, and priority terms are not provenance-compatible"
        )
    return summary


def _term_rows(
    rows: list[dict[str, Any]], priority_terms: list[str]
) -> list[dict[str, Any]]:
    result = []
    for term in priority_terms:
        normalized_term = normalize_text(term).replace(" ", "")
        true_positive = false_negative = false_insertion = 0
        for row in rows:
            reference = normalize_text(str(row["reference"])).replace(" ", "")
            hypothesis = normalize_text(str(row["hypothesis"])).replace(" ", "")
            in_reference = normalized_term in reference
            in_hypothesis = normalized_term in hypothesis
            true_positive += int(in_reference and in_hypothesis)
            false_negative += int(in_reference and not in_hypothesis)
            false_insertion += int(not in_reference and in_hypothesis)
        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_insertion
        recall = true_positive / recall_denominator if recall_denominator else None
        precision = (
            true_positive / precision_denominator if precision_denominator else None
        )
        f1 = (
            2 * recall * precision / (recall + precision)
            if recall is not None and precision is not None and recall + precision
            else None
        )
        result.append(
            {
                "term": term,
                "reference_positive_count": recall_denominator,
                "true_positive": true_positive,
                "false_negative": false_negative,
                "false_insertion": false_insertion,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )
    return result


def _segment_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for segment in row["segments"]:
            if isinstance(segment, dict) and isinstance(
                segment.get(field), (int, float)
            ):
                values.append(float(segment[field]))
    return values


def analyze_failures(
    rows: list[dict[str, Any]], priority_terms: list[str]
) -> dict[str, Any]:
    cer_values: list[float] = []
    wer_values: list[float] = []
    rtf_values: list[float] = []
    empty_hypothesis_count = 0
    shortened_count = 0
    expanded_count = 0
    exact_match_count = 0
    failed_record_count = 0
    for row in rows:
        reference = normalize_text(str(row["reference"]))
        hypothesis = normalize_text(str(row["hypothesis"]))
        metric = score_record(reference, hypothesis)
        cer_values.append(metric.cer)
        wer_values.append(metric.word_edits / max(1, metric.reference_words))
        reference_characters = len(reference.replace(" ", ""))
        hypothesis_characters = len(hypothesis.replace(" ", ""))
        length_ratio = hypothesis_characters / max(1, reference_characters)
        empty_hypothesis_count += not hypothesis
        shortened_count += length_ratio < 0.7
        expanded_count += length_ratio > 1.3
        exact_match_count += reference == hypothesis
        failed_record_count += row.get("status") != "completed"
        audio_seconds = row.get("audio_seconds")
        processing_seconds = row.get("processing_seconds")
        if (
            isinstance(audio_seconds, (int, float))
            and audio_seconds > 0
            and isinstance(processing_seconds, (int, float))
        ):
            rtf_values.append(float(processing_seconds) / float(audio_seconds))

    record_count = len(rows)
    term_rows = _term_rows(rows, priority_terms)
    repeat_candidates = [
        {
            "term": row["term"],
            "reference_positive_count": row["reference_positive_count"],
            "false_negative": row["false_negative"],
            "recall": row["recall"],
        }
        for row in term_rows
        if row["reference_positive_count"] >= 5
        and row["recall"] is not None
        and row["recall"] < 0.8
    ]
    return {
        "record_count": record_count,
        "failed_record_count": failed_record_count,
        "exact_normalized_transcript": {
            "count": exact_match_count,
            "rate": exact_match_count / record_count,
        },
        "empty_hypothesis": {
            "count": empty_hypothesis_count,
            "rate": empty_hypothesis_count / record_count,
        },
        "hypothesis_length_below_70_percent": {
            "count": shortened_count,
            "rate": shortened_count / record_count,
            "interpretation": "length-based deletion signal, not aligned deletion count",
        },
        "hypothesis_length_above_130_percent": {
            "count": expanded_count,
            "rate": expanded_count / record_count,
            "interpretation": "length-based insertion signal, not aligned insertion count",
        },
        "record_cer": {
            **_distribution(cer_values),
            "aggregation": (
                "macro distribution over records; not corpus-level character-weighted CER"
            ),
        },
        "record_wer": {
            **_distribution(wer_values),
            "aggregation": (
                "macro distribution over records; not corpus-level word-weighted WER"
            ),
        },
        "record_rtf": _distribution(rtf_values),
        "segment_quality_signals": {
            "avg_log_probability": _distribution(
                _segment_values(rows, "avg_log_probability")
            ),
            "no_speech_probability": _distribution(
                _segment_values(rows, "no_speech_probability")
            ),
            "compression_ratio": _distribution(
                _segment_values(rows, "compression_ratio")
            ),
            "interpretation": (
                "모델 원시 디코딩 신호의 집계이며 정확도 확률이나 안전 확률이 아닙니다."
            ),
        },
        "priority_term_by_term": term_rows,
        "lora_repeat_error_candidates": repeat_candidates,
        "lora_decision": "NOT_DECIDABLE_WITHOUT_CROSS_CONDITION_COMPARISON",
    }


def build_failure_report(
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    summary_sha256: str,
    private_records_sha256: str,
    evaluator_revision: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if not GIT_REVISION_PATTERN.fullmatch(evaluator_revision):
        raise FailureAnalysisError("evaluator revision must be a full 40-character Git SHA")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fact_status": "구현 완료",
        "evaluation_name": "비공개 STT 전사 실패 유형 집계",
        "evidence_scope": summary["evidence_scope"],
        "dataset": summary["dataset"],
        "runtime": summary["runtime"],
        "input_artifacts": {
            "summary_sha256": summary_sha256,
            "private_records_sha256": private_records_sha256,
            "private_records_committed_to_git": False,
        },
        "evaluator": {
            "repository": "chemicheck119-lab/speech-service",
            "git_revision": evaluator_revision,
        },
        "metrics": metrics,
        "privacy": {
            "transcripts_in_report": False,
            "record_keys_in_report": False,
            "public_priority_terms_only": True,
        },
        "claims_not_allowed": [
            "집계 디코딩 신호를 정확도 확률로 해석",
            "길이 축소·팽창 신호를 정렬된 deletion·insertion 개수로 해석",
            "한 지역의 용어 누락만으로 Whisper LoRA 필요성을 확정",
            "실제 현장 무전 실패 유형으로 일반화",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-private", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--priority-terms", type=Path, required=True)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    terms = load_hotwords(args.priority_terms)
    rows = load_variant_records(args.records_private, variant=args.variant)
    summary = load_bound_summary(
        args.summary,
        expected_records=len(rows),
        priority_terms=terms,
        variant=args.variant,
    )
    metrics = analyze_failures(rows, terms)
    report = build_failure_report(
        summary=summary,
        metrics=metrics,
        summary_sha256=sha256_file(args.summary),
        private_records_sha256=sha256_file(args.records_private),
        evaluator_revision=args.evaluator_revision,
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
                "record_count": metrics["record_count"],
                "lora_decision": metrics["lora_decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
