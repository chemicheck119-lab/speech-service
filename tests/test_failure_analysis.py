from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.failure_analysis import (
    FailureAnalysisError,
    analyze_failures,
    build_failure_report,
    load_bound_summary,
    load_variant_records,
)


def _row(
    key: str,
    reference: str,
    hypothesis: str,
    *,
    variant: str = "baseline",
) -> dict:
    return {
        "record_key": key,
        "variant": variant,
        "status": "completed",
        "reference": reference,
        "hypothesis": hypothesis,
        "segments": [
            {
                "avg_log_probability": -0.5,
                "no_speech_probability": 0.1,
                "compression_ratio": 1.2,
            }
        ],
        "audio_seconds": 10.0,
        "processing_seconds": 2.0,
    }


class FailureAnalysisTest(unittest.TestCase):
    def test_aggregates_public_terms_without_transcripts(self) -> None:
        rows = [
            _row("a" * 16, "가스 폭발", "가스 폭발"),
            _row("b" * 16, "가스 누출", "누출"),
            _row("c" * 16, "연기 신고", "가스 연기 신고"),
        ]
        metrics = analyze_failures(rows, ["가스", "폭발"])
        terms = {row["term"]: row for row in metrics["priority_term_by_term"]}

        self.assertEqual(0.5, terms["가스"]["recall"])
        self.assertEqual(1, terms["가스"]["false_insertion"])
        self.assertEqual(0, metrics["empty_hypothesis"]["count"])
        self.assertEqual(0.2, metrics["record_rtf"]["mean"])
        self.assertIn("macro", metrics["record_cer"]["aggregation"])
        self.assertIn("character-weighted", metrics["record_cer"]["aggregation"])
        self.assertIn("word-weighted", metrics["record_wer"]["aggregation"])
        serialized = json.dumps(metrics, ensure_ascii=False)
        self.assertNotIn("가스 누출", serialized)
        self.assertNotIn("연기 신고", serialized)

    def test_loader_filters_variant_and_rejects_duplicate_hashed_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.private.jsonl"
            rows = [
                _row("a" * 16, "가스", "가스"),
                _row("a" * 16, "가스", "가스", variant="hotwords"),
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            selected = load_variant_records(path)
            self.assertEqual(1, len(selected))

            path.write_text(
                "".join(
                    json.dumps(rows[0], ensure_ascii=False) + "\n" for _ in range(2)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FailureAnalysisError, "contract"):
                load_variant_records(path)

    def test_summary_binding_and_report_claim_boundaries(self) -> None:
        terms = ["가스", "폭발"]
        summary = {
            "schema_version": "1.0.0",
            "usage_role": "evaluation",
            "evidence_scope": "AIHub call proxy; not field-radio validation",
            "dataset": {"record_count": 1},
            "runtime": {"model": "small"},
            "priority_terms": terms,
            "variants": {"baseline": {"record_count": 1}},
        }
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            loaded = load_bound_summary(
                summary_path,
                expected_records=1,
                priority_terms=terms,
            )
        report = build_failure_report(
            summary=loaded,
            metrics={"record_count": 1},
            summary_sha256="a" * 64,
            private_records_sha256="b" * 64,
            generated_at="2026-09-05T00:00:00Z",
        )
        self.assertFalse(report["privacy"]["transcripts_in_report"])
        self.assertFalse(report["privacy"]["record_keys_in_report"])
        self.assertIn("LoRA", report["claims_not_allowed"][2])


if __name__ == "__main__":
    unittest.main()
