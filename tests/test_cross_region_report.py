from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.cross_region_report import (
    CrossRegionReportError,
    EXPECTED_EVALUATIONS,
    EXPECTED_RUNTIME,
    build_cross_region_report,
)


class CrossRegionReportTest(unittest.TestCase):
    def _summary(
        self,
        region: str,
        *,
        recall: float = 0.9,
        precision: float = 1.0,
        false_insertion: int = 0,
        terms: list[str] | None = None,
        add_cross_region_hotwords: bool = False,
    ) -> dict:
        evaluation_id, record_count = EXPECTED_EVALUATIONS[region]
        true_positive = 90
        false_negative = round(true_positive * (1 - recall) / recall)
        runtime = dict(EXPECTED_RUNTIME)
        variants = {
            "baseline": {
                "record_count": record_count,
                "failed_record_count": 0,
                "cer": 0.4,
                "wer": 0.6,
                "audio_seconds": record_count * 60,
                "processing_seconds": record_count * 12,
                "real_time_factor": 0.2,
                "priority_term_presence": {
                    "true_positive": true_positive,
                    "false_negative": false_negative,
                    "false_insertion": false_insertion,
                    "recall": recall,
                    "precision": precision,
                    "f1": 2 * precision * recall / (precision + recall),
                },
            }
        }
        if region == "gwangju" or add_cross_region_hotwords:
            variants["hotwords"] = variants["baseline"]
        if region in {"incheon", "seoul"}:
            runtime["variants"] = (
                ["baseline", "hotwords"] if add_cross_region_hotwords else ["baseline"]
            )
        dataset = {
            "evaluation_id": evaluation_id,
            "record_count": record_count,
            "expected_record_count": record_count,
        }
        if region == "gwangju":
            dataset.pop("expected_record_count")
        return {
            "schema_version": "1.0.0",
            "usage_role": "evaluation",
            "evidence_scope": "AIHub call proxy; not field-radio validation",
            "dataset": dataset,
            "runtime": runtime,
            "variants": variants,
            "priority_terms": terms or ["가스", "폭발"],
        }

    def _paths(self, root: Path, summaries: dict[str, dict]) -> dict[str, Path]:
        paths = {}
        for region, summary in summaries.items():
            path = root / f"{region}.json"
            path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            paths[region] = path
        return paths

    def test_builds_fixed_comparison_and_keeps_lora_on_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(
                Path(directory),
                {region: self._summary(region) for region in EXPECTED_EVALUATIONS},
            )
            report = build_cross_region_report(
                gwangju_summary_path=paths["gwangju"],
                incheon_summary_path=paths["incheon"],
                seoul_summary_path=paths["seoul"],
                generated_at="2026-09-05T00:00:00Z",
            )

        self.assertTrue(report["cross_region_gate"]["passed"])
        self.assertEqual(
            "CONDITIONALLY_ACCEPT_BASELINE_FOR_CALL_PROXY_EVALUATION",
            report["cross_region_gate"]["decision"],
        )
        self.assertEqual(
            "HOLD_PENDING_DISTORTION_AND_ERROR_TAXONOMY",
            report["whisper_lora_gate"]["decision"],
        )
        self.assertFalse(report["whisper_lora_gate"]["same_error_type_repeat_proven"])
        self.assertTrue(
            report["schema_compatibility"]["gwangju_missing_expected_record_count"]
        )
        self.assertIn("CAS", report["claims_not_allowed"][1])

    def test_false_insertion_rate_uses_term_negative_opportunities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summaries = {
                region: self._summary(region) for region in EXPECTED_EVALUATIONS
            }
            summaries["incheon"] = self._summary("incheon", false_insertion=3)
            paths = self._paths(Path(directory), summaries)
            report = build_cross_region_report(
                gwangju_summary_path=paths["gwangju"],
                incheon_summary_path=paths["incheon"],
                seoul_summary_path=paths["seoul"],
            )

        terms = report["regions"]["incheon"]["priority_terms"]
        self.assertEqual(3, terms["false_insertion"])
        self.assertEqual(
            3 / terms["negative_opportunities"],
            terms["false_insertion_rate_on_negative_opportunities"],
        )

    def test_rejects_changed_priority_terms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summaries = {
                region: self._summary(region) for region in EXPECTED_EVALUATIONS
            }
            summaries["seoul"] = self._summary("seoul", terms=["다른용어"])
            paths = self._paths(Path(directory), summaries)
            with self.assertRaisesRegex(CrossRegionReportError, "priority terms"):
                build_cross_region_report(
                    gwangju_summary_path=paths["gwangju"],
                    incheon_summary_path=paths["incheon"],
                    seoul_summary_path=paths["seoul"],
                )

    def test_rejects_hotword_cross_region_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summaries = {
                region: self._summary(region) for region in EXPECTED_EVALUATIONS
            }
            summaries["seoul"] = self._summary("seoul", add_cross_region_hotwords=True)
            paths = self._paths(Path(directory), summaries)
            with self.assertRaisesRegex(CrossRegionReportError, "baseline-only"):
                build_cross_region_report(
                    gwangju_summary_path=paths["gwangju"],
                    incheon_summary_path=paths["incheon"],
                    seoul_summary_path=paths["seoul"],
                )


if __name__ == "__main__":
    unittest.main()
