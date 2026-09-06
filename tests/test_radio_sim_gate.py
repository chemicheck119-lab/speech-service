from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.radio_sim_gate import (
    EXPECTED_JOBS,
    EXPECTED_REGIONS,
    RadioSimGateError,
    _write_exclusive,
    build_radio_sim_gate,
    sha256_file,
)
from chemicheck119_speech.robustness import REGISTERED_VARIANTS


class RadioSimGateTest(unittest.TestCase):
    digest = "sha256:" + "1" * 64
    evaluator_commit = "2" * 40
    model_commit = "3" * 40
    manifest_digest = "4" * 64
    priority_digest = "5" * 64

    def _runtime(self) -> dict:
        return {
            "implementation": "faster-whisper",
            "version": "1.2.1",
            "model": "small",
            "requested_device": "cpu",
            "device": "cpu",
            "compute_type": "int8",
            "initialization_fallback": None,
            "language": "ko (configured, not detected)",
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "variants": ["baseline"],
        }

    def _summary(self, region: str, *, signal: bool = True) -> dict:
        variants = {}
        for condition in REGISTERED_VARIANTS:
            is_signal = signal and condition == "wind_snr0"
            variants[condition] = {
                "record_count": 40,
                "failed_record_count": 0,
                "cer": 0.5 if is_signal else 0.3,
                "wer": 0.6 if is_signal else 0.4,
                "real_time_factor": 0.2,
                "priority_term_presence": {
                    "true_positive": 12 if is_signal else 18,
                    "false_negative": 8 if is_signal else 2,
                    "false_insertion": 0,
                    "recall": 0.6 if is_signal else 0.9,
                    "precision": 1.0,
                    "f1": 0.75 if is_signal else 0.947,
                },
            }
        return {
            "schema_version": "1.0.0",
            "usage_role": "evaluation",
            "evidence_scope": "simulated communication distortion; not field-radio validation",
            "record_count": 40,
            "runtime": self._runtime(),
            "simulation_run": {
                "profile_id": "radio-sim-v1",
                "run_summary_sha256": ("6" if region == "incheon" else "7") * 64,
                "source_manifest_sha256": ("8" if region == "incheon" else "9") * 64,
                "priority_terms_sha256": self.priority_digest,
                "seed": 119,
                "selected": {
                    "priority_term_positive": 20,
                    "priority_term_negative": 20,
                    "total": 40,
                },
                "variant_count": len(REGISTERED_VARIANTS),
            },
            "variants": variants,
        }

    def _downstream(
        self,
        summary: dict,
        summary_sha256: str,
        *,
        signal: bool = True,
        safety_violation: int = 0,
    ) -> dict:
        by_condition = {}
        for condition in REGISTERED_VARIANTS:
            is_signal = signal and condition == "wind_snr0"
            by_condition[condition] = {
                "priority_term_by_term": [
                    {
                        "term": "연기",
                        "reference_positive_count": 10,
                        "true_positive": 6 if is_signal else 9,
                        "false_negative": 4 if is_signal else 1,
                        "false_insertion": 0,
                        "recall": 0.6 if is_signal else 0.9,
                        "precision": 1.0,
                        "f1": 0.75 if is_signal else 0.947,
                    }
                ]
            }
        safety = {
            "candidate_promotion_violation_count": safety_violation,
            "rule_execution_before_confirmation_count": 0,
            "two_cas_gate_violation_count": 0,
            "unconfirmed_risk_output_violation_count": 0,
        }
        gate_passed = safety_violation == 0
        return {
            "schema_version": "stt-radio-sim-downstream-silver-eval-v1",
            "fact_status": "부분 구현 또는 개발용 데모",
            "evidence_scope": "모의 통신 왜곡 평가; 현장 무전 성능 검증 아님",
            "dataset": {
                "profile_id": "radio-sim-v1",
                "source_manifest_sha256": summary["simulation_run"][
                    "source_manifest_sha256"
                ],
                "record_count_per_condition": 40,
                "condition_count": len(REGISTERED_VARIANTS),
                "derived_data": True,
            },
            "input_artifacts": {
                "speech_summary_sha256": summary_sha256,
                "private_records_sha256": "a" * 64,
                "priority_terms_sha256": self.priority_digest,
            },
            "evaluation_runtime": {
                "repository": "chemicheck119-lab/analysis-engine",
                "git_commit": self.evaluator_commit,
            },
            "stt_runtime": self._runtime(),
            "speech_evaluator_artifact": {
                "repository": "chemicheck119-lab/speech-service",
                "container_image_digest": self.digest,
            },
            "model_api_runtime": {
                "service_revision": "preview-revision",
                "service_git_commit": self.model_commit,
                "runtime_manifest_sha256": self.manifest_digest,
                "api_schema": "chemiguard119-api-v1",
            },
            "metrics": {
                "profile_id": "radio-sim-v1",
                "condition_count": len(REGISTERED_VARIANTS),
                "record_count_per_condition": 40,
                "by_condition": by_condition,
                "safety_violation_totals": safety,
                "evaluation_integrity_gate": {"passed": gate_passed},
                "analysis_coverage_gate": {"passed": gate_passed},
                "safety_contract_gate": {"passed": gate_passed},
                "downstream_evaluation_gate": {"passed": gate_passed},
                "cas_ground_truth_available": False,
                "is_cas_accuracy_evaluation": False,
                "wrong_single_cas_promotion_ground_truth_count": None,
            },
        }

    def _snapshot(self, region: str, execution_name: str) -> dict:
        return {
            "metadata": {
                "name": execution_name,
                "labels": {"run.googleapis.com/job": EXPECTED_JOBS[region]},
                "annotations": {
                    "private-note": "결과에 포함하면 안 됨",
                    "creator": "sensitive@example.invalid",
                },
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "image": "registry.invalid/private/image@" + self.digest,
                                "env": [{"name": "SECRET", "value": "hidden"}],
                            }
                        ]
                    }
                }
            },
            "status": {
                "startTime": "2026-09-06T00:00:00Z",
                "completionTime": "2026-09-06T01:00:00Z",
                "conditions": [{"type": "Completed", "status": "True"}],
            },
        }

    def _build(
        self,
        root: Path,
        *,
        seoul_signal: bool = True,
        seoul_source_same: bool = False,
        seoul_safety_violation: int = 0,
    ) -> dict:
        summary_paths = {}
        downstream_paths = {}
        execution_names = {}
        snapshots = {}
        for region in EXPECTED_REGIONS:
            signal = region == "incheon" or seoul_signal
            summary = self._summary(region, signal=signal)
            if region == "seoul" and seoul_source_same:
                summary["simulation_run"]["source_manifest_sha256"] = "8" * 64
            summary_path = root / f"{region}-summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            summary_paths[region] = summary_path
            report = self._downstream(
                summary,
                sha256_file(summary_path),
                signal=signal,
                safety_violation=(
                    seoul_safety_violation if region == "seoul" else 0
                ),
            )
            downstream_path = root / f"{region}-downstream.json"
            downstream_path.write_text(json.dumps(report), encoding="utf-8")
            downstream_paths[region] = downstream_path
            execution_name = f"{EXPECTED_JOBS[region]}-abcde"
            execution_names[region] = execution_name
            snapshots[execution_name] = self._snapshot(region, execution_name)

        return build_radio_sim_gate(
            summary_paths=summary_paths,
            downstream_paths=downstream_paths,
            execution_names=execution_names,
            describe_execution=lambda name: snapshots[name],
            evaluator_git_commit="b" * 40,
            generated_at="2026-09-06T02:00:00Z",
        )

    def test_allows_bounded_lora_only_for_repeated_specific_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._build(Path(directory))

        self.assertTrue(report["comparability_gate"]["passed"])
        self.assertTrue(report["downstream_gate"]["passed"])
        self.assertEqual(1, len(report["repeated_specific_signals"]))
        self.assertEqual(
            "PROCEED_TO_BOUNDED_LORA_EXPERIMENT",
            report["whisper_lora_gate"]["decision"],
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("sensitive@example.invalid", serialized)
        self.assertNotIn("SECRET", serialized)

    def test_rejects_when_signal_is_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._build(Path(directory), seoul_signal=False)
        self.assertEqual(
            "DO_NOT_RUN_LORA_NO_REPEATED_SPECIFIC_ERROR",
            report["whisper_lora_gate"]["decision"],
        )

    def test_rejects_incomparable_source_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._build(Path(directory), seoul_source_same=True)
        self.assertFalse(report["comparability_gate"]["passed"])
        self.assertEqual(
            "DO_NOT_RUN_LORA_INCOMPARABLE_EVALUATIONS",
            report["whisper_lora_gate"]["decision"],
        )

    def test_rejects_failed_safety_gate_even_with_repeat_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._build(Path(directory), seoul_safety_violation=1)
        self.assertFalse(report["downstream_gate"]["passed"])
        self.assertEqual(
            "DO_NOT_RUN_LORA_SAFETY_OR_INTEGRITY_GATE_FAILED",
            report["whisper_lora_gate"]["decision"],
        )

    def test_rejects_summary_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = {
                region: self._summary(region) for region in EXPECTED_REGIONS
            }
            summary_paths = {}
            downstream_paths = {}
            execution_names = {}
            snapshots = {}
            for region, summary in summaries.items():
                summary_path = root / f"{region}.json"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                summary_paths[region] = summary_path
                downstream_path = root / f"{region}-downstream.json"
                downstream_path.write_text(
                    json.dumps(self._downstream(summary, "0" * 64)),
                    encoding="utf-8",
                )
                downstream_paths[region] = downstream_path
                execution_name = f"{EXPECTED_JOBS[region]}-abcde"
                execution_names[region] = execution_name
                snapshots[execution_name] = self._snapshot(region, execution_name)
            with self.assertRaisesRegex(RadioSimGateError, "결합되지"):
                build_radio_sim_gate(
                    summary_paths=summary_paths,
                    downstream_paths=downstream_paths,
                    execution_names=execution_names,
                    describe_execution=lambda name: snapshots[name],
                    evaluator_git_commit="b" * 40,
                )

    def test_private_output_is_exclusive_and_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "gate.json"
            _write_exclusive(path, {"safe": True})
            self.assertEqual(0o600, os.stat(path).st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                _write_exclusive(path, {"safe": False})


if __name__ == "__main__":
    unittest.main()
