from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.radio_sim_provenance import (
    EXPECTED_JOBS,
    EXPECTED_REGIONS,
    RadioSimProvenanceError,
    capture_radio_sim_provenance,
    write_provenance,
)
from chemicheck119_speech.robustness import REGISTERED_VARIANTS


class RadioSimProvenanceTest(unittest.TestCase):
    digest = "sha256:" + "1" * 64

    def _summary(self, region: str) -> dict:
        return {
            "schema_version": "1.0.0",
            "usage_role": "evaluation",
            "evidence_scope": "simulated communication distortion; not field-radio validation",
            "record_count": 40,
            "runtime": {
                "implementation": "faster-whisper",
                "version": "1.2.1",
                "model": "small",
                "device": "cpu",
                "compute_type": "int8",
                "language": "ko (configured, not detected)",
                "beam_size": 5,
                "temperature": 0.0,
                "vad_filter": True,
                "condition_on_previous_text": False,
                "variants": ["baseline"],
            },
            "simulation_run": {
                "profile_id": "radio-sim-v1",
                "variant_count": len(REGISTERED_VARIANTS),
                "run_summary_sha256": ("2" if region == "incheon" else "3") * 64,
                "source_manifest_sha256": ("4" if region == "incheon" else "5") * 64,
                "priority_terms_sha256": "6" * 64,
                "selected": {"total": 40},
            },
            "variants": {
                condition: {"record_count": 40}
                for condition in REGISTERED_VARIANTS
            },
        }

    def _snapshot(
        self, region: str, execution_name: str, *, completed: bool = True
    ) -> dict:
        return {
            "metadata": {
                "name": execution_name,
                "labels": {"run.googleapis.com/job": EXPECTED_JOBS[region]},
                "annotations": {
                    "creator": "sensitive@example.invalid",
                    "private-note": "결과에 포함하면 안 됨",
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
                "completionTime": (
                    "2026-09-06T01:00:00Z" if completed else None
                ),
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True" if completed else "Unknown",
                    }
                ],
            },
        }

    def _capture(
        self,
        root: Path,
        *,
        same_source: bool = False,
        seoul_completed: bool = True,
    ) -> dict:
        summary_paths = {}
        execution_names = {}
        snapshots = {}
        for region in EXPECTED_REGIONS:
            summary = self._summary(region)
            if same_source and region == "seoul":
                summary["simulation_run"]["source_manifest_sha256"] = "4" * 64
            path = root / f"{region}-summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            summary_paths[region] = path
            execution_name = f"{EXPECTED_JOBS[region]}-abcde"
            execution_names[region] = execution_name
            snapshots[execution_name] = self._snapshot(
                region,
                execution_name,
                completed=seoul_completed if region == "seoul" else True,
            )
        return capture_radio_sim_provenance(
            summary_paths=summary_paths,
            execution_names=execution_names,
            describe_execution=lambda name: snapshots[name],
            collector_git_commit="7" * 40,
            captured_at="2026-09-06T02:00:00Z",
        )

    def test_binds_completed_executions_without_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._capture(Path(directory))
        self.assertTrue(payload["comparability_gate"]["passed"])
        self.assertFalse(
            payload["comparability_gate"]["final_lora_decision_made_here"]
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("sensitive@example.invalid", serialized)
        self.assertNotIn("SECRET", serialized)

    def test_records_incomparable_duplicate_source_without_deciding_lora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._capture(Path(directory), same_source=True)
        self.assertFalse(payload["comparability_gate"]["passed"])
        self.assertFalse(
            payload["comparability_gate"]["checks"]["different_source_manifests"]
        )

    def test_rejects_running_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RadioSimProvenanceError, "완료된 seoul"
            ):
                self._capture(Path(directory), seoul_completed=False)

    def test_output_is_owner_only_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "provenance.json"
            write_provenance(path, {"safe": True})
            self.assertEqual(0o600, os.stat(path).st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                write_provenance(path, {"safe": False})


if __name__ == "__main__":
    unittest.main()
