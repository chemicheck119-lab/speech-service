from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.cross_region_report import EXPECTED_EVALUATIONS
from chemicheck119_speech.runtime_provenance import (
    RuntimeProvenanceCaptureError,
    capture_runtime_provenance,
    write_runtime_provenance,
)


class RuntimeProvenanceTest(unittest.TestCase):
    def _summaries(self, root: Path) -> dict[str, Path]:
        summaries: dict[str, Path] = {}
        for region in EXPECTED_EVALUATIONS:
            path = root / f"{region}-summary.json"
            path.write_text(
                json.dumps({"region": region}, ensure_ascii=False), encoding="utf-8"
            )
            summaries[region] = path
        return summaries

    def _snapshot(
        self,
        region: str,
        execution_name: str,
        *,
        completed: bool = True,
        digest: str | None = None,
    ) -> dict:
        jobs = {
            "gwangju": "chemicheck119-speech-eval-cpu",
            "incheon": "chemicheck119-speech-cross-region-cpu",
            "seoul": "chemicheck119-speech-seoul-cpu",
        }
        return {
            "metadata": {
                "name": execution_name,
                "labels": {"run.googleapis.com/job": jobs[region]},
                "annotations": {
                    "run.googleapis.com/creator": "sensitive@example.invalid",
                    "private-note": "절대 결과에 포함하면 안 됨",
                },
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "image": "asia-northeast3-docker.pkg.dev/private/image@"
                                + (digest or "sha256:" + "2" * 64),
                                "env": [{"name": "SECRET", "value": "hidden"}],
                            }
                        ]
                    }
                }
            },
            "status": {
                "startTime": "2026-09-05T00:00:00Z",
                "completionTime": (
                    "2026-09-05T01:00:00Z" if completed else None
                ),
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True" if completed else "Unknown",
                    }
                ],
            },
        }

    def _capture(self, root: Path, *, seoul_completed: bool = True) -> dict:
        execution_names = {
            "gwangju": "chemicheck119-speech-eval-cpu-abcde",
            "incheon": "chemicheck119-speech-cross-region-cpu-abcde",
            "seoul": "chemicheck119-speech-seoul-cpu-abcde",
        }
        snapshots = {
            region: self._snapshot(
                region,
                execution_names[region],
                completed=seoul_completed if region == "seoul" else True,
                digest=("sha256:" + "1" * 64) if region == "gwangju" else None,
            )
            for region in EXPECTED_EVALUATIONS
        }

        def describe(execution_name: str) -> dict:
            region = next(
                item
                for item, expected_name in execution_names.items()
                if expected_name == execution_name
            )
            return snapshots[region]

        return capture_runtime_provenance(
            execution_names=execution_names,
            summary_paths=self._summaries(root),
            describe_execution=describe,
            captured_at="2026-09-05T02:00:00Z",
        )

    def test_capture_keeps_only_allowlisted_non_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._capture(Path(directory))

        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("annotations", serialized)
        self.assertNotIn("sensitive@example.invalid", serialized)
        self.assertNotIn("SECRET", serialized)
        self.assertEqual(
            {
                "execution_name",
                "job_name",
                "container_image_digest",
                "start_time",
                "completion_time",
                "completion_succeeded",
                "summary_sha256",
            },
            set(payload["regions"]["seoul"]),
        )

    def test_capture_rejects_running_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeProvenanceCaptureError, "완료된 고정 execution"
            ):
                self._capture(Path(directory), seoul_completed=False)

    def test_write_uses_owner_only_permissions_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private" / "runtime-provenance.json"
            write_runtime_provenance(output, {"safe": True})
            self.assertEqual(0o600, os.stat(output).st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                write_runtime_provenance(output, {"safe": False})
            self.assertEqual({"safe": True}, json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
