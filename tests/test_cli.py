from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from chemicheck119_speech import cli


class CliTest(unittest.TestCase):
    def test_records_requested_and_actual_runtime_after_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hotwords = root / "hotwords.txt"
            hotwords.write_text("가스\n", encoding="utf-8")
            summary_path = root / "summary.json"
            records_path = root / "records.private.jsonl"
            summary_path.write_text("{}\n", encoding="utf-8")
            records_path.write_text("", encoding="utf-8")

            transcriber = Mock()
            transcriber.actual_device = "cpu"
            transcriber.actual_compute_type = "int8"
            transcriber.initialization_fallback = "CudaUnavailable"
            provenance = {
                "dataset_id": "fixture",
                "dataset_version": "1",
                "evaluation_id": "fixture-evaluation",
                "record_count": 77,
            }
            summary = {"dataset": {"record_count": 77}}

            with (
                patch.object(cli, "materialize", side_effect=lambda _, target: target),
                patch.object(
                    cli, "validate_evaluation_manifest", return_value=provenance
                ),
                patch.object(
                    cli, "FasterWhisperTranscriber", return_value=transcriber
                ) as transcriber_factory,
                patch.object(cli, "evaluate_archives", return_value=(summary, [])) as evaluate,
                patch.object(
                    cli, "_write_results", return_value=(summary_path, records_path)
                ),
            ):
                result = cli.main(
                    [
                        "--audio-archive",
                        "gs://private/audio.zip",
                        "--label-archive",
                        "gs://private/labels.zip",
                        "--dataset-manifest",
                        "gs://private/manifest.json",
                        "--hotwords-file",
                        str(hotwords),
                        "--device",
                        "cuda",
                        "--compute-type",
                        "float16",
                    ]
                )

            self.assertEqual(0, result)
            transcriber_factory.assert_called_once_with(
                model="small",
                device="cuda",
                compute_type="float16",
                cpu_threads=4,
                download_root=None,
                local_files_only=False,
            )
            evaluation_arguments = evaluate.call_args.kwargs
            self.assertEqual("cuda", evaluation_arguments["requested_device"])
            self.assertEqual("cpu", evaluation_arguments["device"])
            self.assertEqual("int8", evaluation_arguments["compute_type"])
            self.assertEqual(
                "CudaUnavailable", evaluation_arguments["initialization_fallback"]
            )
            self.assertEqual(provenance, evaluation_arguments["dataset_provenance"])


if __name__ == "__main__":
    unittest.main()
