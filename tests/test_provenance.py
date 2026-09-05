import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.evaluation import EVALUATION_ID
from chemicheck119_speech.provenance import validate_evaluation_manifest


class ProvenanceTest(unittest.TestCase):
    def test_binds_archives_to_versioned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.zip"
            labels = root / "labels.zip"
            audio.write_bytes(b"audio")
            labels.write_bytes(b"labels")
            manifest = {
                "dataset_id": "aihub_71768_gwangju_fire",
                "dataset_version": "version-1",
                "usage_role": "evaluation",
                "evaluation": {"id": EVALUATION_ID, "record_count": 77},
                "artifacts": [
                    {
                        "path": "gs://private/audio.zip",
                        "sha256": hashlib.sha256(b"audio").hexdigest(),
                    },
                    {
                        "path": "gs://private/labels.zip",
                        "sha256": hashlib.sha256(b"labels").hexdigest(),
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_evaluation_manifest(manifest_path, audio, labels)
            self.assertEqual("version-1", result["dataset_version"])
            self.assertEqual(64, len(result["manifest_sha256"]))

    def test_rejects_changed_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.zip"
            labels = root / "labels.zip"
            audio.write_bytes(b"changed")
            labels.write_bytes(b"labels")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "aihub_71768_gwangju_fire",
                        "dataset_version": "version-1",
                        "usage_role": "evaluation",
                        "evaluation": {"id": EVALUATION_ID, "record_count": 77},
                        "artifacts": [
                            {"path": "audio.zip", "sha256": "0" * 64},
                            {
                                "path": "labels.zip",
                                "sha256": hashlib.sha256(b"labels").hexdigest(),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_evaluation_manifest(manifest_path, audio, labels)


if __name__ == "__main__":
    unittest.main()
