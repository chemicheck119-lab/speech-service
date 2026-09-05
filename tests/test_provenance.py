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

    def test_accepts_a_versioned_cross_region_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "VS_서울_화재.zip"
            labels = root / "VL_서울_화재.zip"
            audio.write_bytes(b"seoul-audio")
            labels.write_bytes(b"seoul-labels")
            manifest = {
                "dataset_id": "aihub_71768_seoul_fire",
                "dataset_version": "dataset-71768-downloaded-2026-09-05",
                "usage_role": "evaluation",
                "evaluation": {
                    "id": "speech_aihub119_seoul_fire_validation_900",
                    "record_count": 900,
                },
                "inventory": {"paired_count": 900},
                "artifacts": [
                    {
                        "path": "gs://private/VS_서울_화재.zip",
                        "sha256": hashlib.sha256(b"seoul-audio").hexdigest(),
                    },
                    {
                        "path": "gs://private/VL_서울_화재.zip",
                        "sha256": hashlib.sha256(b"seoul-labels").hexdigest(),
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_evaluation_manifest(manifest_path, audio, labels)
            self.assertEqual("aihub_71768_seoul_fire", result["dataset_id"])


if __name__ == "__main__":
    unittest.main()
