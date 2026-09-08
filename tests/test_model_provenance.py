from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chemicheck119_speech.model_provenance import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    prepare_model_artifact,
    verify_model_provenance,
)


class ModelProvenanceTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        model_directory = root / "faster-whisper-small"
        model_directory.mkdir()
        model_path = model_directory / "model.bin"
        model_path.write_bytes(b"bounded-test-model")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest_path = model_directory / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "repository": "Systran/faster-whisper-small",
                    "revision": "5" * 40,
                    "model_bin_sha256": digest,
                }
            ),
            encoding="utf-8",
        )
        return model_directory, manifest_path

    def test_verifies_pinned_model_file(self) -> None:
        with TemporaryDirectory() as directory:
            model_directory, manifest_path = self._fixture(Path(directory))

            result = verify_model_provenance(model_directory, manifest_path)

        self.assertTrue(result["model_artifact_verified"])
        self.assertEqual("5" * 40, result["revision"])

    def test_rejects_tampered_model_file(self) -> None:
        with TemporaryDirectory() as directory:
            model_directory, manifest_path = self._fixture(Path(directory))
            (model_directory / "model.bin").write_bytes(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                verify_model_provenance(model_directory, manifest_path)

    def test_rejects_manifest_outside_model_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory, manifest_path = self._fixture(root)
            external_manifest = root / "external.json"
            external_manifest.write_bytes(manifest_path.read_bytes())

            with self.assertRaisesRegex(RuntimeError, "outside"):
                verify_model_provenance(model_directory, external_manifest)

    def test_prepare_pins_download_and_writes_verified_manifest(self) -> None:
        model_payload = b"prepared-test-model"
        expected_sha256 = hashlib.sha256(model_payload).hexdigest()

        def fake_download(repository: str, *, output_dir: str, revision: str) -> str:
            self.assertEqual("Systran/faster-whisper-small", repository)
            self.assertEqual("5" * 40, revision)
            destination = Path(output_dir)
            (destination / "model.bin").write_bytes(model_payload)
            return str(destination)

        with TemporaryDirectory() as directory:
            output_directory = Path(directory) / "model"
            result = prepare_model_artifact(
                repository="Systran/faster-whisper-small",
                revision="5" * 40,
                expected_model_bin_sha256=expected_sha256,
                output_directory=output_directory,
                downloader=fake_download,
            )
            manifest = json.loads(
                (output_directory / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(expected_sha256, result["model_bin_sha256"])
        self.assertEqual(result, manifest)


if __name__ == "__main__":
    unittest.main()
