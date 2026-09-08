"""Pinned faster-whisper artifact preparation and runtime verification."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MANIFEST_SCHEMA_VERSION = "chemicheck119-speech-model-provenance-v1"
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MODEL_FILENAME = "model.bin"
MANIFEST_FILENAME = "chemicheck119-model-provenance.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("model provenance manifest must be an object")
    return payload


def verify_model_provenance(
    model_directory: Path,
    manifest_path: Path,
) -> dict[str, str | bool]:
    """Bind a runtime model directory to a pinned repository revision and hash."""

    model_directory = model_directory.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path.parent != model_directory:
        raise RuntimeError("model provenance manifest is outside the model directory")
    payload = _load_manifest(manifest_path)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported model provenance manifest")
    repository = str(payload.get("repository") or "")
    revision = str(payload.get("revision") or "")
    expected_sha256 = str(payload.get("model_bin_sha256") or "")
    if "/" not in repository or not repository.strip():
        raise RuntimeError("model repository is invalid")
    if GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError("model revision is not pinned")
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise RuntimeError("model.bin SHA-256 is invalid")
    model_path = model_directory / MODEL_FILENAME
    if not model_path.is_file() or model_path.is_symlink():
        raise RuntimeError("model.bin is missing or is a symlink")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("model.bin SHA-256 does not match provenance manifest")
    return {
        "repository": repository,
        "revision": revision,
        "model_bin_sha256": actual_sha256,
        "model_artifact_verified": True,
    }


def prepare_model_artifact(
    *,
    repository: str,
    revision: str,
    expected_model_bin_sha256: str,
    output_directory: Path,
    downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Download one pinned model and write a deterministic verified manifest."""

    if "/" not in repository or not repository.strip():
        raise ValueError("repository must be a Hugging Face repository ID")
    if GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a 40-character lowercase Git commit")
    if SHA256_PATTERN.fullmatch(expected_model_bin_sha256) is None:
        raise ValueError("expected model.bin SHA-256 is invalid")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("model output directory is not empty")
    output_directory.mkdir(parents=True, exist_ok=True)

    if downloader is None:
        from faster_whisper.utils import download_model

        downloader = download_model

    downloaded = Path(
        downloader(
            repository,
            output_dir=str(output_directory),
            revision=revision,
        )
    ).resolve()
    if downloaded != output_directory.resolve():
        raise RuntimeError("downloaded model path does not match output directory")
    model_path = output_directory / MODEL_FILENAME
    if not model_path.is_file() or model_path.is_symlink():
        raise RuntimeError("downloaded model.bin is missing or is a symlink")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_model_bin_sha256:
        raise RuntimeError("downloaded model.bin SHA-256 does not match pinned value")

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repository": repository,
        "revision": revision,
        "model_bin_sha256": actual_sha256,
    }
    manifest_path = output_directory / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_model_provenance(output_directory, manifest_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-bin-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    prepare_model_artifact(
        repository=args.repository,
        revision=args.revision,
        expected_model_bin_sha256=args.model_bin_sha256,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
