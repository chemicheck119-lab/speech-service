"""Bind a fixed evaluation ID to manifest and archive content digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .evaluation import EVALUATION_ID, EXPECTED_EVALUATION_RECORDS


EXPECTED_DATASET_ID = "aihub_71768_gwangju_fire"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evaluation_manifest(
    manifest_path: Path,
    audio_archive: Path,
    label_archive: Path,
) -> dict[str, object]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation manifest is missing evaluation metadata")
    if evaluation.get("id") != EVALUATION_ID:
        raise ValueError("unexpected fixed evaluation ID")
    if evaluation.get("record_count") != EXPECTED_EVALUATION_RECORDS:
        raise ValueError("unexpected fixed evaluation record count")
    if manifest.get("usage_role") != "evaluation":
        raise ValueError("manifest usage_role must be evaluation")
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("unexpected dataset ID")
    if not isinstance(manifest.get("dataset_version"), str) or not manifest[
        "dataset_version"
    ].strip():
        raise ValueError("dataset version must be declared")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")
    expected: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        digest = artifact.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            expected[PurePosixPath(path).name] = digest
    observed = {
        "audio": sha256_file(audio_archive),
        "labels": sha256_file(label_archive),
    }
    expected_digests = set(expected.values())
    for role, digest in observed.items():
        if digest not in expected_digests:
            raise ValueError(f"archive digest does not match manifest: {role}")

    return {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_version": manifest.get("dataset_version"),
        "evaluation_id": evaluation["id"],
        "record_count": evaluation["record_count"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_sha256": observed,
    }
