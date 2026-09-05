"""Bind an evaluation ID to a versioned manifest and archive digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

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
    evaluation_id = evaluation.get("id")
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ValueError("evaluation ID must be declared")
    record_count = evaluation.get("record_count")
    if type(record_count) is not int or record_count <= 0:
        raise ValueError("evaluation record count must be a positive integer")
    if manifest.get("usage_role") != "evaluation":
        raise ValueError("manifest usage_role must be evaluation")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset ID must be declared")
    if not isinstance(manifest.get("dataset_version"), str) or not manifest[
        "dataset_version"
    ].strip():
        raise ValueError("dataset version must be declared")
    inventory = manifest.get("inventory")
    if isinstance(inventory, dict) and inventory.get("paired_count") != record_count:
        raise ValueError("evaluation and inventory record counts do not match")

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
        "dataset_id": dataset_id,
        "dataset_version": manifest.get("dataset_version"),
        "evaluation_id": evaluation_id,
        "record_count": record_count,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_sha256": observed,
    }
