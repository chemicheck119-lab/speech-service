"""Explicit GCS materialization and upload helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def materialize(source: str, destination: Path) -> Path:
    if not source.startswith("gs://"):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from google.cloud import storage

    bucket_name, object_name = split_gcs_uri(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(object_name).download_to_filename(
        destination
    )
    return destination


def upload_file(source: Path, destination_uri: str) -> None:
    from google.cloud import storage

    bucket_name, object_name = split_gcs_uri(destination_uri)
    storage.Client().bucket(bucket_name).blob(object_name).upload_from_filename(source)
