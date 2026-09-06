"""Validate immutable LoRA data artifacts without exposing restricted labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from .lora_protocol import load_experiment_config


EXECUTION_PROTOCOL_ID = "whisper-small-lora-gwangju-execution-v1"
ARTIFACT_PROTOCOL_ID = "whisper-lora-clean-wind-snr0-v1"
REGISTERED_EXECUTION_CONFIG_SHA256 = (
    "7c47b6bd15423263f189aa3bf2dea9ef0ef87bcc8f576ceb6241b016035d7d01"
)
EXPECTED_EXPERIMENT_CONFIG_SHA256 = (
    "273284641b807936bd333c90e4e0e697e443c8caf3055efe7404348cf9ad663d"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "ad56d29958069719651b4f73c37fd29a79d5edcf03bfc9adac6a05b20fc1272b"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "3ef9d791c090d11a249c66468e4836f79323d4496242c5c1dd74ce071ce7300d"
)
EXPECTED_PRIORITY_TERMS_SHA256 = (
    "1269cc8e61f8299061e061e77badcbee0670fe63421b67e733b99c252d1b67b3"
)
EXPECTED_EVIDENCE_SCOPE = (
    "AIHub 신고전화와 절차적 모의 통신 왜곡; 실제 현장 무전 검증 아님"
)
PARTITIONS = ("train", "dev")
CONDITIONS = ("clean", "wind_snr0")
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_AUDIO_MEMBER_BYTES = 32 * 1024 * 1024
MAX_LABEL_MEMBER_BYTES = 4 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


@dataclass(frozen=True)
class FileSnapshot:
    sha256: str
    bytes: int
    inode: int
    mtime_ns: int


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _safe_filename(value: object, name: str) -> str:
    filename = _non_empty_string(value, name)
    if PurePosixPath(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"{name} must be a plain filename")
    return filename


def _read_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(f"{name} exceeds the bounded JSON size")
    content = path.read_bytes()
    if len(content) != size:
        raise ValueError(f"{name} changed while reading")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    return _object(payload, name), content


def _snapshot(path: Path, cache: dict[Path, FileSnapshot]) -> FileSnapshot:
    resolved = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path.name}")
    before = path.stat()
    if before.st_size <= 0 or before.st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"artifact size is outside the bounded range: {path.name}")
    if stat.S_IMODE(before.st_mode) & 0o077:
        raise ValueError(f"private artifact permissions are too broad: {path.name}")
    if resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"artifact changed while hashing: {path.name}")
    result = FileSnapshot(
        digest.hexdigest(), before.st_size, before.st_ino, before.st_mtime_ns
    )
    cache[resolved] = result
    return result


def _assert_snapshots_unchanged(cache: dict[Path, FileSnapshot]) -> None:
    for path, snapshot in cache.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact changed after validation: {path.name}")
        current = path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (
            snapshot.inode,
            snapshot.bytes,
            snapshot.mtime_ns,
        ):
            raise ValueError(f"artifact changed after validation: {path.name}")


def _timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an ISO-8601 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{name} must be an ISO-8601 timestamp with timezone"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be an ISO-8601 timestamp with timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_members(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    members: dict[str, str] = {}
    maximum = MAX_AUDIO_MEMBER_BYTES if suffix == ".wav" else MAX_LABEL_MEMBER_BYTES
    for info in archive.infolist():
        if not info.filename.lower().endswith(suffix):
            continue
        if info.file_size <= 0 or info.file_size > maximum:
            raise ValueError("archive member size is outside the bounded range")
        if info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
            raise ValueError("archive member compression ratio is unsafe")
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("archive member path is unsafe")
        stem = path.stem
        if stem in members:
            raise ValueError("archive contains duplicate member stems")
        members[stem] = info.filename
    if not members:
        raise ValueError(f"archive contains no {suffix} members")
    return members


def _read_member(archive: zipfile.ZipFile, name: str, maximum: int) -> bytes:
    with archive.open(name) as source:
        content = source.read(maximum + 1)
    if not content or len(content) > maximum:
        raise ValueError("archive member exceeds the bounded read")
    return content


def _membership_digest(record_ids: set[str]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(record_ids):
        encoded = record_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_execution_config(path: Path) -> tuple[dict[str, object], bytes]:
    config, content = _read_json(path, "execution config")
    if hashlib.sha256(content).hexdigest() != REGISTERED_EXECUTION_CONFIG_SHA256:
        raise ValueError("execution config SHA-256 is not the registered artifact")
    if (
        config.get("schema_version") != "1.0.0"
        or config.get("protocol_id") != EXECUTION_PROTOCOL_ID
        or config.get("fact_status") != "설계 완료·구현 전"
        or config.get("evidence_scope") != EXPECTED_EVIDENCE_SCOPE
        or config.get("automatic_training_allowed") is not False
    ):
        raise ValueError("execution config safety boundary does not match")

    experiment = _object(config.get("experiment_config"), "experiment config binding")
    if experiment.get("sha256") != EXPECTED_EXPERIMENT_CONFIG_SHA256:
        raise ValueError("experiment config binding does not match")
    data = _object(config.get("derived_data"), "derived data config")
    if (
        data.get("artifact_protocol_id") != ARTIFACT_PROTOCOL_ID
        or data.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or data.get("split_manifest_sha256") != EXPECTED_SPLIT_MANIFEST_SHA256
        or data.get("priority_terms_sha256") != EXPECTED_PRIORITY_TERMS_SHA256
        or data.get("required_partitions") != list(PARTITIONS)
        or data.get("required_conditions") != list(CONDITIONS)
    ):
        raise ValueError("derived data binding does not match")
    _safe_filename(data.get("run_summary_file"), "run summary file")
    _non_empty_string(data.get("run_summary_sha256"), "run summary SHA-256")

    selection = _object(config.get("training_selection"), "training selection")
    if (
        selection.get("unit") != "recordId group"
        or selection.get("algorithm")
        != "lowest SHA-256(assignment:seed:recordId) ranks are clean"
        or selection.get("seed") != 9119
        or selection.get("clean_fraction") != 0.6
        or selection.get("wind_snr0_fraction") != 0.4
        or selection.get("rounding") != "nearest integer, half up"
        or selection.get("each_utterance_seen_once") is not True
    ):
        raise ValueError("training selection is not the registered algorithm")
    segment = _object(config.get("segment_contract"), "segment contract")
    if (
        segment.get("timestamp_unit") != "milliseconds"
        or segment.get("max_audio_seconds") != 12.0
        or segment.get("max_label_tokens") != 160
        or segment.get("over_limit_action") != "fail"
        or segment.get("empty_label_action") != "fail"
        or segment.get("resample_hz") != 16000
    ):
        raise ValueError("segment contract does not match")
    runtime = _object(config.get("runtime"), "runtime config")
    if (
        runtime.get("provider") != "local_owned_hardware"
        or runtime.get("location") != "local"
        or runtime.get("architecture") != "arm64"
        or runtime.get("python_major_minor") != "3.11"
        or runtime.get("accelerator_backend") != "mps"
        or runtime.get("machine_type") != "apple-m4"
        or runtime.get("gpu_type") != "apple-mps"
        or runtime.get("gpu_count") != 1
        or runtime.get("vcpu_count") != 10
        or runtime.get("memory_gib") != 24
        or runtime.get("boot_disk_gib") != 0
        or runtime.get("max_processes") != 1
        or runtime.get("max_runtime_seconds") != 43200
        or runtime.get("external_timeout_seconds") != 42900
        or runtime.get("internal_deadline_seconds") != 42600
        or runtime.get("retry_count") != 0
        or runtime.get("require_mps") is not True
        or runtime.get("cpu_training_fallback") is not False
        or runtime.get("dataloader_pin_memory") is not False
        or runtime.get("gradient_checkpointing_use_reentrant") is not False
    ):
        raise ValueError("runtime cost or MPS boundary does not match")
    cost = _object(config.get("cost_guard"), "cost guard")
    if (
        cost.get("current_quote_required_before_gpu") is not True
        or cost.get("single_use_remote_claim_required") is not True
        or cost.get("experiment_hard_cap_krw") != 0
        or cost.get("compute_billing_ceiling_usd_per_hour") != 0
        or cost.get("boot_disk_billing_ceiling_usd") != 0
        or cost.get("network_transfer_billing_ceiling_usd") != 0
    ):
        raise ValueError("cost authorization boundary does not match")
    output = _object(config.get("private_output"), "private output config")
    if any(
        output.get(field) is not False
        for field in (
            "git_commit_allowed",
            "transcript_logging_allowed",
            "record_id_logging_allowed",
            "address_logging_allowed",
            "overwrite_allowed",
        )
    ):
        raise ValueError("private output boundary must fail closed")
    return config, content


def _artifact_index(
    manifest: dict[str, object],
    *,
    gcs_prefix: str,
) -> dict[str, dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("derived manifest artifacts must be an array")
    indexed: dict[str, dict[str, object]] = {}
    expected_prefix = gcs_prefix.rstrip("/") + "/"
    for item in artifacts:
        artifact = _object(item, "derived manifest artifact")
        path = _non_empty_string(artifact.get("path"), "derived artifact path")
        if not path.startswith(expected_prefix):
            raise ValueError("derived artifact is outside the registered GCS prefix")
        filename = _safe_filename(PurePosixPath(path).name, "derived artifact filename")
        if filename in indexed:
            raise ValueError("derived manifest contains duplicate artifacts")
        indexed[filename] = artifact
    return indexed


def _validate_manifest(
    *,
    manifest: dict[str, object],
    partition: str,
    condition: str,
    summary_entry: dict[str, object],
    artifact_root: Path,
    gcs_prefix: str,
    cache: dict[Path, FileSnapshot],
) -> tuple[Path, Path, str]:
    expected_role = "training" if partition == "train" else "development"
    if (
        manifest.get("dataset_id") != f"aihub_71768_gwangju_fire_lora_{partition}_{condition}"
        or manifest.get("usage_role") != expected_role
        or manifest.get("classification") != "derived"
        or manifest.get("evidence_scope")
        != "AIHub emergency-call Training derivative with procedural wind; not field-radio validation"
    ):
        raise ValueError("derived manifest identity or evidence scope does not match")
    split = _object(manifest.get("split"), "derived manifest split")
    parameters = _object(split.get("parameters"), "derived manifest split parameters")
    if (
        parameters.get("protocol_id") != ARTIFACT_PROTOCOL_ID
        or parameters.get("partition") != partition
        or parameters.get("condition") != condition
        or parameters.get("clean_and_derived_share_partition") is not True
        or parameters.get("used_for_tuning") is not True
    ):
        raise ValueError("derived manifest split contract does not match")
    membership_sha256 = _non_empty_string(
        parameters.get("membership_sha256"), "membership SHA-256"
    )
    inventory = _object(manifest.get("inventory"), "derived manifest inventory")
    if (
        inventory.get("paired_count") != summary_entry.get("record_count")
        or inventory.get("utterance_count") != summary_entry.get("utterance_count")
    ):
        raise ValueError("derived manifest inventory does not match run summary")

    audio_name = f"{partition}-{condition}.zip"
    label_name = f"{partition}-labels.zip"
    ledger_name = "provenance.private.jsonl"
    indexed = _artifact_index(manifest, gcs_prefix=gcs_prefix)
    if set(indexed) != {audio_name, label_name, ledger_name}:
        raise ValueError("derived manifest artifact set does not match")
    for filename, summary_field in (
        (audio_name, "audio_sha256"),
        (label_name, "labels_sha256"),
    ):
        declared = indexed[filename]
        local = _snapshot(artifact_root / filename, cache)
        if (
            declared.get("sha256") != local.sha256
            or declared.get("bytes") != local.bytes
            or summary_entry.get(summary_field) != local.sha256
            or declared.get("access") != "private"
        ):
            raise ValueError(f"{filename} does not match its immutable manifest")
    ledger = _snapshot(artifact_root / ledger_name, cache)
    ledger_declared = indexed[ledger_name]
    if (
        ledger_declared.get("sha256") != ledger.sha256
        or ledger_declared.get("bytes") != ledger.bytes
        or ledger_declared.get("access") != "private"
    ):
        raise ValueError("private provenance ledger does not match")
    return artifact_root / audio_name, artifact_root / label_name, membership_sha256


def _partition_labels(
    *,
    partition: str,
    label_archive: Path,
    audio_archives: dict[str, Path],
    expected_membership_sha256: str,
    max_audio_seconds: float,
) -> tuple[set[str], dict[str, int], int, float]:
    with zipfile.ZipFile(label_archive) as labels:
        label_members = _safe_members(labels, ".json")
        audio_member_sets: list[set[str]] = []
        for condition in CONDITIONS:
            with zipfile.ZipFile(audio_archives[condition]) as audio:
                audio_member_sets.append(set(_safe_members(audio, ".wav")))
        if any(members != set(label_members) for members in audio_member_sets):
            raise ValueError(f"{partition} clean/wind/label archive pairing differs")

        record_ids: set[str] = set()
        utterance_counts: dict[str, int] = {}
        total_utterances = 0
        maximum_duration = 0.0
        for stem in sorted(label_members):
            content = _read_member(
                labels, label_members[stem], MAX_LABEL_MEMBER_BYTES
            )
            try:
                document = json.loads(content.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("label archive contains invalid JSON") from error
            document = _object(document, "label record")
            record_id = _non_empty_string(document.get("recordId"), "label recordId")
            if record_id in record_ids:
                raise ValueError(f"{partition} label archive contains duplicate recordId")
            record_ids.add(record_id)
            utterances = document.get("utterances")
            if not isinstance(utterances, list) or not utterances:
                raise ValueError("label utterances must be a non-empty array")
            utterance_counts[record_id] = len(utterances)
            total_utterances += len(utterances)
            for utterance in utterances:
                item = _object(utterance, "utterance")
                text = item.get("text")
                start = item.get("startAt")
                end = item.get("endAt")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("utterance text must be non-empty")
                if (
                    not isinstance(start, (int, float))
                    or isinstance(start, bool)
                    or not isinstance(end, (int, float))
                    or isinstance(end, bool)
                    or start < 0
                    or end <= start
                ):
                    raise ValueError("utterance timestamps are invalid")
                duration = (float(end) - float(start)) / 1000.0
                maximum_duration = max(maximum_duration, duration)
                if duration > max_audio_seconds:
                    raise ValueError("utterance exceeds the registered audio limit")

    if _membership_digest(record_ids) != expected_membership_sha256:
        raise ValueError(f"{partition} membership digest does not match")
    return record_ids, utterance_counts, total_utterances, maximum_duration


def training_condition_assignments(
    record_ids: set[str],
    *,
    seed: int,
    clean_fraction: float,
) -> dict[str, str]:
    """Return the registered record-level arm without exposing it in reports."""

    ordered = sorted(
        record_ids,
        key=lambda record_id: (
            hashlib.sha256(
                f"assignment:{seed}:{record_id}".encode("utf-8")
            ).digest(),
            record_id,
        ),
    )
    clean_count = int(len(ordered) * clean_fraction + 0.5)
    if clean_count <= 0 or clean_count >= len(ordered):
        raise ValueError("training condition assignment produced an empty arm")
    clean = set(ordered[:clean_count])
    return {
        record_id: "clean" if record_id in clean else "wind_snr0"
        for record_id in ordered
    }


def _training_assignment(
    record_ids: set[str],
    utterance_counts: dict[str, int],
    *,
    seed: int,
    clean_fraction: float,
) -> dict[str, dict[str, int]]:
    assignments = training_condition_assignments(
        record_ids,
        seed=seed,
        clean_fraction=clean_fraction,
    )
    result = {
        condition: {"record_count": 0, "utterance_count": 0}
        for condition in CONDITIONS
    }
    for record_id, condition in assignments.items():
        result[condition]["record_count"] += 1
        result[condition]["utterance_count"] += utterance_counts[record_id]
    if sum(item["record_count"] for item in result.values()) != len(record_ids):
        raise AssertionError("condition assignment lost a training record")
    if sum(item["utterance_count"] for item in result.values()) != sum(
        utterance_counts.values()
    ):
        raise AssertionError("condition assignment lost a training utterance")
    return result


def validate_lora_data_preflight(
    *,
    execution_config_path: Path,
    experiment_config_path: Path,
    artifact_root: Path,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Return an aggregate-only readiness report; never authorizes training."""

    execution, execution_bytes = _load_execution_config(execution_config_path)
    _, experiment_bytes = load_experiment_config(experiment_config_path)
    if hashlib.sha256(experiment_bytes).hexdigest() != EXPECTED_EXPERIMENT_CONFIG_SHA256:
        raise ValueError("experiment config does not match the execution binding")
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("artifact root must be a non-symlink directory")

    derived = _object(execution["derived_data"], "derived data config")
    summary_name = _safe_filename(derived["run_summary_file"], "run summary file")
    summary, summary_bytes = _read_json(artifact_root / summary_name, "run summary")
    observed_summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    if observed_summary_sha256 != derived["run_summary_sha256"]:
        raise ValueError("run summary SHA-256 does not match the execution config")
    if (
        summary.get("protocol_id") != ARTIFACT_PROTOCOL_ID
        or summary.get("status") != "completed"
        or summary.get("fact_status") != "구현 완료"
        or summary.get("source_manifest_sha256")
        != derived["source_manifest_sha256"]
        or summary.get("split_manifest_sha256") != derived["split_manifest_sha256"]
        or summary.get("priority_terms_sha256") != derived["priority_terms_sha256"]
        or summary.get("automatic_training_allowed") is not False
    ):
        raise ValueError("run summary safety or provenance contract does not match")
    privacy = _object(summary.get("privacy"), "run summary privacy")
    if (
        privacy.get("git_commit_allowed") is not False
        or privacy.get("private_storage_required") is not True
        or privacy.get("console_contains_record_ids_or_transcripts") is not False
    ):
        raise ValueError("run summary privacy contract does not match")

    entries = summary.get("manifests")
    if not isinstance(entries, list):
        raise ValueError("run summary manifests must be an array")
    indexed_entries: dict[tuple[str, str], dict[str, object]] = {}
    for entry_value in entries:
        entry = _object(entry_value, "run summary manifest entry")
        key = (str(entry.get("partition")), str(entry.get("condition")))
        if key in indexed_entries:
            raise ValueError("run summary contains duplicate manifest entries")
        indexed_entries[key] = entry
    expected_keys = {(p, c) for p in PARTITIONS for c in CONDITIONS}
    if set(indexed_entries) != expected_keys:
        raise ValueError("run summary does not contain the four registered arms")

    cache: dict[Path, FileSnapshot] = {}
    archives: dict[str, dict[str, Path]] = {partition: {} for partition in PARTITIONS}
    label_archives: dict[str, Path] = {}
    membership: dict[str, str] = {}
    manifest_snapshots: list[dict[str, object]] = []
    for partition in PARTITIONS:
        for condition in CONDITIONS:
            entry = indexed_entries[(partition, condition)]
            manifest_name = _safe_filename(entry.get("manifest"), "manifest filename")
            expected_manifest_name = f"{partition}-{condition}.manifest.json"
            if manifest_name != expected_manifest_name:
                raise ValueError("run summary manifest filename does not match its arm")
            manifest, manifest_bytes = _read_json(
                artifact_root / manifest_name, "derived manifest"
            )
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if manifest_sha256 != entry.get("manifest_sha256"):
                raise ValueError("derived manifest SHA-256 does not match run summary")
            audio, labels, membership_sha256 = _validate_manifest(
                manifest=manifest,
                partition=partition,
                condition=condition,
                summary_entry=entry,
                artifact_root=artifact_root,
                gcs_prefix=_non_empty_string(derived.get("gcs_prefix"), "data GCS prefix"),
                cache=cache,
            )
            archives[partition][condition] = audio
            label_archives[partition] = labels
            previous_membership = membership.setdefault(partition, membership_sha256)
            if previous_membership != membership_sha256:
                raise ValueError("clean and wind manifests declare different membership")
            manifest_snapshots.append(
                {
                    "partition": partition,
                    "condition": condition,
                    "sha256": manifest_sha256,
                }
            )

    ledger = _snapshot(artifact_root / "provenance.private.jsonl", cache)
    if ledger.sha256 != summary.get("private_ledger_sha256"):
        raise ValueError("private provenance ledger does not match run summary")

    segment = _object(execution["segment_contract"], "segment contract")
    partition_records: dict[str, set[str]] = {}
    utterance_counts: dict[str, dict[str, int]] = {}
    partition_summary: dict[str, dict[str, object]] = {}
    for partition in PARTITIONS:
        records, per_record, utterances, maximum_duration = _partition_labels(
            partition=partition,
            label_archive=label_archives[partition],
            audio_archives=archives[partition],
            expected_membership_sha256=membership[partition],
            max_audio_seconds=float(segment["max_audio_seconds"]),
        )
        expected = indexed_entries[(partition, "clean")]
        if (
            len(records) != expected.get("record_count")
            or utterances != expected.get("utterance_count")
        ):
            raise ValueError("observed label inventory does not match run summary")
        partition_records[partition] = records
        utterance_counts[partition] = per_record
        partition_summary[partition] = {
            "record_count": len(records),
            "utterance_count": utterances,
            "membership_sha256": membership[partition],
            "max_utterance_seconds": round(maximum_duration, 6),
        }
    if partition_records["train"] & partition_records["dev"]:
        raise ValueError("train/dev recordId overlap detected")

    selection = _object(execution["training_selection"], "training selection")
    assignment = _training_assignment(
        partition_records["train"],
        utterance_counts["train"],
        seed=int(selection["seed"]),
        clean_fraction=float(selection["clean_fraction"]),
    )
    created = _timestamp(
        generated_at or datetime.now(timezone.utc).isoformat(), "generated_at"
    )
    _assert_snapshots_unchanged(cache)
    return {
        "schema_version": "1.0.0",
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "status": "limited",
        "fact_status": "구현 완료",
        "training_fact_status": "설계 완료·구현 전",
        "generated_at": created,
        "execution_config_sha256": hashlib.sha256(execution_bytes).hexdigest(),
        "experiment_config_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
        "run_summary_sha256": observed_summary_sha256,
        "artifact_snapshots": sorted(
            [
                {"file": path.name, "sha256": snapshot.sha256, "bytes": snapshot.bytes}
                for path, snapshot in cache.items()
            ],
            key=lambda item: str(item["file"]),
        ),
        "manifest_snapshots": manifest_snapshots,
        "partitions": partition_summary,
        "training_condition_assignment": assignment,
        "integrity": {
            "record_overlap_count": 0,
            "speaker_overlap_status": "not_evaluated",
            "event_overlap_status": "not_evaluated",
            "token_limit_status": "pending_runtime_tokenizer_validation",
        },
        "privacy": {
            "contains_record_ids": False,
            "contains_transcripts": False,
            "contains_addresses": False,
        },
        "limitations": [
            "speaker and cross-record incident overlap cannot be evaluated from provider labels",
            "label token limits still require the pinned Whisper tokenizer at accelerator runtime",
            "this is emergency-call development data with procedural wind, not field-radio evidence",
        ],
        "automatic_training_allowed": False,
        "next_gate": "pinned tokenizer dry-run, reviewed local MPS trainer, and current zero-cost attestation",
        "claim_scope": "data readiness only; no LoRA performance or field-radio claim",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    report = validate_lora_data_preflight(
        execution_config_path=args.execution_config,
        experiment_config_path=args.experiment_config,
        artifact_root=args.artifact_root,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite LoRA data preflight: {args.output}"
        ) from error
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_fact_status": report["training_fact_status"],
                "automatic_training_allowed": report["automatic_training_allowed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
