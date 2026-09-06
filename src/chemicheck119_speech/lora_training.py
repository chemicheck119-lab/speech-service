"""Fail-closed GPU harness for one bounded Whisper LoRA training run.

The harness intentionally trains only. It does not evaluate, merge, convert, adopt,
or deploy the resulting adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib.metadata import version as distribution_version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sys
import time
from typing import Callable, Mapping, Sequence
import wave
import zipfile

from .lora_data_preflight import (
    MAX_AUDIO_MEMBER_BYTES,
    MAX_LABEL_MEMBER_BYTES,
    _load_execution_config,
    _non_empty_string,
    _object,
    _read_member,
    _safe_members,
    _timestamp,
    training_condition_assignments,
    validate_lora_data_preflight,
)
from .lora_protocol import load_experiment_config
from .lora_tokenizer_preflight import validate_lora_tokenizer_preflight


TRAINING_PROTOCOL_ID = "whisper-small-lora-training-v1"
COST_QUOTE_PROTOCOL_ID = "whisper-small-lora-cost-quote-v1"
CONFIRMATION_PHRASE = "RUN_BOUNDED_LORA_ONCE"
MAX_QUOTE_BYTES = 64 * 1024
MAX_QUOTE_AGE = timedelta(hours=24)
ALLOWED_TRAIN_METRICS = {
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "train_loss",
    "epoch",
}


@dataclass(frozen=True)
class TrainingExample:
    audio_path: Path
    start_ms: float
    end_ms: float
    text: str = field(repr=False)


@dataclass(frozen=True)
class CostDecision:
    authorization_id: str
    authorized_speech_revision: str
    cumulative_development_cost_before_krw: int
    quote_sha256: str
    quoted_compute_usd_per_hour: float
    quoted_boot_disk_usd: float
    quoted_network_transfer_usd: float
    quoted_total_usd: float
    quoted_total_krw_with_contingency: int
    independent_experiment_ceiling_krw: int
    independent_total_ceiling_krw: int
    generated_at: str
    expires_at: str


@dataclass(frozen=True)
class AuthorizationClaim:
    claim_sha256: str
    remote_object_uri: str
    claimed_at: str


def _read_bounded_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_QUOTE_BYTES:
        raise ValueError(f"{name} exceeds the bounded JSON size")
    content = path.read_bytes()
    if len(content) != size:
        raise ValueError(f"{name} changed while reading")
    try:
        return _object(json.loads(content.decode("utf-8")), name), content
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error


def _finite_positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return number


def _parse_utc(value: object, name: str) -> datetime:
    canonical = _timestamp(value, name)
    return datetime.fromisoformat(canonical.replace("Z", "+00:00"))


def validate_cost_quote(
    *,
    quote_path: Path,
    execution_config: dict[str, object],
    runner_revision: str,
    now: datetime | None = None,
) -> CostDecision:
    """Validate a current itemized quote and both registered budget ceilings."""

    quote, quote_bytes = _read_bounded_json(quote_path, "cost quote")
    if (
        quote.get("schema_version") != "1.0.0"
        or quote.get("protocol_id") != COST_QUOTE_PROTOCOL_ID
        or quote.get("currency") != "USD"
    ):
        raise ValueError("cost quote identity does not match")
    authorization = _object(quote.get("authorization"), "cost authorization")
    authorization_id = _non_empty_string(
        authorization.get("id"), "cost authorization id"
    )
    authorized_revision = _non_empty_string(
        authorization.get("speech_revision"), "authorized speech revision"
    )
    cumulative_before = authorization.get("cumulative_development_cost_before_krw")
    if (
        re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", authorization_id) is None
        or re.fullmatch(r"[0-9a-f]{40}", runner_revision) is None
        or authorized_revision != runner_revision
        or authorization.get("authorized_run_count") != 1
        or authorization.get("remote_claim_required") is not True
        or not isinstance(cumulative_before, int)
        or isinstance(cumulative_before, bool)
        or cumulative_before < 0
    ):
        raise ValueError("cost authorization does not match the single-use run")
    generated = _parse_utc(quote.get("generated_at"), "cost quote generated_at")
    expires = _parse_utc(quote.get("expires_at"), "cost quote expires_at")
    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None or observed_now.utcoffset() is None:
        raise ValueError("cost validation time must include a timezone")
    observed_now = observed_now.astimezone(timezone.utc)
    if expires <= generated or expires - generated > MAX_QUOTE_AGE:
        raise ValueError("cost quote validity window must be at most 24 hours")
    if observed_now < generated - timedelta(minutes=5) or observed_now >= expires:
        raise ValueError("cost quote is not current")

    runtime = _object(execution_config.get("runtime"), "runtime config")
    resource = _object(quote.get("resource"), "cost quote resource")
    expected_resource = {
        "provider": runtime["provider"],
        "location": runtime["location"],
        "machine_type": runtime["machine_type"],
        "gpu_type": runtime["gpu_type"],
        "gpu_count": runtime["gpu_count"],
        "vcpu_count": runtime["vcpu_count"],
        "memory_gib": runtime["memory_gib"],
        "boot_disk_gib": runtime["boot_disk_gib"],
        "runtime_hours": float(runtime["max_runtime_seconds"]) / 3600.0,
    }
    if resource != expected_resource:
        raise ValueError("cost quote resource does not match the registered runtime")

    pricing = _object(quote.get("pricing"), "cost quote pricing")
    if pricing.get("model") != "local_owned_hardware":
        raise ValueError("cost quote pricing model does not match the registered runtime")
    compute_hour = _finite_positive(
        pricing.get("machine_usd_per_hour"), "machine price", allow_zero=True
    )
    disk_month = _finite_positive(
        pricing.get("boot_disk_usd_per_gib_month"),
        "boot disk price",
        allow_zero=True,
    )
    network_total = _finite_positive(
        pricing.get("network_transfer_usd"),
        "network transfer price",
        allow_zero=True,
    )
    fx = _finite_positive(quote.get("fx_krw_per_usd"), "FX rate")
    sources = quote.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources) < 2
        or any(
            not isinstance(source, str) or not source.startswith("https://")
            for source in sources
        )
    ):
        raise ValueError("cost quote must contain HTTPS pricing and FX sources")

    hours = expected_resource["runtime_hours"]
    boot_total = disk_month * float(runtime["boot_disk_gib"])
    total_usd = compute_hour * hours + boot_total + network_total
    cost_guard = _object(execution_config.get("cost_guard"), "cost guard")
    contingency = _finite_positive(
        cost_guard.get("contingency_fraction"),
        "contingency fraction",
        allow_zero=True,
    )
    quoted_krw = math.ceil(total_usd * fx * (1.0 + contingency))
    if compute_hour > float(cost_guard["compute_billing_ceiling_usd_per_hour"]):
        raise ValueError("quoted compute price exceeds the registered ceiling")
    if boot_total > float(cost_guard["boot_disk_billing_ceiling_usd"]):
        raise ValueError("quoted boot disk price exceeds the registered ceiling")
    if network_total > float(cost_guard["network_transfer_billing_ceiling_usd"]):
        raise ValueError("quoted network transfer price exceeds the registered ceiling")
    if fx > float(cost_guard["fx_ceiling_krw_per_usd"]):
        raise ValueError("quoted FX rate exceeds the registered ceiling")
    if quoted_krw > int(cost_guard["experiment_hard_cap_krw"]):
        raise ValueError("quoted experiment cost exceeds the hard cap")

    independent_experiment_ceiling = math.ceil(
        (
            float(cost_guard["compute_billing_ceiling_usd_per_hour"]) * hours
            + float(cost_guard["boot_disk_billing_ceiling_usd"])
            + float(cost_guard["network_transfer_billing_ceiling_usd"])
        )
        * float(cost_guard["fx_ceiling_krw_per_usd"])
        * (1.0 + contingency)
    )
    if cumulative_before < int(
        cost_guard["tracked_prior_development_cost_ceiling_krw"]
    ):
        raise ValueError(
            "cost authorization understates tracked prior development cost"
        )
    independent_total_ceiling = cumulative_before + independent_experiment_ceiling
    if independent_experiment_ceiling > int(cost_guard["experiment_hard_cap_krw"]):
        raise ValueError("independent experiment ceiling exceeds the hard cap")
    if independent_total_ceiling > int(cost_guard["total_development_server_cap_krw"]):
        raise ValueError("independent total ceiling exceeds the development cap")
    return CostDecision(
        authorization_id=authorization_id,
        authorized_speech_revision=authorized_revision,
        cumulative_development_cost_before_krw=cumulative_before,
        quote_sha256=hashlib.sha256(quote_bytes).hexdigest(),
        quoted_compute_usd_per_hour=round(compute_hour, 6),
        quoted_boot_disk_usd=round(boot_total, 6),
        quoted_network_transfer_usd=round(network_total, 6),
        quoted_total_usd=round(total_usd, 6),
        quoted_total_krw_with_contingency=quoted_krw,
        independent_experiment_ceiling_krw=independent_experiment_ceiling,
        independent_total_ceiling_krw=independent_total_ceiling,
        generated_at=generated.isoformat().replace("+00:00", "Z"),
        expires_at=expires.isoformat().replace("+00:00", "Z"),
    )


def validate_authorization_claim(
    *,
    claim_path: Path,
    cost: CostDecision,
    runner_revision: str,
    expected_gcs_prefix: str,
    now: datetime | None = None,
) -> AuthorizationClaim:
    """Bind training to the receipt of an atomically created remote claim."""

    claim, claim_bytes = _read_bounded_json(claim_path, "authorization claim")
    claimed_at = _parse_utc(claim.get("claimed_at"), "authorization claimed_at")
    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None or observed_now.utcoffset() is None:
        raise ValueError("claim validation time must include a timezone")
    observed_now = observed_now.astimezone(timezone.utc)
    remote_uri = _non_empty_string(
        claim.get("remote_object_uri"), "authorization remote object URI"
    )
    expected_suffix = f"/authorizations/{cost.authorization_id}.claimed.json"
    if (
        claim.get("schema_version") != "1.0.0"
        or claim.get("protocol_id") != "whisper-small-lora-authorization-claim-v1"
        or claim.get("authorization_id") != cost.authorization_id
        or claim.get("speech_revision") != runner_revision
        or claim.get("cost_quote_sha256") != cost.quote_sha256
        or claim.get("remote_claim_created") is not True
        or not remote_uri.startswith(expected_gcs_prefix.rstrip("/") + "/")
        or not remote_uri.endswith(expected_suffix)
        or claimed_at > observed_now + timedelta(minutes=5)
        or observed_now - claimed_at > MAX_QUOTE_AGE
    ):
        raise ValueError("authorization claim does not match the single-use cost quote")
    return AuthorizationClaim(
        claim_sha256=hashlib.sha256(claim_bytes).hexdigest(),
        remote_object_uri=remote_uri,
        claimed_at=claimed_at.isoformat().replace("+00:00", "Z"),
    )


def validate_gpu_runtime(
    execution_config: dict[str, object],
    *,
    torch_module: object | None = None,
    installed_version: Callable[[str], str] = distribution_version,
) -> dict[str, object]:
    """Reject any runtime other than the registered local Apple MPS environment."""

    runtime = _object(execution_config.get("runtime"), "runtime config")
    if (
        f"{sys.version_info.major}.{sys.version_info.minor}"
        != runtime["python_major_minor"]
    ):
        raise RuntimeError("Python runtime does not match the execution config")
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as error:
            raise RuntimeError("PyTorch is unavailable") from error
    packages = _object(runtime.get("packages"), "runtime packages")
    torch_version = str(getattr(torch_module, "__version__", ""))
    if not torch_version.startswith(str(packages["torch_expected_prefix"])):
        raise RuntimeError("PyTorch runtime does not match the execution config")
    if platform.machine() != runtime["architecture"]:
        raise RuntimeError("machine architecture does not match the execution config")
    if runtime.get("accelerator_backend") != "mps":
        raise RuntimeError("accelerator backend does not match the execution config")
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not mps.is_built() or not mps.is_available():
        raise RuntimeError("MPS is required; CPU training fallback is disabled")
    device_name = "Apple MPS"
    observed_packages: dict[str, str] = {}
    for package in ("transformers", "peft", "accelerate", "numpy", "scipy"):
        observed = installed_version(package)
        if observed != str(packages[package]):
            raise RuntimeError(f"{package} runtime does not match the execution config")
        observed_packages[package] = observed
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch_version,
        "accelerator_backend": "mps",
        "gpu_count": int(runtime["gpu_count"]),
        "gpu_name": device_name,
        "packages": observed_packages,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_snapshot_map(data_report: Mapping[str, object]) -> dict[str, str]:
    snapshots = data_report.get("artifact_snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("data preflight artifact snapshots are missing")
    result: dict[str, str] = {}
    for value in snapshots:
        item = _object(value, "artifact snapshot")
        name = _non_empty_string(item.get("file"), "artifact snapshot file")
        digest = _non_empty_string(item.get("sha256"), "artifact snapshot SHA-256")
        result[name] = digest
    return result


def _verify_artifacts(artifact_root: Path, expected: Mapping[str, str]) -> None:
    for name, digest in expected.items():
        path = artifact_root / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise ValueError("private artifact changed after preflight")


def _secure_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def materialize_training_examples(
    *,
    artifact_root: Path,
    output_dir: Path,
    execution_config: dict[str, object],
    expected_snapshots: Mapping[str, str],
) -> tuple[list[TrainingExample], dict[str, dict[str, int]]]:
    """Materialize each selected full-call WAV once and retain labels only in memory."""

    if output_dir.exists():
        raise FileExistsError("private materialization directory already exists")
    output_dir.mkdir(mode=0o700, parents=False)
    _verify_artifacts(artifact_root, expected_snapshots)
    selection = _object(execution_config["training_selection"], "training selection")
    label_path = artifact_root / "train-labels.zip"
    archive_paths = {
        condition: artifact_root / f"train-{condition}.zip"
        for condition in ("clean", "wind_snr0")
    }
    examples: list[TrainingExample] = []
    counts = {
        condition: {"record_count": 0, "utterance_count": 0}
        for condition in archive_paths
    }
    with (
        zipfile.ZipFile(label_path) as labels,
        zipfile.ZipFile(archive_paths["clean"]) as clean_audio,
        zipfile.ZipFile(archive_paths["wind_snr0"]) as wind_audio,
    ):
        label_members = _safe_members(labels, ".json")
        audio_archives = {"clean": clean_audio, "wind_snr0": wind_audio}
        audio_members = {
            condition: _safe_members(archive, ".wav")
            for condition, archive in audio_archives.items()
        }
        if any(
            set(members) != set(label_members) for members in audio_members.values()
        ):
            raise ValueError("training archive pairing differs after preflight")
        documents: list[tuple[str, str, dict[str, object]]] = []
        record_ids: set[str] = set()
        for stem in sorted(label_members):
            content = _read_member(labels, label_members[stem], MAX_LABEL_MEMBER_BYTES)
            try:
                document = _object(
                    json.loads(content.decode("utf-8-sig")), "training label record"
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "training label archive contains invalid JSON"
                ) from error
            record_id = _non_empty_string(document.get("recordId"), "recordId")
            if record_id in record_ids:
                raise ValueError("training labels contain duplicate recordId")
            record_ids.add(record_id)
            documents.append((stem, record_id, document))
        assignments = training_condition_assignments(
            record_ids,
            seed=int(selection["seed"]),
            clean_fraction=float(selection["clean_fraction"]),
        )
        maximum_seconds = float(
            execution_config["segment_contract"]["max_audio_seconds"]
        )
        for stem, record_id, document in documents:
            condition = assignments[record_id]
            audio_content = _read_member(
                audio_archives[condition],
                audio_members[condition][stem],
                MAX_AUDIO_MEMBER_BYTES,
            )
            filename = (
                hashlib.sha256(
                    f"private-audio:{condition}:{stem}".encode("utf-8")
                ).hexdigest()
                + ".wav"
            )
            audio_path = output_dir / filename
            _secure_write(audio_path, audio_content)
            counts[condition]["record_count"] += 1
            utterances = document.get("utterances")
            if not isinstance(utterances, list) or not utterances:
                raise ValueError("training utterances must be a non-empty array")
            for value in utterances:
                utterance = _object(value, "training utterance")
                text = utterance.get("text")
                start = utterance.get("startAt")
                end = utterance.get("endAt")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("training text must be non-empty")
                if (
                    not isinstance(start, (int, float))
                    or isinstance(start, bool)
                    or not isinstance(end, (int, float))
                    or isinstance(end, bool)
                    or start < 0
                    or end <= start
                    or (float(end) - float(start)) / 1000.0 > maximum_seconds
                ):
                    raise ValueError("training utterance timestamps are invalid")
                examples.append(
                    TrainingExample(audio_path, float(start), float(end), text)
                )
                counts[condition]["utterance_count"] += 1
    _verify_artifacts(artifact_root, expected_snapshots)
    if len(examples) != sum(item["utterance_count"] for item in counts.values()):
        raise AssertionError("training materialization lost an utterance")
    return examples, counts


class WhisperSegmentDataset:
    def __init__(
        self,
        examples: Sequence[TrainingExample],
        processor: object,
        *,
        sample_rate: int,
        max_label_tokens: int,
    ) -> None:
        self._examples = examples
        self._processor = processor
        self._sample_rate = sample_rate
        self._max_label_tokens = max_label_tokens

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        import numpy as np
        from scipy.signal import resample_poly

        example = self._examples[index]
        with wave.open(str(example.audio_path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("training WAV must be mono PCM16")
            source_rate = audio.getframerate()
            start_frame = round(example.start_ms * source_rate / 1000.0)
            end_frame = round(example.end_ms * source_rate / 1000.0)
            if (
                start_frame < 0
                or end_frame <= start_frame
                or end_frame > audio.getnframes()
            ):
                raise ValueError("training segment is outside its WAV bounds")
            audio.setpos(start_frame)
            frames = audio.readframes(end_frame - start_frame)
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if source_rate != self._sample_rate:
            divisor = math.gcd(source_rate, self._sample_rate)
            samples = resample_poly(
                samples,
                self._sample_rate // divisor,
                source_rate // divisor,
            ).astype(np.float32, copy=False)
        features = self._processor.feature_extractor(
            samples,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        ).input_features[0]
        encoded = self._processor.tokenizer(
            example.text,
            add_special_tokens=True,
            truncation=False,
        )
        labels = list(encoded.input_ids)
        if not labels or len(labels) > self._max_label_tokens:
            raise ValueError("training label violates the registered token limit")
        return {"input_features": features, "labels": labels}


class WhisperDataCollator:
    def __init__(self, processor: object, decoder_start_token_id: int) -> None:
        self._processor = processor
        self._decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: Sequence[Mapping[str, object]]) -> dict[str, object]:
        feature_batch = self._processor.feature_extractor.pad(
            [{"input_features": item["input_features"]} for item in features],
            return_tensors="pt",
        )
        label_batch = self._processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features],
            return_tensors="pt",
        )
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        if labels.shape[1] and (labels[:, 0] == self._decoder_start_token_id).all():
            labels = labels[:, 1:]
        feature_batch["labels"] = labels
        return feature_batch


def _hash_tree(root: Path) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return artifacts


def _harden_tree(root: Path) -> None:
    root.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("training output contains an unexpected symlink")
        path.chmod(0o700 if path.is_dir() else 0o600)


def _clean_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in sorted(ALLOWED_TRAIN_METRICS):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                result[key] = round(number, 8)
    return result


def build_whisper_lora_config(
    lora: dict[str, object], factory: Callable[..., object]
) -> object:
    """Use generic PEFT forwarding because Whisper consumes input_features."""

    if lora.get("peft_wrapper") != "generic":
        raise ValueError("Whisper LoRA requires the registered generic PEFT wrapper")
    return factory(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        target_modules=list(lora["target_modules"]),
        task_type=None,
    )


def _install_deadline(seconds: int) -> Callable[[], None]:
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError(
            "this harness requires a process deadline on the GPU runtime"
        )

    def timeout_handler(_: int, __: object) -> None:
        raise TimeoutError("registered LoRA runtime limit reached")

    previous_alarm = signal.signal(signal.SIGALRM, timeout_handler)
    previous_term = signal.signal(signal.SIGTERM, timeout_handler)
    signal.alarm(seconds)

    def restore() -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)

    return restore


def run_bounded_training(
    *,
    execution_config_path: Path,
    experiment_config_path: Path,
    artifact_root: Path,
    cost_quote_path: Path,
    authorization_claim_path: Path,
    output_dir: Path,
    confirmation: str,
    runner_revision: str,
) -> dict[str, object]:
    if confirmation != CONFIRMATION_PHRASE:
        raise PermissionError("explicit bounded-experiment confirmation is required")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite LoRA training output")
    execution, execution_bytes = _load_execution_config(execution_config_path)
    experiment, experiment_bytes = load_experiment_config(experiment_config_path)
    cost = validate_cost_quote(
        quote_path=cost_quote_path,
        execution_config=execution,
        runner_revision=runner_revision,
    )
    claim = validate_authorization_claim(
        claim_path=authorization_claim_path,
        cost=cost,
        runner_revision=runner_revision,
        expected_gcs_prefix=str(execution["private_output"]["gcs_prefix"]),
    )
    runtime_report = validate_gpu_runtime(execution)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    # Keep an unrelated TensorFlow installation out of the ASR-only runtime.
    # This must be set before importing PEFT/Transformers model modules.
    os.environ["USE_TF"] = "0"
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as error:
        raise RuntimeError(
            "pinned LoRA training dependencies are unavailable"
        ) from error

    models = _object(experiment["models"], "experiment models")
    base = _object(models["transformers_base"], "transformers base")
    processor = WhisperProcessor.from_pretrained(
        str(base["id"]),
        revision=str(base["revision"]),
        language=str(experiment["training"]["language"]),
        task=str(experiment["training"]["task"]),
    )
    tokenizer_report = validate_lora_tokenizer_preflight(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
        tokenizer=processor.tokenizer,
        tokenizer_version=distribution_version("transformers"),
    )
    if (
        tokenizer_report["status"] != "limited"
        or not tokenizer_report["training_eligible_by_token_limit"]
    ):
        raise RuntimeError("tokenizer preflight rejected training")
    data_report = validate_lora_data_preflight(
        execution_config_path=execution_config_path,
        experiment_config_path=experiment_config_path,
        artifact_root=artifact_root,
    )
    expected_snapshots = _expected_snapshot_map(data_report)

    output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_dir.parent.chmod(0o700)
    stage = output_dir.parent / f".{output_dir.name}.stage"
    work = output_dir.parent / f".{output_dir.name}.work"
    if any(path.exists() or path.is_symlink() for path in (stage, work)):
        raise FileExistsError("refusing to reuse LoRA private staging paths")
    stage.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    runtime_limit = int(execution["runtime"]["internal_deadline_seconds"])
    remaining_seconds = runtime_limit - math.ceil(time.monotonic() - started_monotonic)
    if remaining_seconds <= 0:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise TimeoutError(
            "registered LoRA runtime limit reached during initialization"
        )
    restore_deadline = _install_deadline(remaining_seconds)
    try:
        audio_root = work / "audio"
        examples, assignment_counts = materialize_training_examples(
            artifact_root=artifact_root,
            output_dir=audio_root,
            execution_config=execution,
            expected_snapshots=expected_snapshots,
        )
        expected_train = data_report["partitions"]["train"]["utterance_count"]
        if len(examples) != expected_train:
            raise ValueError("materialized training count does not match preflight")

        training = _object(experiment["training"], "experiment training")
        segment = _object(execution["segment_contract"], "segment contract")
        dataset = WhisperSegmentDataset(
            examples,
            processor,
            sample_rate=int(segment["resample_hz"]),
            max_label_tokens=int(segment["max_label_tokens"]),
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(base["id"]),
            revision=str(base["revision"]),
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
        lora = _object(experiment["lora"], "LoRA config")
        model = get_peft_model(
            model,
            build_whisper_lora_config(lora, LoraConfig),
        )
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        if trainable <= 0 or trainable >= total:
            raise RuntimeError("LoRA trainable-parameter boundary is invalid")

        trainer_root = work / "trainer"
        arguments = Seq2SeqTrainingArguments(
            output_dir=str(trainer_root),
            overwrite_output_dir=False,
            num_train_epochs=float(training["epochs"]),
            per_device_train_batch_size=int(training["per_device_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]),
            warmup_ratio=float(training["warmup_ratio"]),
            weight_decay=float(training["weight_decay"]),
            max_grad_norm=float(training["max_gradient_norm"]),
            fp16=bool(training["fp16"]),
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(
                    execution["runtime"]["gradient_checkpointing_use_reentrant"]
                )
            },
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=25,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=int(execution["runtime"]["dataloader_num_workers"]),
            dataloader_pin_memory=bool(
                execution["runtime"]["dataloader_pin_memory"]
            ),
            seed=int(training["seed"]),
            data_seed=int(training["seed"]),
            full_determinism=bool(execution["runtime"]["deterministic_algorithms"]),
            tf32=bool(execution["runtime"]["allow_tf32"]),
            predict_with_generate=False,
        )
        torch.use_deterministic_algorithms(True)
        collator = WhisperDataCollator(processor, model.config.decoder_start_token_id)
        trainer = Seq2SeqTrainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=processor,
        )
        result = trainer.train()

        adapter_dir = stage / "adapter"
        processor_dir = stage / "processor"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(processor_dir)
        completed = datetime.now(timezone.utc)
        report = {
            "schema_version": "1.0.0",
            "protocol_id": TRAINING_PROTOCOL_ID,
            "status": "trained_unvalidated",
            "fact_status": "부분 구현 또는 개발용 데모",
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "completed_at": completed.isoformat().replace("+00:00", "Z"),
            "wall_seconds": round(time.monotonic() - started_monotonic, 3),
            "execution_config_sha256": hashlib.sha256(execution_bytes).hexdigest(),
            "experiment_config_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
            "data_run_summary_sha256": data_report["run_summary_sha256"],
            "tokenizer_preflight": {
                "status": tokenizer_report["status"],
                "observed_max_label_tokens": tokenizer_report[
                    "observed_max_label_tokens"
                ],
                "over_limit_count": tokenizer_report["over_limit_count"],
            },
            "runtime": runtime_report,
            "cost_guard": {
                "authorization_id": cost.authorization_id,
                "authorized_speech_revision": cost.authorized_speech_revision,
                "cumulative_development_cost_before_krw": cost.cumulative_development_cost_before_krw,
                "authorization_claim_sha256": claim.claim_sha256,
                "authorization_claim_uri": claim.remote_object_uri,
                "authorization_claimed_at": claim.claimed_at,
                "quote_sha256": cost.quote_sha256,
                "quoted_compute_usd_per_hour": cost.quoted_compute_usd_per_hour,
                "quoted_boot_disk_usd": cost.quoted_boot_disk_usd,
                "quoted_network_transfer_usd": cost.quoted_network_transfer_usd,
                "quoted_total_usd": cost.quoted_total_usd,
                "quoted_total_krw_with_contingency": cost.quoted_total_krw_with_contingency,
                "independent_experiment_ceiling_krw": cost.independent_experiment_ceiling_krw,
                "independent_total_ceiling_krw": cost.independent_total_ceiling_krw,
            },
            "training": {
                "record_and_utterance_counts": assignment_counts,
                "utterance_count": len(examples),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "metrics": _clean_metrics(result.metrics),
            },
            "privacy": {
                "contains_record_ids": False,
                "contains_transcripts": False,
                "contains_addresses": False,
                "private_storage_required": True,
                "git_commit_allowed": False,
            },
            "automatic_adoption_allowed": False,
            "next_gate": "A/B/C conversion and locked dev plus downstream safety evaluation",
            "claim_scope": "adapter trained; no accuracy, safety, field-radio, adoption, or deployment claim",
        }
        report["output_artifacts"] = _hash_tree(stage)
        report_path = stage / "training-report.json"
        _secure_write(
            report_path,
            (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        _harden_tree(stage)
        stage.rename(output_dir)
        return report
    finally:
        restore_deadline()
        shutil.rmtree(work, ignore_errors=True)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cost-quote", type=Path, required=True)
    parser.add_argument("--authorization-claim", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-bounded-experiment", required=True)
    parser.add_argument("--runner-revision", required=True)
    args = parser.parse_args(argv)
    report = run_bounded_training(
        execution_config_path=args.execution_config,
        experiment_config_path=args.experiment_config,
        artifact_root=args.artifact_root,
        cost_quote_path=args.cost_quote,
        authorization_claim_path=args.authorization_claim,
        output_dir=args.output_dir,
        confirmation=args.confirm_bounded_experiment,
        runner_revision=args.runner_revision,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "fact_status": report["fact_status"],
                "automatic_adoption_allowed": report["automatic_adoption_allowed"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
