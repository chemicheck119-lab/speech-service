from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import call, patch

from chemicheck119_speech import lora_data_preflight, lora_training
from chemicheck119_speech.lora_data_preflight import validate_lora_data_preflight
from chemicheck119_speech.lora_training import (
    CONFIRMATION_PHRASE,
    WhisperSegmentDataset,
    materialize_training_examples,
    run_bounded_training,
    validate_authorization_claim,
    validate_cost_quote,
    validate_gpu_runtime,
)
from tests.test_lora_data_preflight import _fixture


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG = ROOT / "config" / "whisper_lora_execution_v1.json"
EXPERIMENT_CONFIG = ROOT / "config" / "whisper_lora_experiment_v1.json"
RUNNER_REVISION = "a" * 40
LORA_NUMERIC_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None for package in ("numpy", "scipy")
)


def _quote(
    root: Path,
    *,
    gpu_hour: float = 0.35,
    hours_old: int = 0,
    authorized_revision: str = RUNNER_REVISION,
    cumulative_before: int = 50_000,
) -> Path:
    generated = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc) - timedelta(
        hours=hours_old
    )
    payload = {
        "schema_version": "1.0.0",
        "protocol_id": "whisper-small-lora-cost-quote-v1",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "expires_at": (generated + timedelta(hours=24))
        .isoformat()
        .replace("+00:00", "Z"),
        "currency": "USD",
        "authorization": {
            "id": "lora-20260907-001",
            "speech_revision": authorized_revision,
            "authorized_run_count": 1,
            "cumulative_development_cost_before_krw": cumulative_before,
            "remote_claim_required": True,
        },
        "resource": {
            "gcp_region": "asia-northeast3",
            "machine_type": "n1-standard-4",
            "gpu_type": "nvidia-tesla-t4",
            "gpu_count": 1,
            "vcpu_count": 4,
            "memory_gib": 15,
            "boot_disk_gib": 100,
            "runtime_hours": 3.0,
        },
        "pricing": {
            "gpu_usd_per_hour": gpu_hour,
            "vcpu_usd_per_hour": 0.031611,
            "memory_gib_usd_per_hour": 0.004237,
            "boot_disk_usd_per_gib_month": 0.13,
            "month_hours": 730,
        },
        "fx_krw_per_usd": 1400,
        "sources": [
            "https://cloud.google.com/compute/gpus-pricing",
            "https://www.google.com/finance/quote/USD-KRW",
        ],
    }
    path = root / "quote.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _claim(root: Path, quote: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "protocol_id": "whisper-small-lora-authorization-claim-v1",
        "authorization_id": "lora-20260907-001",
        "speech_revision": RUNNER_REVISION,
        "cost_quote_sha256": hashlib.sha256(quote.read_bytes()).hexdigest(),
        "claimed_at": "2026-09-07T00:30:00Z",
        "remote_claim_created": True,
        "remote_object_uri": (
            "gs://chemi-check-ml-data-181872008704/experiments/speech/lora/"
            "gwangju-v1/authorizations/lora-20260907-001.claimed.json"
        ),
    }
    path = root / "claim.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(_: int) -> str:
        return "Tesla T4"


class FakeTorch:
    __version__ = "2.9.1+cu129"
    version = SimpleNamespace(cuda="12.9")
    cuda = FakeCuda()


class FakeFeatureResult:
    def __init__(self, values: list[float]) -> None:
        self.input_features = [values]


class FakeEncoding:
    input_ids = [1, 2, 3]


class FakeProcessor:
    def __init__(self) -> None:
        self.seen_samples = 0
        self.feature_extractor = self
        self.tokenizer = self

    def __call__(self, value: object, **kwargs: object) -> object:
        if isinstance(value, str):
            return FakeEncoding()
        self.seen_samples = len(value)  # type: ignore[arg-type]
        self.sampling_rate = kwargs["sampling_rate"]
        return FakeFeatureResult([0.0])


class LoraTrainingTest(unittest.TestCase):
    def test_cost_quote_is_below_both_independent_caps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            quote = _quote(Path(directory))
            execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
            decision = validate_cost_quote(
                quote_path=quote,
                execution_config=execution,
                runner_revision=RUNNER_REVISION,
                now=datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
            )
            quote_sha256 = hashlib.sha256(quote.read_bytes()).hexdigest()
        self.assertLess(decision.quoted_total_krw_with_contingency, 20_000)
        self.assertEqual(8_500, decision.independent_experiment_ceiling_krw)
        self.assertEqual(58_500, decision.independent_total_ceiling_krw)
        self.assertEqual(quote_sha256, decision.quote_sha256)

    def test_cost_quote_rejects_expiry_and_compute_ceiling(self) -> None:
        execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
        for gpu_hour, now, message in (
            (0.35, datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc), "current"),
            (1.1, datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc), "compute"),
        ):
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                quote = _quote(Path(directory), gpu_hour=gpu_hour)
                with self.assertRaisesRegex(ValueError, message):
                    validate_cost_quote(
                        quote_path=quote,
                        execution_config=execution,
                        runner_revision=RUNNER_REVISION,
                        now=now,
                    )

    def test_cost_quote_rejects_revision_mismatch_and_understated_prior_cost(
        self,
    ) -> None:
        execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
        for options, message in (
            ({"authorized_revision": "b" * 40}, "single-use"),
            ({"cumulative_before": 49_999}, "understates"),
        ):
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as directory,
            ):
                quote = _quote(Path(directory), **options)
                with self.assertRaisesRegex(ValueError, message):
                    validate_cost_quote(
                        quote_path=quote,
                        execution_config=execution,
                        runner_revision=RUNNER_REVISION,
                        now=datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
                    )

    def test_authorization_claim_binds_quote_revision_and_remote_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quote = _quote(root)
            execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
            cost = validate_cost_quote(
                quote_path=quote,
                execution_config=execution,
                runner_revision=RUNNER_REVISION,
                now=datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
            )
            claim_path = _claim(root, quote)
            claim = validate_authorization_claim(
                claim_path=claim_path,
                cost=cost,
                runner_revision=RUNNER_REVISION,
                expected_gcs_prefix=(
                    "gs://chemi-check-ml-data-181872008704/experiments/speech/lora/"
                    "gwangju-v1"
                ),
                now=datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
            )
            payload = json.loads(claim_path.read_text())
            payload["cost_quote_sha256"] = "0" * 64
            claim_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "single-use"):
                validate_authorization_claim(
                    claim_path=claim_path,
                    cost=cost,
                    runner_revision=RUNNER_REVISION,
                    expected_gcs_prefix=(
                        "gs://chemi-check-ml-data-181872008704/experiments/speech/"
                        "lora/gwangju-v1"
                    ),
                    now=datetime(2026, 9, 7, 1, 0, tzinfo=timezone.utc),
                )
        self.assertTrue(claim.remote_object_uri.endswith(".claimed.json"))

    def test_runtime_accepts_only_pinned_single_t4(self) -> None:
        execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
        packages = execution["runtime"]["packages"]
        versions = {
            "transformers": packages["transformers"],
            "peft": packages["peft"],
            "accelerate": packages["accelerate"],
            "numpy": packages["numpy"],
            "scipy": packages["scipy"],
        }
        with patch.object(
            lora_training.sys,
            "version_info",
            SimpleNamespace(major=3, minor=12),
        ):
            report = validate_gpu_runtime(
                execution,
                torch_module=FakeTorch(),
                installed_version=versions.__getitem__,
            )
        self.assertEqual("Tesla T4", report["gpu_name"])
        self.assertEqual(1, report["gpu_count"])

        class NoCudaTorch(FakeTorch):
            cuda = SimpleNamespace(is_available=lambda: False)

        with (
            patch.object(
                lora_training.sys,
                "version_info",
                SimpleNamespace(major=3, minor=12),
            ),
            self.assertRaisesRegex(RuntimeError, "CPU training fallback"),
        ):
            validate_gpu_runtime(
                execution,
                torch_module=NoCudaTorch(),
                installed_version=versions.__getitem__,
            )

    def test_materializes_each_selected_call_once_and_keeps_ids_out_of_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, config_sha256 = _fixture(root)
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                config_sha256,
            ):
                report = validate_lora_data_preflight(**inputs)
            execution = json.loads(inputs["execution_config_path"].read_text())
            expected = {
                item["file"]: item["sha256"] for item in report["artifact_snapshots"]
            }
            examples, counts = materialize_training_examples(
                artifact_root=inputs["artifact_root"],
                output_dir=root / "private-work",
                execution_config=execution,
                expected_snapshots=expected,
            )
            audio_files = list((root / "private-work").glob("*.wav"))
        self.assertEqual(4, len(examples))
        self.assertEqual(4, len(audio_files))
        self.assertEqual(4, sum(item["record_count"] for item in counts.values()))
        self.assertNotIn("train-", json.dumps(counts))

    @unittest.skipUnless(
        LORA_NUMERIC_RUNTIME_AVAILABLE,
        "LoRA optional NumPy/SciPy dependencies are not installed",
    )
    def test_segment_dataset_resamples_8khz_to_16khz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, config_sha256 = _fixture(root)
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                config_sha256,
            ):
                report = validate_lora_data_preflight(**inputs)
            execution = json.loads(inputs["execution_config_path"].read_text())
            expected = {
                item["file"]: item["sha256"] for item in report["artifact_snapshots"]
            }
            examples, _ = materialize_training_examples(
                artifact_root=inputs["artifact_root"],
                output_dir=root / "private-work",
                execution_config=execution,
                expected_snapshots=expected,
            )
            processor = FakeProcessor()
            dataset = WhisperSegmentDataset(
                examples,
                processor,
                sample_rate=16_000,
                max_label_tokens=160,
            )
            item = dataset[0]
        self.assertEqual([1, 2, 3], item["labels"])
        self.assertEqual(14_400, processor.seen_samples)
        self.assertEqual(16_000, processor.sampling_rate)

    def test_training_requires_exact_confirmation_before_any_gpu_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(PermissionError, "confirmation"):
                run_bounded_training(
                    execution_config_path=EXECUTION_CONFIG,
                    experiment_config_path=EXPERIMENT_CONFIG,
                    artifact_root=root,
                    cost_quote_path=root / "missing.json",
                    authorization_claim_path=root / "missing-claim.json",
                    output_dir=root / "output",
                    confirmation=CONFIRMATION_PHRASE.lower(),
                    runner_revision=RUNNER_REVISION,
                )

    def test_metric_report_drops_unregistered_or_nonfinite_values(self) -> None:
        cleaned = lora_training._clean_metrics(
            {
                "train_loss": 1.25,
                "train_runtime": 3,
                "eval_private_text": "절대 기록하지 않음",
                "epoch": float("nan"),
            }
        )
        self.assertEqual({"train_loss": 1.25, "train_runtime": 3.0}, cleaned)
        self.assertNotIn("절대 기록하지 않음", json.dumps(cleaned, ensure_ascii=False))

    def test_deadline_handles_term_and_alarm_then_restores_both(self) -> None:
        with (
            patch.object(
                lora_training.signal, "signal", return_value="previous"
            ) as set_signal,
            patch.object(lora_training.signal, "alarm") as alarm,
        ):
            restore = lora_training._install_deadline(9600)
            restore()
        registered = [call.args[0] for call in set_signal.call_args_list[:2]]
        restored = [call.args[0] for call in set_signal.call_args_list[2:]]
        self.assertEqual(
            [lora_training.signal.SIGALRM, lora_training.signal.SIGTERM], registered
        )
        self.assertEqual(registered, restored)
        self.assertEqual([call(9600), call(0)], alarm.call_args_list)


if __name__ == "__main__":
    unittest.main()
