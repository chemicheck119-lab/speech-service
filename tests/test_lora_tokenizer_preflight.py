from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chemicheck119_speech import lora_data_preflight
from chemicheck119_speech.lora_data_preflight import _load_execution_config
from chemicheck119_speech.lora_tokenizer_preflight import (
    validate_lora_tokenizer_preflight,
)
from tests.test_lora_data_preflight import _fixture


class FakeTokenizer:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if not text:
            raise AssertionError("fixture text must not be empty")
        return {"input_ids": list(range(self.token_count))}


class LoraTokenizerPreflightTest(unittest.TestCase):
    def test_registered_transformers_5_execution_config(self) -> None:
        root = Path(__file__).resolve().parents[1]

        execution, _ = _load_execution_config(
            root / "config" / "whisper_lora_execution_v2.json"
        )

        self.assertEqual(
            "whisper-small-lora-gwangju-execution-v2",
            execution["protocol_id"],
        )
        self.assertEqual(
            "5.10.1", execution["runtime"]["packages"]["transformers"]
        )

    def test_passes_token_limit_without_exposing_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, execution_sha256 = _fixture(Path(directory))
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                execution_sha256,
            ):
                report = validate_lora_tokenizer_preflight(
                    **inputs,
                    tokenizer=FakeTokenizer(12),
                    tokenizer_version="4.57.6",
                    generated_at="2026-09-06T00:00:00Z",
                )
        self.assertEqual("limited", report["status"])
        self.assertEqual(12, report["observed_max_label_tokens"])
        self.assertEqual(0, report["over_limit_count"])
        self.assertTrue(report["training_eligible_by_token_limit"])
        self.assertIs(report["automatic_training_allowed"], False)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("연기 테스트 문장", serialized)
        self.assertNotIn("train-0", serialized)

    def test_rejects_over_limit_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, execution_sha256 = _fixture(Path(directory))
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                execution_sha256,
            ):
                report = validate_lora_tokenizer_preflight(
                    **inputs,
                    tokenizer=FakeTokenizer(161),
                    tokenizer_version="4.57.6",
                )
        self.assertEqual("rejected", report["status"])
        self.assertGreater(report["over_limit_count"], 0)
        self.assertFalse(report["training_eligible_by_token_limit"])

    def test_rejects_unregistered_tokenizer_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, execution_sha256 = _fixture(Path(directory))
            with (
                patch.object(
                    lora_data_preflight,
                    "REGISTERED_EXECUTION_CONFIG_SHA256",
                    execution_sha256,
                ),
                self.assertRaisesRegex(RuntimeError, "version"),
            ):
                validate_lora_tokenizer_preflight(
                    **inputs,
                    tokenizer=FakeTokenizer(12),
                    tokenizer_version="5.16.1",
                )

    def test_uses_transformers_version_from_execution_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, _ = _fixture(Path(directory))
            execution_path = inputs["execution_config_path"]
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["runtime"]["packages"]["transformers"] = "5.10.1"
            execution_path.write_text(json.dumps(execution), encoding="utf-8")
            execution_sha256 = lora_data_preflight.hashlib.sha256(
                execution_path.read_bytes()
            ).hexdigest()
            with patch.object(
                lora_data_preflight,
                "REGISTERED_EXECUTION_CONFIG_SHA256",
                execution_sha256,
            ):
                report = validate_lora_tokenizer_preflight(
                    **inputs,
                    tokenizer=FakeTokenizer(12),
                    tokenizer_version="5.10.1",
                )

        self.assertEqual(
            "5.10.1", report["tokenizer"]["transformers_version"]
        )


if __name__ == "__main__":
    unittest.main()
