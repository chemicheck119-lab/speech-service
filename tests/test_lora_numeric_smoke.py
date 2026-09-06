from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from chemicheck119_speech.lora_numeric_smoke import run_numeric_smoke


class LoraNumericSmokeTest(unittest.TestCase):
    def test_requires_confirmation_before_reading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(PermissionError, "confirmation"):
                run_numeric_smoke(
                    execution_config_path=root / "missing-execution.json",
                    experiment_config_path=root / "missing-experiment.json",
                    artifact_root=root / "missing-artifacts",
                    cost_quote_path=root / "missing-quote.json",
                    authorization_claim_path=root / "missing-claim.json",
                    output_dir=root / "output",
                    confirmation="NO",
                    runner_revision="a" * 40,
                )


if __name__ == "__main__":
    unittest.main()
