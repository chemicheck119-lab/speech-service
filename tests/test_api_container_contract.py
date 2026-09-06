from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SpeechApiContainerContractTest(unittest.TestCase):
    def test_runtime_is_non_root_offline_and_api_only(self) -> None:
        dockerfile = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")

        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn('ENTRYPOINT ["chemicheck119-speech-api"]', dockerfile)
        self.assertIn("CHEMICHECK119_SPEECH_LOCAL_FILES_ONLY=true", dockerfile)
        self.assertIn("CHEMICHECK119_SPEECH_DEVICE=cpu", dockerfile)
        self.assertIn("CHEMICHECK119_SPEECH_COMPUTE_TYPE=int8", dockerfile)
        self.assertNotIn("CHEMICHECK119_SPEECH_API_KEY=", dockerfile)
        self.assertNotIn("ALLOW_ANONYMOUS=true", dockerfile)

    def test_build_context_cannot_include_data_audio_or_model_weights(self) -> None:
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

        self.assertEqual("*", dockerignore[0])
        self.assertEqual(
            {
                "!pyproject.toml",
                "!README.md",
                "!src/",
                "!src/**",
                "!config/",
                "!config/**",
            },
            set(dockerignore[1:]),
        )

    def test_smoke_uses_model_absence_as_fail_closed_readiness_check(self) -> None:
        smoke = (ROOT / "scripts/smoke_speech_api_container.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("EMBED_WHISPER_MODEL=false", smoke)
        self.assertIn("--read-only", smoke)
        self.assertIn('[[ "${ready_status}" == "503" ]]', smoke)
        self.assertIn('[[ "${unauthorized_status}" == "401" ]]', smoke)
        self.assertNotIn("ALLOW_ANONYMOUS", smoke)


if __name__ == "__main__":
    unittest.main()
