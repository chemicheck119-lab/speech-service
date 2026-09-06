from __future__ import annotations

import unittest

from chemicheck119_speech.api import create_app


class SpeechOpenApiTest(unittest.TestCase):
    def test_transcription_contract_declares_binary_audio_and_safety_boundary(
        self,
    ) -> None:
        schema = create_app().openapi()
        operation = schema["paths"]["/api/v1/transcriptions"]["post"]
        audio = operation["requestBody"]["content"]["audio/wav"]["schema"]
        self.assertEqual("string", audio["type"])
        self.assertEqual("binary", audio["format"])
        response_ref = operation["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        self.assertEqual("#/components/schemas/TranscriptionResponse", response_ref)
        self.assertEqual([{"APIKeyHeader": []}], operation["security"])
        security_scheme = schema["components"]["securitySchemes"]["APIKeyHeader"]
        self.assertEqual("apiKey", security_scheme["type"])
        self.assertEqual("X-API-Key", security_scheme["name"])
        self.assertEqual("header", security_scheme["in"])

        boundary = schema["components"]["schemas"]["SafetyBoundaryResponse"]
        required = set(boundary["required"])
        self.assertIn("chemical_identification_performed", required)
        self.assertIn("cas_confirmation_performed", required)
        self.assertIn("risk_assessment_performed", required)
        self.assertEqual(
            False,
            boundary["properties"]["cas_confirmation_performed"]["const"],
        )


if __name__ == "__main__":
    unittest.main()
