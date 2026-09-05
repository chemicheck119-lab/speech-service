import unittest

from chemicheck119_speech.storage import split_gcs_uri


class StorageTest(unittest.TestCase):
    def test_splits_gcs_uri(self) -> None:
        self.assertEqual(
            ("private-bucket", "raw/audio.zip"),
            split_gcs_uri("gs://private-bucket/raw/audio.zip"),
        )

    def test_rejects_non_gcs_uri(self) -> None:
        with self.assertRaises(ValueError):
            split_gcs_uri("https://example.invalid/audio.zip")


if __name__ == "__main__":
    unittest.main()
