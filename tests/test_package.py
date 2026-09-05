import unittest

import chemicheck119_speech


class PackageTest(unittest.TestCase):
    def test_version_is_declared(self) -> None:
        self.assertEqual("0.1.0", chemicheck119_speech.__version__)


if __name__ == "__main__":
    unittest.main()
