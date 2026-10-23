"""Regression checks for adopting an already downloaded verified checkpoint.

Uses a tiny temporary fixture with patched expected size and digest. It never
downloads model weights or allocates a model, and leaves the real ckpt untouched.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import download_weights as downloader


class ExistingCheckpointTests(unittest.TestCase):
    payload = b"small official-checkpoint stand-in\x00\x01"

    @contextmanager
    def fixture_directory(self):
        # Keep transient files inside the permitted workspace on Windows.
        parent = Path(__file__).resolve().parent
        with TemporaryDirectory(prefix="weights-test-", dir=parent) as directory:
            self.assertEqual(Path(directory).resolve().parent, parent)
            yield directory

    def prepare(self, directory, *, valid_digest=True, with_config=True):
        root = Path(directory)
        final = root / "model.safetensors"
        final.write_bytes(self.payload)
        if with_config:
            (root / "config.json").write_text("{}", encoding="utf-8")
        stack = ExitStack()
        stack.enter_context(patch.object(downloader, "CKPT", root))
        stack.enter_context(patch.object(downloader, "SIZE", len(self.payload)))
        expected = hashlib.sha256(self.payload).hexdigest() if valid_digest else "0" * 64
        stack.enter_context(patch.object(downloader, "SHA256", expected))
        # The existing-file path must never contact the model download endpoint.
        stack.enter_context(patch.object(downloader.urllib.request, "urlopen",
                                        side_effect=AssertionError("Unexpected model download")))
        return root, final, stack

    def assert_usable_ledger(self, root):
        ledger = json.loads((root / "download_ranges.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["revision"], downloader.REVISION)
        self.assertEqual(ledger["size"], len(self.payload))
        self.assertTrue(ledger["ranges"])
        # This is the readiness condition consumed by run_low_memory.py.
        self.assertTrue(all(item["next"] == item["end"] for item in ledger["ranges"]))
        self.assertEqual(ledger["prefix"] + sum(item["end"] - item["start"]
                                              for item in ledger["ranges"]), len(self.payload))

    def test_verified_file_without_ledger_becomes_inference_ready(self):
        with self.fixture_directory() as directory:
            root, final, stack = self.prepare(directory)
            with stack, patch.object(downloader.urllib.request, "urlretrieve") as config_download:
                self.assertFalse((root / "download_ranges.json").exists())
                downloader.main()
                self.assert_usable_ledger(root)
                self.assertEqual(final.read_bytes(), self.payload)
                config_download.assert_not_called()

    def test_verified_file_can_retrieve_only_missing_config(self):
        with self.fixture_directory() as directory:
            root, final, stack = self.prepare(directory, with_config=False)

            def write_config(url, target):
                Path(target).write_text("{}", encoding="utf-8")

            with stack, patch.object(downloader.urllib.request, "urlretrieve",
                                     side_effect=write_config) as config_download:
                downloader.main()
                self.assert_usable_ledger(root)
                config_download.assert_called_once_with(
                    f"{downloader.BASE}/config.json", root / "config.json")
                self.assertEqual(final.read_bytes(), self.payload)

    def test_wrong_digest_never_marks_file_complete(self):
        with self.fixture_directory() as directory:
            root, final, stack = self.prepare(directory, valid_digest=False, with_config=False)
            with stack, patch.object(downloader.urllib.request, "urlretrieve") as config_download:
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    downloader.main()
                self.assertFalse((root / "download_ranges.json").exists())
                self.assertEqual(final.read_bytes(), self.payload)
                config_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
