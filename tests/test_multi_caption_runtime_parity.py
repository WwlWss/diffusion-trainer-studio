import unittest
from pathlib import Path


class MultiCaptionRuntimeParityTests(unittest.TestCase):
    def test_stable_and_dev_multi_caption_core_are_byte_identical(self):
        root = Path(__file__).resolve().parents[1] / "scripts"
        stable = (root / "stable" / "library" / "multi_caption.py").read_bytes()
        dev = (root / "dev" / "library" / "multi_caption.py").read_bytes()
        self.assertEqual(stable, dev)

    def test_dev_common_network_trainer_and_full_trainer_are_wired(self):
        root = Path(__file__).resolve().parents[1] / "scripts" / "dev"
        for filename in ("train_network.py", "flux_train.py"):
            with self.subTest(filename=filename):
                text = (root / filename).read_text(encoding="utf-8")
                self.assertIn("configure_multi_caption_dataset_groups", text)
                self.assertIn("args.multi_caption_config", text)


if __name__ == "__main__":
    unittest.main()
