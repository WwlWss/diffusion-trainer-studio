import unittest
from pathlib import Path

from mikazuki.multi_caption_config import build_multi_caption_sidecar


class MultiCaptionRequestContractTests(unittest.TestCase):
    def test_standard_is_a_true_noop_for_trainer_fields(self):
        config = {
            "caption_mode": "standard",
            "multi_caption_storage": "files",
            "multi_caption_file_groups": {"stale": {"extension": ".txt", "weight": 1}},
            "caption_extension": ".txt",
            "shuffle_caption": True,
        }
        path, sidecars, policy = build_multi_caption_sidecar(config, "lora-master")
        self.assertIsNone(path)
        self.assertEqual(sidecars, {})
        self.assertIsNone(policy)
        self.assertEqual(config, {"caption_extension": ".txt", "shuffle_caption": True})

    def test_all_supported_training_schema_sources_expose_multi_caption(self):
        root = Path(__file__).resolve().parents[1] / "mikazuki" / "schema"
        expectations = {
            "lora-basic.ts": "CAPTION_MODE_BASIC",
            "lora-master.ts": "CAPTION_MODE_SHARED",
            "dreambooth.ts": "CAPTION_MODE_SHARED",
            "sdxl-full.ts": "CAPTION_MODE_SHARED",
            "flux-finetune.ts": "CAPTION_MODE_SHARED",
            "sd3-lora.ts": "CAPTION_MODE_SHARED",
            "flux-lora.ts": "CAPTION_MODE_ANIMA",
        }
        for filename, needle in expectations.items():
            with self.subTest(filename=filename):
                self.assertIn(needle, (root / filename).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
