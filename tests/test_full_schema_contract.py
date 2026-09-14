from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FLUX = (ROOT / "mikazuki/schema/flux-finetune.ts").read_text(encoding="utf-8")
SDXL = (ROOT / "mikazuki/schema/sdxl-full.ts").read_text(encoding="utf-8")


class FullSchemaContractTests(unittest.TestCase):
    def test_flux_exposes_working_advanced_paths_only(self):
        for field in ("fused_backward_pass", "blockwise_fused_optimizers", "deepspeed"):
            self.assertIn(f"{field}:", FLUX)
        self.assertNotIn("fused_optimizer_groups:", FLUX)

    def test_sdxl_text_encoder_cache_defaults_are_safe(self):
        self.assertIn("cache_text_encoder_outputs: Schema.boolean().default(false)", SDXL)
        self.assertIn("cache_text_encoder_outputs_to_disk: Schema.boolean().default(false)", SDXL)

    def test_both_full_pages_document_step_precedence(self):
        self.assertIn("后端会移除默认 max_train_epochs", FLUX)
        self.assertIn("后端会移除默认 max_train_epochs", SDXL)


if __name__ == "__main__":
    unittest.main()
