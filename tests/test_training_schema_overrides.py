from pathlib import Path
import unittest

from mikazuki.training_schema_overrides import (
    fixed_flux_family_schema,
    fixed_sd_schema,
    override_raw_schema,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "mikazuki" / "schema"


class TrainingSchemaOverrideTests(unittest.TestCase):
    def test_dreambooth_uses_real_save_formats_token_mode_and_shared_logging(self):
        source = (SCHEMA / "dreambooth.ts").read_text(encoding="utf-8")
        fixed = fixed_sd_schema(source, "sd-dreambooth")
        self.assertNotIn('"pt", "ckpt"', fixed)
        self.assertIn('"diffusers", "diffusers_safetensors"', fixed)
        self.assertIn("sd_max_token_length_mode", fixed)
        self.assertNotIn("max_token_length: Schema.number().default(255)", fixed)
        self.assertIn("SHARED_SCHEMAS.LOG_SETTINGS", fixed)
        self.assertNotIn('log_with: Schema.union(["tensorboard", "wandb"])', fixed)
        self.assertIn("memory_mode", fixed)

    def test_sdxl_lora_does_not_render_clip_skip(self):
        source = (SCHEMA / "lora-master.ts").read_text(encoding="utf-8")
        fixed = fixed_sd_schema(source, "sdxl-lora")
        self.assertNotIn("SHARED_SCHEMAS.OTHER,", fixed)
        self.assertNotIn("clip_skip", fixed)
        self.assertIn("memory_mode", fixed)

    def test_flux_and_chroma_lora_gain_flux_shift_and_memory_mode(self):
        source = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        fixed = fixed_flux_family_schema(source, "flux", "flux-lora")
        self.assertIn('"shift", "flux_shift"', fixed)
        self.assertIn("memory_mode", fixed)

    def test_anima_lora_target_is_not_locked_to_dit(self):
        source = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        fixed = fixed_flux_family_schema(source, "anima", "anima-lora", "lora")
        self.assertIn("anima_lora_target", fixed)
        self.assertIn('Schema.union(["dit", "qwen3", "dit_qwen3"])', fixed)
        self.assertNotIn("固定为仅训练 Anima DiT LoRA", fixed)
        self.assertIn("memory_mode", fixed)

    def test_sdxl_full_removes_fake_vpred_controls_and_adds_compile(self):
        source = (SCHEMA / "sdxl-full.ts").read_text(encoding="utf-8")
        fixed = override_raw_schema("sdxl-full", source)
        self.assertNotIn("v_parameterization: Schema.boolean", fixed)
        self.assertNotIn("scale_v_pred_loss_like_noise_pred: Schema.boolean", fixed)
        self.assertIn("torch_compile", fixed)
        self.assertIn("dynamo_backend", fixed)
        self.assertIn("memory_mode", fixed)

    def test_flux_full_adds_compile_without_inventing_backend_enum(self):
        source = (SCHEMA / "flux-finetune.ts").read_text(encoding="utf-8")
        fixed = override_raw_schema("flux-finetune", source)
        self.assertIn("torch_compile", fixed)
        self.assertIn('dynamo_backend: Schema.string().default("inductor")', fixed)
        self.assertIn("memory_mode", fixed)


if __name__ == "__main__":
    unittest.main()
