from pathlib import Path
import json
import shutil
import subprocess
import unittest

from mikazuki.training_schema_overrides import (
    fixed_flux_family_schema,
    fixed_sd_schema,
    override_raw_schema,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "mikazuki" / "schema"


def assert_js_expression_compiles(testcase: unittest.TestCase, source: str, label: str) -> None:
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node is required for transformed schema syntax checks")
    # new Function compiles without evaluating the expression, so runtime globals
    # such as Schema / SHARED_SCHEMAS / UpdateSchema need not be stubbed here.
    body = "return (\n" + source + "\n);"
    completed = subprocess.run(
        [node, "-e", f"new Function({json.dumps(body)});"],
        capture_output=True,
        text=True,
    )
    testcase.assertEqual(
        completed.returncode,
        0,
        f"{label} transformed schema is not valid JavaScript:\n{completed.stderr}",
    )


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

    def test_anima_full_uses_same_mutually_exclusive_memory_mode(self):
        source = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        fixed = fixed_flux_family_schema(source, "anima", "anima-finetune", "finetune")
        self.assertIn('memory_mode: Schema.union(["auto", "lowram", "highvram"])', fixed)
        self.assertNotIn(
            'highvram: Schema.boolean().default(false).description("启用 sd-scripts High VRAM 模式',
            fixed,
        )

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

    def test_all_transformed_training_schemas_remain_valid_javascript(self):
        lora_master = (SCHEMA / "lora-master.ts").read_text(encoding="utf-8")
        dreambooth = (SCHEMA / "dreambooth.ts").read_text(encoding="utf-8")
        flux_lora = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        transformed = {
            "sd-lora": fixed_sd_schema(lora_master, "sd-lora"),
            "sdxl-lora": fixed_sd_schema(lora_master, "sdxl-lora"),
            "sd-dreambooth": fixed_sd_schema(dreambooth, "sd-dreambooth"),
            "flux-lora": fixed_flux_family_schema(flux_lora, "flux", "flux-lora"),
            "chroma-lora": fixed_flux_family_schema(flux_lora, "chroma", "chroma-lora"),
            "anima-lora": fixed_flux_family_schema(flux_lora, "anima", "anima-lora", "lora"),
            "anima-finetune": fixed_flux_family_schema(flux_lora, "anima", "anima-finetune", "finetune"),
            "flux-finetune": override_raw_schema(
                "flux-finetune",
                (SCHEMA / "flux-finetune.ts").read_text(encoding="utf-8"),
            ),
            "sdxl-finetune": override_raw_schema(
                "sdxl-full",
                (SCHEMA / "sdxl-full.ts").read_text(encoding="utf-8"),
            ),
        }
        for label, source in transformed.items():
            with self.subTest(label=label):
                assert_js_expression_compiles(self, source, label)


if __name__ == "__main__":
    unittest.main()
