from pathlib import Path
import json
import shutil
import subprocess
import unittest

from mikazuki.parameter_policy_schema import PARAMETER_POLICY_EDITOR_MARKER
from mikazuki.training_schema_overrides import fixed_flux_family_schema, fixed_sd_schema, override_raw_schema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "mikazuki" / "schema"


def assert_js_expression_compiles(testcase: unittest.TestCase, source: str, label: str) -> None:
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node is required for transformed schema syntax checks")
    expression = source.strip()
    if expression.endswith(";"):
        expression = expression[:-1].rstrip()
    body = "return (\n" + expression + "\n);"
    completed = subprocess.run([node, "-e", f"new Function({json.dumps(body)});"], capture_output=True, text=True)
    testcase.assertEqual(completed.returncode, 0, f"{label} transformed schema is not valid JavaScript:\n{completed.stderr}")


class TrainingSchemaOverrideTests(unittest.TestCase):
    def test_shared_wandb_api_key_is_optional(self):
        source = (SCHEMA / "shared.ts").read_text(encoding="utf-8")
        fixed = override_raw_schema("shared", source)
        self.assertIn("wandb_api_key: Schema.string().description", fixed)
        self.assertNotIn("wandb_api_key: Schema.string().required()", fixed)

    def test_dreambooth_uses_real_save_formats_token_mode_and_shared_logging(self):
        source = (SCHEMA / "dreambooth.ts").read_text(encoding="utf-8")
        fixed = fixed_sd_schema(source, "sd-dreambooth")
        self.assertNotIn('"pt", "ckpt"', fixed)
        self.assertIn('"diffusers", "diffusers_safetensors"', fixed)
        self.assertIn("sd_max_token_length_mode", fixed)
        self.assertNotIn("max_token_length: Schema.number().default(255)", fixed)
        self.assertIn("SHARED_SCHEMAS.LOG_SETTINGS", fixed)
        self.assertIn("dataset_source", fixed)
        self.assertIn("dataset_config", fixed)
        self.assertIn("memory_mode", fixed)

    def test_sd_lora_uses_semantic_token_target_and_dataset_source(self):
        source = (SCHEMA / "lora-master.ts").read_text(encoding="utf-8")
        fixed = fixed_sd_schema(source, "sd-lora")
        self.assertIn("sd_max_token_length_mode", fixed)
        self.assertIn("lora_target", fixed)
        self.assertNotIn("network_train_unet_only: Schema.boolean", fixed)
        self.assertNotIn("network_train_text_encoder_only: Schema.boolean", fixed)
        self.assertIn("dataset_source", fixed)
        self.assertIn("dataset_config", fixed)

    def test_sdxl_lora_defaults_and_no_clip_skip(self):
        source = (SCHEMA / "lora-master.ts").read_text(encoding="utf-8")
        fixed = fixed_sd_schema(source, "sdxl-lora")
        self.assertNotIn("SHARED_SCHEMAS.OTHER,", fixed)
        self.assertNotIn("clip_skip", fixed)
        self.assertIn('resolution: Schema.string().default("1024,1024")', fixed)
        self.assertIn("max_bucket_reso: Schema.number().default(2048)", fixed)
        self.assertIn("bucket_reso_steps: Schema.number().default(32)", fixed)
        self.assertIn("memory_mode", fixed)

    def test_sd3_lora_uses_single_semantic_target_control(self):
        source = (SCHEMA / "sd3-lora.ts").read_text(encoding="utf-8")
        fixed = override_raw_schema("sd3-lora", source)
        self.assertIn("sd3_lora_target", fixed)
        self.assertIn('Schema.union(["mmdit", "text_encoder", "mmdit_text_encoder"])', fixed)
        self.assertNotIn("network_train_unet_only: Schema.boolean", fixed)
        self.assertNotIn("network_train_text_encoder_only: Schema.boolean", fixed)
        self.assertIn("train_t5xxl", fixed)

    def test_flux_and_chroma_lora_gain_semantic_target_dataset_and_memory(self):
        source = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        flux = fixed_flux_family_schema(source, "flux", "flux-lora")
        chroma = fixed_flux_family_schema(source, "chroma", "chroma-lora")
        self.assertIn('"shift", "flux_shift"', flux)
        self.assertIn('Schema.union(["dit", "dit_clip_l", "dit_clip_l_t5xxl"])', flux)
        self.assertIn('Schema.union(["dit", "dit_t5xxl"])', chroma)
        self.assertIn("dataset_config", flux)
        self.assertIn("memory_mode", flux)

    def test_anima_lora_target_is_not_locked_to_dit(self):
        source = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        fixed = fixed_flux_family_schema(source, "anima", "anima-lora", "lora")
        self.assertIn("anima_lora_target", fixed)
        self.assertIn('Schema.union(["dit", "qwen3", "dit_qwen3"])', fixed)
        self.assertNotIn("固定为仅训练 Anima DiT LoRA", fixed)
        self.assertIn("memory_mode", fixed)

    def test_anima_full_removes_unimplemented_lowram_option(self):
        source = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        fixed = fixed_flux_family_schema(source, "anima", "anima-finetune", "finetune")
        self.assertIn('memory_mode: Schema.union(["auto", "highvram"])', fixed)
        self.assertNotIn('memory_mode: Schema.union(["auto", "lowram", "highvram"])', fixed)
        self.assertNotIn('highvram: Schema.boolean().default(false).description("启用 sd-scripts High VRAM 模式', fixed)

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

    def test_release_pages_receive_exactly_one_parameter_policy_editor(self):
        lora_master = (SCHEMA / "lora-master.ts").read_text(encoding="utf-8")
        dreambooth = (SCHEMA / "dreambooth.ts").read_text(encoding="utf-8")
        flux_lora = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        final = {
            "lora-master": override_raw_schema(
                "lora-master",
                fixed_sd_schema(lora_master, "sd-lora"),
            ),
            "sdxl-lora": override_raw_schema(
                "sdxl-lora",
                fixed_sd_schema(lora_master, "sdxl-lora"),
            ),
            "dreambooth": override_raw_schema(
                "dreambooth",
                fixed_sd_schema(dreambooth, "sd-dreambooth"),
            ),
            "sdxl-finetune": override_raw_schema(
                "sdxl-finetune",
                fixed_sd_schema(dreambooth, "sdxl-finetune"),
            ),
            "flux-lora": override_raw_schema(
                "flux-lora",
                fixed_flux_family_schema(flux_lora, "flux", "flux-lora"),
            ),
            "chroma-lora": override_raw_schema(
                "chroma-lora",
                fixed_flux_family_schema(flux_lora, "chroma", "chroma-lora"),
            ),
            "anima-lora": override_raw_schema(
                "anima-lora",
                fixed_flux_family_schema(flux_lora, "anima", "anima-lora", "lora"),
            ),
            "anima-finetune": override_raw_schema(
                "anima-finetune",
                fixed_flux_family_schema(flux_lora, "anima", "anima-finetune", "finetune"),
            ),
            "sd3-lora": override_raw_schema(
                "sd3-lora",
                (SCHEMA / "sd3-lora.ts").read_text(encoding="utf-8"),
            ),
            "flux-finetune": override_raw_schema(
                "flux-finetune",
                (SCHEMA / "flux-finetune.ts").read_text(encoding="utf-8"),
            ),
            "sdxl-full": override_raw_schema(
                "sdxl-full",
                (SCHEMA / "sdxl-full.ts").read_text(encoding="utf-8"),
            ),
            "lora-basic": override_raw_schema(
                "lora-basic",
                (SCHEMA / "lora-basic.ts").read_text(encoding="utf-8"),
            ),
        }
        for label, source in final.items():
            with self.subTest(label=label):
                self.assertEqual(source.count(PARAMETER_POLICY_EDITOR_MARKER), 1)
                self.assertIn("parameter_policy_profiles", source)
                self.assertIn("parameter_policy_components", source)
                self.assertIn('default("standard")', source)

    def test_non_release_pages_do_not_receive_parameter_policy_editor(self):
        for name in ("shared", "lumina2-lora", "tagger"):
            with self.subTest(name=name):
                source = (SCHEMA / f"{name}.ts").read_text(encoding="utf-8")
                fixed = override_raw_schema(name, source)
                self.assertNotIn(PARAMETER_POLICY_EDITOR_MARKER, fixed)
                self.assertNotIn("parameter_policy_profiles", fixed)

    def test_all_transformed_training_schemas_remain_valid_javascript(self):
        lora_master = (SCHEMA / "lora-master.ts").read_text(encoding="utf-8")
        dreambooth = (SCHEMA / "dreambooth.ts").read_text(encoding="utf-8")
        flux_lora = (SCHEMA / "flux-lora.ts").read_text(encoding="utf-8")
        transformed = {
            "sd-lora": override_raw_schema("lora-master", fixed_sd_schema(lora_master, "sd-lora")),
            "sdxl-lora": override_raw_schema("sdxl-lora", fixed_sd_schema(lora_master, "sdxl-lora")),
            "sd-dreambooth": override_raw_schema("dreambooth", fixed_sd_schema(dreambooth, "sd-dreambooth")),
            "flux-lora": override_raw_schema("flux-lora", fixed_flux_family_schema(flux_lora, "flux", "flux-lora")),
            "chroma-lora": override_raw_schema("chroma-lora", fixed_flux_family_schema(flux_lora, "chroma", "chroma-lora")),
            "anima-lora": override_raw_schema("anima-lora", fixed_flux_family_schema(flux_lora, "anima", "anima-lora", "lora")),
            "anima-finetune": override_raw_schema("anima-finetune", fixed_flux_family_schema(flux_lora, "anima", "anima-finetune", "finetune")),
            "flux-finetune": override_raw_schema("flux-finetune", (SCHEMA / "flux-finetune.ts").read_text(encoding="utf-8")),
            "sdxl-finetune": override_raw_schema("sdxl-full", (SCHEMA / "sdxl-full.ts").read_text(encoding="utf-8")),
            "sd3-lora": override_raw_schema("sd3-lora", (SCHEMA / "sd3-lora.ts").read_text(encoding="utf-8")),
            "lora-basic": override_raw_schema("lora-basic", (SCHEMA / "lora-basic.ts").read_text(encoding="utf-8")),
        }
        for label, source in transformed.items():
            with self.subTest(label=label):
                assert_js_expression_compiles(self, source, label)


if __name__ == "__main__":
    unittest.main()
