import unittest

from mikazuki.anima_effective_config import _normalize_lora_target
from mikazuki.training_config import prepare_training_config


def fake_resolve_backend(config, requested):
    return requested, f"./{requested}.py"


class EffectiveTrainingConfigTests(unittest.TestCase):
    def prepare(self, config, page_type):
        return prepare_training_config(
            config,
            page_train_type=page_type,
            resolve_backend=fake_resolve_backend,
        )

    def test_outer_page_backend_wins_over_embedded_routing(self):
        prepared = self.prepare(
            {
                "model_train_type": "sd-lora",
                "model_type": "flux",
                "learning_rate": "1e-5",
            },
            "flux-finetune",
        )
        self.assertEqual(prepared.train_type, "flux-finetune")
        self.assertNotIn("model_train_type", prepared.config)
        self.assertTrue(any("页面固定后端" in warning for warning in prepared.warnings))

    def test_flux_t5xxl_semantic_control_becomes_network_arg(self):
        prepared = self.prepare(
            {
                "model_type": "flux",
                "train_t5xxl": True,
                "network_train_unet_only": False,
                "network_args": ["foo=bar", "train_t5xxl=False"],
            },
            "flux-lora",
        )
        self.assertIn("foo=bar", prepared.config["network_args"])
        self.assertIn("train_t5xxl=True", prepared.config["network_args"])
        self.assertNotIn("train_t5xxl", prepared.config)

    def test_flux_t5xxl_rejects_text_encoder_cache(self):
        with self.assertRaisesRegex(ValueError, "不能缓存"):
            self.prepare(
                {
                    "model_type": "flux",
                    "train_t5xxl": True,
                    "network_train_unet_only": False,
                    "cache_text_encoder_outputs": True,
                },
                "flux-lora",
            )

    def test_dreambooth_legacy_255_migrates_to_225(self):
        prepared = self.prepare({"max_token_length": 255}, "dreambooth")
        self.assertEqual(prepared.config["max_token_length"], 225)
        self.assertTrue(any("255" in warning for warning in prepared.warnings))

    def test_dreambooth_75_uses_trainer_default(self):
        prepared = self.prepare({"sd_max_token_length_mode": "75"}, "dreambooth")
        self.assertNotIn("max_token_length", prepared.config)
        self.assertNotIn("sd_max_token_length_mode", prepared.config)

    def test_dreambooth_rejects_obsolete_pt_output(self):
        with self.assertRaisesRegex(ValueError, "save_model_as=pt"):
            self.prepare({"save_model_as": "pt"}, "dreambooth")

    def test_full_step_control_removes_default_epoch(self):
        prepared = self.prepare(
            {
                "model_type": "flux",
                "max_train_steps": 100,
                "max_train_epochs": 1,
                "mixed_precision": "bf16",
            },
            "flux-finetune",
        )
        self.assertEqual(prepared.config["max_train_steps"], 100)
        self.assertNotIn("max_train_epochs", prepared.config)

    def test_full_mode_strips_legacy_lora_learning_rates(self):
        prepared = self.prepare(
            {
                "model_type": "flux",
                "optimizer_type": "DAdaptation",
                "learning_rate": 1,
                "unet_lr": 1,
                "text_encoder_lr": 1,
                "mixed_precision": "bf16",
            },
            "flux-finetune",
        )
        self.assertNotIn("unet_lr", prepared.config)
        self.assertNotIn("text_encoder_lr", prepared.config)

    def test_memory_mode_is_mutually_exclusive(self):
        high = self.prepare({"memory_mode": "highvram"}, "sd-lora")
        self.assertFalse(high.config["lowram"])
        self.assertTrue(high.config["highvram"])
        low = self.prepare({"memory_mode": "lowram"}, "sd-lora")
        self.assertTrue(low.config["lowram"])
        self.assertFalse(low.config["highvram"])

    def test_flux_compile_rejects_unverified_block_swap_combination(self):
        with self.assertRaisesRegex(ValueError, "torch_compile.*blocks_to_swap"):
            self.prepare(
                {
                    "model_type": "flux",
                    "mixed_precision": "bf16",
                    "torch_compile": True,
                    "dynamo_backend": "inductor",
                    "blocks_to_swap": 1,
                },
                "flux-finetune",
            )

    def test_anima_lora_targets_map_to_historical_flags(self):
        dit = {"anima_lora_target": "dit", "text_encoder_lr": 1e-5}
        _normalize_lora_target(dit)
        self.assertTrue(dit["network_train_unet_only"])
        self.assertNotIn("network_train_text_encoder_only", dit)
        self.assertNotIn("text_encoder_lr", dit)

        qwen = {"anima_lora_target": "qwen3", "anima_lora_text_encoder_lr": "5e-6"}
        _normalize_lora_target(qwen)
        self.assertFalse(qwen["network_train_unet_only"])
        self.assertTrue(qwen["network_train_text_encoder_only"])
        self.assertEqual(qwen["text_encoder_lr"], 5e-6)

        both = {"anima_lora_target": "dit_qwen3"}
        _normalize_lora_target(both)
        self.assertFalse(both["network_train_unet_only"])
        self.assertFalse(both["network_train_text_encoder_only"])

    def test_anima_qwen_lora_rejects_cached_outputs(self):
        with self.assertRaisesRegex(ValueError, "必须关闭"):
            _normalize_lora_target(
                {
                    "anima_lora_target": "qwen3",
                    "cache_text_encoder_outputs": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
