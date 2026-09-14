import unittest

from mikazuki.anima_effective_config import _normalize_lora_target
from mikazuki.training_config import PreparedTrainingConfig, prepare_training_config
from mikazuki.training_validation import validate_prepared_config


def fake_resolve_backend(config, requested):
    return requested, f"./{requested}.py"


def discriminator_sensitive_resolver(config, requested):
    """Mimic the old shared resolver closely enough to catch stale selectors."""
    model_type = config.get("model_type")
    anima_mode = config.get("anima_training_mode")
    if requested in {"flux-lora", "flux-finetune"} and model_type != "flux":
        raise ValueError(f"expected flux but got {model_type}")
    if requested == "chroma-lora" and model_type != "chroma":
        raise ValueError(f"expected chroma but got {model_type}")
    if requested == "anima-lora" and (model_type != "anima" or anima_mode != "lora"):
        raise ValueError("stale Anima LoRA discriminator")
    if requested == "anima-finetune" and (model_type != "anima" or anima_mode != "finetune"):
        raise ValueError("stale Anima finetune discriminator")
    if requested.startswith("sd") and model_type not in (None, ""):
        raise ValueError(f"SD page inherited model_type={model_type}")
    return requested, f"./{requested}.py"


class EffectiveTrainingConfigTests(unittest.TestCase):
    def prepare(self, config, page_type, resolver=fake_resolve_backend):
        return prepare_training_config(
            config,
            page_train_type=page_type,
            resolve_backend=resolver,
        )

    def test_preview_does_not_require_runtime_assets(self):
        prepared = PreparedTrainingConfig(
            train_type="anima-lora",
            trainer_file="./missing/anima_train_network.py",
            config={
                "pretrained_model_name_or_path": "",
                "train_data_dir": "",
                "qwen3": "",
                "vae": "",
                "network_module": "networks.lora_anima",
            },
        )
        validate_prepared_config(prepared, False)
        self.assertEqual(prepared.config["qwen3"], "")
        self.assertEqual(prepared.config["vae"], "")

    def test_outer_page_backend_wins_over_embedded_routing(self):
        prepared = self.prepare(
            {
                "model_train_type": "sd-lora",
                "model_type": "anima",
                "anima_training_mode": "lora",
                "learning_rate": "1e-5",
                "mixed_precision": "bf16",
            },
            "flux-finetune",
            discriminator_sensitive_resolver,
        )
        self.assertEqual(prepared.train_type, "flux-finetune")
        self.assertEqual(prepared.config["model_type"], "flux")
        self.assertNotIn("model_train_type", prepared.config)
        self.assertNotIn("anima_training_mode", prepared.config)
        self.assertEqual(prepared.warnings, [])

    def test_sd_page_drops_stale_flux_family_selectors(self):
        prepared = self.prepare(
            {
                "model_type": "anima",
                "anima_training_mode": "finetune",
                "model_train_type": "sdxl-lora",
            },
            "lora-master",
            discriminator_sensitive_resolver,
        )
        self.assertEqual(prepared.train_type, "sd-lora")
        self.assertNotIn("model_type", prepared.config)
        self.assertNotIn("anima_training_mode", prepared.config)
        self.assertNotIn("model_train_type", prepared.config)
        self.assertEqual(prepared.warnings, [])

    def test_anima_page_overwrites_stale_flux_and_wrong_mode(self):
        prepared = self.prepare(
            {
                "model_type": "flux",
                "anima_training_mode": "lora",
                "anima_model_variant": "base",
                "anima_finetune_learning_rate": "1e-5",
                "anima_precision_mode": "mixed_bf16",
                "anima_latent_cache_mode": "off",
                "anima_text_encoder_cache_mode": "off",
                "anima_checkpoint_mode": "off",
                "optimizer_type": "AdamW8bit",
                "lr_scheduler": "constant",
            },
            "anima-finetune",
            discriminator_sensitive_resolver,
        )
        self.assertEqual(prepared.train_type, "anima-finetune")
        self.assertNotIn("model_type", prepared.config)
        self.assertNotIn("anima_training_mode", prepared.config)
        self.assertEqual(prepared.warnings, [])

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
