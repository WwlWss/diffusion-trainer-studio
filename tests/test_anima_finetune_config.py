import copy
import unittest

from mikazuki.anima_finetune_config import (
    normalize_anima_finetune_config,
    validate_anima_finetune_config,
)


class AnimaFinetuneConfigTests(unittest.TestCase):
    def test_semantic_modes_expand_to_raw_sd_scripts_args(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "anima_precision_mode": "full_bf16",
            "anima_latent_cache_mode": "disk",
            "anima_text_encoder_cache_mode": "memory",
            "anima_checkpoint_mode": "unsloth",
        }

        normalize_anima_finetune_config(config, "finetune")

        self.assertEqual(config["learning_rate"], 1e-5)
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertFalse(config["full_fp16"])
        self.assertTrue(config["full_bf16"])
        self.assertTrue(config["cache_latents"])
        self.assertTrue(config["cache_latents_to_disk"])
        self.assertTrue(config["cache_text_encoder_outputs"])
        self.assertFalse(config["cache_text_encoder_outputs_to_disk"])
        self.assertTrue(config["gradient_checkpointing"])
        self.assertFalse(config["cpu_offload_checkpointing"])
        self.assertTrue(config["unsloth_offload_checkpointing"])

        for semantic_key in (
            "anima_finetune_learning_rate",
            "anima_precision_mode",
            "anima_latent_cache_mode",
            "anima_text_encoder_cache_mode",
            "anima_checkpoint_mode",
        ):
            self.assertNotIn(semantic_key, config)

    def test_normalization_is_idempotent(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "anima_precision_mode": "mixed_bf16",
            "anima_latent_cache_mode": "disk",
            "anima_text_encoder_cache_mode": "off",
            "anima_checkpoint_mode": "standard",
        }
        normalize_anima_finetune_config(config, "finetune")
        once = copy.deepcopy(config)
        normalize_anima_finetune_config(config, "finetune")
        self.assertEqual(config, once)

    def test_legacy_disk_cache_is_normalized_to_effective_cache_state(self):
        config = {
            "learning_rate": "1e-5",
            "mixed_precision": "bf16",
            "cache_latents": False,
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs": False,
            "cache_text_encoder_outputs_to_disk": True,
        }
        normalize_anima_finetune_config(config, "finetune")
        self.assertTrue(config["cache_latents"])
        self.assertTrue(config["cache_text_encoder_outputs"])

    def test_semantic_precision_conflicting_raw_value_is_rejected(self):
        config = {
            "learning_rate": "1e-5",
            "anima_precision_mode": "full_bf16",
            "full_fp16": True,
        }
        with self.assertRaisesRegex(ValueError, "训练精度"):
            normalize_anima_finetune_config(config, "finetune")

    def test_semantic_cache_conflicting_raw_value_is_rejected(self):
        config = {
            "learning_rate": "1e-5",
            "anima_latent_cache_mode": "off",
            "cache_latents_to_disk": True,
        }
        with self.assertRaisesRegex(ValueError, "Latent"):
            normalize_anima_finetune_config(config, "finetune")

    def test_semantic_learning_rate_conflict_is_rejected(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "learning_rate": "5e-5",
        }
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            normalize_anima_finetune_config(config, "finetune")

    def test_lora_discards_finetune_semantic_fields_without_touching_lora_args(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "anima_precision_mode": "full_bf16",
            "anima_latent_cache_mode": "disk",
            "anima_text_encoder_cache_mode": "off",
            "anima_checkpoint_mode": "standard",
            "learning_rate": "5e-5",
            "network_dim": 32,
            "optimizer_type": "Prodigy",
        }
        normalize_anima_finetune_config(config, "lora")
        self.assertEqual(config["learning_rate"], "5e-5")
        self.assertEqual(config["network_dim"], 32)
        self.assertEqual(config["optimizer_type"], "Prodigy")
        self.assertNotIn("anima_finetune_learning_rate", config)
        self.assertNotIn("anima_precision_mode", config)

    def test_full_finetune_requires_positive_global_lr(self):
        for value in (0, "0", -1e-5):
            with self.subTest(value=value):
                config = {"learning_rate": value}
                with self.assertRaisesRegex(ValueError, "learning_rate > 0"):
                    validate_anima_finetune_config(config, "finetune")

    def test_precision_conflicts_are_rejected(self):
        bad_configs = [
            {"learning_rate": 1e-5, "mixed_precision": "bf16", "full_fp16": True},
            {"learning_rate": 1e-5, "mixed_precision": "fp16", "full_bf16": True},
            {
                "learning_rate": 1e-5,
                "mixed_precision": "bf16",
                "full_fp16": True,
                "full_bf16": True,
            },
        ]
        for config in bad_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_anima_finetune_config(config, "finetune")

    def test_latent_cache_rejects_incompatible_augmentation(self):
        for key in ("color_aug", "random_crop"):
            with self.subTest(key=key):
                config = {
                    "learning_rate": 1e-5,
                    "cache_latents": True,
                    key: True,
                }
                with self.assertRaisesRegex(ValueError, key):
                    validate_anima_finetune_config(config, "finetune")

    def test_flip_aug_remains_allowed_with_latent_cache(self):
        config = {
            "learning_rate": 1e-5,
            "cache_latents": True,
            "flip_aug": True,
        }
        validate_anima_finetune_config(config, "finetune")
        self.assertTrue(config["flip_aug"])

    def test_unsupported_noop_controls_are_rejected(self):
        for key in ("no_half_vae", "lowram", "compile"):
            with self.subTest(key=key):
                config = {"learning_rate": 1e-5, key: True}
                with self.assertRaisesRegex(ValueError, "尚未实现"):
                    validate_anima_finetune_config(config, "finetune")

    def test_independent_llm_adapter_path_is_rejected_until_trainer_support_exists(self):
        config = {"learning_rate": 1e-5, "llm_adapter_path": "adapter.safetensors"}
        with self.assertRaisesRegex(ValueError, "llm_adapter_path"):
            validate_anima_finetune_config(config, "finetune")

    def test_fake_preview_sampler_is_removed(self):
        config = {"learning_rate": 1e-5, "sample_sampler": "euler_a"}
        validate_anima_finetune_config(config, "finetune")
        self.assertNotIn("sample_sampler", config)

    def test_anima_save_format_is_fixed_to_safetensors(self):
        ok = {"learning_rate": 1e-5, "save_model_as": "safetensors"}
        validate_anima_finetune_config(ok, "finetune")
        self.assertNotIn("save_model_as", ok)

        bad = {"learning_rate": 1e-5, "save_model_as": "ckpt"}
        with self.assertRaisesRegex(ValueError, "safetensors"):
            validate_anima_finetune_config(bad, "finetune")

    def test_only_real_anima_weighting_schemes_are_allowed(self):
        for scheme in ("uniform", "none", "sigma_sqrt", "cosmap"):
            with self.subTest(scheme=scheme):
                config = {"learning_rate": 1e-5, "weighting_scheme": scheme}
                validate_anima_finetune_config(config, "finetune")

        for scheme in ("mode", "logit_normal"):
            with self.subTest(scheme=scheme):
                config = {"learning_rate": 1e-5, "weighting_scheme": scheme}
                with self.assertRaisesRegex(ValueError, "loss-weighting"):
                    validate_anima_finetune_config(config, "finetune")

    def test_legacy_cpu_offload_explicitly_enables_gradient_checkpointing(self):
        config = {
            "learning_rate": 1e-5,
            "gradient_checkpointing": False,
            "cpu_offload_checkpointing": True,
        }
        normalize_anima_finetune_config(config, "finetune")
        validate_anima_finetune_config(config, "finetune")
        self.assertTrue(config["gradient_checkpointing"])

    def test_cpu_and_unsloth_checkpoint_offload_conflict_is_rejected_early(self):
        config = {
            "learning_rate": 1e-5,
            "cpu_offload_checkpointing": True,
            "unsloth_offload_checkpointing": True,
        }
        with self.assertRaisesRegex(ValueError, "不能同时启用"):
            normalize_anima_finetune_config(config, "finetune")


if __name__ == "__main__":
    unittest.main()
