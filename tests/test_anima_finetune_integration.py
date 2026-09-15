import unittest

from mikazuki.anima_finetune_config import (
    normalize_anima_finetune_config,
    validate_anima_finetune_config,
)
from mikazuki.anima_qwen_config import normalize_qwen_training_config


class AnimaFinetuneIntegrationTests(unittest.TestCase):
    def test_semantic_qwen_disk_cache_is_rejected_before_launch(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "anima_text_encoder_cache_mode": "disk",
            "train_qwen3_text_encoder": True,
            "qwen3_lr": "5e-7",
            "optimizer_type": "AdamW8bit",
        }
        normalize_anima_finetune_config(config, "finetune")
        with self.assertRaisesRegex(ValueError, "cache_text_encoder_outputs"):
            normalize_qwen_training_config(config, "finetune")

    def test_semantic_qwen_off_cache_off_produces_safe_raw_config(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "anima_precision_mode": "full_bf16",
            "anima_latent_cache_mode": "disk",
            "anima_text_encoder_cache_mode": "off",
            "anima_checkpoint_mode": "standard",
            "train_qwen3_text_encoder": True,
            "qwen3_lr": "5e-7",
            "optimizer_type": "AdamW8bit",
        }
        normalize_anima_finetune_config(config, "finetune")
        enabled = normalize_qwen_training_config(config, "finetune")
        validate_anima_finetune_config(config, "finetune")

        self.assertTrue(enabled)
        self.assertFalse(config["cache_text_encoder_outputs"])
        self.assertFalse(config["cache_text_encoder_outputs_to_disk"])
        self.assertEqual(config["learning_rate"], 1e-5)
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertTrue(config["full_bf16"])

    def test_legacy_full_finetune_cuda_switches_fail_instead_of_noop(self):
        for key in ("cuda_allow_tf32", "cuda_cudnn_benchmark"):
            with self.subTest(key=key):
                config = {"learning_rate": 1e-5, key: True}
                with self.assertRaisesRegex(ValueError, key):
                    validate_anima_finetune_config(config, "finetune")

    def test_lora_does_not_apply_finetune_cuda_restrictions(self):
        config = {
            "learning_rate": "5e-5",
            "cuda_allow_tf32": True,
            "cuda_cudnn_benchmark": True,
        }
        normalize_anima_finetune_config(config, "lora")
        validate_anima_finetune_config(config, "lora")
        self.assertTrue(config["cuda_allow_tf32"])
        self.assertTrue(config["cuda_cudnn_benchmark"])


if __name__ == "__main__":
    unittest.main()
