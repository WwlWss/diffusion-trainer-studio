import unittest

from mikazuki.anima_qwen_config import ANIMA_QWEN_TRAINING_KEYS, normalize_qwen_training_config


class AnimaQwenOffCompatibilityTests(unittest.TestCase):
    def _legacy_style_config(self):
        return {
            "train_qwen3_text_encoder": False,
            "qwen3_lr": "5e-7",
            "qwen3_gradient_checkpointing": True,
            "qwen3_output_dir": "unused",
            "learning_rate": "1e-5",
            "optimizer_type": "Prodigy",
            "deepspeed": True,
            "cache_text_encoder_outputs": True,
            "cache_text_encoder_outputs_to_disk": True,
            "sentinel_existing_option": "keep-me",
        }

    def _assert_existing_options_unchanged(self, config):
        self.assertEqual(config["optimizer_type"], "Prodigy")
        self.assertTrue(config["deepspeed"])
        self.assertTrue(config["cache_text_encoder_outputs"])
        self.assertTrue(config["cache_text_encoder_outputs_to_disk"])
        self.assertEqual(config["sentinel_existing_option"], "keep-me")
        self.assertTrue(ANIMA_QWEN_TRAINING_KEYS.isdisjoint(config))

    def test_finetune_with_qwen_disabled_keeps_existing_options(self):
        config = self._legacy_style_config()
        self.assertFalse(normalize_qwen_training_config(config, "finetune"))
        self._assert_existing_options_unchanged(config)

    def test_lora_strips_stale_qwen_fields_without_applying_joint_limits(self):
        config = self._legacy_style_config()
        config["train_qwen3_text_encoder"] = True
        self.assertFalse(normalize_qwen_training_config(config, "lora"))
        self._assert_existing_options_unchanged(config)


if __name__ == "__main__":
    unittest.main()
