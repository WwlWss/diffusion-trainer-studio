import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_qwen_config import (
    ANIMA_QWEN_TRAINING_KEYS,
    normalize_qwen_training_config,
    text_encoder_cache_enabled,
    trainer_supports_qwen_training,
)


class AnimaQwenConfigTests(unittest.TestCase):
    def test_text_encoder_disk_cache_counts_as_enabled(self):
        self.assertTrue(text_encoder_cache_enabled({"cache_text_encoder_outputs_to_disk": True}))
        self.assertTrue(text_encoder_cache_enabled({"cache_text_encoder_outputs": True}))
        self.assertFalse(text_encoder_cache_enabled({}))

    def test_lora_strips_all_qwen_training_fields(self):
        config = {
            "train_qwen3_text_encoder": True,
            "qwen3_lr": "5e-7",
            "qwen3_gradient_checkpointing": True,
            "qwen3_output_dir": "out",
        }
        self.assertFalse(normalize_qwen_training_config(config, "lora"))
        self.assertTrue(ANIMA_QWEN_TRAINING_KEYS.isdisjoint(config))

    def test_disabled_finetune_strips_all_qwen_training_fields(self):
        config = {
            "train_qwen3_text_encoder": False,
            "qwen3_lr": "5e-7",
            "qwen3_gradient_checkpointing": True,
            "qwen3_output_dir": "out",
        }
        self.assertFalse(normalize_qwen_training_config(config, "finetune"))
        self.assertTrue(ANIMA_QWEN_TRAINING_KEYS.isdisjoint(config))

    def test_non_anima_strips_all_qwen_training_fields(self):
        config = {
            "train_qwen3_text_encoder": True,
            "qwen3_lr": "5e-7",
        }
        self.assertFalse(normalize_qwen_training_config(config, "disabled"))
        self.assertTrue(ANIMA_QWEN_TRAINING_KEYS.isdisjoint(config))

    def test_memory_cache_is_rejected(self):
        config = self._valid_config(cache_text_encoder_outputs=True)
        with self.assertRaisesRegex(ValueError, "cache_text_encoder_outputs"):
            normalize_qwen_training_config(config, "finetune")

    def test_disk_cache_is_rejected_even_when_memory_cache_is_false(self):
        config = self._valid_config(
            cache_text_encoder_outputs=False,
            cache_text_encoder_outputs_to_disk=True,
        )
        with self.assertRaisesRegex(ValueError, "cache_text_encoder_outputs_to_disk"):
            normalize_qwen_training_config(config, "finetune")

    def test_missing_qwen_lr_is_rejected(self):
        config = self._valid_config()
        config.pop("qwen3_lr")
        with self.assertRaisesRegex(ValueError, "qwen3_lr"):
            normalize_qwen_training_config(config, "finetune")

    def test_non_positive_qwen_lr_is_rejected(self):
        for value in ("0", "-1e-7"):
            with self.subTest(value=value):
                config = self._valid_config(qwen3_lr=value)
                with self.assertRaisesRegex(ValueError, "qwen3_lr"):
                    normalize_qwen_training_config(config, "finetune")

    def test_supported_optimizer_is_accepted(self):
        config = self._valid_config(optimizer_type="AdamW8bit")
        self.assertTrue(normalize_qwen_training_config(config, "finetune"))
        self.assertEqual(config["qwen3_lr"], "5e-7")

    def test_unsafe_multi_lr_optimizers_are_rejected(self):
        for optimizer in ("Prodigy", "DAdaptAdam", "AdaFactor"):
            with self.subTest(optimizer=optimizer):
                config = self._valid_config(optimizer_type=optimizer)
                with self.assertRaisesRegex(ValueError, "联合训练仅支持"):
                    normalize_qwen_training_config(config, "finetune")

    def test_deepspeed_is_rejected_only_for_qwen_training(self):
        config = self._valid_config(deepspeed=True)
        with self.assertRaisesRegex(ValueError, "DeepSpeed"):
            normalize_qwen_training_config(config, "finetune")

        disabled = {"train_qwen3_text_encoder": False, "deepspeed": True}
        self.assertFalse(normalize_qwen_training_config(disabled, "finetune"))
        self.assertTrue(disabled["deepspeed"])

    def test_fused_backward_is_rejected_only_for_qwen_training(self):
        config = self._valid_config(fused_backward_pass=True)
        with self.assertRaisesRegex(ValueError, "fused_backward_pass"):
            normalize_qwen_training_config(config, "finetune")

        disabled = {"train_qwen3_text_encoder": False, "fused_backward_pass": True}
        self.assertFalse(normalize_qwen_training_config(disabled, "finetune"))
        self.assertTrue(disabled["fused_backward_pass"])

    def test_empty_output_dir_is_omitted(self):
        config = self._valid_config(qwen3_output_dir="")
        self.assertTrue(normalize_qwen_training_config(config, "finetune"))
        self.assertNotIn("qwen3_output_dir", config)

    def test_existing_file_cannot_be_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "not-a-directory"
            file_path.write_text("x", encoding="utf-8")
            config = self._valid_config(qwen3_output_dir=str(file_path))
            with self.assertRaisesRegex(ValueError, "不是目录"):
                normalize_qwen_training_config(config, "finetune")

    def test_trainer_capability_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Path(temp_dir) / "anima_train.py"
            trainer.write_text("parser.add_argument('--train_qwen3_text_encoder')", encoding="utf-8")
            self.assertTrue(trainer_supports_qwen_training(str(trainer)))
            trainer.write_text("print('old trainer')", encoding="utf-8")
            self.assertFalse(trainer_supports_qwen_training(str(trainer)))

    @staticmethod
    def _valid_config(**overrides):
        config = {
            "train_qwen3_text_encoder": True,
            "qwen3_lr": "5e-7",
            "qwen3_gradient_checkpointing": True,
            "optimizer_type": "AdamW8bit",
            "cache_text_encoder_outputs": False,
            "cache_text_encoder_outputs_to_disk": False,
        }
        config.update(overrides)
        return config


if __name__ == "__main__":
    unittest.main()
