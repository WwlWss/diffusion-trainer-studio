import unittest

from mikazuki.anima_finetune_config import (
    normalize_anima_finetune_config,
    validate_anima_finetune_config,
)
from mikazuki.anima_qwen_config import normalize_qwen_training_config


class AnimaFinetuneTrainerCoverageTests(unittest.TestCase):
    def test_preview_cadence_epoch_maps_to_only_epoch_arg(self):
        config = {
            "anima_preview_cadence": "epoch",
            "anima_preview_interval": 2,
            "sample_every_n_steps": 50,
        }
        normalize_anima_finetune_config(config, "finetune")
        self.assertEqual(config["sample_every_n_epochs"], 2)
        self.assertNotIn("sample_every_n_steps", config)
        self.assertNotIn("anima_preview_cadence", config)
        self.assertNotIn("anima_preview_interval", config)

    def test_preview_cadence_step_maps_to_only_step_arg(self):
        config = {
            "anima_preview_cadence": "step",
            "anima_preview_interval": 100,
            "sample_every_n_epochs": 1,
        }
        normalize_anima_finetune_config(config, "finetune")
        self.assertEqual(config["sample_every_n_steps"], 100)
        self.assertNotIn("sample_every_n_epochs", config)

    def test_custom_optimizer_maps_to_sd_scripts_optimizer_type(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "optimizer_type": "Custom",
            "anima_custom_optimizer_type": "bitsandbytes.optim.PagedAdEMAMix8bit",
        }
        normalize_anima_finetune_config(config, "finetune")
        self.assertEqual(config["optimizer_type"], "bitsandbytes.optim.PagedAdEMAMix8bit")
        self.assertNotIn("anima_custom_optimizer_type", config)

    def test_custom_scheduler_maps_to_lr_scheduler_type(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "lr_scheduler": "custom",
            "anima_custom_lr_scheduler_type": "torch.optim.lr_scheduler.OneCycleLR",
        }
        normalize_anima_finetune_config(config, "finetune")
        self.assertEqual(config["lr_scheduler_type"], "torch.optim.lr_scheduler.OneCycleLR")
        self.assertEqual(config["lr_scheduler"], "constant")
        self.assertNotIn("anima_custom_lr_scheduler_type", config)

    def test_fused_backward_requires_adafactor_and_no_accumulation(self):
        bad_optimizer = {
            "learning_rate": 1e-5,
            "optimizer_type": "AdamW8bit",
            "fused_backward_pass": True,
        }
        with self.assertRaisesRegex(ValueError, "AdaFactor"):
            validate_anima_finetune_config(bad_optimizer, "finetune")

        bad_accumulation = {
            "learning_rate": 1e-5,
            "optimizer_type": "AdaFactor",
            "gradient_accumulation_steps": 2,
            "fused_backward_pass": True,
        }
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps=1"):
            validate_anima_finetune_config(bad_accumulation, "finetune")

        ok = {
            "learning_rate": 1e-5,
            "optimizer_type": "AdaFactor",
            "gradient_accumulation_steps": 1,
            "fused_backward_pass": True,
        }
        validate_anima_finetune_config(ok, "finetune")

    def test_deepspeed_normalizes_worker_count(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "deepspeed": True,
            "zero_stage": 2,
            "max_data_loader_n_workers": 8,
        }
        normalize_anima_finetune_config(config, "finetune")
        self.assertEqual(config["max_data_loader_n_workers"], 1)
        validate_anima_finetune_config(config, "finetune")

    def test_deepspeed_offload_stage_rules_are_checked(self):
        bad_optimizer_offload = {
            "learning_rate": 1e-5,
            "deepspeed": True,
            "zero_stage": 1,
            "offload_optimizer_device": "cpu",
        }
        with self.assertRaisesRegex(ValueError, "stage 2/3"):
            validate_anima_finetune_config(bad_optimizer_offload, "finetune")

        bad_param_offload = {
            "learning_rate": 1e-5,
            "deepspeed": True,
            "zero_stage": 2,
            "offload_param_device": "cpu",
        }
        with self.assertRaisesRegex(ValueError, "stage 3"):
            validate_anima_finetune_config(bad_param_offload, "finetune")

    def test_zero3_flags_require_zero3(self):
        config = {
            "learning_rate": 1e-5,
            "deepspeed": True,
            "zero_stage": 2,
            "zero3_init_flag": True,
        }
        with self.assertRaisesRegex(ValueError, "ZeRO stage 3"):
            validate_anima_finetune_config(config, "finetune")

    def test_fp16_master_weights_require_exact_deepspeed_mode(self):
        config = {
            "learning_rate": 1e-5,
            "mixed_precision": "bf16",
            "deepspeed": True,
            "zero_stage": 2,
            "offload_optimizer_device": "cpu",
            "fp16_master_weights_and_gradients": True,
        }
        with self.assertRaisesRegex(ValueError, "FP16"):
            validate_anima_finetune_config(config, "finetune")

    def test_qwen_joint_training_rejects_deepspeed_and_fused_backward(self):
        base = {
            "learning_rate": 1e-5,
            "train_qwen3_text_encoder": True,
            "qwen3_lr": 5e-7,
            "optimizer_type": "AdamW8bit",
        }
        with self.assertRaisesRegex(ValueError, "DeepSpeed"):
            normalize_qwen_training_config({**base, "deepspeed": True}, "finetune")
        with self.assertRaisesRegex(ValueError, "fused_backward_pass"):
            normalize_qwen_training_config({**base, "fused_backward_pass": True}, "finetune")

    def test_highvram_and_ddp_static_graph_are_real_passthrough_args(self):
        config = {
            "anima_finetune_learning_rate": "1e-5",
            "highvram": True,
            "ddp_static_graph": True,
        }
        normalize_anima_finetune_config(config, "finetune")
        validate_anima_finetune_config(config, "finetune")
        self.assertTrue(config["highvram"])
        self.assertTrue(config["ddp_static_graph"])

    def test_torch_compile_is_distinct_from_unsupported_per_block_compile(self):
        config = {
            "learning_rate": 1e-5,
            "torch_compile": True,
            "dynamo_backend": "inductor",
        }
        validate_anima_finetune_config(config, "finetune")
        self.assertTrue(config["torch_compile"])

        bad = {"learning_rate": 1e-5, "compile": True}
        with self.assertRaisesRegex(ValueError, "尚未实现"):
            validate_anima_finetune_config(bad, "finetune")


if __name__ == "__main__":
    unittest.main()
