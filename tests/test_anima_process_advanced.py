import unittest

from mikazuki.anima_finetune_advanced import (
    normalize_anima_save_schedule,
    validate_anima_finetune_advanced_combinations,
    validate_effective_text_encoder_cache,
)


class AnimaProcessAdvancedTests(unittest.TestCase):
    def test_text_encoder_cache_rejects_token_warmup(self):
        config = {
            "cache_text_encoder_outputs": True,
            "token_warmup_step": 0.25,
        }
        with self.assertRaisesRegex(ValueError, "token_warmup_step"):
            validate_effective_text_encoder_cache(config)

    def test_text_encoder_cache_allows_zero_token_warmup(self):
        config = {
            "cache_text_encoder_outputs_to_disk": True,
            "token_warmup_step": 0,
        }
        validate_effective_text_encoder_cache(config)

    def test_save_epoch_ratio_removes_competing_epoch_interval(self):
        config = {
            "save_n_epoch_ratio": 5,
            "save_every_n_epochs": 1,
        }
        normalize_anima_save_schedule(config)
        self.assertEqual(config["save_n_epoch_ratio"], 5)
        self.assertNotIn("save_every_n_epochs", config)

    def test_dataset_config_and_metadata_json_are_mutually_exclusive(self):
        config = {
            "dataset_config": "dataset.toml",
            "in_json": "metadata.json",
        }
        with self.assertRaisesRegex(ValueError, "dataset_config.*in_json"):
            validate_anima_finetune_advanced_combinations(config)

    def test_masked_loss_requires_conditioning_source(self):
        with self.assertRaisesRegex(ValueError, "masked_loss"):
            validate_anima_finetune_advanced_combinations({"masked_loss": True})

        validate_anima_finetune_advanced_combinations(
            {"masked_loss": True, "conditioning_data_dir": "masks"}
        )
        validate_anima_finetune_advanced_combinations(
            {"masked_loss": True, "dataset_config": "dataset.toml"}
        )

    def test_deepspeed_rejects_unverified_execution_combinations(self):
        for key, value in (
            ("fused_backward_pass", True),
            ("blocks_to_swap", 1),
            ("torch_compile", True),
        ):
            with self.subTest(key=key):
                config = {"deepspeed": True, key: value}
                with self.assertRaises(ValueError):
                    validate_anima_finetune_advanced_combinations(config)


if __name__ == "__main__":
    unittest.main()
