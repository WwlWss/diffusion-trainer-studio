import unittest

from mikazuki.full_trainer_contract import normalize_validate_flux_full, normalize_validate_sdxl_full


class FullTrainerContractTests(unittest.TestCase):
    def test_steps_override_epochs_explicitly(self):
        for normalizer in (normalize_validate_sdxl_full, normalize_validate_flux_full):
            config = {"max_train_steps": 100, "max_train_epochs": 1}
            normalizer(config)
            self.assertNotIn("max_train_epochs", config)

    def test_disk_cache_enables_base_cache(self):
        config = {"cache_latents_to_disk": True, "cache_text_encoder_outputs_to_disk": True}
        normalize_validate_flux_full(config)
        self.assertTrue(config["cache_latents"])
        self.assertTrue(config["cache_text_encoder_outputs"])

    def test_sdxl_text_encoder_training_rejects_cache(self):
        with self.assertRaises(ValueError):
            normalize_validate_sdxl_full({"train_text_encoder": True, "cache_text_encoder_outputs": True})

    def test_sdxl_block_lr_requires_23_numbers(self):
        with self.assertRaises(ValueError):
            normalize_validate_sdxl_full({"block_lr": "1e-6,1e-6"})
        valid = {"block_lr": ",".join(["1e-6"] * 23)}
        normalize_validate_sdxl_full(valid)

    def test_sdxl_fused_backward_requires_adafactor_and_no_accumulation(self):
        with self.assertRaises(ValueError):
            normalize_validate_sdxl_full({"fused_backward_pass": True, "optimizer_type": "AdamW"})
        with self.assertRaises(ValueError):
            normalize_validate_sdxl_full({
                "fused_backward_pass": True,
                "optimizer_type": "AdaFactor",
                "gradient_accumulation_steps": 2,
            })

    def test_cacheability_guards(self):
        with self.assertRaises(ValueError):
            normalize_validate_sdxl_full({"cache_latents": True, "color_aug": True})
        with self.assertRaises(ValueError):
            normalize_validate_flux_full({"cache_text_encoder_outputs": True, "shuffle_caption": True})

    def test_flux_block_swap_rejects_cpu_checkpoint_offload(self):
        with self.assertRaises(ValueError):
            normalize_validate_flux_full({"blocks_to_swap": 1, "cpu_offload_checkpointing": True})

    def test_flux_blockwise_fused_rejects_accumulation_and_schedulefree(self):
        with self.assertRaises(ValueError):
            normalize_validate_flux_full({"blockwise_fused_optimizers": True, "gradient_accumulation_steps": 2})
        with self.assertRaises(ValueError):
            normalize_validate_flux_full({"blockwise_fused_optimizers": True, "optimizer_type": "RAdamScheduleFree"})


if __name__ == "__main__":
    unittest.main()
