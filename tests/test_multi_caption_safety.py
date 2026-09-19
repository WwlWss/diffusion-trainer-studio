import unittest

from mikazuki.training_config import PreparedTrainingConfig
from mikazuki.training_validation import validate_prepared_config


class MultiCaptionSafetyTests(unittest.TestCase):
    def prepared(self, config):
        return PreparedTrainingConfig(
            train_type="flux-lora",
            trainer_file="./scripts/dev/flux_train_network.py",
            config=dict(config),
        )

    def test_multi_rejects_text_encoder_output_cache_in_preview_too(self):
        for key in ("cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"):
            with self.subTest(key=key):
                prepared = self.prepared({
                    "multi_caption_config": "config/autosave/multi-caption/policy.json",
                    key: True,
                })
                with self.assertRaisesRegex(ValueError, "Text Encoder Output Cache"):
                    validate_prepared_config(prepared, False)

    def test_multi_allows_latent_cache(self):
        prepared = self.prepared({
            "multi_caption_config": "config/autosave/multi-caption/policy.json",
            "cache_latents": True,
            "cache_latents_to_disk": True,
        })
        validate_prepared_config(prepared, False)
        self.assertTrue(prepared.config["cache_latents"])

    def test_multi_rejects_custom_dataset_class(self):
        prepared = self.prepared({
            "multi_caption_config": "config/autosave/multi-caption/policy.json",
            "dataset_class": "custom.module.Dataset",
        })
        with self.assertRaisesRegex(ValueError, "dataset_class"):
            validate_prepared_config(prepared, False)

    def test_standard_does_not_change_existing_cache_behavior(self):
        prepared = self.prepared({
            "cache_text_encoder_outputs": True,
            "cache_text_encoder_outputs_to_disk": True,
        })
        validate_prepared_config(prepared, False)
        self.assertNotIn("multi_caption_config", prepared.config)

    def test_multi_validation_does_not_disable_cache_silently(self):
        prepared = self.prepared({
            "multi_caption_config": "config/autosave/multi-caption/policy.json",
            "cache_text_encoder_outputs": True,
        })
        with self.assertRaises(ValueError):
            validate_prepared_config(prepared, False)
        self.assertTrue(prepared.config["cache_text_encoder_outputs"])


if __name__ == "__main__":
    unittest.main()
