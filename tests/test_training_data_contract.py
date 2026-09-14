import unittest

from mikazuki.training_data_contract import (
    DATASET_CONFIG_TRAIN_TYPES,
    normalize_optional_dataset_paths,
    validate_dataset_source,
)


class DatasetSourceContractTests(unittest.TestCase):
    def test_dataset_config_full_trainers(self):
        self.assertEqual(
            DATASET_CONFIG_TRAIN_TYPES,
            {"sdxl-finetune", "flux-finetune", "anima-finetune"},
        )
        for train_type in DATASET_CONFIG_TRAIN_TYPES:
            config = {
                "dataset_config": "config.toml",
                "train_data_dir": "stale-images",
                "in_json": "stale-meta.json",
            }
            validate_dataset_source(
                config,
                train_type,
                is_file=lambda path: path == "config.toml",
                validate_data_dir=lambda path: False,
            )
            self.assertNotIn("train_data_dir", config)
            self.assertNotIn("in_json", config)

    def test_dataset_config_rejected_elsewhere(self):
        for train_type in ("sd-lora", "sdxl-lora", "sd-dreambooth", "flux-lora", "chroma-lora", "anima-lora"):
            with self.subTest(train_type=train_type):
                with self.assertRaises(ValueError):
                    validate_dataset_source(
                        {"dataset_config": "config.toml"},
                        train_type,
                        is_file=lambda path: True,
                        validate_data_dir=lambda path: True,
                    )

    def test_sdxl_without_dataset_config_requires_train_dir(self):
        with self.assertRaises(ValueError):
            validate_dataset_source(
                {"in_json": "meta.json"},
                "sdxl-finetune",
                is_file=lambda path: path == "meta.json",
                validate_data_dir=lambda path: False,
            )

    def test_sdxl_metadata_still_uses_train_dir(self):
        validate_dataset_source(
            {"train_data_dir": "images", "in_json": "meta.json"},
            "sdxl-finetune",
            is_file=lambda path: path == "meta.json",
            validate_data_dir=lambda path: path == "images",
        )

    def test_empty_optional_paths_are_removed(self):
        config = {
            "dataset_config": "",
            "in_json": None,
            "conditioning_data_dir": "",
            "train_data_dir": "images",
        }
        normalize_optional_dataset_paths(config)
        for key in ("dataset_config", "in_json", "conditioning_data_dir"):
            self.assertNotIn(key, config)


if __name__ == "__main__":
    unittest.main()
