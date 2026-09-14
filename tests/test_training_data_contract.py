import unittest

from mikazuki.training_data_contract import (
    DATASET_CONFIG_TRAIN_TYPES,
    normalize_optional_dataset_paths,
    validate_dataset_source,
)


def _inspect(path):
    return (path == "images", "invalid dataset")


class DatasetSourceContractTests(unittest.TestCase):
    def test_dataset_config_supported_by_network_and_full_trainers(self):
        expected = {
            "sd-lora", "sdxl-lora", "sd-dreambooth", "sdxl-finetune", "flux-lora",
            "chroma-lora", "flux-finetune", "anima-lora", "anima-finetune", "sd3-lora",
        }
        self.assertEqual(DATASET_CONFIG_TRAIN_TYPES, expected)
        for train_type in expected:
            config = {
                "dataset_config": "config.toml",
                "train_data_dir": "stale-images",
                "reg_data_dir": "stale-reg",
                "in_json": "stale-meta.json",
            }
            validate_dataset_source(
                config, train_type,
                is_file=lambda path: path == "config.toml",
                is_dir=lambda path: False,
                inspect_data_dir=_inspect,
            )
            self.assertNotIn("train_data_dir", config)
            self.assertNotIn("reg_data_dir", config)
            self.assertNotIn("in_json", config)

    def test_unknown_backend_rejects_dataset_config(self):
        with self.assertRaises(ValueError):
            validate_dataset_source(
                {"dataset_config": "config.toml"}, "unknown",
                is_file=lambda path: True, is_dir=lambda path: True, inspect_data_dir=_inspect,
            )

    def test_without_dataset_config_requires_train_dir(self):
        with self.assertRaises(ValueError):
            validate_dataset_source(
                {"in_json": "meta.json"}, "sdxl-finetune",
                is_file=lambda path: path == "meta.json", is_dir=lambda path: True, inspect_data_dir=_inspect,
            )

    def test_metadata_requires_real_image_directory(self):
        validate_dataset_source(
            {"train_data_dir": "images", "in_json": "meta.json"}, "sdxl-finetune",
            is_file=lambda path: path == "meta.json", is_dir=lambda path: path == "images", inspect_data_dir=_inspect,
        )
        with self.assertRaises(ValueError):
            validate_dataset_source(
                {"train_data_dir": "missing", "in_json": "meta.json"}, "sdxl-finetune",
                is_file=lambda path: path == "meta.json", is_dir=lambda path: False, inspect_data_dir=_inspect,
            )

    def test_classic_folder_uses_read_only_inspector(self):
        calls = []
        def inspect(path):
            calls.append(path)
            return True, "ok"
        validate_dataset_source(
            {"train_data_dir": "images"}, "sd-lora",
            is_file=lambda path: False, is_dir=lambda path: True, inspect_data_dir=inspect,
        )
        self.assertEqual(calls, ["images"])

    def test_empty_optional_paths_are_removed(self):
        config = {"dataset_config": "", "in_json": None, "conditioning_data_dir": "", "train_data_dir": "images"}
        normalize_optional_dataset_paths(config)
        for key in ("dataset_config", "in_json", "conditioning_data_dir"):
            self.assertNotIn(key, config)


if __name__ == "__main__":
    unittest.main()
