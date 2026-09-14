import unittest

from mikazuki.training_config import prepare_training_config
from mikazuki.training_gui_args import apply_raw_gui_semantics
from mikazuki.training_rehydrate import rehydrate_trainer_config


def _resolve(config, requested):
    return requested, f"trainer/{requested}.py"


class TrainingSemanticContractTests(unittest.TestCase):
    def test_legacy_network_optimizer_and_paths_are_compiled_in_python(self):
        raw = {
            "pretrained_model_name_or_path": r"C:\models\base.safetensors",
            "network_module": "lycoris.kohya",
            "lycoris_algo": "lokr",
            "conv_dim": 16,
            "conv_alpha": 8,
            "lokr_factor": 4,
            "enable_block_weights": True,
            "down_lr_weight": "1,0.5",
            "optimizer_type": "Prodigy",
            "prodigy_d0": "1e-6",
            "prodigy_d_coef": 2,
            "lr_warmup_steps": 10,
        }
        got = apply_raw_gui_semantics(raw, page_train_type="lora-master")
        self.assertEqual(got["pretrained_model_name_or_path"], "C:/models/base.safetensors")
        self.assertIn("algo=lokr", got["network_args"])
        self.assertIn("factor=4", got["network_args"])
        self.assertIn("down_lr_weight=1,0.5", got["network_args"])
        self.assertIn("decouple=True", got["optimizer_args"])
        self.assertIn("use_bias_correction=True", got["optimizer_args"])
        self.assertIn("d_coef=2", got["optimizer_args"])
        self.assertIn("safeguard_warmup=True", got["optimizer_args"])

    def test_basic_hidden_defaults_moved_out_of_frontend(self):
        got = apply_raw_gui_semantics({"optimizer_type": "AdamW8bit"}, page_train_type="lora-basic")
        self.assertEqual(got["network_module"], "networks.lora")
        self.assertEqual(got["save_model_as"], "safetensors")
        self.assertTrue(got["enable_bucket"])
        self.assertEqual(got["caption_extension"], ".txt")
        self.assertNotIn("max_token_length", got)

    def test_workers_zero_disables_persistent(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "max_data_loader_n_workers": 0,
                "persistent_data_loader_workers": True,
            },
            page_train_type="lora-master",
            resolve_backend=_resolve,
        )
        self.assertFalse(prepared.config["persistent_data_loader_workers"])

    def test_dataset_config_excludes_folder_fields(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "dataset_source": "config",
                "dataset_config": "dataset.toml",
                "train_data_dir": "ignored",
                "reg_data_dir": "ignored-reg",
                "in_json": "ignored.json",
            },
            page_train_type="lora-master",
            resolve_backend=_resolve,
        )
        self.assertEqual(prepared.config["dataset_config"], "dataset.toml")
        self.assertNotIn("train_data_dir", prepared.config)
        self.assertNotIn("reg_data_dir", prepared.config)
        self.assertNotIn("in_json", prepared.config)

    def test_sd_token_75_is_omitted_and_legacy_255_migrates(self):
        first = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "sd_max_token_length_mode": "75"},
            page_train_type="lora-master", resolve_backend=_resolve,
        )
        self.assertNotIn("max_token_length", first.config)
        second = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "max_token_length": 255},
            page_train_type="lora-master", resolve_backend=_resolve,
        )
        self.assertEqual(second.config["max_token_length"], 225)

    def test_lora_target_is_single_semantic_control(self):
        prepared = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "lora_target": "unet"},
            page_train_type="sdxl-lora", resolve_backend=_resolve,
        )
        self.assertTrue(prepared.config["network_train_unet_only"])
        self.assertNotIn("network_train_text_encoder_only", prepared.config)

    def test_flux_target_materializes_t5_network_arg(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "flux_lora_target": "dit_clip_l_t5xxl",
            },
            page_train_type="flux-lora", resolve_backend=_resolve,
        )
        self.assertFalse(prepared.config["network_train_unet_only"])
        self.assertIn("train_t5xxl=True", prepared.config["network_args"])

    def test_anima_full_lowram_rejected_before_trainer(self):
        with self.assertRaisesRegex(ValueError, "Low RAM|lowram"):
            prepare_training_config(
                {"memory_mode": "lowram"},
                page_train_type="anima-finetune",
                resolve_backend=_resolve,
            )

    def test_rehydrate_consumes_network_args_and_recompiles_equivalently(self):
        effective = {
            "optimizer_type": "Prodigy",
            "network_module": "lycoris.kohya",
            "network_args": ["algo=lokr", "factor=4", "custom=ok"],
            "optimizer_args": ["decouple=True", "weight_decay=0.01", "use_bias_correction=True", "d_coef=2"],
            "network_train_unet_only": True,
            "max_token_length": 225,
        }
        gui = rehydrate_trainer_config(effective, "lora-master")
        self.assertNotIn("network_args", gui)
        self.assertEqual(gui["lycoris_algo"], "lokr")
        self.assertEqual(gui["lora_target"], "unet")
        self.assertEqual(gui["sd_max_token_length_mode"], "225")
        self.assertIn("custom=ok", gui["network_args_custom"])
        rebuilt = prepare_training_config(gui, page_train_type="lora-master", resolve_backend=_resolve).config
        self.assertIn("algo=lokr", rebuilt["network_args"])
        self.assertIn("factor=4", rebuilt["network_args"])
        self.assertIn("custom=ok", rebuilt["network_args"])
        self.assertTrue(rebuilt["network_train_unet_only"])
        self.assertEqual(rebuilt["max_token_length"], 225)


if __name__ == "__main__":
    unittest.main()
