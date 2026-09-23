import unittest
from copy import deepcopy

from mikazuki.training_config import prepare_training_config
from mikazuki.training_gui_args import apply_raw_gui_semantics
from mikazuki.training_gui_semantics import (
    normalize_flux_lora_target,
    normalize_sd3_lora_target,
    normalize_sd_lora_target,
)
from mikazuki.training_rehydrate import rehydrate_trainer_config


def _resolve(config, requested):
    return requested, f"trainer/{requested}.py"


class TrainingSemanticContractTests(unittest.TestCase):
    def test_legacy_network_optimizer_and_paths_are_compiled_in_python(self):
        raw = {
            "pretrained_model_name_or_path": "C:\\models\\base.safetensors",
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
        self.assertEqual(got["save_precision"], "fp16")
        self.assertTrue(got["enable_bucket"])
        self.assertEqual(got["caption_extension"], ".txt")
        self.assertNotIn("max_token_length", got)

    def test_legacy_float_strings_are_numeric_in_effective_config(self):
        got = apply_raw_gui_semantics(
            {"learning_rate": "1e-4", "guidance_scale": "3.5"},
            page_train_type="lora-master",
        )
        self.assertEqual(got["learning_rate"], 1e-4)
        self.assertEqual(got["guidance_scale"], 3.5)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            apply_raw_gui_semantics({"learning_rate": "not-a-number"}, page_train_type="lora-master")

    def test_optional_legacy_zero_fields_are_omitted(self):
        got = apply_raw_gui_semantics(
            {"noise_offset": 0, "network_dropout": 0, "vae": "", "optimizer_type": "AdamW8bit"},
            page_train_type="lora-master",
        )
        self.assertNotIn("noise_offset", got)
        self.assertNotIn("network_dropout", got)
        self.assertNotIn("vae", got)

    def test_workers_zero_disables_persistent(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "lora_target": "unet",
                "max_data_loader_n_workers": 0,
                "persistent_data_loader_workers": True,
            },
            page_train_type="lora-master",
            resolve_backend=_resolve,
        )
        self.assertFalse(prepared.config["persistent_data_loader_workers"])

    def test_workers_zero_cannot_be_reenabled_by_custom_toml(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "lora_target": "unet",
                "max_data_loader_n_workers": 0,
                "ui_custom_params": "persistent_data_loader_workers = true",
            },
            page_train_type="lora-master",
            resolve_backend=_resolve,
        )
        self.assertFalse(prepared.config["persistent_data_loader_workers"])

    def test_dataset_config_excludes_folder_fields(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "lora_target": "unet",
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

    def test_folder_dataset_source_discards_stale_dataset_config(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "lora_target": "unet",
                "dataset_source": "folder",
                "dataset_config": "stale.toml",
                "train_data_dir": "train/10_test",
            },
            page_train_type="lora-master",
            resolve_backend=_resolve,
        )
        self.assertNotIn("dataset_config", prepared.config)
        self.assertEqual(prepared.config["train_data_dir"], "train/10_test")

    def test_sd_token_75_is_omitted_and_legacy_255_migrates(self):
        first = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "lora_target": "unet", "sd_max_token_length_mode": "75"},
            page_train_type="lora-master", resolve_backend=_resolve,
        )
        self.assertNotIn("max_token_length", first.config)
        second = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "lora_target": "unet", "max_token_length": 255},
            page_train_type="lora-master", resolve_backend=_resolve,
        )
        self.assertEqual(second.config["max_token_length"], 225)

    def test_invalid_sd_token_length_is_rejected_even_from_custom_toml(self):
        with self.assertRaisesRegex(ValueError, "max_token_length"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW8bit",
                    "lora_target": "unet",
                    "ui_custom_params": "max_token_length = 77",
                },
                page_train_type="lora-master",
                resolve_backend=_resolve,
            )

    def test_lora_target_is_single_semantic_control(self):
        prepared = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "lora_target": "unet"},
            page_train_type="sdxl-lora", resolve_backend=_resolve,
        )
        self.assertTrue(prepared.config["network_train_unet_only"])
        self.assertNotIn("network_train_text_encoder_only", prepared.config)

    def test_custom_toml_cannot_make_both_sd_lora_only_flags_true(self):
        with self.assertRaisesRegex(ValueError, "不能同时"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW8bit",
                    "lora_target": "unet",
                    "ui_custom_params": (
                        "network_train_unet_only = true\n"
                        "network_train_text_encoder_only = true\n"
                    ),
                },
                page_train_type="sdxl-lora",
                resolve_backend=_resolve,
            )

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

    def test_sd3_target_materializes_real_trainer_flags_and_t5_network_arg(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW",
                "sd3_lora_target": "mmdit_text_encoder",
                "train_t5xxl": True,
                "cache_text_encoder_outputs": False,
            },
            page_train_type="sd3-lora",
            resolve_backend=_resolve,
        )
        self.assertFalse(prepared.config.get("network_train_unet_only", False))
        self.assertFalse(prepared.config.get("network_train_text_encoder_only", False))
        self.assertNotIn("train_t5xxl", prepared.config)
        self.assertIn("train_t5xxl=True", prepared.config["network_args"])

    def test_sd3_semantic_target_is_reasserted_after_custom_overrides(self):
        with self.assertRaisesRegex(ValueError, "不能同时"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW",
                    "sd3_lora_target": "mmdit",
                    "train_t5xxl": False,
                    "ui_custom_params": (
                        "network_train_unet_only = true\n"
                        "network_train_text_encoder_only = true\n"
                    ),
                },
                page_train_type="sd3-lora",
                resolve_backend=_resolve,
            )

    def test_sd3_t5_custom_override_rewrites_existing_network_arg(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW",
                "sd3_lora_target": "mmdit_text_encoder",
                "train_t5xxl": False,
                "network_args_custom": ["train_t5xxl=True", "custom=ok"],
                "ui_custom_params": "train_t5xxl = true",
            },
            page_train_type="sd3-lora",
            resolve_backend=_resolve,
        )
        self.assertEqual(
            [item for item in prepared.config["network_args"] if item.startswith("train_t5xxl=")],
            ["train_t5xxl=True"],
        )
        self.assertIn("custom=ok", prepared.config["network_args"])

    def test_sd3_rehydrate_round_trip_preserves_target_and_t5(self):
        effective = {
            "optimizer_type": "AdamW",
            "network_module": "networks.lora_sd3",
            "network_train_unet_only": False,
            "network_train_text_encoder_only": False,
            "network_args": ["train_t5xxl=True", "custom=ok"],
        }
        gui = rehydrate_trainer_config(effective, "sd3-lora")
        self.assertEqual(gui["sd3_lora_target"], "mmdit_text_encoder")
        self.assertTrue(gui["train_t5xxl"])
        self.assertIn("custom=ok", gui["network_args_custom"])

        rebuilt = prepare_training_config(
            gui,
            page_train_type="sd3-lora",
            resolve_backend=_resolve,
        ).config
        self.assertIn("train_t5xxl=True", rebuilt["network_args"])
        self.assertIn("custom=ok", rebuilt["network_args"])
        self.assertNotIn("train_t5xxl", rebuilt)

    def test_rehydrate_preserves_lowercase_true_as_disabled_t5(self):
        cases = (
            (
                "flux-lora",
                {
                    "optimizer_type": "AdamW",
                    "network_module": "networks.lora_flux",
                    "network_train_unet_only": False,
                    "network_args": ["train_t5xxl=true", "custom=ok"],
                },
            ),
            (
                "sd3-lora",
                {
                    "optimizer_type": "AdamW",
                    "network_module": "networks.lora_sd3",
                    "network_train_unet_only": False,
                    "network_train_text_encoder_only": False,
                    "network_args": ["train_t5xxl=true", "custom=ok"],
                },
            ),
        )
        for train_type, effective in cases:
            with self.subTest(train_type=train_type):
                gui = rehydrate_trainer_config(effective, train_type)
                if train_type == "sd3-lora":
                    self.assertFalse(gui["train_t5xxl"])
                else:
                    self.assertEqual(gui["flux_lora_target"], "dit_clip_l")
                rebuilt = prepare_training_config(
                    gui,
                    page_train_type=train_type,
                    resolve_backend=_resolve,
                ).config
                self.assertNotIn(
                    "train_t5xxl=True",
                    rebuilt.get("network_args", []),
                )
                self.assertIn("custom=ok", rebuilt["network_args"])

    def test_lora_target_normalizers_are_idempotent(self):
        cases = (
            (
                normalize_sd_lora_target,
                {"lora_target": "unet_text_encoder"},
                {},
            ),
            (
                normalize_flux_lora_target,
                {
                    "flux_lora_target": "dit_clip_l_t5xxl",
                    "network_args": ["custom=ok"],
                },
                {"chroma": False},
            ),
            (
                normalize_flux_lora_target,
                {
                    "flux_lora_target": "dit_t5xxl",
                    "network_args": ["custom=ok"],
                },
                {"chroma": True},
            ),
            (
                normalize_sd3_lora_target,
                {
                    "sd3_lora_target": "mmdit_text_encoder",
                    "train_t5xxl": True,
                    "network_args": ["custom=ok"],
                },
                {},
            ),
        )
        for normalizer, raw, kwargs in cases:
            with self.subTest(normalizer=normalizer.__name__, raw=raw):
                config = deepcopy(raw)
                normalizer(config, **kwargs)
                once = deepcopy(config)
                normalizer(config, **kwargs)
                self.assertEqual(config, once)

    def test_flux_custom_network_args_cannot_bypass_target_invariant(self):
        with self.assertRaisesRegex(ValueError, "train_t5xxl"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit",
                    "ui_custom_params": 'network_args = ["train_t5xxl=True"]',
                },
                page_train_type="flux-lora",
                resolve_backend=_resolve,
            )

    def test_chroma_custom_network_args_cannot_bypass_target_invariant(self):
        with self.assertRaisesRegex(ValueError, "train_t5xxl"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit",
                    "ui_custom_params": 'network_args = ["train_t5xxl=True"]',
                },
                page_train_type="chroma-lora",
                resolve_backend=_resolve,
            )

    def test_sd3_custom_network_args_cannot_bypass_target_invariant(self):
        with self.assertRaisesRegex(ValueError, "Train T5XXL"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW",
                    "sd3_lora_target": "mmdit",
                    "train_t5xxl": False,
                    "ui_custom_params": 'network_args = ["train_t5xxl=True"]',
                },
                page_train_type="sd3-lora",
                resolve_backend=_resolve,
            )

    def test_sd3_standard_target_t5_matrix_and_round_trip(self):
        cases = (
            ("mmdit", False, True, False, False),
            ("text_encoder", False, False, True, False),
            ("text_encoder", True, False, True, True),
            ("mmdit_text_encoder", False, False, False, False),
            ("mmdit_text_encoder", True, False, False, True),
        )
        for target, train_t5, unet_only, te_only, effective_t5 in cases:
            with self.subTest(target=target, train_t5=train_t5):
                prepared = prepare_training_config(
                    {
                        "optimizer_type": "AdamW",
                        "sd3_lora_target": target,
                        "train_t5xxl": train_t5,
                    },
                    page_train_type="sd3-lora",
                    resolve_backend=_resolve,
                )
                effective = prepared.config
                self.assertEqual(
                    bool(effective.get("network_train_unet_only", False)),
                    unet_only,
                )
                self.assertEqual(
                    bool(effective.get("network_train_text_encoder_only", False)),
                    te_only,
                )
                self.assertEqual(
                    "train_t5xxl=True" in effective.get("network_args", []),
                    effective_t5,
                )

                gui = rehydrate_trainer_config(effective, "sd3-lora")
                rebuilt = prepare_training_config(
                    gui,
                    page_train_type="sd3-lora",
                    resolve_backend=_resolve,
                ).config
                for key in (
                    "network_train_unet_only",
                    "network_train_text_encoder_only",
                    "network_args",
                ):
                    self.assertEqual(
                        effective.get(key),
                        rebuilt.get(key),
                        (target, train_t5, key),
                    )

    def test_sd3_mmdit_target_rejects_t5_training(self):
        with self.assertRaisesRegex(ValueError, "Train T5XXL"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW",
                    "sd3_lora_target": "mmdit",
                    "train_t5xxl": True,
                },
                page_train_type="sd3-lora",
                resolve_backend=_resolve,
            )

    def test_standard_text_encoder_cache_capability_matrix(self):
        allowed = (
            (
                "flux-lora",
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "flux-lora",
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit_clip_l",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "chroma-lora",
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "sd3-lora",
                {
                    "optimizer_type": "AdamW",
                    "sd3_lora_target": "mmdit_text_encoder",
                    "train_t5xxl": False,
                    "cache_text_encoder_outputs": True,
                },
            ),
        )
        for train_type, raw in allowed:
            with self.subTest(train_type=train_type, raw=raw, expected="allowed"):
                prepared = prepare_training_config(
                    raw,
                    page_train_type=train_type,
                    resolve_backend=_resolve,
                )
                self.assertTrue(prepared.config["cache_text_encoder_outputs"])

        blocked = (
            (
                "sdxl-lora",
                {
                    "optimizer_type": "AdamW",
                    "lora_target": "unet_text_encoder",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "flux-lora",
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit_clip_l_t5xxl",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "chroma-lora",
                {
                    "optimizer_type": "AdamW",
                    "flux_lora_target": "dit_t5xxl",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "sd3-lora",
                {
                    "optimizer_type": "AdamW",
                    "sd3_lora_target": "mmdit_text_encoder",
                    "train_t5xxl": True,
                    "cache_text_encoder_outputs": True,
                },
            ),
        )
        for train_type, raw in blocked:
            with self.subTest(train_type=train_type, raw=raw, expected="blocked"):
                with self.assertRaisesRegex(
                    ValueError,
                    "Text Encoder|T5XXL|文本编码器|缓存",
                ):
                    prepare_training_config(
                        raw,
                        page_train_type=train_type,
                        resolve_backend=_resolve,
                    )

    def test_lora_t5_network_arg_preserves_exact_true_wire_semantics(self):
        for train_type in ("flux-lora", "sd3-lora"):
            with self.subTest(train_type=train_type):
                prepared = prepare_training_config(
                    {
                        "optimizer_type": "AdamW",
                        "network_args_custom": ["train_t5xxl=true", "custom=ok"],
                    },
                    page_train_type=train_type,
                    resolve_backend=_resolve,
                )
                self.assertNotIn(
                    "train_t5xxl=True",
                    prepared.config.get("network_args", []),
                )
                self.assertIn("custom=ok", prepared.config["network_args"])

    def test_flux_and_chroma_custom_false_clears_first_pass_t5_target(self):
        cases = (
            ("flux-lora", "dit_clip_l_t5xxl"),
            ("chroma-lora", "dit_t5xxl"),
        )
        for train_type, target in cases:
            with self.subTest(train_type=train_type):
                prepared = prepare_training_config(
                    {
                        "optimizer_type": "AdamW",
                        "flux_lora_target": target,
                        "ui_custom_params": "train_t5xxl = false",
                    },
                    page_train_type=train_type,
                    resolve_backend=_resolve,
                )
                self.assertNotIn(
                    "train_t5xxl=True",
                    prepared.config.get("network_args", []),
                )
                if train_type == "chroma-lora":
                    self.assertTrue(prepared.config["network_train_unet_only"])

    def test_dadapt_rewrites_active_lr_but_prodigy_does_not(self):
        dadapt = prepare_training_config(
            {
                "optimizer_type": "DAdaptation",
                "lora_target": "unet",
                "learning_rate": "0.001",
                "unet_lr": "0.002",
                "text_encoder_lr": "0.003",
            },
            page_train_type="lora-master", resolve_backend=_resolve,
        )
        self.assertEqual(dadapt.config["learning_rate"], 1.0)
        self.assertEqual(dadapt.config["unet_lr"], 1.0)
        self.assertEqual(dadapt.config["text_encoder_lr"], 1.0)

        prodigy = prepare_training_config(
            {
                "optimizer_type": "Prodigy",
                "lora_target": "unet",
                "learning_rate": "0.001",
                "unet_lr": "0.002",
                "text_encoder_lr": "0.003",
            },
            page_train_type="lora-master", resolve_backend=_resolve,
        )
        self.assertEqual(prodigy.config["learning_rate"], 0.001)
        self.assertEqual(prodigy.config["unet_lr"], 0.002)
        self.assertEqual(prodigy.config["text_encoder_lr"], 0.003)

    def test_legacy_frontend_conflicts_are_backend_contracts(self):
        cases = (
            ({"cache_text_encoder_outputs": True, "shuffle_caption": True}, "shuffle_caption"),
            ({"cache_latents": True, "color_aug": True}, "color_aug"),
            ({"cache_latents": True, "random_crop": True}, "random_crop"),
            ({"noise_offset": 0.1, "multires_noise_iterations": 6}, "noise_offset"),
        )
        for extra, expected in cases:
            with self.subTest(extra=extra):
                raw = {"optimizer_type": "AdamW8bit", "lora_target": "unet", **extra}
                with self.assertRaisesRegex(ValueError, expected):
                    prepare_training_config(raw, page_train_type="lora-master", resolve_backend=_resolve)

    def test_oft_is_sdxl_only(self):
        with self.assertRaisesRegex(ValueError, "OFT"):
            prepare_training_config(
                {"optimizer_type": "AdamW8bit", "network_module": "networks.oft", "lora_target": "unet"},
                page_train_type="lora-master", resolve_backend=_resolve,
            )
        prepared = prepare_training_config(
            {"optimizer_type": "AdamW8bit", "network_module": "networks.oft", "lora_target": "unet"},
            page_train_type="sdxl-lora", resolve_backend=_resolve,
        )
        self.assertEqual(prepared.config["network_module"], "networks.oft")

    def test_custom_toml_cannot_bypass_legacy_conflicts(self):
        with self.assertRaisesRegex(ValueError, "color_aug"):
            prepare_training_config(
                {
                    "optimizer_type": "AdamW8bit",
                    "lora_target": "unet",
                    "cache_latents": True,
                    "ui_custom_params": "color_aug = true",
                },
                page_train_type="lora-master",
                resolve_backend=_resolve,
            )

    def test_full_page_custom_network_keys_are_stripped_again(self):
        prepared = prepare_training_config(
            {
                "optimizer_type": "AdamW8bit",
                "mixed_precision": "bf16",
                "ui_custom_params": 'network_module = "networks.lora"\nnetwork_dim = 64',
            },
            page_train_type="sdxl-full",
            resolve_backend=_resolve,
        )
        self.assertNotIn("network_module", prepared.config)
        self.assertNotIn("network_dim", prepared.config)

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
