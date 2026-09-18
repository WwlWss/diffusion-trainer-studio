import unittest
from pathlib import Path

from mikazuki.training_config import PreparedTrainingConfig, prepare_training_config
from mikazuki.training_gui_args import apply_raw_gui_semantics
from mikazuki.training_rehydrate import rehydrate_trainer_config
from mikazuki.training_validation import validate_prepared_config


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "mikazuki/schema/flux-lora.ts").read_text(encoding="utf-8")
ANIMA_ARGS = (ROOT / "sd-scripts/library/anima_train_utils.py").read_text(encoding="utf-8-sig")
ANIMA_NETWORK = (ROOT / "sd-scripts/networks/lora_anima.py").read_text(encoding="utf-8-sig")
ANIMA_NETWORK_TRAINER = (ROOT / "sd-scripts/anima_train_network.py").read_text(encoding="utf-8-sig")
ANIMA_FULL_TRAINER = (ROOT / "sd-scripts/anima_train.py").read_text(encoding="utf-8-sig")
GENERIC_ARGS = (ROOT / "sd-scripts/library/args.py").read_text(encoding="utf-8-sig")
TRAIN_NETWORK = (ROOT / "sd-scripts/train_network.py").read_text(encoding="utf-8-sig")


def fake_resolve_backend(config, requested):
    return requested, f"./{requested}.py"


class AnimaUiParityTests(unittest.TestCase):
    def prepare(self, config, page):
        return prepare_training_config(
            config,
            page_train_type=page,
            resolve_backend=fake_resolve_backend,
        )

    def test_scientific_notation_gui_values_become_toml_numbers(self):
        raw = apply_raw_gui_semantics(
            {
                "self_attn_lr": "1e-6",
                "cross_attn_lr": "2e-6",
                "mlp_lr": "3e-6",
                "mod_lr": "4e-6",
                "llm_adapter_lr": "5e-7",
                "qwen3_lr": "2e-7",
                "ip_noise_gamma": "0.1",
                "logit_mean": "-0.2",
                "logit_std": "1.1",
                "mode_scale": "1.29",
            }
        )
        for key in (
            "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr",
            "llm_adapter_lr", "qwen3_lr", "ip_noise_gamma",
            "logit_mean", "logit_std", "mode_scale",
        ):
            with self.subTest(key=key):
                self.assertIsInstance(raw[key], float)

    def test_post_override_numeric_values_are_also_coerced(self):
        prepared = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigmoid",
                "ui_custom_params": 'ip_noise_gamma = "0.15"\n',
            },
            "anima-lora",
        )
        self.assertEqual(prepared.config["ip_noise_gamma"], 0.15)
        self.assertIsInstance(prepared.config["ip_noise_gamma"], float)

    def test_full_component_lrs_survive_as_numeric_trainer_values(self):
        prepared = self.prepare(
            {
                "anima_model_variant": "base",
                "anima_finetune_learning_rate": "1e-6",
                "anima_precision_mode": "mixed_bf16",
                "anima_latent_cache_mode": "off",
                "anima_text_encoder_cache_mode": "off",
                "anima_checkpoint_mode": "standard",
                "optimizer_type": "AdamW8bit",
                "lr_scheduler": "constant",
                "timestep_sampling": "sigmoid",
                "self_attn_lr": "1e-6",
                "cross_attn_lr": "8e-7",
                "mlp_lr": "9e-7",
                "mod_lr": "7e-7",
                "llm_adapter_lr": "5e-7",
            },
            "anima-finetune",
        )
        expected = {
            "self_attn_lr": 1e-6,
            "cross_attn_lr": 8e-7,
            "mlp_lr": 9e-7,
            "mod_lr": 7e-7,
            "llm_adapter_lr": 5e-7,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(prepared.config[key], value)
                self.assertIsInstance(prepared.config[key], float)

    def test_logit_normal_is_effective_only_with_sigma_sampling(self):
        prepared = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigma",
                "weighting_scheme": "logit_normal",
                "logit_mean": "-0.3",
                "logit_std": "1.2",
            },
            "anima-lora",
        )
        self.assertEqual(prepared.config["weighting_scheme"], "logit_normal")
        self.assertEqual(prepared.config["logit_mean"], -0.3)
        self.assertEqual(prepared.config["logit_std"], 1.2)
        with self.assertRaisesRegex(ValueError, "timestep_sampling=sigma"):
            self.prepare(
                {
                    "anima_model_variant": "base",
                    "learning_rate": "5e-5",
                    "anima_lora_target": "dit",
                    "timestep_sampling": "sigmoid",
                    "weighting_scheme": "logit_normal",
                },
                "anima-lora",
            )

    def test_lora_network_gui_materializes_real_network_args(self):
        prepared = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigmoid",
                "anima_lora_train_llm_adapter": True,
                "anima_lora_rank_dropout": "0.1",
                "anima_lora_module_dropout": "0.2",
                "anima_lora_include_patterns": "blocks\\.0\nblocks\\.1",
                "anima_lora_exclude_patterns": "final_layer",
                "anima_lora_network_reg_dims": "blocks.0=16,blocks.1=8",
                "anima_lora_network_reg_lrs": "blocks.0=1e-4,blocks.1=5e-5",
                "anima_lora_loraplus_lr_ratio": "2.0",
            },
            "anima-lora",
        )
        args = prepared.config["network_args"]
        self.assertIn("train_llm_adapter=true", args)
        self.assertIn("rank_dropout=0.1", args)
        self.assertIn("module_dropout=0.2", args)
        self.assertTrue(any(x.startswith("include_patterns=") for x in args))
        self.assertTrue(any(x.startswith("exclude_patterns=") for x in args))
        self.assertIn("network_reg_dims=blocks.0=16,blocks.1=8", args)
        self.assertIn("network_reg_lrs=blocks.0=1e-4,blocks.1=5e-5", args)
        self.assertIn("loraplus_lr_ratio=2.0", args)
        for semantic in (
            "anima_lora_train_llm_adapter",
            "anima_lora_rank_dropout",
            "anima_lora_module_dropout",
            "anima_lora_include_patterns",
        ):
            self.assertNotIn(semantic, prepared.config)

    def test_lora_checkpoint_and_compile_modes_materialize_raw_args(self):
        cpu = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigmoid",
                "anima_lora_checkpoint_mode": "cpu",
                "anima_lora_compile_mode": "off",
            },
            "anima-lora",
        ).config
        self.assertTrue(cpu["gradient_checkpointing"])
        self.assertTrue(cpu["cpu_offload_checkpointing"])
        self.assertFalse(cpu["unsloth_offload_checkpointing"])

        compiled = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigmoid",
                "anima_lora_checkpoint_mode": "standard",
                "anima_lora_compile_mode": "per_block",
                "compile_backend": "inductor",
                "compile_mode": "default",
                "compile_dynamic": "auto",
            },
            "anima-lora",
        ).config
        self.assertTrue(compiled["compile"])
        self.assertFalse(compiled["torch_compile"])
        self.assertEqual(compiled["compile_backend"], "inductor")

    def test_timestep_diagnostic_launch_does_not_require_training_assets(self):
        prepared = PreparedTrainingConfig(
            train_type="anima-lora",
            trainer_file="./sd-scripts/anima_train_network.py",
            config={"show_timesteps": "console", "timestep_sampling": "sigmoid"},
        )
        validate_prepared_config(
            prepared,
            True,
            exists=lambda path: path == "./sd-scripts/anima_train_network.py",
            is_file=lambda path: False,
            is_dir=lambda path: False,
            inspect_data_dir=lambda path: (False, "must not inspect data"),
            validate_model=lambda path, train_type: (_ for _ in ()).throw(AssertionError("model validation must not run")),
        )

    def test_anima_lora_rehydrate_round_trips_new_semantic_controls(self):
        effective = {
            "learning_rate": 5e-5,
            "network_module": "networks.lora_anima",
            "network_train_unet_only": True,
            "gradient_checkpointing": True,
            "cpu_offload_checkpointing": True,
            "unsloth_offload_checkpointing": False,
            "compile": True,
            "compile_backend": "inductor",
            "compile_mode": "default",
            "network_args": [
                "train_llm_adapter=true",
                "rank_dropout=0.1",
                "module_dropout=0.2",
                "include_patterns=['blocks\\\\.0', 'blocks\\\\.1']",
                "network_reg_dims=blocks.0=16,blocks.1=8",
                "loraplus_lr_ratio=2.0",
                "custom_future_arg=keep-me",
            ],
            "timestep_sampling": "sigmoid",
        }
        gui = rehydrate_trainer_config(effective, "anima-lora")
        self.assertEqual(gui["anima_lora_checkpoint_mode"], "cpu")
        self.assertEqual(gui["anima_lora_compile_mode"], "per_block")
        self.assertTrue(gui["anima_lora_train_llm_adapter"])
        self.assertEqual(gui["anima_lora_rank_dropout"], "0.1")
        self.assertEqual(gui["anima_lora_module_dropout"], "0.2")
        self.assertEqual(gui["anima_lora_include_patterns"], "blocks\\.0\nblocks\\.1")
        self.assertEqual(gui["anima_lora_network_reg_dims"], "blocks.0=16,blocks.1=8")
        self.assertEqual(gui["anima_lora_loraplus_lr_ratio"], "2.0")
        self.assertIn("custom_future_arg=keep-me", gui["network_args_custom"])

        prepared = self.prepare(gui, "anima-lora")
        self.assertTrue(prepared.config["cpu_offload_checkpointing"])
        self.assertTrue(prepared.config["compile"])
        self.assertIn("train_llm_adapter=true", prepared.config["network_args"])
        self.assertIn("rank_dropout=0.1", prepared.config["network_args"])
        self.assertIn("custom_future_arg=keep-me", prepared.config["network_args"])

    def test_timestep_diagnostic_off_is_not_written_to_trainer_toml(self):
        off = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigmoid",
                "show_timesteps": "off",
            },
            "anima-lora",
        ).config
        self.assertNotIn("show_timesteps", off)

        console = self.prepare(
            {
                "anima_model_variant": "base",
                "learning_rate": "5e-5",
                "anima_lora_target": "dit",
                "timestep_sampling": "sigmoid",
                "show_timesteps": "console",
                "show_timesteps_resolution": "1024,768",
                "show_timesteps_offset": 0.1,
            },
            "anima-lora",
        ).config
        self.assertEqual(console["show_timesteps"], "console")
        self.assertEqual(console["show_timesteps_resolution"], "1024,768")
        self.assertEqual(console["show_timesteps_offset"], 0.1)

    def test_low_frequency_anima_sections_default_to_collapsed(self):
        collapsed_markers = (
            'description("分组件学习率（高级；大多数训练留空）").collapse()',
            'description("优化器与调度器高级参数（通常留空）").collapse()',
            'description("高级数据集控制（通常不需要）").collapse()',
            'description("LoRA 网络高级选项（大多数训练保持默认/留空）").collapse()',
            'description("LoRA 性能与编译高级选项（通常保持默认）").collapse()',
            'description("Caption 高级增强（通常保持默认/留空）").collapse()',
            'description("模型 Metadata（发布时再填写）").collapse()',
            'description("Hugging Face 保存/恢复（不用云端训练时保持折叠）").collapse()',
            'description("其他高级设置").collapse()',
        )
        for marker in collapsed_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, SCHEMA)

        # Frequently adjusted training controls stay immediately visible.
        self.assertIn('description("Anima 全参微调学习率与优化器"),', SCHEMA)
        self.assertIn('description("Anima LoRA 学习率与优化器"),', SCHEMA)

    def test_schema_controls_are_backed_by_pinned_sd_scripts(self):
        schema_fields = (
            "llm_adapter_lr",
            "logit_mean", "logit_std", "mode_scale",
            "ip_noise_gamma", "ip_noise_gamma_random_strength",
            "show_timesteps", "show_timesteps_resolution", "show_timesteps_offset",
            "dataset_repeats", "debug_dataset", "dataset_class",
            "text_encoder_batch_size", "skip_cache_check",
            "validation_split", "validate_every_n_steps", "validate_every_n_epochs",
        )
        for field in schema_fields:
            with self.subTest(field=field):
                self.assertIn(field, SCHEMA)

        self.assertIn("--llm_adapter_lr", ANIMA_ARGS)
        for arg in (
            "--show_timesteps", "--show_timesteps_resolution", "--show_timesteps_offset",
            "--ip_noise_gamma", "--ip_noise_gamma_random_strength",
            "--dataset_repeats", "--debug_dataset", "--dataset_class",
            "--text_encoder_batch_size", "--skip_cache_check",
        ):
            self.assertIn(arg, GENERIC_ARGS)
        for arg in (
            "--validation_split", "--validate_every_n_steps", "--validate_every_n_epochs",
        ):
            self.assertIn(arg, TRAIN_NETWORK)

        for marker in (
            'kwargs.get("train_llm_adapter"',
            'kwargs.get("rank_dropout"',
            'kwargs.get("module_dropout"',
            'kwargs.get("network_reg_dims"',
            'kwargs.get("network_reg_lrs"',
            'kwargs.get("loraplus_lr_ratio"',
        ):
            self.assertIn(marker, ANIMA_NETWORK)
        self.assertIn("if args.compile:", ANIMA_NETWORK_TRAINER)
        self.assertIn("args.llm_adapter_lr", ANIMA_FULL_TRAINER)

    def test_non_anima_page_does_not_receive_anima_semantics(self):
        prepared = self.prepare(
            {
                "learning_rate": "1e-4",
                "mixed_precision": "bf16",
            },
            "flux-finetune",
        )
        self.assertEqual(prepared.train_type, "flux-finetune")
        self.assertNotIn("anima_lora_compile_mode", prepared.config)
        self.assertNotIn("llm_adapter_lr", prepared.config)


if __name__ == "__main__":
    unittest.main()
