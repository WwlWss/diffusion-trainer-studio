import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "mikazuki/schema/flux-lora.ts").read_text(encoding="utf-8")
TRAINER = (ROOT / "sd-scripts/anima_train.py").read_text(encoding="utf-8-sig")
ARGS = (ROOT / "sd-scripts/library/args.py").read_text(encoding="utf-8-sig")
ANIMA_ARGS = (ROOT / "sd-scripts/library/anima_train_utils.py").read_text(encoding="utf-8-sig")
DEEPSPEED = (ROOT / "sd-scripts/library/deepspeed_utils.py").read_text(encoding="utf-8-sig")
ACCELERATOR = (ROOT / "sd-scripts/library/accelerator_setup.py").read_text(encoding="utf-8-sig")


class AnimaGuiTrainerContractTests(unittest.TestCase):
    def test_full_trainer_has_no_lora_network_hyperparameters(self):
        # Full finetune uses DiT parameter groups, not the train_network LoRA
        # parser. If these ever appear here, the GUI split must be reviewed.
        for arg in (
            "--network_module",
            "--network_weights",
            "--network_dim",
            "--network_alpha",
            "--network_dropout",
            "--network_args",
        ):
            with self.subTest(arg=arg):
                self.assertNotIn(arg, TRAINER)

    def test_component_lr_contract_matches_full_trainer(self):
        for field in (
            "self_attn_lr",
            "cross_attn_lr",
            "mlp_lr",
            "mod_lr",
            "llm_adapter_lr",
        ):
            with self.subTest(field=field):
                self.assertIn(field, SCHEMA)
                self.assertIn(f"--{field}", ANIMA_ARGS)
                self.assertIn(f"args.{field}", TRAINER)

    def test_effective_full_finetune_controls_are_exposed(self):
        # These are not every generic argparse field; they are the controls that
        # materially affect the Anima full-finetune path and are appropriate for
        # an expert training GUI.
        gui_fields = (
            "max_train_steps",
            "max_train_epochs",
            "train_batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "max_data_loader_n_workers",
            "persistent_data_loader_workers",
            "fused_backward_pass",
            "highvram",
            "torch_compile",
            "dynamo_backend",
            "ddp_timeout",
            "ddp_gradient_as_bucket_view",
            "ddp_static_graph",
            "optimizer_type",
            "lr_scheduler",
            "lr_warmup_steps",
            "lr_decay_steps",
            "lr_scheduler_num_cycles",
            "lr_scheduler_power",
            "lr_scheduler_timescale",
            "lr_scheduler_min_lr_ratio",
            "lr_scheduler_args",
            "loss_type",
            "huber_schedule",
            "huber_c",
            "huber_scale",
            "weighting_scheme",
            "text_encoder_batch_size",
            "vae_batch_size",
            "dataset_config",
            "in_json",
            "skip_image_resolution",
            "resize_interpolation",
            "caption_separator",
            "secondary_separator",
            "enable_wildcard",
            "caption_prefix",
            "caption_suffix",
            "token_warmup_min",
            "token_warmup_step",
            "alpha_mask",
            "face_crop_aug_range",
            "masked_loss",
            "conditioning_data_dir",
            "skip_cache_check",
            "save_precision",
            "save_every_n_epochs",
            "save_every_n_steps",
            "save_n_epoch_ratio",
            "save_last_n_epochs",
            "save_last_n_steps",
            "save_state",
            "save_state_on_train_end",
            "save_last_n_epochs_state",
            "save_last_n_steps_state",
            "deepspeed",
            "zero_stage",
            "offload_optimizer_device",
            "offload_optimizer_nvme_path",
            "offload_param_device",
            "offload_param_nvme_path",
            "zero3_init_flag",
            "zero3_save_16bit_model",
            "fp16_master_weights_and_gradients",
        )
        for field in gui_fields:
            with self.subTest(field=field):
                self.assertIn(field, SCHEMA)

    def test_advanced_gui_features_are_backed_by_real_trainer_paths(self):
        self.assertIn("args.fused_backward_pass", TRAINER)
        self.assertIn("args.deepspeed", TRAINER)
        self.assertIn("--deepspeed", DEEPSPEED)
        self.assertIn("--zero_stage", DEEPSPEED)
        self.assertIn("args.highvram", ACCELERATOR)
        self.assertIn("args.torch_compile", ACCELERATOR)
        self.assertIn("static_graph=args.ddp_static_graph", ACCELERATOR)

    def test_anima_specific_model_controls_are_backed_by_parser(self):
        for field in (
            "qwen3",
            "llm_adapter_lr",
            "t5_tokenizer_path",
            "qwen3_max_token_length",
            "t5_max_token_length",
            "discrete_flow_shift",
            "timestep_sampling",
            "sigmoid_scale",
            "attn_mode",
            "split_attn",
            "vae_chunk_size",
            "vae_disable_cache",
            "qwen_image_vae_2d",
        ):
            with self.subTest(field=field):
                self.assertIn(field, SCHEMA)
                self.assertIn(f"--{field}", ANIMA_ARGS)

    def test_preview_uses_semantic_single_cadence(self):
        self.assertIn("anima_preview_cadence", SCHEMA)
        self.assertIn("anima_preview_interval", SCHEMA)
        # The raw fields still exist in sd-scripts, but are generated by the
        # host normalizer rather than shown as two competing Anima GUI inputs.
        self.assertIn("--sample_every_n_steps", ARGS)
        self.assertIn("--sample_every_n_epochs", ARGS)


if __name__ == "__main__":
    unittest.main()
