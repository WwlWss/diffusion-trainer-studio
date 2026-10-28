"""Host-only D1-SD3C2-A SD3.5 Medium scoped Full-BF16 admission tests.

There is no qualification promotion, model value load, Torch, or CUDA here.
A temporarily patched test-only qualification record exercises the future
release path while the real sd3-lora backend remains pending.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mikazuki.parameter_policy import serialize_parameter_policy, validate_parameter_policy
from tests.test_parameter_policy_request_contract import _real_prepare_request

from mikazuki import parameter_policy_execution as execution
from mikazuki import sd3_full_bf16_scope as sd3
from mikazuki.sd3_architecture_contract import (
    SD3Architecture,
    expected_sd3_mmdit_shapes,
)
from tools.prepare_sd3_component_baseline import make_policy


ROOT = Path(__file__).resolve().parents[1]


def r2_policy() -> dict:
    policy = make_policy()
    policy["optimizer_profiles"] = {
        "muon_primary": {"type": "Muon", "args": {}},
        "adamw_fallback": {"type": "AdamW", "args": {}},
    }
    for name in sd3.SD3_FULL_BF16_TRAINED_COMPONENTS:
        policy["components"][name] = {
            "train": True,
            "optimizer_profile": "muon_primary",
            "learning_rate": 0.0001,
            "fallback_optimizer_profile": "adamw_fallback",
            "fallback_learning_rate": 0.0001,
        }
    return validate_parameter_policy(policy)


def config(model: Path, *, accumulation: int = 1) -> dict:
    return {
        "full_bf16": True,
        "mixed_precision": "bf16",
        "pretrained_model_name_or_path": str(model),
        "network_module": "networks.lora_sd3",
        "network_train_unet_only": True,
        "network_train_text_encoder_only": False,
        "network_dim": 4,
        "network_alpha": 4,
        "train_batch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "resolution": "512,512",
        "max_data_loader_n_workers": 0,
        "save_model_as": "safetensors",
        "optimizer_type": "AdamW",
        "lr_scheduler": "constant",
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
    }


def shape_fixture(*, variant: str = "sd35-medium", embedded: bool = True) -> dict:
    specs = {
        "sd35-medium": (24, 1536, 384, "rms", tuple(range(13))),
        "sd3-medium": (24, 1536, 192, None, ()),
        "sd35-large-topology": (38, 2432, 192, "rms", ()),
    }
    depth, hidden, grid, norm, extras = specs[variant]
    arch = SD3Architecture(
        variant=variant, depth=depth, hidden_size=hidden,
        num_heads=hidden // 64, head_dim=64, patch_size=2,
        pos_embed_grid=grid, qk_norm=norm,
        extra_attention_layers=extras, shape_index_sha256="test-only",
    )
    shapes = {
        "model.diffusion_model." + name: list(shape)
        for name, shape in expected_sd3_mmdit_shapes(arch).items()
    }
    if embedded:
        shapes.update({key: [1] for key in sd3._EMBEDDED_SOURCE_KEYS.values()})
        shapes["first_stage_model.encoder.conv_in.weight"] = [1]
    return shapes


class SD3C2AScopeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "sd3.5_medium.safetensors"
        self.model.write_bytes(b"fake-header-fixture-never-loaded")
        self.shape_reader = patch.object(
            sd3, "_read_safetensors_shape_index", return_value=shape_fixture()
        )
        self.reader = self.shape_reader.start()
        self.addCleanup(self.shape_reader.stop)

    def test_r1_adamw_and_r2_muon_explicit_fallback_have_exact_scope(self):
        self.assertEqual(sd3.sd3_full_bf16_scope_blockers(make_policy(), config(self.model)), [])
        self.assertEqual(
            sd3.sd3_full_bf16_scope_blockers(r2_policy(), config(self.model, accumulation=2)),
            [],
        )
        self.assertEqual(self.reader.call_count, 2)

    def test_real_backend_remains_pending_even_with_valid_inputs(self):
        self.assertEqual(execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"].status, "pending")
        self.reader.reset_mock()
        blockers = execution.parameter_policy_execution_blockers(
            make_policy(), train_type="sd3-lora", effective_config=config(self.model),
        )
        self.assertEqual(len(blockers), 1)
        self.assertIn("pending", blockers[0])
        self.reader.assert_not_called()

    def test_test_only_qualified_record_uses_shared_preview_start_direct_gate(self):
        fake = execution.ExecutionFeatureQualification(
            "qualified", "test-only scoped admission", "test:sd3-c2a-scope",
        )
        with patch.dict(execution.FULL_BF16_BACKEND_QUALIFICATIONS, {"sd3-lora": fake}):
            self.assertEqual(
                execution.parameter_policy_execution_blockers(
                    make_policy(), train_type="sd3-lora",
                    effective_config=config(self.model),
                ),
                [],
            )
            self.assertEqual(
                execution.parameter_policy_execution_blockers(
                    r2_policy(), train_type="sd3-lora",
                    effective_config=config(self.model, accumulation=2),
                ),
                [],
            )
            bad_config = config(self.model)
            bad_config["network_train_unet_only"] = False
            self.assertIn(
                "network_train_unet_only=true",
                " ".join(execution.parameter_policy_execution_blockers(
                    make_policy(), train_type="sd3-lora", effective_config=bad_config,
                )),
            )
        self.assertEqual(execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"].status, "pending")

    def test_seven_components_and_exact_train_freeze_matrix(self):
        cases = {
            "train_clip_l": ("clip_l.adapter", True),
            "train_other": ("mmdit.other.adapter", True),
            "freeze_attention": ("mmdit.attention.adapter", False),
            "freeze_modulation": ("mmdit.modulation_norm.adapter", False),
        }
        for name, (component, trained) in cases.items():
            with self.subTest(name=name):
                policy = copy.deepcopy(make_policy())
                policy["components"][component]["train"] = trained
                self.assertIn("MMDiT", " ".join(sd3.sd3_full_bf16_scope_blockers(policy, config(self.model))))
        policy = copy.deepcopy(make_policy())
        del policy["components"]["t5xxl.adapter"]
        self.assertIn("seven", " ".join(sd3.sd3_full_bf16_scope_blockers(policy, config(self.model))))

    def test_r1_no_fallback_and_exact_optimizer_profile(self):
        policy = make_policy()
        policy["components"]["mmdit.attention.adapter"]["fallback_optimizer_profile"] = "baseline"
        self.assertIn("without fallback", " ".join(sd3.sd3_full_bf16_scope_blockers(policy, config(self.model))))
        policy = make_policy()
        policy["optimizer_profiles"]["baseline"]["type"] = "AdamW8bit"
        self.assertIn("permits only", " ".join(sd3.sd3_full_bf16_scope_blockers(policy, config(self.model))))
        policy = make_policy()
        policy["optimizer_profiles"]["unused"] = {"type": "AdamW", "args": {}}
        self.assertIn("permits only", " ".join(sd3.sd3_full_bf16_scope_blockers(policy, config(self.model))))

    def test_r2_requires_both_children_and_explicit_fallback_for_every_component(self):
        for mutation in ("no_adamw_child", "no_muon", "missing_fallback", "bad_fallback_lr", "misrouted_modulation"):
            with self.subTest(mutation=mutation):
                policy = r2_policy()
                if mutation == "no_adamw_child":
                    del policy["optimizer_profiles"]["adamw_fallback"]
                elif mutation == "no_muon":
                    del policy["optimizer_profiles"]["muon_primary"]
                elif mutation == "missing_fallback":
                    del policy["components"]["mmdit.mlp.adapter"]["fallback_optimizer_profile"]
                elif mutation == "bad_fallback_lr":
                    policy["components"]["mmdit.attention.adapter"]["fallback_learning_rate"] = 0
                elif mutation == "misrouted_modulation":
                    policy["components"]["mmdit.modulation_norm.adapter"]["optimizer_profile"] = "adamw_fallback"
                self.assertTrue(sd3.sd3_full_bf16_scope_blockers(policy, config(self.model, accumulation=2)))

    def test_case_specific_accumulation_cannot_be_swapped(self):
        for policy, accum in ((make_policy(), 2), (r2_policy(), 1), (r2_policy(), 4)):
            with self.subTest(accum=accum, types=list(policy["optimizer_profiles"])):
                blockers = sd3.sd3_full_bf16_scope_blockers(policy, config(self.model, accumulation=accum))
                self.assertIn("gradient_accumulation_steps", " ".join(blockers))

    def test_required_explicit_training_contract_fields_fail_closed(self):
        mutations = (
            ("network_module", "networks.lora"),
            ("network_train_unet_only", False),
            ("network_train_unet_only", "maybe"),
            ("network_train_unet_only", 2),
            ("network_dim", 8),
            ("network_alpha", 8),
            ("resolution", 1024),
            ("train_batch_size", 2),
            ("max_data_loader_n_workers", 4),
            ("lr_scheduler", "cosine"),
            ("optimizer_type", "Muon"),
            ("save_model_as", "ckpt"),
            ("cache_latents", False),
            ("cache_text_encoder_outputs", False),
            ("cache_text_encoder_outputs", "maybe"),
            ("gradient_checkpointing", True),
            ("fp8_base", True),
            ("fp8_base_unet", True),
            ("deepspeed", True),
            ("cpu_offload_checkpointing", True),
            ("blocks_to_swap", 2),
            ("fused_optimizer_groups", 2),
            ("enable_scaled_pos_embed", True),
            ("pos_emb_random_crop_rate", 0.1),
            ("network_dropout", 0.1),
            ("network_args", ["train_t5xxl=true"]),
            ("network_weights", str(self.model)),
            ("base_weights", str(self.model)),
            ("optimizer_args", ["betas=(0.9,0.99)"]),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                config_ = config(self.model)
                config_[field] = value
                self.assertTrue(sd3.sd3_full_bf16_scope_blockers(make_policy(), config_))

    def test_rejects_missing_relative_or_wrong_extension_model_path(self):
        for bad in (None, "", "sd3.5_medium.safetensors", str(self.root), str(self.root / "missing.safetensors")):
            with self.subTest(path=bad):
                config_ = config(self.model)
                config_["pretrained_model_name_or_path"] = bad
                self.assertIn("model", " ".join(sd3.sd3_full_bf16_scope_blockers(make_policy(), config_)).lower())

    def test_rejects_other_variants_and_incomplete_or_wrong_loader_roots(self):
        for variant in ("sd3-medium", "sd35-large-topology"):
            with self.subTest(variant=variant):
                self.reader.return_value = shape_fixture(variant=variant)
                self.assertIn("only", " ".join(sd3.sd3_full_bf16_scope_blockers(make_policy(), config(self.model))))
        shapes = shape_fixture()
        del shapes["model.diffusion_model.x_embedder.proj.weight"]
        self.reader.return_value = shapes
        self.assertIn("audit failed", " ".join(sd3.sd3_full_bf16_scope_blockers(make_policy(), config(self.model))))
        self.reader.return_value = shape_fixture()
        with patch.object(sd3, "_read_safetensors_shape_index", side_effect=OSError("corrupt header")):
            self.assertIn("audit failed", " ".join(sd3.sd3_full_bf16_scope_blockers(make_policy(), config(self.model))))

    def test_external_encoder_and_vae_paths_required_if_not_embedded(self):
        self.reader.return_value = shape_fixture(embedded=False)
        config_ = config(self.model)
        for name in ("clip_l", "clip_g", "t5xxl", "vae"):
            with self.subTest(missing=name):
                self.assertIn(name, " ".join(sd3.sd3_full_bf16_scope_blockers(make_policy(), config_)))
                file = self.root / (name + ".safetensors")
                file.write_bytes(b"test-fixture")
                config_[name] = str(file)
        self.assertEqual(sd3.sd3_full_bf16_scope_blockers(make_policy(), config_), [])

    def test_missing_safetensors_dependency_or_unreadable_header_is_fail_closed(self):
        with patch.object(sd3, "_read_safetensors_shape_index", side_effect=ImportError("missing safetensors")):
            blockers = sd3.sd3_full_bf16_scope_blockers(make_policy(), config(self.model))
            self.assertIn("audit failed", " ".join(blockers))


    def test_config_only_does_not_touch_checkpoint_and_strict_default_does(self):
        missing = self.root / "absent.safetensors"
        self.reader.reset_mock()
        with patch.object(sd3, "_read_safetensors_shape_index", side_effect=AssertionError("model I/O from Preview")):
            self.assertEqual(sd3.sd3_full_bf16_config_blockers(make_policy(), config(missing)), [])
            self.assertEqual(
                sd3.sd3_full_bf16_scope_blockers(
                    make_policy(), config(missing), require_assets=False,
                ),
                [],
            )
        self.reader.assert_not_called()
        self.assertIn(
            "existing absolute local",
            " ".join(sd3.sd3_full_bf16_scope_blockers(make_policy(), config(missing))),
        )
        self.assertIn(
            "require_assets",
            " ".join(sd3.sd3_full_bf16_scope_blockers(
                make_policy(), config(missing), require_assets="false",
            )),
        )

    def test_unreviewed_dataset_inputs_are_rejected_without_reading_external_toml(self):
        dataset_toml = self.root / "override.toml"
        dataset_toml.write_text(
            "[[datasets]]\nbatch_size = 4\nresolution = 1024\n",
            encoding="utf-8",
        )
        changes = {
            "dataset_config": str(dataset_toml),
            "dataset_class": "my_loader.CustomDataset",
            "in_json": "images.json",
            "reg_data_dir": str(self.root),
            "multi_caption_config": "captions.json",
            "conditioning_data_dir": str(self.root),
            "masked_loss": True,
        }
        for field, value in changes.items():
            with self.subTest(field=field), patch.object(
                sd3, "_read_safetensors_shape_index",
                side_effect=AssertionError("Unqualified config must fail before model I/O"),
            ):
                spec = config(self.model)
                spec[field] = value
                issues = sd3.sd3_full_bf16_config_blockers(make_policy(), spec)
                self.assertIn(field, " ".join(issues), issues)

    def test_sd3_real_parser_spelling_conditioning_and_dataloader_fail_closed(self):
        cases = (
            ("pos_emb_random_crop_rate", 0.1),
            ("pos_emb_random_crop_rate", float("nan")),
            ("clip_l_dropout_rate", 0.1),
            ("clip_g_dropout_rate", 0.1),
            ("t5_dropout_rate", 0.1),
            ("training_shift", 2.0),
            ("t5xxl_max_token_length", 512),
            ("apply_lg_attn_mask", True),
            ("apply_t5_attn_mask", True),
            ("persistent_data_loader_workers", True),
            ("weighting_scheme", "cosmap"),
            ("loss_type", "huber"),
            ("t5xxl_dtype", "fp16"),
            ("t5xxl_device", "cuda:1"),
            ("resume_from_huggingface", True),
            ("min_timestep", 5),
            ("max_timestep", 900),
        )
        for field, value in cases:
            with self.subTest(field=field):
                candidate = config(self.model)
                candidate[field] = value
                self.assertIn(field, " ".join(
                    sd3.sd3_full_bf16_config_blockers(make_policy(), candidate)
                ))
        for field in ("training_shift", "t5xxl_max_token_length", "pos_emb_random_crop_rate"):
            config_ = config(self.model)
            defaults = {"training_shift": 1.0, "t5xxl_max_token_length": 256,
                        "pos_emb_random_crop_rate": 0}
            config_[field] = defaults[field]
            self.assertEqual(sd3.sd3_full_bf16_config_blockers(make_policy(), config_), [])

    def test_reference_optimizer_arguments_and_rates_are_strict(self):
        r1 = make_policy()
        r2 = r2_policy()
        self.assertEqual(validate_parameter_policy(r1), r1)
        self.assertEqual(validate_parameter_policy(r2), r2)
        for case, mutation, marker in (
            ("r1", lambda p: p["optimizer_profiles"]["baseline"]["args"].update({"betas": [0.8, 0.999]}), "Profile args"),
            ("r1", lambda p: p["optimizer_profiles"]["baseline"].update({"args": {}}), "Profile args"),
            ("r1", lambda p: p["components"]["mmdit.attention.adapter"].update({"learning_rate": 0.01}), "reference learning rate"),
            ("r2", lambda p: p["optimizer_profiles"]["muon_primary"]["args"].update({"momentum": 0.88}), "Profile args"),
            ("r2", lambda p: p["optimizer_profiles"]["adamw_fallback"]["args"].update({"weight_decay": 0.01}), "Profile args"),
            ("r2", lambda p: p["components"]["mmdit.mlp.adapter"].update({"learning_rate": 0.01}), "reference learning rates"),
            ("r2", lambda p: p["components"]["mmdit.mlp.adapter"].update({"fallback_learning_rate": 0.01}), "reference learning rates"),
        ):
            with self.subTest(case=case, marker=marker):
                policy = copy.deepcopy(r1 if case == "r1" else r2)
                mutation(policy)
                issues = sd3.sd3_full_bf16_config_blockers(
                    policy, config(self.model, accumulation=2 if case == "r2" else 1),
                )
                self.assertIn(marker, " ".join(issues), issues)

    def _ui_request_config(
        self, model: Path, *, policy: dict | None = None,
        accumulation: int = 1,
    ) -> dict:
        if policy is None:
            policy = make_policy()
        return {
            "optimization_mode": "component",
            "parameter_policy_profiles": copy.deepcopy(policy["optimizer_profiles"]),
            "parameter_policy_components": copy.deepcopy(policy["components"]),
            "sd3_lora_target": "mmdit",
            "train_data_dir": str(self.root),
            "max_train_steps": 2,
            **config(model, accumulation=accumulation),
        }


    def test_reference_resolution_production_forms_and_invalid_inputs(self):
        cases = (
            ("512", True),
            ("512,512", True),
            ("512, 512", True),
            ((512, 512), True),
            ("768,768", False),
            ("512,768", False),
            ((512, 768), False),
            ("512,512,512", False),
            ([512, 512], False),
            (512, False),
            (True, False),
            (None, False),
            (float("nan"), False),
            ("512,garbage", False),
            ((True, 512), False),
        )
        for value, allowed in cases:
            with self.subTest(value=value):
                candidate = config(self.model)
                candidate["resolution"] = value
                blockers = sd3.sd3_full_bf16_config_blockers(make_policy(), candidate)
                self.assertEqual(not blockers, allowed, blockers)

    def test_r1_r2_network_dropout_stock_none_and_bucket_flag(self):
        for label, policy, accumulation in (
            ("r1", make_policy(), 1),
            ("r2", r2_policy(), 2),
        ):
            with self.subTest(case=label):
                reference = config(self.model, accumulation=accumulation)
                reference["resolution"] = "512,512"
                reference["network_dropout"] = None
                reference["enable_bucket"] = False
                before = copy.deepcopy(reference)
                self.assertEqual(
                    sd3.sd3_full_bf16_config_blockers(policy, reference), [],
                )
                self.assertEqual(reference, before, "scope must not mutate effective config")
                positive_zero = dict(reference, network_dropout=0.0)
                self.assertEqual(
                    sd3.sd3_full_bf16_config_blockers(policy, positive_zero), [],
                )
                for field, value in (
                    ("enable_bucket", True),
                    ("enable_bucket", "maybe"),
                    ("network_dropout", 0.1),
                    ("network_dropout", float("nan")),
                ):
                    with self.subTest(case=label, field=field, value=value):
                        bad = dict(reference, **{field: value})
                        blockers = sd3.sd3_full_bf16_config_blockers(policy, bad)
                        self.assertIn(field, " ".join(blockers), blockers)

    def test_r1_r2_ui_request_uses_real_resolution_and_config_path(self):
        prepare = _real_prepare_request()
        qualified = execution.ExecutionFeatureQualification(
            "qualified", "test-only future R1/R2", "test:sd3c2a-production-shape",
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sd3-lora": qualified},
        ), patch.object(
            sd3, "_read_safetensors_shape_index",
            side_effect=AssertionError("Preview must not read model header"),
        ):
            for label, policy, accumulation in (
                ("r1", make_policy(), 1),
                ("r2", r2_policy(), 2),
            ):
                with self.subTest(case=label):
                    raw = self._ui_request_config(
                        self.root / "ui-model-not-mounted.safetensors",
                        policy=policy, accumulation=accumulation,
                    )
                    raw.update(
                        resolution="512,512",
                        network_dropout=None,
                        enable_bucket=False,
                    )
                    prepared = prepare(raw, "sd3-lora", launch=False)
                    self.assertEqual(prepared.runtime_blockers, [])
                    self.assertEqual(prepared.config["resolution"], "512,512")

    def test_production_prepare_dataset_args_yields_qualifying_trainer_tuple(self):
        # Run the actual production conversion function without importing its
        # heavyweight CUDA/torch module in pure Host CI.
        train_util_src = (ROOT / "scripts/dev/library/train_util.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(train_util_src, filename="scripts/dev/library/train_util.py")
        func = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_dataset_args"
        )
        extracted = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(extracted)
        namespace = {"argparse": argparse}
        exec(compile(extracted, "scripts/dev/library/train_util.py", "exec"), namespace)
        prepare_dataset_args = namespace["prepare_dataset_args"]

        for label, policy, accumulation in (
            ("r1", make_policy(), 1),
            ("r2", r2_policy(), 2),
        ):
            with self.subTest(case=label):
                values = config(self.model, accumulation=accumulation)
                values.update(
                    resolution="512", network_dropout=None,
                    enable_bucket=False, caption_extention=None,
                    face_crop_aug_range=None, in_json=None,
                    color_aug=False, random_crop=False,
                )
                args = argparse.Namespace(**values)
                prepare_dataset_args(args, True)
                self.assertEqual(args.resolution, (512, 512))
                # Session consumes exactly vars(args), not the pre-parser
                # or manually assembled shorthand fixture.
                self.assertEqual(
                    sd3.sd3_full_bf16_config_blockers(policy, vars(args)), [],
                )

    def test_real_request_preview_keeps_asset_free_start_remains_strict(self):
        missing = self.root / "missing.safetensors"
        materialized = []
        prepare = _real_prepare_request(
            materialize_sidecars=lambda sidecars: materialized.append(
                dict(sidecars)
            ),
        )
        output_dir = self.root / "new-output"
        toml_path = output_dir / "training.toml"
        fake = execution.ExecutionFeatureQualification(
            "qualified", "test-only qualified", "test:sd3c2a-future",
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS, {"sd3-lora": fake},
        ), patch.object(sd3, "_read_safetensors_shape_index", side_effect=AssertionError(
            "Preview/Export must not open a model"
        )):
            preview = prepare(self._ui_request_config(missing), "sd3-lora", launch=False)
            self.assertEqual(preview.runtime_blockers, [])
        self.assertNotIn("parameter_policy_config", preview.sidecars)
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS, {"sd3-lora": fake},
        ), self.assertRaisesRegex(ValueError, "existing absolute local"):
            start_config = self._ui_request_config(missing)
            start_config["output_dir"] = str(output_dir)
            prepare(
                start_config, "sd3-lora",
                launch=True, materialize=True, toml_path=str(toml_path),
            )
        self.assertEqual(materialized, [], "Start must not stage sidecars after asset failure")
        self.assertFalse(output_dir.exists(), "Start must not create output directory")
        self.assertFalse(toml_path.exists(), "Start must not emit training TOML")
        self.assertFalse(list(self.root.rglob("*parameter-policy*.json")))

    def test_direct_trainer_preserves_independent_strict_asset_gate(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("direct-trainer integration runs in pinned CPU runtime smoke")
        from mikazuki.parameter_policy_trainer import (
            ParameterPolicyTrainerRuntimeError,
            create_parameter_policy_session,
        )
        policy = make_policy()
        _name, canonical_text = serialize_parameter_policy(policy)
        sidecar = self.root / "component-policy.json"
        sidecar.write_text(canonical_text, encoding="utf-8")
        missing = self.root / "missing.safetensors"
        spec = config(missing)
        spec["parameter_policy_config"] = str(sidecar)
        fake = execution.ExecutionFeatureQualification(
            "qualified", "test-only qualified", "test:sd3c2a-future",
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS, {"sd3-lora": fake},
        ), patch(
            "mikazuki.parameter_policy_trainer.scan_parameter_roots",
            side_effect=AssertionError("must reject before scanning roots"),
        ), self.assertRaisesRegex(
            ParameterPolicyTrainerRuntimeError, "existing absolute local",
        ):
            create_parameter_policy_session(
                args=spec, train_type="sd3-lora", roots={},
            )

    def test_direct_trainer_r1_r2_production_namespace_reaches_parameter_scan(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("direct-trainer integration runs in pinned CPU runtime smoke")
        from mikazuki.parameter_policy_trainer import create_parameter_policy_session

        qualified = execution.ExecutionFeatureQualification(
            "qualified", "test-only qualified", "test:sd3c2a-future",
        )
        for label, policy, accumulation in (
            ("r1", make_policy(), 1),
            ("r2", r2_policy(), 2),
        ):
            with self.subTest(case=label):
                _name, canonical = serialize_parameter_policy(policy)
                sidecar = self.root / (label + "-policy.json")
                sidecar.write_text(canonical, encoding="utf-8")
                # This is the representation the production Trainer passes to
                # create_parameter_policy_session after prepare_dataset_args.
                spec = config(self.model, accumulation=accumulation)
                spec.update(
                    parameter_policy_config=str(sidecar),
                    resolution=(512, 512),
                    network_dropout=None,
                    enable_bucket=False,
                )
                sentinel = "R1_R2_SCOPE_ACCEPTED_REACHED_PARAMETER_SCAN"
                with patch.dict(
                    execution.FULL_BF16_BACKEND_QUALIFICATIONS,
                    {"sd3-lora": qualified},
                ), patch(
                    "mikazuki.parameter_policy_trainer.scan_parameter_roots",
                    side_effect=RuntimeError(sentinel),
                ) as scan, self.assertRaisesRegex(RuntimeError, sentinel):
                    create_parameter_policy_session(
                        args=argparse.Namespace(**spec),
                        train_type="sd3-lora",
                        roots={},
                    )
                scan.assert_called_once()

    def test_actual_safetensors_header_reader_on_small_cpu_fixture(self):
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("safetensors") is None
        ):
            self.skipTest("small header test runs in pinned CPU runtime smoke")
        import torch
        from safetensors.torch import save_file

        model = self.root / "header-only.safetensors"
        save_file({"tiny.weight": torch.zeros((2, 3), dtype=torch.float32)}, str(model))
        # setUp mocks the reader for full SD3 shape fixtures. Explicitly stop
        # that mock so this test exercises the real safetensors C++ header path.
        self.shape_reader.stop()
        self.assertEqual(
            sd3._read_safetensors_shape_index(model), {"tiny.weight": (2, 3)},
        )

    def test_host_module_and_source_never_promote_backend(self):
        paths = [ROOT / "mikazuki" / "sd3_full_bf16_scope.py",
                 ROOT / "mikazuki" / "parameter_policy_execution.py"]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(any(i == "torch" or i.startswith("torch.") for i in imports))
        scope_source = paths[0].read_text(encoding="utf-8")
        self.assertNotIn("FULL_BF16_BACKEND_QUALIFICATIONS", scope_source)


if __name__ == "__main__":
    unittest.main()
