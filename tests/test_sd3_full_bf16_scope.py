"""Host-only D1-SD3C2-A SD3.5 Medium scoped Full-BF16 admission tests.

There is no qualification promotion, model value load, Torch, or CUDA here.
A temporarily patched test-only qualification record exercises the future
release path while the real sd3-lora backend remains pending.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
    return policy


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
        "resolution": 512,
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
            ("pos_embed_random_crop_rate", 0.1),
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
