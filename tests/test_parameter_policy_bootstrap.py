import copy
import math
import unittest
from unittest.mock import patch

from mikazuki.parameter_policy_bootstrap import (
    LEGACY_BOOTSTRAP_PROFILE,
    _bootstrap_component_rows,
    _component_route_from_legacy_lr,
    bootstrap_parameter_policy_from_standard,
    _expand_text_encoder_lrs,
    _parse_legacy_lr,
    _prepare_standard_snapshot,
    _resolve_legacy_lr,
    bootstrap_parameter_policy_optimizer_profile,
)
from mikazuki.training_config import PreparedTrainingConfig


def _resolver(config, requested):
    return requested, f"trainer/{requested}.py"


class ParameterPolicyBootstrapPrimitiveTests(unittest.TestCase):
    def test_prepare_standard_snapshot_does_not_mutate_caller(self):
        raw = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {"stale": {"type": "Muon"}},
            "parameter_policy_components": {"stale": {"train": False}},
            "parameter_policy_config": "stale-sidecar.json",
            "optimizer_type": "AdamW",
            "learning_rate": "1e-4",
            "lora_target": "unet",
        }
        before = copy.deepcopy(raw)

        prepared = _prepare_standard_snapshot(
            raw,
            "lora-master",
            resolve_backend=_resolver,
        )

        self.assertEqual(raw, before)
        self.assertEqual(prepared.train_type, "sd-lora")
        self.assertNotIn("optimization_mode", prepared.config)
        self.assertNotIn("parameter_policy_profiles", prepared.config)
        self.assertNotIn("parameter_policy_components", prepared.config)
        self.assertNotIn("parameter_policy_config", prepared.config)
        self.assertEqual(prepared.config["learning_rate"], 1e-4)

    def test_prepare_standard_snapshot_preserves_standard_custom_override_semantics(self):
        prepared = _prepare_standard_snapshot(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "lora_target": "unet",
                "ui_custom_params": (
                    'optimizer_type = "Lion"\n'
                    'learning_rate = 0.0003'
                ),
            },
            "lora-master",
            resolve_backend=_resolver,
        )
        self.assertEqual(prepared.config["optimizer_type"], "Lion")
        self.assertEqual(prepared.config["learning_rate"], 3e-4)

    def test_prepare_standard_snapshot_calls_effective_compiler_once_without_launch(self):
        prepared = PreparedTrainingConfig(
            train_type="flux-finetune",
            trainer_file="trainer/flux.py",
            config={"learning_rate": 1e-4},
        )
        raw = {
            "optimization_mode": "component",
            "parameter_policy_config": "stale.json",
            "learning_rate": 1e-4,
        }

        with patch(
            "mikazuki.parameter_policy_bootstrap.prepare_training_config",
            return_value=prepared,
        ) as mocked:
            result = _prepare_standard_snapshot(
                raw,
                "flux-finetune",
                resolve_backend=_resolver,
            )

        self.assertIs(result, prepared)
        mocked.assert_called_once()
        call = mocked.call_args
        candidate = call.args[0]
        self.assertNotIn("optimization_mode", candidate)
        self.assertNotIn("parameter_policy_config", candidate)
        self.assertEqual(call.kwargs["page_train_type"], "flux-finetune")
        self.assertFalse(call.kwargs["launch"])
        self.assertIsNone(call.kwargs["toml_path"])

    def test_existing_profile_only_bootstrap_still_uses_compiled_standard_semantics(self):
        profiles = bootstrap_parameter_policy_optimizer_profile(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "lora_target": "unet",
                "ui_custom_params": 'optimizer_type = "Lion"',
            },
            "lora-master",
            resolve_backend=_resolver,
        )
        self.assertEqual(set(profiles), {LEGACY_BOOTSTRAP_PROFILE})
        self.assertEqual(profiles[LEGACY_BOOTSTRAP_PROFILE]["type"], "Lion")

    def test_parse_legacy_lr_accepts_finite_nonnegative_values(self):
        for raw, expected in (
            (0, 0.0),
            ("0", 0.0),
            (1e-4, 1e-4),
            ("5e-7", 5e-7),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    _parse_legacy_lr(raw, field="learning_rate"),
                    expected,
                )

    def test_parse_legacy_lr_rejects_negative_nonfinite_bool_and_malformed(self):
        for raw in (
            -1,
            "-1e-4",
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            None,
            "",
            "bad",
            [],
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError,
                "有限非负数字",
            ):
                _parse_legacy_lr(raw, field="learning_rate")

    def test_resolve_legacy_lr_uses_only_explicit_fallback(self):
        self.assertEqual(
            _resolve_legacy_lr(None, field="unet_lr", fallback=2e-4),
            2e-4,
        )
        self.assertEqual(
            _resolve_legacy_lr("", field="unet_lr", fallback=2e-4),
            2e-4,
        )
        self.assertEqual(
            _resolve_legacy_lr(0, field="unet_lr", fallback=2e-4),
            0.0,
        )
        self.assertEqual(
            _resolve_legacy_lr(3e-4, field="unet_lr", fallback=2e-4),
            3e-4,
        )
        with self.assertRaisesRegex(ValueError, "不会猜测"):
            _resolve_legacy_lr(None, field="learning_rate")

    def test_component_route_converts_zero_to_explicit_freeze(self):
        self.assertEqual(
            _component_route_from_legacy_lr(0),
            {"train": False},
        )
        self.assertEqual(
            _component_route_from_legacy_lr(2e-4),
            {
                "train": True,
                "optimizer_profile": LEGACY_BOOTSTRAP_PROFILE,
                "learning_rate": 2e-4,
            },
        )

    def test_component_route_never_emits_train_true_with_zero_lr(self):
        for raw in (0, "0", 0.0):
            with self.subTest(raw=raw):
                route = _component_route_from_legacy_lr(raw)
                self.assertEqual(route, {"train": False})
                self.assertNotIn("learning_rate", route)

    def test_expand_two_encoder_lrs_matches_flux_semantics(self):
        cases = (
            (None, (1e-4, 1e-4)),
            ("", (1e-4, 1e-4)),
            ([], (1e-4, 1e-4)),
            (2e-4, (2e-4, 2e-4)),
            ("2e-4", (2e-4, 2e-4)),
            ([2e-4], (2e-4, 2e-4)),
            ([2e-4, 3e-4], (2e-4, 3e-4)),
            ([2e-4, 3e-4, 4e-4], (2e-4, 3e-4)),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    _expand_text_encoder_lrs(
                        raw,
                        count=2,
                        base_lr=1e-4,
                    ),
                    expected,
                )

    def test_expand_three_encoder_lrs_matches_sd3_semantics(self):
        cases = (
            (None, (1e-4, 1e-4, 1e-4)),
            (2e-4, (2e-4, 2e-4, 2e-4)),
            ([2e-4], (2e-4, 2e-4, 2e-4)),
            ([2e-4, 3e-4], (2e-4, 3e-4, 3e-4)),
            ([2e-4, 3e-4, 4e-4], (2e-4, 3e-4, 4e-4)),
            ([2e-4, 3e-4, 4e-4, 5e-4], (2e-4, 3e-4, 4e-4)),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    _expand_text_encoder_lrs(
                        raw,
                        count=3,
                        base_lr=1e-4,
                    ),
                    expected,
                )

    def test_expand_text_encoder_lrs_validates_every_value(self):
        bad_values = (
            [1e-4, -1],
            [1e-4, float("nan")],
            [1e-4, True],
            ["1e-4", "bad"],
        )
        for raw in bad_values:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _expand_text_encoder_lrs(raw, count=2, base_lr=1e-4)

        with self.assertRaises(ValueError):
            _expand_text_encoder_lrs(None, count=2, base_lr=-1)
        with self.assertRaisesRegex(ValueError, "正整数"):
            _expand_text_encoder_lrs(None, count=0, base_lr=1e-4)
        with self.assertRaisesRegex(ValueError, "正整数"):
            _expand_text_encoder_lrs(None, count=True, base_lr=1e-4)

    def test_comma_separated_text_encoder_lr_string_is_not_silently_guessed(self):
        with self.assertRaises(ValueError):
            _expand_text_encoder_lrs(
                "1e-4,2e-4",
                count=2,
                base_lr=1e-4,
            )


class ParameterPolicyBackendMappingTests(unittest.TestCase):
    def assert_complete_schema(self, train_type, rows):
        from mikazuki.model_component_profiles import get_model_component_profile

        self.assertEqual(
            set(rows),
            set(get_model_component_profile(train_type).components),
        )
        self.assertEqual(list(rows), sorted(rows))
        for route in rows.values():
            self.assertIn("train", route)
            if route["train"]:
                self.assertEqual(route["optimizer_profile"], LEGACY_BOOTSTRAP_PROFILE)
                self.assertGreater(route["learning_rate"], 0)
            else:
                self.assertEqual(route, {"train": False})

    def test_every_backend_mapper_returns_complete_component_schema(self):
        cases = {
            "sd-lora": {"learning_rate": 1e-4},
            "sdxl-lora": {"learning_rate": 1e-4},
            "flux-lora": {
                "learning_rate": 1e-4,
                "network_args": ["train_t5xxl=true"],
            },
            "chroma-lora": {
                "learning_rate": 1e-4,
                "network_args": ["train_t5xxl=true"],
            },
            "sd3-lora": {
                "learning_rate": 1e-4,
                "network_args": ["train_t5xxl=true"],
            },
            "anima-lora": {"learning_rate": 1e-4},
            "sd-dreambooth": {"learning_rate": 1e-6},
            "sdxl-finetune": {
                "learning_rate": 1e-6,
                "train_text_encoder": True,
            },
            "flux-finetune": {"learning_rate": 1e-6},
            "anima-finetune": {
                "learning_rate": 1e-6,
                "train_qwen3_text_encoder": False,
            },
        }
        for train_type, config in cases.items():
            with self.subTest(train_type=train_type):
                rows = _bootstrap_component_rows(config, train_type)
                self.assert_complete_schema(train_type, rows)

    def test_sd_lora_maps_unet_and_text_encoder_lr_with_zero_freeze(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "unet_lr": 2e-4,
                "text_encoder_lr": 0,
            },
            "sd-lora",
        )
        for component_id in (
            "unet.attention.adapter",
            "unet.feed_forward.adapter",
            "unet.conv.adapter",
            "unet.other.adapter",
        ):
            self.assertEqual(rows[component_id]["learning_rate"], 2e-4)
        self.assertEqual(rows["text_encoder.adapter"], {"train": False})

    def test_sd_lora_target_unavailable_overrides_stale_invalid_lr(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "network_train_unet_only": True,
                "text_encoder_lr": "stale-invalid",
            },
            "sd-lora",
        )
        self.assertEqual(rows["text_encoder.adapter"], {"train": False})

        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "network_train_text_encoder_only": True,
                "unet_lr": "stale-invalid",
            },
            "sd-lora",
        )
        for component_id in (
            "unet.attention.adapter",
            "unet.feed_forward.adapter",
            "unet.conv.adapter",
            "unet.other.adapter",
        ):
            self.assertEqual(rows[component_id], {"train": False})

    def test_sdxl_lora_uses_one_legacy_text_encoder_lr_for_both_encoders(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "text_encoder_lr": 7e-5,
            },
            "sdxl-lora",
        )
        self.assertEqual(rows["text_encoder_1.adapter"]["learning_rate"], 7e-5)
        self.assertEqual(rows["text_encoder_2.adapter"]["learning_rate"], 7e-5)

    def test_flux_lora_expands_text_encoder_lr_slots(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "unet_lr": 3e-4,
                "text_encoder_lr": [5e-5, 7e-5],
                "network_args": ["train_t5xxl=true"],
            },
            "flux-lora",
        )
        self.assertEqual(
            rows["transformer.double_stream.adapter"]["learning_rate"],
            3e-4,
        )
        self.assertEqual(rows["clip_l.adapter"]["learning_rate"], 5e-5)
        self.assertEqual(rows["t5xxl.adapter"]["learning_rate"], 7e-5)

    def test_chroma_t5_uses_flux_text_encoder_slot_one(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "text_encoder_lr": [5e-5, 7e-5],
                "network_args": ["train_t5xxl=true"],
            },
            "chroma-lora",
        )
        self.assertEqual(rows["t5xxl.adapter"]["learning_rate"], 7e-5)
        self.assertNotIn("clip_l.adapter", rows)

    def test_flux_and_chroma_t5_target_off_skips_stale_text_lr(self):
        for train_type in ("flux-lora", "chroma-lora"):
            with self.subTest(train_type=train_type):
                config = {
                    "learning_rate": 1e-4,
                    "text_encoder_lr": [1e-4, "stale-invalid"],
                }
                if train_type == "flux-lora":
                    config["network_train_unet_only"] = True
                rows = _bootstrap_component_rows(config, train_type)
                self.assertEqual(rows["t5xxl.adapter"], {"train": False})

    def test_sd3_text_encoder_lr_expansion_uses_three_real_slots(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "text_encoder_lr": [5e-5, 7e-5],
                "network_args": ["train_t5xxl=true"],
            },
            "sd3-lora",
        )
        self.assertEqual(rows["clip_l.adapter"]["learning_rate"], 5e-5)
        self.assertEqual(rows["clip_g.adapter"]["learning_rate"], 7e-5)
        self.assertEqual(rows["t5xxl.adapter"]["learning_rate"], 7e-5)
        for component_id in (
            "mmdit.attention.adapter",
            "mmdit.mlp.adapter",
            "mmdit.modulation_norm.adapter",
            "mmdit.other.adapter",
        ):
            self.assertEqual(rows[component_id]["learning_rate"], 1e-4)

    def test_sd3_mmdit_components_use_real_sd3_target_ids(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "unet_lr": 2e-4,
            },
            "sd3-lora",
        )
        for component_id in (
            "mmdit.attention.adapter",
            "mmdit.mlp.adapter",
            "mmdit.modulation_norm.adapter",
            "mmdit.other.adapter",
        ):
            self.assertTrue(rows[component_id]["train"])
            self.assertEqual(rows[component_id]["learning_rate"], 2e-4)

    def test_sd3_unet_only_does_not_parse_stale_text_encoder_lrs(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "network_train_unet_only": True,
                "text_encoder_lr": ["bad", "bad", "bad"],
            },
            "sd3-lora",
        )
        self.assertEqual(rows["clip_l.adapter"], {"train": False})
        self.assertEqual(rows["clip_g.adapter"], {"train": False})
        self.assertEqual(rows["t5xxl.adapter"], {"train": False})

    def test_anima_lora_maps_dit_qwen_and_llm_adapter_target_semantics(self):
        joint = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "unet_lr": 2e-4,
                "text_encoder_lr": 5e-6,
                "network_args": ["train_llm_adapter=true"],
            },
            "anima-lora",
        )
        self.assertEqual(
            joint["dit.self_attention.adapter"]["learning_rate"],
            2e-4,
        )
        self.assertEqual(joint["llm_adapter.adapter"]["learning_rate"], 2e-4)
        self.assertEqual(joint["qwen3.adapter"]["learning_rate"], 5e-6)

        dit_only = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "network_train_unet_only": True,
                "text_encoder_lr": "stale-invalid",
            },
            "anima-lora",
        )
        self.assertEqual(dit_only["qwen3.adapter"], {"train": False})
        self.assertEqual(dit_only["llm_adapter.adapter"], {"train": False})

        qwen_only = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "network_train_text_encoder_only": True,
                "unet_lr": "stale-invalid",
                "text_encoder_lr": 9e-6,
            },
            "anima-lora",
        )
        for component_id in (
            "dit.self_attention.adapter",
            "dit.cross_attention.adapter",
            "dit.mlp.adapter",
            "dit.modulation.adapter",
            "dit.other.adapter",
            "llm_adapter.adapter",
        ):
            self.assertEqual(qwen_only[component_id], {"train": False})
        self.assertEqual(qwen_only["qwen3.adapter"]["learning_rate"], 9e-6)

    def test_dreambooth_maps_static_text_encoder_states(self):
        active = _bootstrap_component_rows(
            {
                "learning_rate": 1e-6,
                "learning_rate_te": 5e-7,
            },
            "sd-dreambooth",
        )
        self.assertEqual(active["text_encoder"]["learning_rate"], 5e-7)
        for component_id in (
            "unet.transformer",
            "unet.conv_resnet",
            "unet.norm_bias_other",
            "unet.base_other",
        ):
            self.assertEqual(active[component_id]["learning_rate"], 1e-6)

        disabled = _bootstrap_component_rows(
            {
                "learning_rate": 1e-6,
                "stop_text_encoder_training": -1,
                "learning_rate_te": "stale-invalid",
            },
            "sd-dreambooth",
        )
        self.assertEqual(disabled["text_encoder"], {"train": False})

    def test_sdxl_full_maps_independent_text_encoder_lrs_and_target_off(self):
        active = _bootstrap_component_rows(
            {
                "learning_rate": 1e-6,
                "train_text_encoder": True,
                "learning_rate_te1": 5e-7,
                "learning_rate_te2": 0,
            },
            "sdxl-finetune",
        )
        self.assertEqual(active["text_encoder_1"]["learning_rate"], 5e-7)
        self.assertEqual(active["text_encoder_2"], {"train": False})

        disabled = _bootstrap_component_rows(
            {
                "learning_rate": 1e-6,
                "train_text_encoder": False,
                "learning_rate_te1": "stale-invalid",
                "learning_rate_te2": "stale-invalid",
            },
            "sdxl-finetune",
        )
        self.assertEqual(disabled["text_encoder_1"], {"train": False})
        self.assertEqual(disabled["text_encoder_2"], {"train": False})

    def test_flux_full_uses_base_lr_for_every_component(self):
        rows = _bootstrap_component_rows(
            {"learning_rate": 3e-6},
            "flux-finetune",
        )
        self.assertTrue(rows)
        self.assertTrue(all(route["train"] for route in rows.values()))
        self.assertEqual(
            {route["learning_rate"] for route in rows.values()},
            {3e-6},
        )

    def test_anima_full_maps_optional_sub_lrs_and_zero_freeze(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-6,
                "self_attn_lr": 2e-6,
                "cross_attn_lr": 0,
                "mlp_lr": None,
                "mod_lr": 3e-6,
                "llm_adapter_lr": None,
                "train_qwen3_text_encoder": False,
                "qwen3_lr": "stale-invalid",
            },
            "anima-finetune",
        )
        self.assertEqual(rows["dit.base_other"]["learning_rate"], 1e-6)
        self.assertEqual(rows["dit.self_attention"]["learning_rate"], 2e-6)
        self.assertEqual(rows["dit.cross_attention"], {"train": False})
        self.assertEqual(rows["dit.mlp"]["learning_rate"], 1e-6)
        self.assertEqual(rows["dit.modulation"]["learning_rate"], 3e-6)
        self.assertEqual(rows["dit.llm_adapter"]["learning_rate"], 1e-6)
        self.assertEqual(rows["qwen3"], {"train": False})

    def test_anima_full_qwen_requires_explicit_qwen_lr_when_available(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-6,
                "train_qwen3_text_encoder": True,
                "qwen3_lr": 5e-7,
            },
            "anima-finetune",
        )
        self.assertEqual(rows["qwen3"]["learning_rate"], 5e-7)

        with self.assertRaisesRegex(ValueError, "qwen3_lr"):
            _bootstrap_component_rows(
                {
                    "learning_rate": 1e-6,
                    "train_qwen3_text_encoder": True,
                },
                "anima-finetune",
            )

    def test_target_unavailable_rows_never_emit_hidden_optimizer_or_lr_fields(self):
        rows = _bootstrap_component_rows(
            {
                "learning_rate": 1e-4,
                "network_train_unet_only": True,
            },
            "sdxl-lora",
        )
        self.assertEqual(rows["text_encoder_1.adapter"], {"train": False})
        self.assertEqual(rows["text_encoder_2.adapter"], {"train": False})

    def test_missing_base_lr_fails_closed_for_every_backend_mapper(self):
        for train_type in (
            "sd-lora",
            "sdxl-lora",
            "flux-lora",
            "chroma-lora",
            "sd3-lora",
            "anima-lora",
            "sd-dreambooth",
            "sdxl-finetune",
            "flux-finetune",
            "anima-finetune",
        ):
            config = {}
            if train_type == "sdxl-finetune":
                config["train_text_encoder"] = False
            if train_type == "anima-finetune":
                config["train_qwen3_text_encoder"] = False
            with self.subTest(train_type=train_type), self.assertRaisesRegex(
                ValueError,
                "learning_rate",
            ):
                _bootstrap_component_rows(config, train_type)

    def test_mapper_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            _bootstrap_component_rows({"learning_rate": 1e-4}, "unknown-backend")


class ParameterPolicyFullBootstrapTests(unittest.TestCase):
    def test_public_bootstrap_returns_strict_canonical_policy(self):
        from mikazuki.parameter_policy import validate_parameter_policy

        raw = {
            "optimizer_type": "AdamW",
            "learning_rate": "1e-4",
            "unet_lr": "2e-4",
            "text_encoder_lr": "0",
            "lora_target": "both",
        }
        before = copy.deepcopy(raw)

        policy = bootstrap_parameter_policy_from_standard(
            raw,
            "lora-master",
            resolve_backend=_resolver,
        )

        self.assertEqual(raw, before)
        self.assertEqual(policy, validate_parameter_policy(policy))
        self.assertEqual(policy["version"], 1)
        self.assertEqual(set(policy["optimizer_profiles"]), {LEGACY_BOOTSTRAP_PROFILE})
        self.assertEqual(
            policy["optimizer_profiles"][LEGACY_BOOTSTRAP_PROFILE]["type"],
            "AdamW",
        )
        self.assertEqual(
            policy["components"]["unet.attention.adapter"]["learning_rate"],
            2e-4,
        )
        self.assertEqual(
            policy["components"]["text_encoder.adapter"],
            {"train": False},
        )

    def test_public_bootstrap_honors_standard_custom_override_before_migration(self):
        policy = bootstrap_parameter_policy_from_standard(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "lora_target": "unet",
                "ui_custom_params": (
                    'optimizer_type = "Lion"\n'
                    'learning_rate = 0.0003'
                ),
            },
            "lora-master",
            resolve_backend=_resolver,
        )
        self.assertEqual(
            policy["optimizer_profiles"][LEGACY_BOOTSTRAP_PROFILE]["type"],
            "Lion",
        )
        for component_id, route in policy["components"].items():
            if component_id.startswith("unet."):
                self.assertEqual(route["learning_rate"], 3e-4)
            else:
                self.assertEqual(route, {"train": False})

    def test_public_bootstrap_runs_compatibility_gate_before_profile_or_mapping(self):
        prepared = PreparedTrainingConfig(
            train_type="sdxl-finetune",
            trainer_file="trainer/sdxl.py",
            config={
                "learning_rate": 1e-6,
                "block_lr": ",".join(["1e-6"] * 23),
                "optimizer_type": "Custom",
            },
        )

        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ), patch(
            "mikazuki.parameter_policy_bootstrap.bootstrap_legacy_optimizer_profile"
        ) as profile_bootstrap, patch(
            "mikazuki.parameter_policy_bootstrap._bootstrap_component_rows"
        ) as component_bootstrap:
            with self.assertRaisesRegex(ValueError, "block_lr"):
                bootstrap_parameter_policy_from_standard(
                    {},
                    "sdxl-finetune",
                    resolve_backend=_resolver,
                )

        profile_bootstrap.assert_not_called()
        component_bootstrap.assert_not_called()

    def test_public_bootstrap_reports_all_compatibility_blockers_in_order(self):
        prepared = PreparedTrainingConfig(
            train_type="flux-lora",
            trainer_file="trainer/flux.py",
            config={
                "learning_rate": 1e-4,
                "network_module": "custom.network",
                "network_args": [
                    "loraplus_lr_ratio=2",
                    "network_reg_lrs=foo=1e-4",
                ],
                "fused_backward_pass": True,
                "deepspeed": True,
            },
        )
        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ):
            with self.assertRaises(ValueError) as ctx:
                bootstrap_parameter_policy_from_standard(
                    {},
                    "flux-lora",
                    resolve_backend=_resolver,
                )

        message = str(ctx.exception)
        markers = (
            "fused_backward_pass",
            "DeepSpeed",
            "LoRA+",
            "network_reg_lrs",
            "network_module",
        )
        positions = [message.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_public_bootstrap_never_invents_fallback_optimizer(self):
        prepared = PreparedTrainingConfig(
            train_type="flux-finetune",
            trainer_file="trainer/flux.py",
            config={
                "optimizer_type": "Muon",
                "optimizer_args": [],
                "learning_rate": 2e-4,
            },
        )
        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ):
            policy = bootstrap_parameter_policy_from_standard(
                {},
                "flux-finetune",
                resolve_backend=_resolver,
            )

        self.assertEqual(set(policy["optimizer_profiles"]), {LEGACY_BOOTSTRAP_PROFILE})
        self.assertEqual(
            policy["optimizer_profiles"][LEGACY_BOOTSTRAP_PROFILE]["type"],
            "Muon",
        )
        for route in policy["components"].values():
            self.assertNotIn("fallback_optimizer_profile", route)
            self.assertNotIn("fallback_learning_rate", route)

    def test_public_bootstrap_is_deterministic_for_equivalent_input(self):
        first = {
            "optimizer_type": "AdamW",
            "learning_rate": "1e-4",
            "lora_target": "unet",
        }
        second = {
            "lora_target": "unet",
            "learning_rate": "1e-4",
            "optimizer_type": "AdamW",
        }
        policy_a = bootstrap_parameter_policy_from_standard(
            first,
            "lora-master",
            resolve_backend=_resolver,
        )
        policy_b = bootstrap_parameter_policy_from_standard(
            second,
            "lora-master",
            resolve_backend=_resolver,
        )
        self.assertEqual(policy_a, policy_b)
        self.assertEqual(
            list(policy_a["components"]),
            sorted(policy_a["components"]),
        )

    def test_public_bootstrap_profile_and_component_keys_match_every_backend(self):
        from mikazuki.model_component_profiles import get_model_component_profile

        cases = {
            "sd-lora": {"learning_rate": 1e-4},
            "sdxl-lora": {"learning_rate": 1e-4},
            "flux-lora": {"learning_rate": 1e-4},
            "chroma-lora": {"learning_rate": 1e-4},
            "sd3-lora": {"learning_rate": 1e-4},
            "anima-lora": {"learning_rate": 1e-4},
            "sd-dreambooth": {"learning_rate": 1e-6},
            "sdxl-finetune": {
                "learning_rate": 1e-6,
                "train_text_encoder": False,
            },
            "flux-finetune": {"learning_rate": 1e-6},
            "anima-finetune": {
                "learning_rate": 1e-6,
                "train_qwen3_text_encoder": False,
            },
        }

        for train_type, effective in cases.items():
            with self.subTest(train_type=train_type):
                prepared = PreparedTrainingConfig(
                    train_type=train_type,
                    trainer_file=f"trainer/{train_type}.py",
                    config={
                        "optimizer_type": "AdamW",
                        **effective,
                    },
                )
                with patch(
                    "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
                    return_value=prepared,
                ):
                    policy = bootstrap_parameter_policy_from_standard(
                        {},
                        train_type,
                        resolve_backend=_resolver,
                    )

                self.assertEqual(
                    set(policy["components"]),
                    set(get_model_component_profile(train_type).components),
                )
                self.assertEqual(
                    set(policy["optimizer_profiles"]),
                    {LEGACY_BOOTSTRAP_PROFILE},
                )

    def test_public_bootstrap_rejects_dreambooth_temporal_text_encoder_stop(self):
        prepared = PreparedTrainingConfig(
            train_type="sd-dreambooth",
            trainer_file="trainer/sd.py",
            config={
                "optimizer_type": "AdamW",
                "learning_rate": 1e-6,
                "stop_text_encoder_training": 100,
            },
        )
        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "stop_text_encoder_training",
            ):
                bootstrap_parameter_policy_from_standard(
                    {},
                    "sd-dreambooth",
                    resolve_backend=_resolver,
                )

    def test_public_bootstrap_rejects_loraplus_instead_of_approximating(self):
        prepared = PreparedTrainingConfig(
            train_type="sd-lora",
            trainer_file="trainer/sd-lora.py",
            config={
                "optimizer_type": "AdamW",
                "learning_rate": 1e-4,
                "network_module": "networks.lora",
                "network_args": ["loraplus_lr_ratio=1"],
            },
        )
        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ):
            with self.assertRaisesRegex(ValueError, "LoRA\\+"):
                bootstrap_parameter_policy_from_standard(
                    {},
                    "sd-lora",
                    resolve_backend=_resolver,
                )

    def test_public_bootstrap_uses_strict_validator_as_final_round_trip(self):
        prepared = PreparedTrainingConfig(
            train_type="flux-finetune",
            trainer_file="trainer/flux.py",
            config={
                "optimizer_type": "AdamW",
                "learning_rate": 1e-6,
            },
        )
        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ), patch(
            "mikazuki.parameter_policy_bootstrap.validate_parameter_policy",
            wraps=__import__(
                "mikazuki.parameter_policy",
                fromlist=["validate_parameter_policy"],
            ).validate_parameter_policy,
        ) as validator:
            policy = bootstrap_parameter_policy_from_standard(
                {},
                "flux-finetune",
                resolve_backend=_resolver,
            )

        validator.assert_called_once()
        from mikazuki.parameter_policy import validate_parameter_policy

        self.assertEqual(policy, validate_parameter_policy(policy))

    def test_profile_only_bootstrap_remains_independent_of_compatibility_gate(self):
        prepared = PreparedTrainingConfig(
            train_type="sdxl-finetune",
            trainer_file="trainer/sdxl.py",
            config={
                "optimizer_type": "AdamW",
                "block_lr": "legacy-block-values",
            },
        )
        with patch(
            "mikazuki.parameter_policy_bootstrap._prepare_standard_snapshot",
            return_value=prepared,
        ), patch(
            "mikazuki.parameter_policy_bootstrap.parameter_policy_compatibility_blockers",
            side_effect=AssertionError("profile-only bootstrap must not call compatibility gate"),
        ):
            profiles = bootstrap_parameter_policy_optimizer_profile(
                {},
                "sdxl-finetune",
                resolve_backend=_resolver,
            )

        self.assertEqual(set(profiles), {LEGACY_BOOTSTRAP_PROFILE})


if __name__ == "__main__":
    unittest.main()
