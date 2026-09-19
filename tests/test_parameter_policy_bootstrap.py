import copy
import math
import unittest
from unittest.mock import patch

from mikazuki.parameter_policy_bootstrap import (
    LEGACY_BOOTSTRAP_PROFILE,
    _component_route_from_legacy_lr,
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


if __name__ == "__main__":
    unittest.main()
