import json
import unittest

from mikazuki.parameter_policy import (
    build_parameter_policy_sidecar,
    canonicalize_parameter_policy,
    parameter_policy_runtime_blockers,
    parse_legacy_optimizer_args,
    rehydrate_parameter_policy,
    serialize_parameter_policy,
    validate_parameter_policy,
)
from mikazuki.parameter_policy_bootstrap import bootstrap_parameter_policy_optimizer_profile
from mikazuki.training_rehydrate import rehydrate_trainer_config


def _component_config():
    return {
        "optimization_mode": "component",
        "parameter_policy_profiles": {
            "main_muon": {
                "type": "Muon",
                "args": {
                    "momentum": 0.95,
                    "weight_decay": 0.01,
                    "ns_steps": 5,
                    "ns_coeffs": "original",
                    "use_adjusted_lr": False,
                },
            },
            "aux_adamw": {
                "type": "AdamW",
                "args": {"betas": [0.9, 0.999], "weight_decay": 0.01},
            },
        },
        "parameter_policy_components": {
            "dit.self_attention": {
                "train": True,
                "optimizer_profile": "main_muon",
                "learning_rate": 2e-4,
                "fallback_optimizer_profile": "aux_adamw",
            },
            "qwen3": {
                "train": False,
                "optimizer_profile": "stale-hidden-value",
                "learning_rate": 123,
            },
        },
    }


class ParameterPolicyConfigTests(unittest.TestCase):
    def test_standard_is_true_noop_and_strips_stale_policy_gui_state(self):
        config = {
            "optimization_mode": "standard",
            "parameter_policy_profiles": {"stale": {"type": "Muon"}},
            "parameter_policy_components": {"stale": {"train": True}},
            "optimizer_type": "AdamW8bit",
            "learning_rate": 1e-4,
        }
        path, sidecars, policy = build_parameter_policy_sidecar(config, "lora-master")
        self.assertIsNone(path)
        self.assertEqual(sidecars, {})
        self.assertIsNone(policy)
        self.assertEqual(
            config,
            {"optimizer_type": "AdamW8bit", "learning_rate": 1e-4},
        )

    def test_missing_mode_is_standard(self):
        config = {
            "parameter_policy_profiles": {"stale": {"type": "Muon"}},
            "parameter_policy_components": {"stale": {"train": True}},
            "optimizer_type": "AdamW",
        }
        path, sidecars, policy = build_parameter_policy_sidecar(config)
        self.assertIsNone(path)
        self.assertEqual(sidecars, {})
        self.assertIsNone(policy)
        self.assertEqual(config, {"optimizer_type": "AdamW"})

    def test_manual_sidecar_injection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "托管字段"):
            build_parameter_policy_sidecar(
                {"parameter_policy_config": "evil.json"},
                "lora-master",
                resolve_backend=lambda config, requested: (requested, f"trainer/{requested}.py"),
            )

    def test_muon_requires_fallback(self):
        config = _component_config()
        del config["parameter_policy_components"]["dit.self_attention"]["fallback_optimizer_profile"]
        with self.assertRaisesRegex(ValueError, "fallback_optimizer_profile"):
            build_parameter_policy_sidecar(config)

    def test_muon_fallback_cannot_require_eligibility_again(self):
        config = _component_config()
        config["parameter_policy_profiles"]["second_muon"] = {
            "type": "Muon",
            "args": {},
        }
        config["parameter_policy_components"]["dit.self_attention"][
            "fallback_optimizer_profile"
        ] = "second_muon"
        with self.assertRaisesRegex(ValueError, "二次 eligibility"):
            build_parameter_policy_sidecar(config)

    def test_frozen_component_drops_hidden_stale_values(self):
        config = _component_config()
        _, _, policy = build_parameter_policy_sidecar(config)
        self.assertEqual(policy["components"]["qwen3"], {"train": False})

    def test_fallback_lr_requires_fallback_profile(self):
        config = _component_config()
        route = config["parameter_policy_components"]["dit.self_attention"]
        route["optimizer_profile"] = "aux_adamw"
        route.pop("fallback_optimizer_profile")
        route["fallback_learning_rate"] = 1e-5
        with self.assertRaisesRegex(ValueError, "不能单独设置"):
            build_parameter_policy_sidecar(config)

    def test_learning_rates_must_be_finite_and_nonnegative_but_zero_is_allowed(self):
        config = _component_config()
        config["parameter_policy_components"]["dit.self_attention"]["learning_rate"] = 0
        _, _, policy = build_parameter_policy_sidecar(config)
        self.assertEqual(policy["components"]["dit.self_attention"]["learning_rate"], 0.0)

        for bad in (-1, float("nan"), float("inf"), "bad"):
            broken = _component_config()
            broken["parameter_policy_components"]["dit.self_attention"]["learning_rate"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_parameter_policy_sidecar(broken)

    def test_profile_names_are_case_insensitively_unique(self):
        config = _component_config()
        config["parameter_policy_profiles"]["MAIN_MUON"] = {
            "type": "AdamW",
            "args": {},
        }
        with self.assertRaisesRegex(ValueError, "规范化后重复"):
            build_parameter_policy_sidecar(config)

    def test_component_profile_reference_is_case_insensitive_and_canonical(self):
        config = _component_config()
        config["parameter_policy_components"]["dit.self_attention"]["optimizer_profile"] = "MAIN_MUON"
        _, _, policy = build_parameter_policy_sidecar(config)
        self.assertEqual(
            policy["components"]["dit.self_attention"]["optimizer_profile"],
            "main_muon",
        )

    def test_sidecar_is_content_addressed_and_dict_order_independent(self):
        config_a = _component_config()
        _, _, policy_a = build_parameter_policy_sidecar(config_a)
        path_a, content_a = serialize_parameter_policy(policy_a)

        config_b = _component_config()
        config_b["parameter_policy_profiles"] = dict(
            reversed(list(config_b["parameter_policy_profiles"].items()))
        )
        config_b["parameter_policy_components"] = dict(
            reversed(list(config_b["parameter_policy_components"].items()))
        )
        _, _, policy_b = build_parameter_policy_sidecar(config_b)
        path_b, content_b = serialize_parameter_policy(policy_b)

        self.assertEqual(path_a, path_b)
        self.assertEqual(content_a, content_b)
        self.assertTrue(path_a.startswith("config/autosave/parameter-policy/"))
        self.assertEqual(validate_parameter_policy(json.loads(content_a)), policy_a)

    def test_rehydrate_round_trip_preserves_semantics(self):
        config = _component_config()
        _, _, policy = build_parameter_policy_sidecar(config)
        gui = rehydrate_parameter_policy(policy)
        rebuilt = canonicalize_parameter_policy(
            {
                "optimizer_profiles": gui["parameter_policy_profiles"],
                "components": gui["parameter_policy_components"],
            }
        )
        self.assertEqual(rebuilt, policy)

    def test_restricted_and_planned_profiles_are_structurally_representable(self):
        for optimizer_type, marker in (
            ("AdaFactor", "restricted"),
            ("pytorch_optimizer.CAME", "planned"),
        ):
            config = {
                "optimization_mode": "component",
                "parameter_policy_profiles": {
                    "main": {"type": optimizer_type, "args": {}},
                },
                "parameter_policy_components": {
                    "test.component": {
                        "train": True,
                        "optimizer_profile": "main",
                        "learning_rate": 1e-4,
                    },
                },
            }
            with self.subTest(optimizer_type=optimizer_type):
                _, _, policy = build_parameter_policy_sidecar(config)
                blockers = parameter_policy_runtime_blockers(policy)
                self.assertTrue(any(marker in item for item in blockers))

    def test_runtime_blockers_always_include_step2_runtime_gate(self):
        config = _component_config()
        _, _, policy = build_parameter_policy_sidecar(config)
        blockers = parameter_policy_runtime_blockers(policy)
        self.assertGreaterEqual(len(blockers), 1)
        self.assertIn("runtime 尚未实现", blockers[0])

    def test_component_ui_custom_params_cannot_override_policy_owned_keys(self):
        for text in (
            'optimizer_type = "Lion"',
            'learning_rate = 0.001',
            'use_8bit_adam = true',
            'optimization_mode = "standard"',
            'parameter_policy_config = "evil.json"',
        ):
            config = _component_config()
            config["ui_custom_params"] = text
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "冲突"):
                build_parameter_policy_sidecar(config)

    def test_standard_does_not_change_ui_custom_params_behavior(self):
        config = {
            "optimization_mode": "standard",
            "ui_custom_params": 'optimizer_type = "Lion"',
            "optimizer_type": "AdamW",
        }
        path, sidecars, policy = build_parameter_policy_sidecar(config)
        self.assertIsNone(path)
        self.assertEqual(sidecars, {})
        self.assertIsNone(policy)
        self.assertEqual(config["ui_custom_params"], 'optimizer_type = "Lion"')


    def test_standard_trainer_rehydrate_does_not_invent_hidden_policy_gui_state(self):
        gui = rehydrate_trainer_config(
            {"optimizer_type": "AdamW", "learning_rate": 1e-4},
            "lora-master",
        )
        self.assertNotIn("optimization_mode", gui)
        self.assertNotIn("parameter_policy_profiles", gui)
        self.assertNotIn("parameter_policy_components", gui)

    def test_trainer_rehydrate_restores_component_mode_from_sidecar_content(self):
        config = _component_config()
        path, sidecars, policy = build_parameter_policy_sidecar(config)
        gui = rehydrate_trainer_config(
            {
                "optimizer_type": "AdamW",
                "learning_rate": 1e-4,
                "parameter_policy_config": path,
            },
            "lora-master",
            sidecars=sidecars,
        )
        self.assertEqual(gui["optimization_mode"], "component")
        self.assertEqual(gui["parameter_policy_profiles"], policy["optimizer_profiles"])
        self.assertEqual(gui["parameter_policy_components"], policy["components"])

    def test_trainer_rehydrate_missing_sidecar_never_falls_back_to_standard(self):
        with self.assertRaisesRegex(ValueError, "不能静默回退 Standard"):
            rehydrate_trainer_config(
                {
                    "optimizer_type": "AdamW",
                    "parameter_policy_config": "config/autosave/parameter-policy/missing.json",
                },
                "lora-master",
                sidecars={},
            )

    def test_legacy_optimizer_args_match_sd_scripts_literal_semantics(self):
        parsed = parse_legacy_optimizer_args(
            ["betas=(0.9, 0.999)", "weight_decay=0.01", "weight_decay=0.02"]
        )
        self.assertEqual(parsed["betas"], (0.9, 0.999))
        self.assertEqual(parsed["weight_decay"], 0.02)
        with self.assertRaisesRegex(ValueError, "缺少"):
            parse_legacy_optimizer_args(["broken"])

    def test_bootstrap_uses_compiled_legacy_optimizer_semantics(self):
        profiles = bootstrap_parameter_policy_optimizer_profile(
            {
                "optimizer_type": "Prodigy",
                "prodigy_d0": "1e-6",
                "prodigy_d_coef": 2,
                "lr_warmup_steps": 10,
                "lora_target": "unet",
            },
            "lora-master",
            resolve_backend=lambda config, requested: (requested, f"trainer/{requested}.py"),
        )
        profile = profiles["legacy_main"]
        self.assertEqual(profile["type"], "Prodigy")
        self.assertTrue(profile["args"]["decouple"])
        self.assertEqual(profile["args"]["weight_decay"], 0.01)
        self.assertEqual(profile["args"]["d0"], 1e-6)
        self.assertEqual(profile["args"]["d_coef"], 2)
        self.assertTrue(profile["args"]["safeguard_warmup"])

    def test_bootstrap_honors_legacy_custom_toml_override(self):
        profiles = bootstrap_parameter_policy_optimizer_profile(
            {
                "optimizer_type": "AdamW",
                "lora_target": "unet",
                "ui_custom_params": 'optimizer_type = "Lion"',
            },
            "lora-master",
            resolve_backend=lambda config, requested: (requested, f"trainer/{requested}.py"),
        )
        self.assertEqual(profiles["legacy_main"]["type"], "Lion")

    def test_bootstrap_rejects_unregistered_custom_optimizer_without_affecting_standard(self):
        with self.assertRaisesRegex(ValueError, "无法自动迁移"):
            bootstrap_parameter_policy_optimizer_profile(
                {
                    "optimizer_type": "torch.optim.Adamax",
                    "lora_target": "unet",
                },
                "lora-master",
            )


if __name__ == "__main__":
    unittest.main()
