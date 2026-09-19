import unittest

from mikazuki.training_request import prepare_request_config


def _component_fields():
    return {
        "optimization_mode": "component",
        "parameter_policy_profiles": {
            "main_muon": {"type": "Muon", "args": {}},
            "aux_adamw": {"type": "AdamW", "args": {}},
        },
        "parameter_policy_components": {
            "test.component": {
                "train": True,
                "optimizer_profile": "main_muon",
                "learning_rate": 1e-4,
                "fallback_optimizer_profile": "aux_adamw",
            }
        },
    }


class ParameterPolicyRequestContractTests(unittest.TestCase):
    def test_standard_missing_and_explicit_modes_prepare_identically(self):
        base = {
            "optimizer_type": "AdamW8bit",
            "learning_rate": 1e-4,
            "lora_target": "unet",
        }
        missing = prepare_request_config(dict(base), "lora-basic", launch=False)
        explicit = prepare_request_config(
            {
                **base,
                "optimization_mode": "standard",
                "parameter_policy_profiles": {"stale": {"type": "Muon"}},
                "parameter_policy_components": {"stale": {"train": True}},
            },
            "lora-basic",
            launch=False,
        )
        self.assertEqual(missing.config, explicit.config)
        self.assertEqual(missing.sidecars, explicit.sidecars)
        self.assertEqual(missing.runtime_blockers, [])
        self.assertEqual(explicit.runtime_blockers, [])
        self.assertNotIn("parameter_policy_config", missing.config)

    def test_component_preview_has_sidecar_and_runtime_blocker(self):
        prepared = prepare_request_config(
            {
                "optimizer_type": "AdamW",
                "learning_rate": 1e-4,
                "lora_target": "unet",
                **_component_fields(),
            },
            "lora-basic",
            launch=False,
        )
        policy_path = prepared.config.get("parameter_policy_config")
        self.assertTrue(policy_path)
        self.assertIn(policy_path, prepared.sidecars)
        self.assertTrue(prepared.runtime_blockers)
        self.assertIn("runtime 尚未实现", prepared.runtime_blockers[0])

    def test_component_start_fails_closed_before_runtime_exists(self):
        with self.assertRaisesRegex(ValueError, "runtime 尚未实现"):
            prepare_request_config(
                {
                    "optimizer_type": "AdamW",
                    "learning_rate": 1e-4,
                    "lora_target": "unet",
                    **_component_fields(),
                },
                "lora-basic",
                launch=True,
            )

    def test_component_custom_override_cannot_replace_host_sidecar(self):
        config = {
            "optimizer_type": "AdamW",
            "learning_rate": 1e-4,
            "lora_target": "unet",
            **_component_fields(),
            "ui_custom_params": 'parameter_policy_config = "evil.json"',
        }
        with self.assertRaisesRegex(ValueError, "冲突|托管"):
            prepare_request_config(config, "lora-basic", launch=False)


if __name__ == "__main__":
    unittest.main()
