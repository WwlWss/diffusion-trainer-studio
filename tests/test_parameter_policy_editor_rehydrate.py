from __future__ import annotations

import json
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.training_rehydrate import rehydrate_trainer_config


class ParameterPolicyTrainingRehydrateTests(unittest.TestCase):
    def _policy(self, optimizer_type: str) -> dict:
        components = {}
        for index, component_id in enumerate(
            sorted(get_model_component_profile("sd-lora").components)
        ):
            components[component_id] = (
                {
                    "train": True,
                    "optimizer_profile": "main",
                    "learning_rate": 1e-4,
                }
                if index == 0
                else {"train": False}
            )
        return {
            "version": 1,
            "optimizer_profiles": {
                "main": {
                    "type": optimizer_type,
                    "args": {"eps": 1e-8, "amsgrad": False},
                }
            },
            "components": components,
        }

    def test_rehydrate_emits_legacy_editor_safe_literals(self):
        sidecar_path = "config/parameter-policy/test-policy.json"
        gui = rehydrate_trainer_config(
            {"parameter_policy_config": sidecar_path},
            "sd-lora",
            sidecars={sidecar_path: json.dumps(self._policy("AdamW"))},
        )

        self.assertEqual(gui["optimization_mode"], "component")
        self.assertEqual(
            gui["parameter_policy_profiles"]["main"]["args"],
            {"eps": "1e-08", "amsgrad": "false"},
        )
        trained = next(
            route
            for route in gui["parameter_policy_components"].values()
            if route["train"]
        )
        self.assertEqual(trained["learning_rate"], "0.0001")

    def test_rehydrate_preserves_restricted_optimizer_for_read_only_editor_state(self):
        sidecar_path = "config/parameter-policy/restricted-policy.json"
        policy = self._policy("AdaFactor")
        policy["optimizer_profiles"]["main"]["args"] = {}

        gui = rehydrate_trainer_config(
            {"parameter_policy_config": sidecar_path},
            "sd-lora",
            sidecars={sidecar_path: json.dumps(policy)},
        )

        self.assertEqual(
            gui["parameter_policy_profiles"]["main"]["type"],
            "AdaFactor",
        )


if __name__ == "__main__":
    unittest.main()
