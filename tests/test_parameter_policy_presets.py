from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import tomllib

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.optimizer_profiles import get_optimizer_capability
from mikazuki.parameter_policy import canonicalize_parameter_policy
from mikazuki.parameter_policy_editor import normalize_parameter_policy_editor_state
from mikazuki.training_config import PAGE_BACKEND_MAP
from mikazuki.training_request import prepare_request_config


ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "config" / "presets"

CURATED = {
    "component-flux-finetune-muon.toml": ("flux-finetune", "flux-finetune"),
    "component-anima-finetune-muon.toml": ("anima-finetune", "anima-finetune"),
    "component-sdxl-finetune-split-lr.toml": ("sdxl-full", "sdxl-finetune"),
    "component-sdxl-lora-selective.toml": ("sdxl-lora", "sdxl-lora"),
}


class ParameterPolicyPresetTests(unittest.TestCase):
    def _load(self, filename: str) -> dict:
        with (PRESET_DIR / filename).open("rb") as handle:
            return tomllib.load(handle)

    def _normalized_policy(self, preset: dict) -> tuple[dict, dict]:
        data = deepcopy(preset["data"])
        normalize_parameter_policy_editor_state(data)
        policy = canonicalize_parameter_policy(
            {
                "optimizer_profiles": data["parameter_policy_profiles"],
                "components": data["parameter_policy_components"],
            }
        )
        return data, policy

    def test_curated_component_preset_set_is_exact(self):
        actual = {path.name for path in PRESET_DIR.glob("component-*.toml")}
        self.assertEqual(actual, set(CURATED))

    def test_curated_presets_match_page_backend_and_component_registries(self):
        for filename, (page_type, backend) in CURATED.items():
            with self.subTest(filename=filename):
                preset = self._load(filename)
                self.assertEqual(preset["metadata"]["train_type"], page_type)
                self.assertEqual(PAGE_BACKEND_MAP.get(page_type, page_type), backend)
                self.assertEqual(preset["data"]["optimization_mode"], "component")

                data, policy = self._normalized_policy(preset)
                self.assertTrue(data["parameter_policy_profiles"])
                self.assertTrue(data["parameter_policy_components"])
                self.assertEqual(
                    set(policy["components"]),
                    set(get_model_component_profile(backend).components),
                )

                for profile in policy["optimizer_profiles"].values():
                    capability = get_optimizer_capability(profile["type"])
                    self.assertEqual(
                        capability.component_support,
                        "supported",
                        f"{filename}: {profile['type']} is not Component-supported",
                    )

    def test_muon_routes_have_explicit_non_eligibility_fallback(self):
        for filename in (
            "component-flux-finetune-muon.toml",
            "component-anima-finetune-muon.toml",
        ):
            with self.subTest(filename=filename):
                _data, policy = self._normalized_policy(self._load(filename))
                profiles = policy["optimizer_profiles"]
                for component_id, route in policy["components"].items():
                    if not route.get("train"):
                        continue
                    primary = profiles[route["optimizer_profile"]]
                    if primary["type"] != "Muon":
                        continue
                    fallback_name = route.get("fallback_optimizer_profile")
                    self.assertTrue(fallback_name, component_id)
                    self.assertNotEqual(fallback_name, route["optimizer_profile"])
                    fallback = profiles[fallback_name]
                    fallback_capability = get_optimizer_capability(fallback["type"])
                    self.assertFalse(
                        fallback_capability.requires_parameter_eligibility,
                        component_id,
                    )
                    self.assertGreater(float(route["fallback_learning_rate"]), 0)

    def test_curated_presets_prepare_without_parameter_policy_runtime_blockers(self):
        for filename, (page_type, backend) in CURATED.items():
            with self.subTest(filename=filename):
                preset = self._load(filename)
                prepared = prepare_request_config(
                    deepcopy(preset["data"]),
                    page_type,
                    launch=False,
                )
                self.assertEqual(prepared.train_type, backend)
                self.assertEqual(prepared.runtime_blockers, [])


if __name__ == "__main__":
    unittest.main()
