from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import tomllib

from mikazuki.model_component_profiles import (
    get_model_component_profile,
    resolve_training_target_profile,
)
from mikazuki.optimizer_profiles import get_optimizer_capability
from mikazuki.parameter_policy import (
    build_parameter_policy_sidecar,
    canonicalize_parameter_policy,
    parameter_policy_runtime_blockers,
)
from mikazuki.parameter_policy_editor import normalize_parameter_policy_editor_state
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.training_config import PAGE_BACKEND_MAP, prepare_training_config


ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "config" / "presets"

def _resolve_backend(config: dict, requested: str):
    return requested, f"./{requested}.py"


CURATED = {
    "component-flux-finetune-muon.toml": ("flux-finetune", "flux-finetune"),
    "component-anima-finetune-muon.toml": ("anima-finetune", "anima-finetune"),
    "component-sdxl-finetune-split-lr.toml": ("sdxl-full", "sdxl-finetune"),
    "component-sdxl-lora-selective.toml": ("sdxl-lora", "sdxl-lora"),
}


STANDARD_PRESETS = {
    "anima-2.9b-finetune.toml",
    "anima-2.9b.toml",
    "anima-finetune.toml",
    "anima.toml",
    "chroma.toml",
    "example.toml",
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

    def test_standard_presets_declare_standard_optimization_mode(self):
        actual = {
            path.name
            for path in PRESET_DIR.glob("*.toml")
            if not path.name.startswith("component-")
        }
        self.assertEqual(actual, STANDARD_PRESETS)
        for filename in sorted(STANDARD_PRESETS):
            with self.subTest(filename=filename):
                preset = self._load(filename)
                self.assertEqual(
                    preset["data"].get("optimization_mode"),
                    "standard",
                )

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

    def test_curated_presets_compile_effective_config_without_runtime_or_target_blockers(self):
        for filename, (page_type, backend) in CURATED.items():
            with self.subTest(filename=filename):
                preset = self._load(filename)
                data = deepcopy(preset["data"])
                normalize_parameter_policy_editor_state(data)
                policy_path, sidecars, policy = build_parameter_policy_sidecar(
                    data,
                    page_type,
                )
                self.assertIsNotNone(policy_path)
                self.assertIn(policy_path, sidecars)
                self.assertIsNotNone(policy)

                prepared = prepare_training_config(
                    data,
                    page_train_type=page_type,
                    resolve_backend=_resolve_backend,
                    launch=False,
                )
                self.assertEqual(prepared.train_type, backend)

                blockers = parameter_policy_runtime_blockers(
                    policy,
                    train_type=prepared.train_type,
                    effective_config=prepared.config,
                    integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
                )
                self.assertEqual(blockers, [])

                target = resolve_training_target_profile(
                    prepared.train_type,
                    prepared.config,
                )
                for component_id, route in policy["components"].items():
                    if route.get("train"):
                        self.assertTrue(
                            target.is_available(component_id),
                            (
                                f"{filename}: {component_id} is Train=true but "
                                "unavailable after effective-config normalization"
                            ),
                        )

    def test_sdxl_lora_preset_overrides_stale_public_target_state(self):
        preset = self._load("component-sdxl-lora-selective.toml")
        current = {"lora_target": "text_encoder"}
        merged = {**current, **deepcopy(preset["data"])}
        normalize_parameter_policy_editor_state(merged)
        _path, _sidecars, policy = build_parameter_policy_sidecar(
            merged,
            preset["metadata"]["train_type"],
        )
        prepared = prepare_training_config(
            merged,
            page_train_type=preset["metadata"]["train_type"],
            resolve_backend=_resolve_backend,
            launch=False,
        )
        target = resolve_training_target_profile(
            prepared.train_type,
            prepared.config,
        )
        self.assertTrue(prepared.config["network_train_unet_only"])
        self.assertNotIn("network_train_text_encoder_only", prepared.config)
        self.assertTrue(target.is_available("unet.attention.adapter"))
        self.assertTrue(target.is_available("unet.feed_forward.adapter"))
        self.assertFalse(target.is_available("text_encoder_1.adapter"))
        self.assertFalse(target.is_available("text_encoder_2.adapter"))
        self.assertTrue(policy["components"]["unet.attention.adapter"]["train"])
        self.assertTrue(policy["components"]["unet.feed_forward.adapter"]["train"])


if __name__ == "__main__":
    unittest.main()
