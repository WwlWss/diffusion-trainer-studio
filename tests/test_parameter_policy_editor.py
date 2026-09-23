from __future__ import annotations

import copy
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.optimizer_profiles import list_optimizer_capabilities
from mikazuki.parameter_policy import canonicalize_parameter_policy
from mikazuki.parameter_policy_editor import (
    bootstrap_parameter_policy_editor,
    encode_parameter_policy_editor_state,
    normalize_parameter_policy_editor_state,
    parameter_policy_editor_metadata,
    parameter_policy_editor_preview,
    rehydrate_parameter_policy_editor,
)
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES


EXPECTED_BACKENDS = {
    "sd-lora",
    "sdxl-lora",
    "sd-dreambooth",
    "sdxl-finetune",
    "sd3-lora",
    "flux-lora",
    "chroma-lora",
    "flux-finetune",
    "anima-lora",
    "anima-finetune",
}


def _resolver(config, train_type):
    del config
    return train_type, f"trainer/{train_type}.py"


class ParameterPolicyEditorMetadataTests(unittest.TestCase):
    def test_release_backend_set_is_exact(self):
        self.assertEqual(set(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES), EXPECTED_BACKENDS)

    def test_metadata_is_registry_derived_for_every_release_backend(self):
        supported = [
            capability.name
            for capability in list_optimizer_capabilities()
            if capability.component_support == "supported"
        ]
        for train_type in sorted(EXPECTED_BACKENDS):
            with self.subTest(train_type=train_type):
                metadata = parameter_policy_editor_metadata(train_type)
                profile = get_model_component_profile(train_type)
                self.assertEqual(metadata["version"], 1)
                self.assertEqual(metadata["train_type"], train_type)
                self.assertEqual(
                    [row["type"] for row in metadata["optimizer_types"]],
                    supported,
                )
                self.assertEqual(
                    {row["id"] for row in metadata["components"]},
                    set(profile.components),
                )
                self.assertEqual(
                    {
                        row["id"]: row["label"]
                        for row in metadata["components"]
                    },
                    {
                        component_id: component.display_name
                        for component_id, component in profile.components.items()
                    },
                )

    def test_page_aliases_resolve_to_release_backend(self):
        self.assertEqual(
            parameter_policy_editor_metadata("lora-master")["train_type"],
            "sd-lora",
        )
        self.assertEqual(
            parameter_policy_editor_metadata("sdxl-full")["train_type"],
            "sdxl-finetune",
        )

    def test_release_backend_component_id_sets_are_pairwise_distinct(self):
        seen = {}
        for train_type in sorted(EXPECTED_BACKENDS):
            component_ids = frozenset(
                get_model_component_profile(train_type).components
            )
            self.assertNotIn(
                component_ids,
                seen,
                msg=(
                    f"{train_type} and {seen.get(component_ids)!r} expose the same "
                    "Component ID set; editor backend ownership would become ambiguous."
                ),
            )
            seen[component_ids] = train_type

    def test_unknown_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            parameter_policy_editor_metadata("unknown-backend")


class ParameterPolicyEditorNormalizationTests(unittest.TestCase):
    def test_standard_mode_is_noop_even_with_stale_editor_values(self):
        config = {
            "optimization_mode": "standard",
            "parameter_policy_profiles": {
                "main": {
                    "type": "AdamW",
                    "args": {"eps": "1e-8", "amsgrad": "false"},
                },
                "stale_null": None,
            },
        }
        before = copy.deepcopy(config)
        normalize_parameter_policy_editor_state(config)
        self.assertEqual(config, before)

    def test_named_null_profile_materializes_default_adamw(self):
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "fallback": None,
            },
        }
        normalize_parameter_policy_editor_state(config)
        self.assertEqual(
            config["parameter_policy_profiles"]["fallback"],
            {"type": "AdamW", "args": {}},
        )

    def test_blank_null_profile_placeholder_is_ignored(self):
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "": None,
                "main": {"type": "AdamW", "args": {}},
            },
        }
        normalize_parameter_policy_editor_state(config)
        self.assertEqual(
            config["parameter_policy_profiles"],
            {"main": {"type": "AdamW", "args": {}}},
        )

    def test_non_mapping_profile_still_fails_closed(self):
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "fallback": "AdamW",
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "Optimizer Profile 'fallback' must be an object",
        ):
            normalize_parameter_policy_editor_state(config)

    def test_named_null_fallback_compiles_with_muon_route(self):
        components = {
            component_id: {"train": False}
            for component_id in get_model_component_profile("anima-finetune").components
        }
        components["dit.self_attention"] = {
            "train": True,
            "optimizer_profile": "muon",
            "learning_rate": "1e-4",
            "fallback_optimizer_profile": "fallback",
            "fallback_learning_rate": "2.5e-5",
        }
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "muon": {
                    "type": "Muon",
                    "args": {
                        "momentum": 0.95,
                        "weight_decay": 0.01,
                        "weight_decouple": True,
                        "nesterov": True,
                        "ns_steps": 5,
                        "ns_coeffs": "original",
                        "use_adjusted_lr": False,
                    },
                },
                "fallback": None,
            },
            "parameter_policy_components": components,
        }

        normalize_parameter_policy_editor_state(config)
        policy = canonicalize_parameter_policy(
            {
                "optimizer_profiles": config["parameter_policy_profiles"],
                "components": config["parameter_policy_components"],
            }
        )
        self.assertEqual(policy["optimizer_profiles"]["fallback"]["type"], "AdamW")
        self.assertEqual(policy["optimizer_profiles"]["fallback"]["args"], {})
        self.assertEqual(
            policy["components"]["dit.self_attention"]["fallback_optimizer_profile"],
            "fallback",
        )

    def test_component_optimizer_args_parse_literals_but_preserve_bare_text(self):
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {
                    "type": "AdamW",
                    "args": {
                        "eps": "1e-8",
                        "amsgrad": "false",
                        "foreach": "true",
                        "optional": "none",
                        "betas": "(0.9, 0.95)",
                        "label": "quintic",
                        "quoted": '"hello"',
                    },
                }
            },
        }
        normalize_parameter_policy_editor_state(config)
        args = config["parameter_policy_profiles"]["main"]["args"]
        self.assertEqual(args["eps"], 1e-8)
        self.assertIs(args["amsgrad"], False)
        self.assertIs(args["foreach"], True)
        self.assertIsNone(args["optional"])
        self.assertEqual(args["betas"], (0.9, 0.95))
        self.assertEqual(args["label"], "quintic")
        self.assertEqual(args["quoted"], "hello")

    def test_blank_dict_rows_are_ignored_and_key_value_text_is_recovered(self):
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "muon": {
                    "type": "Muon",
                    "args": {
                        "": "",
                        " ": "momentum = 0.95",
                        "weight_decay": "0.01",
                        "weight_decouple": "true",
                        "nesterov": "true",
                        "ns_steps": "5",
                        "ns_coeffs": "original",
                        "use_adjusted_lr": "false",
                    },
                }
            },
        }
        normalize_parameter_policy_editor_state(config)
        args = config["parameter_policy_profiles"]["muon"]["args"]
        self.assertEqual(
            args,
            {
                "momentum": 0.95,
                "weight_decay": 0.01,
                "weight_decouple": True,
                "nesterov": True,
                "ns_steps": 5,
                "ns_coeffs": "original",
                "use_adjusted_lr": False,
            },
        )

    def test_train_toggle_ignores_and_restores_route_fields_deterministically(self):
        component_ids = sorted(get_model_component_profile("anima-finetune").components)
        target = component_ids[0]
        components = {
            component_id: {"train": False}
            for component_id in component_ids
        }
        components[target] = {
            "train": False,
            "optimizer_profile": "muon",
            "learning_rate": "1e-4",
            "fallback_optimizer_profile": "fallback",
            "fallback_learning_rate": "2.5e-5",
        }
        profiles = {
            "muon": {"type": "Muon", "args": {}},
            "fallback": {"type": "AdamW", "args": {}},
        }

        frozen = canonicalize_parameter_policy(
            {"optimizer_profiles": profiles, "components": components}
        )
        self.assertEqual(frozen["components"][target], {"train": False})

        components[target]["train"] = True
        active = canonicalize_parameter_policy(
            {"optimizer_profiles": profiles, "components": components}
        )
        self.assertTrue(active["components"][target]["train"])
        self.assertEqual(active["components"][target]["optimizer_profile"], "muon")
        self.assertEqual(
            active["components"][target]["fallback_optimizer_profile"],
            "fallback",
        )

        components[target]["train"] = False
        frozen_again = canonicalize_parameter_policy(
            {"optimizer_profiles": profiles, "components": components}
        )
        self.assertEqual(frozen_again["components"][target], {"train": False})

    def test_canonical_editor_round_trip_is_lossless(self):
        component_ids = sorted(get_model_component_profile("sd-lora").components)
        canonical = canonicalize_parameter_policy(
            {
                "optimizer_profiles": {
                    "main": {
                        "type": "AdamW",
                        "args": {
                            "eps": 1e-8,
                            "amsgrad": False,
                            "optional": None,
                            "betas": (0.9, 0.95),
                            "preset": "quintic",
                            "literal_string": "false",
                            "numeric_string": "1e-8",
                            "list_string": "[1, 2]",
                        },
                    }
                },
                "components": {
                    component_id: (
                        {
                            "train": True,
                            "optimizer_profile": "main",
                            "learning_rate": 1e-4,
                        }
                        if index == 0
                        else {"train": False}
                    )
                    for index, component_id in enumerate(component_ids)
                },
            }
        )
        editor = rehydrate_parameter_policy_editor(
            {
                "version": 1,
                "optimizer_profiles": canonical["optimizer_profiles"],
                "components": canonical["components"],
            }
        )
        self.assertIsInstance(
            editor["parameter_policy_components"][component_ids[0]]["learning_rate"],
            str,
        )
        args = editor["parameter_policy_profiles"]["main"]["args"]
        self.assertEqual(args["eps"], "1e-08")
        self.assertEqual(args["amsgrad"], "false")
        self.assertEqual(args["optional"], "null")
        self.assertEqual(args["betas"], "[0.9, 0.95]")
        self.assertEqual(args["preset"], "quintic")
        self.assertEqual(args["literal_string"], "'false'")
        self.assertEqual(args["numeric_string"], "'1e-8'")
        self.assertEqual(args["list_string"], "'[1, 2]'")

        normalized = copy.deepcopy(editor)
        normalize_parameter_policy_editor_state(normalized)
        round_tripped = canonicalize_parameter_policy(
            {
                "optimizer_profiles": normalized["parameter_policy_profiles"],
                "components": normalized["parameter_policy_components"],
            }
        )
        self.assertEqual(round_tripped, canonical)

    def test_muon_editor_rehydrate_preserves_typed_args(self):
        gui = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "muon": {
                    "type": "Muon",
                    "args": {
                        "momentum": 0.95,
                        "weight_decay": 0.01,
                        "weight_decouple": True,
                        "nesterov": True,
                        "ns_steps": 5,
                        "ns_coeffs": "original",
                        "use_adjusted_lr": False,
                    },
                }
            },
            "parameter_policy_components": {
                component_id: {"train": False}
                for component_id in get_model_component_profile("anima-finetune").components
            },
        }
        encoded = encode_parameter_policy_editor_state(gui)
        args = encoded["parameter_policy_profiles"]["muon"]["args"]
        self.assertIsInstance(args["momentum"], float)
        self.assertIsInstance(args["weight_decay"], float)
        self.assertIs(args["weight_decouple"], True)
        self.assertIs(args["nesterov"], True)
        self.assertIsInstance(args["ns_steps"], int)
        self.assertEqual(args["ns_coeffs"], "original")
        self.assertIs(args["use_adjusted_lr"], False)

        canonical = canonicalize_parameter_policy(
            {
                "optimizer_profiles": gui["parameter_policy_profiles"],
                "components": gui["parameter_policy_components"],
            }
        )
        rehydrated = rehydrate_parameter_policy_editor(canonical)
        rehydrated_args = rehydrated["parameter_policy_profiles"]["muon"]["args"]
        self.assertEqual(rehydrated_args, args)
        self.assertIsInstance(rehydrated_args["momentum"], float)
        self.assertIs(rehydrated_args["weight_decouple"], True)
        self.assertIsInstance(rehydrated_args["ns_steps"], int)

        normalized = copy.deepcopy(rehydrated)
        normalize_parameter_policy_editor_state(normalized)
        self.assertEqual(
            normalized["parameter_policy_profiles"]["muon"]["args"],
            gui["parameter_policy_profiles"]["muon"]["args"],
        )

    def test_editor_encoder_does_not_mutate_input(self):
        gui = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {
                    "type": "AdamW",
                    "args": {"eps": 1e-8, "name": "quintic"},
                }
            },
            "parameter_policy_components": {
                component_id: {"train": False}
                for component_id in get_model_component_profile("sd-lora").components
            },
        }
        before = copy.deepcopy(gui)
        encoded = encode_parameter_policy_editor_state(gui)
        self.assertEqual(gui, before)
        self.assertIsNot(encoded, gui)
        self.assertEqual(
            encoded["parameter_policy_profiles"]["main"]["args"]["eps"],
            "1e-08",
        )

    def test_malformed_structured_literal_fails_closed_with_context(self):
        config = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {"type": "AdamW", "args": {"betas": "[0.9, 0.95"}}
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            r"parameter_policy_profiles\.main\.args\.betas",
        ):
            normalize_parameter_policy_editor_state(config)


class ParameterPolicyEditorBootstrapTests(unittest.TestCase):
    def test_standard_bootstrap_returns_component_gui_state(self):
        raw = {
            "optimizer_type": "AdamW",
            "learning_rate": "1e-4",
            "lora_target": "unet",
        }
        before = copy.deepcopy(raw)
        gui = bootstrap_parameter_policy_editor(
            raw,
            "lora-master",
            resolve_backend=_resolver,
        )
        self.assertEqual(raw, before)
        self.assertEqual(gui["optimization_mode"], "component")
        self.assertEqual(set(gui["parameter_policy_profiles"]), {"legacy_main"})
        self.assertTrue(gui["parameter_policy_components"])
        for component_id, route in gui["parameter_policy_components"].items():
            if component_id.startswith("unet."):
                self.assertTrue(route["train"])
            else:
                self.assertEqual(route, {"train": False})

    def test_existing_complete_component_state_is_preserved_without_migration(self):
        components = {}
        for index, component_id in enumerate(
            sorted(get_model_component_profile("flux-finetune").components)
        ):
            components[component_id] = (
                {
                    "train": True,
                    "optimizer_profile": "main",
                    "learning_rate": 2e-4,
                }
                if index == 0
                else {"train": False}
            )
        raw = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {"type": "AdamW", "args": {"eps": "1e-8"}}
            },
            "parameter_policy_components": components,
        }

        def forbidden_resolver(*args, **kwargs):
            raise AssertionError("existing Component state must not re-bootstrap Standard")

        gui = bootstrap_parameter_policy_editor(
            raw,
            "flux-finetune",
            resolve_backend=forbidden_resolver,
        )
        self.assertEqual(gui["optimization_mode"], "component")
        self.assertEqual(
            gui["parameter_policy_profiles"]["main"]["args"]["eps"],
            "1e-08",
        )
        self.assertEqual(
            set(gui["parameter_policy_components"]),
            set(get_model_component_profile("flux-finetune").components),
        )

    def test_existing_component_state_from_another_backend_fails_closed(self):
        raw = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {"type": "AdamW", "args": {}}
            },
            "parameter_policy_components": {
                component_id: {
                    "train": True,
                    "optimizer_profile": "main",
                    "learning_rate": 1e-4,
                }
                for component_id in get_model_component_profile("sd-lora").components
            },
        }
        with self.assertRaisesRegex(ValueError, "does not match backend"):
            bootstrap_parameter_policy_editor(
                raw,
                "flux-finetune",
                resolve_backend=_resolver,
            )

    def test_standard_bootstrap_rejects_restricted_optimizer_profiles(self):
        for optimizer_type in ("AdaFactor", "Prodigy"):
            with self.subTest(optimizer_type=optimizer_type):
                raw = {
                    "optimizer_type": optimizer_type,
                    "learning_rate": 1e-4,
                    "lora_target": "unet",
                }
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot be migrated into a runnable Component-wise profile",
                ):
                    bootstrap_parameter_policy_editor(
                        raw,
                        "lora-master",
                        resolve_backend=_resolver,
                    )

    def test_existing_restricted_component_policy_remains_rehydratable(self):
        components = {
            component_id: {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 1e-4,
            }
            for component_id in get_model_component_profile("sd-lora").components
        }
        raw = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {"type": "AdaFactor", "args": {}}
            },
            "parameter_policy_components": components,
        }

        gui = bootstrap_parameter_policy_editor(
            raw,
            "sd-lora",
            resolve_backend=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("existing restricted Component policy must not re-bootstrap Standard")
            ),
        )
        self.assertEqual(
            gui["parameter_policy_profiles"]["main"]["type"],
            "AdaFactor",
        )

    def test_partial_existing_component_state_fails_closed(self):
        raw = {
            "optimization_mode": "component",
            "parameter_policy_profiles": {
                "main": {"type": "AdamW", "args": {}}
            },
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            bootstrap_parameter_policy_editor(
                raw,
                "flux-finetune",
                resolve_backend=_resolver,
            )

    def test_standard_bootstrap_keeps_compatibility_blockers_authoritative(self):
        raw = {
            "optimizer_type": "AdamW",
            "learning_rate": 1e-4,
            "lora_target": "unet",
            "fused_backward_pass": True,
        }
        with self.assertRaisesRegex(ValueError, "fused_backward_pass"):
            bootstrap_parameter_policy_editor(
                raw,
                "lora-master",
                resolve_backend=_resolver,
            )


class ParameterPolicyEditorPreviewTests(unittest.TestCase):
    def test_preview_is_model_free_and_carries_runtime_readiness(self):
        policy = {
            "version": 1,
            "optimizer_profiles": {
                "main": {"type": "AdamW", "args": {}}
            },
            "components": {
                component_id: (
                    {
                        "train": True,
                        "optimizer_profile": "main",
                        "learning_rate": 1e-6,
                    }
                    if index == 0
                    else {"train": False}
                )
                for index, component_id in enumerate(
                    sorted(get_model_component_profile("flux-finetune").components)
                )
            },
        }
        preview = parameter_policy_editor_preview(
            policy,
            "flux-finetune",
            ["qualification pending"],
        )
        self.assertFalse(preview["runtime_ready"])
        self.assertEqual(preview["runtime_blockers"], ["qualification pending"])
        self.assertEqual(preview["profiles"][0]["name"], "main")
        self.assertEqual(
            {row["id"] for row in preview["components"]},
            set(get_model_component_profile("flux-finetune").components),
        )


if __name__ == "__main__":
    unittest.main()
