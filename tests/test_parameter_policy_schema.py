from __future__ import annotations

import json
import re
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.optimizer_profiles import list_optimizer_capabilities
from mikazuki.parameter_policy_editor import parameter_policy_editor_metadata
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.parameter_policy_schema import (
    PARAMETER_POLICY_EDITOR_MARKER,
    parameter_policy_schema_fragment,
    wrap_parameter_policy_editor,
)


class ParameterPolicySchemaTests(unittest.TestCase):
    def test_fragment_exists_for_every_release_backend(self):
        for train_type in sorted(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES):
            with self.subTest(train_type=train_type):
                fragment = parameter_policy_schema_fragment(train_type)
                self.assertIn("optimization_mode", fragment)
                self.assertIn('default("standard")', fragment)
                self.assertIn("parameter_policy_profiles", fragment)
                self.assertIn("parameter_policy_components", fragment)

    def test_optimizer_choices_are_self_discriminating_and_disable_non_supported(self):
        capabilities = list_optimizer_capabilities()
        for train_type in sorted(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES):
            with self.subTest(train_type=train_type):
                metadata = parameter_policy_editor_metadata(train_type)
                fragment = parameter_policy_schema_fragment(train_type)
                self.assertEqual(
                    [row["type"] for row in metadata["optimizer_types"]],
                    [
                        capability.name
                        for capability in capabilities
                        if capability.component_support == "supported"
                    ],
                )
                self.assertEqual(
                    [row["type"] for row in metadata["optimizer_capabilities"]],
                    [capability.name for capability in capabilities],
                )
                self.assertIn(
                    "parameter_policy_profiles: Schema.dict(\n                    Schema.union([",
                    fragment,
                )
                self.assertNotIn(
                    "parameter_policy_profiles: Schema.dict(Schema.intersect([",
                    fragment,
                )
                for capability in capabilities:
                    option = f'type: Schema.const("{capability.name}").required()'
                    default = f'.default({{ type: "{capability.name}", args: {{}} }})'
                    self.assertIn(option, fragment)
                    self.assertIn(default, fragment)
                    branch_start = fragment.index(option)
                    next_branch = fragment.find("Schema.object({", branch_start + len(option))
                    branch_tail = fragment[branch_start: next_branch if next_branch >= 0 else len(fragment)]
                    if capability.component_support == "supported":
                        self.assertNotIn(".disabled()", branch_tail)
                    else:
                        self.assertIn(".disabled()", branch_tail)

    def test_component_ids_match_model_profile_exactly(self):
        for train_type in sorted(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES):
            with self.subTest(train_type=train_type):
                profile = get_model_component_profile(train_type)
                fragment = parameter_policy_schema_fragment(train_type)
                declared = set(
                    re.findall(
                        r'^\s*"([^"]+)": Schema\.intersect\(\[',
                        fragment,
                        flags=re.MULTILINE,
                    )
                )
                self.assertEqual(declared, set(profile.components))

    def test_component_rows_keep_train_guard_and_offer_profile_choices(self):
        fragment = parameter_policy_schema_fragment("flux-finetune")
        self.assertIn("train: Schema.boolean().default(false)", fragment)
        self.assertIn("train: Schema.const(true).required()", fragment)
        self.assertIn('optimizer_profile: Schema.union(["main", "legacy_main", "muon", "fallback"', fragment)
        self.assertIn('fallback_optimizer_profile: Schema.union(["fallback", "main", "legacy_main"', fragment)
        self.assertIn("learning_rate: Schema.string()", fragment)
        self.assertIn("fallback_learning_rate: Schema.string()", fragment)
        self.assertNotIn("]).collapse()", fragment)

    def test_muon_profile_uses_dedicated_typed_controls(self):
        fragment = parameter_policy_schema_fragment("sd-lora")
        self.assertIn('type: Schema.const("Muon").required()', fragment)
        self.assertIn('.default({ type: "Muon", args: {} })', fragment)
        self.assertIn("momentum: Schema.number().min(0)", fragment)
        self.assertNotIn("momentum: Schema.number().min(0).max(", fragment)
        self.assertNotIn("momentum: Schema.number().min(0).step(", fragment)
        self.assertIn("weight_decay: Schema.number().min(0)", fragment)
        self.assertNotIn("weight_decay: Schema.number().min(0).step(", fragment)
        self.assertIn("weight_decouple: Schema.boolean()", fragment)
        self.assertIn("nesterov: Schema.boolean()", fragment)
        self.assertIn("ns_steps: Schema.number()", fragment)
        self.assertIn(
            'ns_coeffs: Schema.union(["original", "quintic", "polar_express", "polar_express_safer"])',
            fragment,
        )
        self.assertIn("use_adjusted_lr: Schema.boolean()", fragment)
        self.assertIn("args: Schema.dict(Schema.string())", fragment)

    def test_optimizer_switch_cannot_drop_type_via_sibling_union(self):
        fragment = parameter_policy_schema_fragment("anima-finetune")
        profiles = fragment.split("parameter_policy_profiles:", 1)[1].split(
            "parameter_policy_components:", 1
        )[0]
        self.assertIn("Schema.dict(\n                    Schema.union([", profiles)
        self.assertNotIn("type: Schema.union(", profiles)
        self.assertNotIn("Schema.dict(Schema.intersect([", profiles)

    def test_wrapper_is_idempotent_and_marks_exactly_once(self):
        base = "Schema.object({foo: Schema.string()});"
        once = wrap_parameter_policy_editor(base, "sd-lora")
        twice = wrap_parameter_policy_editor(once, "sd-lora")
        self.assertEqual(once, twice)
        self.assertEqual(once.count(PARAMETER_POLICY_EDITOR_MARKER), 1)
        self.assertEqual(once.count("optimization_mode"), 2)


if __name__ == "__main__":
    unittest.main()
