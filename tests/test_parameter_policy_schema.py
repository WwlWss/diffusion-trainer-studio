from __future__ import annotations

import json
import re
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
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

    def test_optimizer_choices_match_editor_metadata_exactly(self):
        pattern = re.compile(
            r'type: Schema\.union\((\[[^\n]+\])\)\.default\("AdamW"\)'
        )
        for train_type in sorted(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES):
            with self.subTest(train_type=train_type):
                metadata = parameter_policy_editor_metadata(train_type)
                fragment = parameter_policy_schema_fragment(train_type)
                match = pattern.search(fragment)
                self.assertIsNotNone(match)
                self.assertEqual(
                    json.loads(match.group(1)),
                    [row["type"] for row in metadata["optimizer_types"]],
                )

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

    def test_component_rows_default_frozen_and_use_string_references(self):
        fragment = parameter_policy_schema_fragment("flux-finetune")
        self.assertIn("train: Schema.boolean().default(false)", fragment)
        self.assertIn("optimizer_profile: Schema.string()", fragment)
        self.assertIn("fallback_optimizer_profile: Schema.string()", fragment)
        self.assertIn("learning_rate: Schema.string()", fragment)
        self.assertIn("fallback_learning_rate: Schema.string()", fragment)

    def test_profile_args_use_string_dictionary_for_legacy_schemastery(self):
        fragment = parameter_policy_schema_fragment("sd-lora")
        self.assertIn("parameter_policy_profiles: Schema.dict", fragment)
        self.assertIn("args: Schema.dict(Schema.string())", fragment)

    def test_wrapper_is_idempotent_and_marks_exactly_once(self):
        base = "Schema.object({foo: Schema.string()});"
        once = wrap_parameter_policy_editor(base, "sd-lora")
        twice = wrap_parameter_policy_editor(once, "sd-lora")
        self.assertEqual(once, twice)
        self.assertEqual(once.count(PARAMETER_POLICY_EDITOR_MARKER), 1)
        self.assertEqual(once.count("optimization_mode"), 2)


if __name__ == "__main__":
    unittest.main()
