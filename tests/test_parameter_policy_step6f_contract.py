from __future__ import annotations

import unittest
from pathlib import Path

from mikazuki.parameter_policy_matrix import (
    PARAMETER_POLICY_BACKEND_MATRIX,
    PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS,
    PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
    PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES,
    parameter_policy_gpu_selection_blockers,
)


ROOT = Path(__file__).resolve().parents[1]
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


class ParameterPolicyStep6FContractTests(unittest.TestCase):
    def test_release_matrix_opens_exact_integrated_backend_set(self):
        self.assertEqual(set(PARAMETER_POLICY_BACKEND_MATRIX), EXPECTED_BACKENDS)
        self.assertEqual(set(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES), EXPECTED_BACKENDS)

    def test_matrix_trainer_paths_match_host_mapping_literals(self):
        source = (ROOT / "mikazuki" / "app" / "api.py").read_text(encoding="utf-8")
        for train_type, row in PARAMETER_POLICY_BACKEND_MATRIX.items():
            with self.subTest(train_type=train_type):
                self.assertIn(f'"{train_type}"', source)
                self.assertIn(row["trainer"], source)

    def test_explicit_multi_gpu_component_start_remains_fail_closed(self):
        self.assertEqual(parameter_policy_gpu_selection_blockers(None), [])
        self.assertEqual(parameter_policy_gpu_selection_blockers([]), [])
        self.assertEqual(parameter_policy_gpu_selection_blockers(["0"]), [])
        blockers = parameter_policy_gpu_selection_blockers(["0", "1"])
        self.assertEqual(len(blockers), 1)
        self.assertIn("多 GPU", blockers[0])

    def test_duplicate_gpu_ids_are_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parameter_policy_gpu_selection_blockers(["0", 0])

    def test_qualification_and_semantic_blockers_are_separate(self):
        self.assertIn("torch_compile", PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS)
        self.assertIn("deepspeed", PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS)
        self.assertIn("blocks_to_swap", PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS)
        self.assertIn("lora_plus", PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES)
        self.assertIn("regex_lr", PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES)
        self.assertNotIn("lora_plus", PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS)

    def test_request_gate_is_matrix_owned_and_component_compile_stays_side_effect_free(self):
        source = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")
        self.assertIn("from mikazuki.parameter_policy_matrix import", source)
        self.assertIn("parameter_policy_gpu_selection_blockers(prepared.gpu_ids)", source)
        self.assertIn("effective_launch = launch and policy is None", source)
        self.assertNotIn("Step 6A must remain closed", source)
        self.assertIn("if launch and blockers:", source)

    def test_component_api_validates_before_runtime_staging_and_materialization(self):
        source = (ROOT / "mikazuki" / "app" / "training_api.py").read_text(encoding="utf-8")
        start = source.index("if component_start:")
        validation = source.index(
            "validate_prepared_config(prepared, True, check_trainer=False)",
            start,
        )
        staging = source.index("prepare_runtime_trainer(prepared)", validation)
        trainer_validation = source.index("validate_prepared_trainer(prepared)", staging)
        launch_finalize = source.index(
            "materialize_anima_launch_side_effects(prepared.config, prepared.train_type)",
            trainer_validation,
        )
        sidecars = source.index("materialize_sidecars(prepared.sidecars)", launch_finalize)
        toml_write = source.index("_write_text_atomic(toml_path", sidecars)
        self.assertLess(validation, staging)
        self.assertLess(staging, trainer_validation)
        self.assertLess(trainer_validation, launch_finalize)
        self.assertLess(launch_finalize, sidecars)
        self.assertLess(sidecars, toml_write)

    def test_standard_launch_order_is_preserved(self):
        source = (ROOT / "mikazuki" / "app" / "training_api.py").read_text(encoding="utf-8")
        standard = source.index("else:\n            # Preserve the Standard-mode launch ordering.")
        staging = source.index("prepare_runtime_trainer(prepared)", standard)
        validation = source.index("validate_prepared_config(prepared, True)", staging)
        self.assertLess(staging, validation)

    def test_anima_component_launch_side_effects_are_explicit(self):
        source = (ROOT / "mikazuki" / "anima_effective_config.py").read_text(encoding="utf-8")
        self.assertIn("def materialize_anima_launch_side_effects(", source)
        helper = source[source.index("def materialize_anima_launch_side_effects("):]
        self.assertIn("os.makedirs", helper)


if __name__ == "__main__":
    unittest.main()
