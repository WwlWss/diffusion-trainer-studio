from __future__ import annotations

import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import mikazuki.parameter_policy_execution as execution
from mikazuki.optimizer_profiles import list_optimizer_capabilities
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_BACKEND_MATRIX


def _policy(*, primary: str = "main", fallback: str | None = None, extra_profiles=None):
    profiles = {
        "main": {"type": "AdamW", "args": {}},
    }
    if fallback is not None:
        profiles["fallback"] = {"type": fallback, "args": {}}
    if extra_profiles:
        profiles.update(extra_profiles)

    route = {
        "train": True,
        "optimizer_profile": primary,
        "learning_rate": 1e-4,
    }
    if fallback is not None:
        route["fallback_optimizer_profile"] = "fallback"
        route["fallback_learning_rate"] = 1e-5

    return {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": {
            "component": route,
        },
    }


class _ExplodingMapping(dict):
    def get(self, *args, **kwargs):
        raise AssertionError("inactive execution feature consulted qualification metadata")


ROOT = Path(__file__).resolve().parents[1]


class ParameterPolicyExecutionIdentityTests(unittest.TestCase):
    def _contract(self, **overrides):
        values = {
            "train_type": "anima-finetune",
            "features": ("full_bf16",),
            "mixed_precision": "bf16",
            "expected_trainable_parameter_dtype": "bfloat16",
            "require_live_root_identity": True,
        }
        values.update(overrides)
        return execution.ParameterPolicyExecutionContract(**values)

    def test_full_bf16_execution_identity_is_exact_and_stable(self):
        contract = self._contract()
        self.assertEqual(
            contract.execution_identity(),
            {
                "schema": "dts.parameter-policy.execution-identity",
                "version": 1,
                "train_type": "anima-finetune",
                "features": {
                    "full_bf16": {
                        "mixed_precision": "bf16",
                        "trainable_parameter_dtype": "bfloat16",
                    }
                },
            },
        )
        self.assertEqual(
            contract.execution_signature(),
            "9f511c8f94192fe7ad4728285157466a9d6cc79488e71b0be569da62451dcca8",
        )

    def test_runtime_enforcement_fields_do_not_change_execution_identity(self):
        strict = self._contract(require_live_root_identity=True)
        relaxed = self._contract(require_live_root_identity=False)
        self.assertEqual(strict.execution_identity(), relaxed.execution_identity())
        self.assertEqual(strict.execution_signature(), relaxed.execution_signature())

    def test_execution_identity_normalizes_train_type(self):
        left = self._contract(train_type=" Anima-Finetune ")
        right = self._contract(train_type="anima-finetune")
        self.assertEqual(left.execution_identity(), right.execution_identity())
        self.assertEqual(left.execution_signature(), right.execution_signature())

    def test_execution_identity_returns_fresh_mutable_payload(self):
        contract = self._contract()
        first = contract.execution_identity()
        first["features"]["full_bf16"]["mixed_precision"] = "mutated"
        second = contract.execution_identity()
        self.assertEqual(
            second["features"]["full_bf16"]["mixed_precision"],
            "bf16",
        )

    def test_execution_identity_rejects_unknown_feature_combinations(self):
        for features in ((), ("future_feature",), ("full_bf16", "future_feature")):
            with self.subTest(features=features), self.assertRaisesRegex(
                ValueError,
                "feature combination",
            ):
                self._contract(features=features).execution_identity()

    def test_execution_identity_rejects_malformed_full_bf16_semantics(self):
        cases = (
            ({"mixed_precision": "fp16"}, "mixed_precision"),
            (
                {"expected_trainable_parameter_dtype": "float32"},
                "bfloat16",
            ),
            ({"train_type": "   "}, "train_type"),
        )
        for overrides, marker in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError,
                marker,
            ):
                self._contract(**overrides).execution_identity()


class ParameterPolicyExecutionContractBuilderTests(unittest.TestCase):
    def test_inactive_execution_contract_is_none_before_other_validation(self):
        for raw in (None, False, 0, "", "false", "off", "no"):
            with self.subTest(raw=raw), patch.object(
                execution,
                "FULL_BF16_BACKEND_QUALIFICATIONS",
                _ExplodingMapping(),
            ), patch.object(
                execution,
                "FULL_BF16_OPTIMIZER_QUALIFICATIONS",
                _ExplodingMapping(),
            ):
                self.assertIsNone(
                    execution.build_parameter_policy_execution_contract(
                        train_type="",
                        effective_config={
                            "full_bf16": raw,
                            "mixed_precision": "not-a-real-mode",
                        },
                    )
                )

    def test_full_bf16_contract_is_compiled_from_effective_config(self):
        contract = execution.build_parameter_policy_execution_contract(
            train_type=" Anima-Finetune ",
            effective_config={
                "full_bf16": "true",
                "mixed_precision": " BF16 ",
            },
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract.train_type, "anima-finetune")
        self.assertEqual(contract.features, ("full_bf16",))
        self.assertEqual(contract.mixed_precision, "bf16")
        self.assertEqual(
            contract.expected_trainable_parameter_dtype,
            "bfloat16",
        )
        self.assertTrue(contract.require_live_root_identity)

    def test_contract_builder_does_not_consult_qualification_metadata(self):
        with patch.object(
            execution,
            "FULL_BF16_BACKEND_QUALIFICATIONS",
            _ExplodingMapping(),
        ), patch.object(
            execution,
            "FULL_BF16_OPTIMIZER_QUALIFICATIONS",
            _ExplodingMapping(),
        ):
            contract = execution.build_parameter_policy_execution_contract(
                train_type="anima-finetune",
                effective_config={
                    "full_bf16": True,
                    "mixed_precision": "bf16",
                },
            )
        self.assertEqual(contract.features, ("full_bf16",))

    def test_contract_builder_does_not_mutate_effective_config(self):
        config = {
            "full_bf16": True,
            "mixed_precision": "bf16",
            "unrelated": {
                "nested": [1, 2, {"value": "keep"}],
            },
        }
        before = copy.deepcopy(config)
        execution.build_parameter_policy_execution_contract(
            train_type="flux-finetune",
            effective_config=config,
        )
        self.assertEqual(config, before)

    def test_unrelated_config_does_not_change_execution_identity_or_signature(self):
        left = execution.build_parameter_policy_execution_contract(
            train_type="anima-finetune",
            effective_config={
                "full_bf16": True,
                "mixed_precision": "bf16",
                "batch_size": 1,
                "learning_rate": 1e-6,
                "gradient_checkpointing": False,
            },
        )
        right = execution.build_parameter_policy_execution_contract(
            train_type=" Anima-Finetune ",
            effective_config={
                "gradient_checkpointing": True,
                "learning_rate": 9e-4,
                "batch_size": 32,
                "mixed_precision": "BF16",
                "full_bf16": "true",
            },
        )
        self.assertEqual(left.execution_identity(), right.execution_identity())
        self.assertEqual(left.execution_signature(), right.execution_signature())

    def test_full_bf16_contract_rejects_invalid_mixed_precision(self):
        for raw in (None, "no", "fp16", "float32"):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError,
                "mixed_precision='bf16'",
            ):
                execution.build_parameter_policy_execution_contract(
                    train_type="anima-finetune",
                    effective_config={
                        "full_bf16": True,
                        "mixed_precision": raw,
                    },
                )

    def test_active_contract_requires_non_empty_train_type(self):
        for raw in (None, "", "   "):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError,
                "train_type",
            ):
                execution.build_parameter_policy_execution_contract(
                    train_type=raw,
                    effective_config={
                        "full_bf16": True,
                        "mixed_precision": "bf16",
                    },
                )

    def test_unknown_future_feature_combination_fails_closed(self):
        with patch.object(
            execution,
            "active_parameter_policy_execution_features",
            return_value=("full_bf16", "future_feature"),
        ), self.assertRaisesRegex(ValueError, "feature combination"):
            execution.build_parameter_policy_execution_contract(
                train_type="anima-finetune",
                effective_config={
                    "full_bf16": True,
                    "mixed_precision": "bf16",
                },
            )


class ParameterPolicyExecutionMetadataTests(unittest.TestCase):
    def test_backend_full_bf16_table_matches_release_matrix_exactly(self):
        self.assertEqual(
            set(execution.FULL_BF16_BACKEND_QUALIFICATIONS),
            set(PARAMETER_POLICY_BACKEND_MATRIX),
        )

    def test_optimizer_full_bf16_table_matches_baseline_supported_optimizers(self):
        supported = {
            capability.name
            for capability in list_optimizer_capabilities()
            if capability.component_support == "supported"
        }
        self.assertEqual(
            set(execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS),
            supported,
        )

    def test_phase_a_has_no_qualified_backend_or_optimizer(self):
        self.assertFalse(
            any(
                row.status == "qualified"
                for row in execution.FULL_BF16_BACKEND_QUALIFICATIONS.values()
            )
        )
        self.assertFalse(
            any(
                row.status == "qualified"
                for row in execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS.values()
            )
        )

    def test_sd_dreambooth_starts_explicitly_unsupported(self):
        row = execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd-dreambooth"]
        self.assertEqual(row.status, "unsupported")
        self.assertTrue(row.reason)

    def test_qualification_rows_are_structurally_valid_and_fail_closed_by_default(self):
        allowed = {"pending", "qualified", "unsupported"}
        rows = (
            list(execution.FULL_BF16_BACKEND_QUALIFICATIONS.values())
            + list(execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS.values())
        )
        for row in rows:
            with self.subTest(status=row.status, reason=row.reason):
                self.assertIn(row.status, allowed)
                self.assertTrue(row.reason.strip())
                if row.status == "qualified":
                    self.assertTrue(str(row.evidence_case_id or "").strip())
                else:
                    self.assertIsNone(row.evidence_case_id)

    def test_execution_module_remains_torch_free_and_host_side(self):
        source = (ROOT / "mikazuki" / "parameter_policy_execution.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        forbidden_prefixes = (
            "torch",
            "accelerate",
            "mikazuki.parameter_policy_trainer",
            "mikazuki.parameter_policy_torch",
        )
        for imported in imported_modules:
            for forbidden in forbidden_prefixes:
                self.assertFalse(
                    imported == forbidden or imported.startswith(forbidden + "."),
                    f"execution module must not import runtime dependency {imported!r}",
                )


class ParameterPolicyExecutionFeatureDetectionTests(unittest.TestCase):
    def test_inactive_full_bf16_returns_before_qualification_tables(self):
        for raw in (None, False, 0, 0.0, "", "0", "false", "off", "no"):
            with self.subTest(raw=raw), patch.object(
                execution,
                "FULL_BF16_BACKEND_QUALIFICATIONS",
                _ExplodingMapping(),
            ), patch.object(
                execution,
                "FULL_BF16_OPTIMIZER_QUALIFICATIONS",
                _ExplodingMapping(),
            ):
                self.assertEqual(
                    execution.parameter_policy_execution_blockers(
                        {"not": "canonical and intentionally unread"},
                        train_type="future-backend",
                        effective_config={"full_bf16": raw},
                    ),
                    [],
                )

    def test_active_feature_detection_accepts_only_strict_boolean_forms(self):
        for raw in (True, 1, 1.0, "1", "true", "yes", "on"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    execution.active_parameter_policy_execution_features(
                        {"full_bf16": raw}
                    ),
                    ("full_bf16",),
                )

        for raw in (None, False, 0, 0.0, "", "0", "false", "no", "off"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    execution.active_parameter_policy_execution_features(
                        {"full_bf16": raw}
                    ),
                    (),
                )

    def test_malformed_full_bf16_values_fail_closed(self):
        for raw in ("maybe", 2, 1.5, [], {}, object()):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError,
                "full_bf16",
            ):
                execution.active_parameter_policy_execution_features(
                    {"full_bf16": raw}
                )

    def test_full_bf16_requires_bf16_mixed_precision_before_metadata(self):
        with patch.object(
            execution,
            "FULL_BF16_BACKEND_QUALIFICATIONS",
            _ExplodingMapping(),
        ):
            blockers = execution.parameter_policy_execution_blockers(
                _policy(),
                train_type="sdxl-finetune",
                effective_config={
                    "full_bf16": True,
                    "mixed_precision": "fp16",
                },
            )
        self.assertEqual(len(blockers), 1)
        self.assertIn("mixed_precision='bf16'", blockers[0])


class ParameterPolicyExecutionQualificationTests(unittest.TestCase):
    def test_pending_and_unsupported_backends_fail_closed(self):
        pending = execution.parameter_policy_execution_blockers(
            _policy(),
            train_type="sdxl-finetune",
            effective_config={"full_bf16": True, "mixed_precision": "bf16"},
        )
        self.assertEqual(len(pending), 1)
        self.assertIn("pending", pending[0])

        unsupported = execution.parameter_policy_execution_blockers(
            _policy(),
            train_type="sd-dreambooth",
            effective_config={"full_bf16": True, "mixed_precision": "bf16"},
        )
        self.assertEqual(len(unsupported), 1)
        self.assertIn("unsupported", unsupported[0])

    def test_unknown_backend_fails_closed(self):
        blockers = execution.parameter_policy_execution_blockers(
            _policy(),
            train_type="future-backend",
            effective_config={"full_bf16": True, "mixed_precision": "bf16"},
        )
        self.assertEqual(len(blockers), 1)
        self.assertIn("no qualification record", blockers[0])

    def test_qualified_without_evidence_still_fails_closed(self):
        qualified_without_evidence = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only state",
            None,
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sdxl-finetune": qualified_without_evidence},
        ):
            blockers = execution.parameter_policy_execution_blockers(
                _policy(),
                train_type="sdxl-finetune",
                effective_config={"full_bf16": True, "mixed_precision": "bf16"},
            )
        self.assertEqual(len(blockers), 1)
        self.assertIn("no evidence_case_id", blockers[0])

    def test_unused_profile_does_not_participate_in_feature_gate(self):
        released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:backend",
        )
        adamw_released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:adamw",
        )
        policy = _policy(
            extra_profiles={
                "unused_lion": {"type": "Lion", "args": {}},
            }
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sdxl-finetune": released},
        ), patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": adamw_released},
        ):
            self.assertEqual(
                execution.parameter_policy_execution_blockers(
                    policy,
                    train_type="sdxl-finetune",
                    effective_config={"full_bf16": True, "mixed_precision": "bf16"},
                ),
                [],
            )

    def test_primary_and_fallback_profiles_both_participate(self):
        released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:backend",
        )
        muon_released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:muon",
        )
        policy = {
            "version": 1,
            "optimizer_profiles": {
                "muon": {"type": "Muon", "args": {}},
                "fallback": {"type": "AdamW", "args": {}},
            },
            "components": {
                "component": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 1e-4,
                    "fallback_optimizer_profile": "fallback",
                    "fallback_learning_rate": 1e-5,
                }
            },
        }
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"anima-finetune": released},
        ), patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"Muon": muon_released},
        ):
            blockers = execution.parameter_policy_execution_blockers(
                policy,
                train_type="anima-finetune",
                effective_config={"full_bf16": True, "mixed_precision": "bf16"},
            )
        self.assertEqual(len(blockers), 1)
        self.assertIn("AdamW", blockers[0])

    def test_baseline_restricted_optimizer_does_not_duplicate_feature_blocker(self):
        released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:backend",
        )
        policy = _policy(
            extra_profiles={
                "restricted": {"type": "AdaFactor", "args": {}},
            }
        )
        policy["components"]["component"]["optimizer_profile"] = "restricted"
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sdxl-finetune": released},
        ):
            self.assertEqual(
                execution.parameter_policy_execution_blockers(
                    policy,
                    train_type="sdxl-finetune",
                    effective_config={"full_bf16": True, "mixed_precision": "bf16"},
                ),
                [],
            )

    def test_adamw_first_release_variant_is_narrow(self):
        released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:backend",
        )
        adamw_released = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only release",
            "test:adamw",
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sdxl-finetune": released},
        ), patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": adamw_released},
        ):
            accepted = _policy()
            accepted["optimizer_profiles"]["main"]["args"] = {
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
            self.assertEqual(
                execution.parameter_policy_execution_blockers(
                    accepted,
                    train_type="sdxl-finetune",
                    effective_config={"full_bf16": True, "mixed_precision": "bf16"},
                ),
                [],
            )

            for key in (
                "fused",
                "foreach",
                "capturable",
                "differentiable",
                "amsgrad",
                "maximize",
            ):
                with self.subTest(key=key):
                    candidate = _policy()
                    candidate["optimizer_profiles"]["main"]["args"] = {key: True}
                    blockers = execution.parameter_policy_execution_blockers(
                        candidate,
                        train_type="sdxl-finetune",
                        effective_config={"full_bf16": True, "mixed_precision": "bf16"},
                    )
                    self.assertEqual(len(blockers), 1)
                    self.assertIn(key, blockers[0])


if __name__ == "__main__":
    unittest.main()
