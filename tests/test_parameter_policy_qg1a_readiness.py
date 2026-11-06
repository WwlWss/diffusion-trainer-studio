"""QG-1A tests-only RED for a pure, proof-independent readiness evaluator.

QG101-QG112 must be the ONLY failures at frozen main:
12 tagged FAIL + four old stable baselines PASS; never expectedFailure/skip.
The module under test is intentionally absent until QG-1A minimum GREEN.
No training request or CUDA/checkpoint behavior is modified by this suite.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, is_dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import unittest

from mikazuki.optimizer_profiles import (
    get_optimizer_capability,
    require_unrestricted_component_optimizer,
)
from mikazuki.parameter_policy_execution import (
    build_parameter_policy_execution_contract,
    PARAMETER_POLICY_EXECUTION_IDENTITY_SCHEMA,
)
from mikazuki.parameter_policy_runtime_bools import (
    strict_component_runtime_bool,
)


MISSING_MARKER = "QG-1A PURE READINESS CONTRACT ABSENT"
NEW_MODULE = "mikazuki.parameter_policy_readiness"
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {
    "decision", "capability", "hard_errors", "warnings", "evidence_tier"
}
FORBIDDEN_PROOF_KEYS = {
    "physical_cuda_qualified", "may_issue", "worker_complete",
    "evidence_case_id", "certificate", "execution_identity",
    "execution_signature", "checkpoint_id", "private_key",
}


class QG1AExistingBaseline(unittest.TestCase):
    def test_b901_full_bf16_identity_remains_stable_without_release_gate(self):
        contract = build_parameter_policy_execution_contract(
            train_type="sd3-lora",
            effective_config={"full_bf16": True, "mixed_precision": "bf16"},
        )
        self.assertEqual(
            contract.execution_identity()["schema"],
            PARAMETER_POLICY_EXECUTION_IDENTITY_SCHEMA,
        )
        self.assertEqual(
            contract.execution_identity()["features"],
            {"full_bf16": {
                "mixed_precision": "bf16",
                "trainable_parameter_dtype": "bfloat16",
            }},
        )
        self.assertEqual(len(contract.execution_signature()), 64)

    def test_b902_invalid_full_bf16_precision_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixed_precision"):
            build_parameter_policy_execution_contract(
                train_type="sd3-lora",
                effective_config={"full_bf16": True, "mixed_precision": "fp16"},
            )

    def test_b903_quoted_false_remains_invalid_real_runtime_bool(self):
        for value in ("false", "0", "true", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    strict_component_runtime_bool(value, field="full_bf16")
        self.assertIs(
            strict_component_runtime_bool(False, field="full_bf16"), False
        )

    def test_b904_planned_optimizer_remains_unimplemented(self):
        self.assertEqual(
            get_optimizer_capability("pytorch_optimizer.CAME").component_support, "planned"
        )
        with self.assertRaises(ValueError):
            require_unrestricted_component_optimizer("pytorch_optimizer.CAME")


class QG1APureReadinessContract(unittest.TestCase):
    def _module(self):
        spec = importlib.util.find_spec(NEW_MODULE)
        if spec is None:
            self.fail(
                f"{MISSING_MARKER}: expected independent pure module {NEW_MODULE}"
            )
        return importlib.import_module(NEW_MODULE)

    def _eval(self, **kwargs):
        fn = getattr(self._module(), "evaluate_execution_readiness", None)
        self.assertTrue(callable(fn), f"{MISSING_MARKER}: missing evaluator")
        return fn(**kwargs)

    def test_qg101_frozen_public_types_and_pure_evaluator_are_present(self):
        module = self._module()
        for name in (
            "ReadinessIssue", "ExecutionReadiness",
            "evaluate_execution_readiness",
        ):
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(module, name), f"{MISSING_MARKER}: missing {name}"
                )
        for name in ("ReadinessIssue", "ExecutionReadiness"):
            item = getattr(module, name)
            self.assertTrue(is_dataclass(item))
            self.assertTrue(item.__dataclass_params__.frozen)
            self.assertTrue(hasattr(item, "__slots__"))
        self.assertTrue(callable(module.evaluate_execution_readiness))

    def test_qg102_issue_has_stable_code_source_and_immutable_fields(self):
        Issue = self._module().ReadinessIssue
        issue = Issue(
            code="runtime.bool.invalid",
            message="Quoted false would be True in the trainer.",
            source="C01",
        )
        self.assertEqual(
            (issue.code, issue.message, issue.source),
            (
                "runtime.bool.invalid",
                "Quoted false would be True in the trainer.",
                "C01",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            issue.code = "qualification.backend.pending"
        for kwargs in (
            {"code": "", "message": "x", "source": "C01"},
            {"code": "1 wrong", "message": "x", "source": "C01"},
            {"code": "runtime.test", "message": "", "source": "C01"},
            {"code": "runtime.test", "message": "x", "source": ""},
            {"code": 123, "message": "x", "source": "C01"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Issue(**kwargs)

    def test_qg103_implemented_and_verified_no_errors_is_allow(self):
        report = self._eval(
            capability="implemented", evidence_tier="formally_qualified"
        )
        self.assertEqual(report.decision, "allow")
        self.assertTrue(report.can_start)
        self.assertEqual(report.capability, "implemented")
        self.assertEqual(report.hard_errors, ())
        self.assertEqual(report.warnings, ())
        self.assertEqual(report.evidence_tier, "formally_qualified")
        self.assertEqual(set(report.to_diagnostics()), REQUIRED_KEYS)

    def test_qg104_evidence_status_alone_never_changes_execution_or_permission(self):
        contract = build_parameter_policy_execution_contract(
            train_type="sd3-lora",
            effective_config={"full_bf16": True, "mixed_precision": "bf16"},
        )
        original_identity = contract.execution_identity()
        original_signature = contract.execution_signature()
        pending = self._eval(
            capability="implemented", evidence_tier="unverified"
        )
        qualified = self._eval(
            capability="implemented", evidence_tier="formally_qualified"
        )
        host = self._eval(
            capability="implemented", evidence_tier="host_cpu_checked"
        )
        self.assertEqual(
            (pending.decision, qualified.decision, host.decision),
            ("allow", "allow", "allow"),
        )
        self.assertTrue(all(x.can_start for x in (pending, qualified, host)))
        self.assertEqual(contract.execution_identity(), original_identity)
        self.assertEqual(contract.execution_signature(), original_signature)
        self.assertNotIn("evidence_tier", original_identity)
        self.assertNotIn("qualification", json.dumps(original_identity))

    def test_qg105_experimental_candidate_is_warn_and_can_start(self):
        report = self._eval(
            capability="experimental_candidate", evidence_tier="unverified"
        )
        self.assertEqual(report.decision, "warn")
        self.assertTrue(report.can_start)
        self.assertFalse(report.hard_errors)
        self.assertTrue(report.warnings)
        self.assertEqual(report.warnings[0].code, "capability.experimental")
        self.assertEqual(report.warnings[0].source, "E02")

    def test_qg106_true_runtime_error_overrides_formal_certificate(self):
        Issue = self._module().ReadinessIssue
        hard = Issue(
            code="runtime.dtype.mismatch",
            message="Expected trainable BF16, got FP32.",
            source="R03",
        )
        result = self._eval(
            capability="implemented",
            hard_errors=[hard],
            evidence_tier="formally_qualified",
        )
        self.assertEqual(result.decision, "block")
        self.assertFalse(result.can_start)
        self.assertEqual(result.hard_errors, (hard,))

    def test_qg107_unimplemented_and_known_incompatible_never_upgrade(self):
        # Resolve the new module once outside subTests: when it is absent,
        # this entire independent contract must be ONE targeted RED case.
        evaluator = self._module().evaluate_execution_readiness
        for cap, expected in (
            ("not_implemented", "capability.not_implemented"),
            ("known_incompatible", "capability.known_incompatible"),
        ):
            with self.subTest(capability=cap):
                result = evaluator(
                    capability=cap,
                    evidence_tier="formally_qualified",
                )
                self.assertEqual(result.decision, "block")
                self.assertFalse(result.can_start)
                self.assertEqual(result.hard_errors[0].code, expected)
                self.assertEqual(result.hard_errors[0].source, "G01")

    def test_qg108_missing_certificate_is_a_warning_not_a_blocker(self):
        Issue = self._module().ReadinessIssue
        warning = Issue(
            code="qualification.backend.pending",
            message="SD3 Full-BF16 R1/R2 physical CUDA evidence pending.",
            source="E02",
        )
        report = self._eval(
            capability="implemented",
            warnings=[warning],
            evidence_tier="unverified",
        )
        self.assertEqual(report.decision, "warn")
        self.assertTrue(report.can_start)
        self.assertEqual(report.warnings, (warning,))
        self.assertEqual(report.hard_errors, ())
        self.assertEqual(report.evidence_tier, "unverified")

    def test_qg109_failed_evidence_alone_does_not_prove_broken_implementation(self):
        Issue = self._module().ReadinessIssue
        report = self._eval(
            capability="implemented", evidence_tier="failed"
        )
        self.assertEqual(report.decision, "allow")
        self.assertTrue(report.can_start)
        self.assertEqual(report.evidence_tier, "failed")
        incompatible = self._eval(
            capability="known_incompatible", evidence_tier="failed"
        )
        self.assertEqual(incompatible.decision, "block")
        blocked = self._eval(
            capability="implemented",
            hard_errors=[Issue(
                code="runtime.checkpoint.invalid",
                message="Checkpoint state mismatch",
                source="R08",
            )],
            evidence_tier="failed",
        )
        self.assertEqual(blocked.decision, "block")

    def test_qg110_malformed_enums_issues_and_duplicate_codes_fail(self):
        Issue = self._module().ReadinessIssue
        wrong = Issue(code="runtime.test", message="bad", source="R03")
        for kwargs in (
            {"capability": ""},
            {"capability": "qualified"},
            {"capability": "unsupported"},
            {"capability": "implemented", "evidence_tier": "pending"},
            {"capability": "implemented", "evidence_tier": None},
            {"capability": "implemented", "hard_errors": "runtime.test"},
            {"capability": "implemented", "hard_errors": [object()]},
            {"capability": "implemented", "hard_errors": [wrong, wrong]},
            {
                "capability": "implemented",
                "hard_errors": [wrong],
                "warnings": [wrong],
            },
        ):
            with self.subTest(kwargs=repr(kwargs)):
                with self.assertRaises(ValueError):
                    self._eval(**kwargs)
        Direct = self._module().ExecutionReadiness
        with self.assertRaises(ValueError):
            Direct(
                decision="allow",
                capability="not_implemented",
                hard_errors=(),
                warnings=(),
                evidence_tier="formally_qualified",
            )

    def test_qg111_results_are_frozen_and_snapshots_do_not_alias_inputs(self):
        Issue = self._module().ReadinessIssue
        warning = Issue(code="qualification.backend.pending",
                        message="pending", source="E02")
        original = [warning]
        report = self._eval(
            capability="implemented",
            warnings=original,
            evidence_tier="host_cpu_checked",
        )
        original.clear()
        self.assertEqual(report.warnings, (warning,))
        self.assertEqual(report.decision, "warn")
        with self.assertRaises(FrozenInstanceError):
            report.evidence_tier = "formally_qualified"
        first = report.to_diagnostics()
        first["warnings"][0]["message"] = "tampered"
        first["warnings"].append({"code": "forged"})
        second = report.to_diagnostics()
        self.assertEqual(second["warnings"][0]["message"], "pending")
        self.assertEqual(len(second["warnings"]), 1)

    def test_qg112_pure_host_module_never_exposes_signing_or_cuda_authority(self):
        module = self._module()
        path = ROOT / "mikazuki" / "parameter_policy_readiness.py"
        text = path.read_text(encoding="utf-8")
        source_ast = ast.parse(text, filename=str(path))
        forbidden_imports = {
            "torch", "accelerate", "safetensors", "subprocess", "socket",
            "ctypes", "os", "pathlib",
        }
        for node in ast.walk(source_ast):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                self.assertFalse(roots & forbidden_imports, roots)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                self.assertNotIn(root, forbidden_imports)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(
                        node.func.id,
                        {"open", "eval", "exec", "__import__"},
                    )
        result = module.evaluate_execution_readiness(
            capability="experimental_candidate",
            evidence_tier="unverified",
        )
        public = result.to_diagnostics()
        self.assertEqual(set(public), REQUIRED_KEYS)
        self.assertFalse(set(public) & FORBIDDEN_PROOF_KEYS)
        self.assertFalse(
            set(vars(module)) & {
                "issue_provisional_fresh_step1_attestation",
                "open_protected_runner_signer",
                "verify_original_provisional_step1_source",
            }
        )


if __name__ == "__main__":
    unittest.main()
