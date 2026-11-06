"""QG-1B1 first-party severity/capability producer — tests-only RED.

16 exact host/CPU-free tests: B101-B112 missing first-party module, B901-B904
existing source baselines PASS. This suite does not enable Full-BF16 training,
change the legacy release matrix or exercise CUDA/real Checkpoint Load.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import inspect
import json
import unittest
from unittest.mock import patch

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.optimizer_profiles import (
    get_optimizer_capability,
    require_unrestricted_component_optimizer,
)
import mikazuki.parameter_policy_execution as execution
from mikazuki.parameter_policy_readiness import (
    ReadinessIssue,
    ExecutionReadiness,
    evaluate_execution_readiness,
)
from mikazuki.parameter_policy_runtime_bools import strict_component_runtime_bool


_MARKER = "QG-1B TRUSTED PRODUCER CONTRACT ABSENT"
_PRODUCER = "mikazuki.parameter_policy_readiness_producer"


def _policy(
    train_type: str = "sd-lora",
    *,
    optimizer: str = "AdamW",
    extra_profiles: dict | None = None,
    fallback: str | None = None,
) -> dict:
    model = get_model_component_profile(train_type)
    available = tuple(sorted(model.components))
    assert available, "test fixture backend has no Component definitions"
    selected = (
        "mmdit.attention.adapter"
        if train_type == "sd3-lora" else
        "unet.adapter"
        if "unet.adapter" in model.components else available[0]
    )
    profiles = {"main": {"type": optimizer, "args": {}}}
    if fallback is not None:
        profiles["fallback"] = {"type": fallback, "args": {}}
    if extra_profiles:
        profiles.update(extra_profiles)
    routes = {key: {"train": False} for key in available}
    routes[selected] = {
        "train": True,
        "optimizer_profile": "main",
        "learning_rate": 1e-4,
    }
    if fallback is not None:
        routes[selected]["fallback_optimizer_profile"] = "fallback"
        routes[selected]["fallback_learning_rate"] = 1e-5
    return {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": routes,
    }


def _config(**overrides) -> dict:
    values = {
        "full_bf16": True,
        "mixed_precision": "bf16",
    }
    values.update(overrides)
    return values


class QG1BExistingSourceBaseline(unittest.TestCase):
    def test_b901_prior_qg1a_pure_reducer_unchanged(self):
        reported = evaluate_execution_readiness(
            capability="implemented",
            warnings=(ReadinessIssue(
                code="qualification.backend.pending",
                message="No exact-head CUDA release evidence",
                source="E02",
            ),),
            evidence_tier="unverified",
        )
        self.assertIsInstance(reported, ExecutionReadiness)
        self.assertTrue(reported.can_start)
        self.assertEqual(reported.decision, "warn")
        self.assertEqual(reported.warnings[0].source, "E02")

    def test_b902_legacy_sd3_pending_gate_remains_unchanged_in_red(self):
        blockers = execution.parameter_policy_execution_blockers(
            _policy("sd3-lora"),
            train_type="sd3-lora",
            effective_config=_config(),
            require_assets=False,
        )
        self.assertEqual(len(blockers), 1, blockers)
        self.assertIn("pending", blockers[0])

    def test_b903_existing_strict_bool_and_bf16_precision_guards(self):
        with self.assertRaises(ValueError):
            strict_component_runtime_bool("false", field="full_bf16")
        with self.assertRaisesRegex(ValueError, "mixed_precision"):
            execution.build_parameter_policy_execution_contract(
                train_type="sd-lora",
                effective_config=_config(mixed_precision="fp16"),
            )
        self.assertIs(
            strict_component_runtime_bool(False, field="full_bf16"), False
        )

    def test_b904_planned_optimizer_came_is_still_forbidden(self):
        self.assertEqual(
            get_optimizer_capability("pytorch_optimizer.CAME").component_support,
            "planned",
        )
        with self.assertRaises(ValueError):
            require_unrestricted_component_optimizer("pytorch_optimizer.CAME")


class QG1BTrustedProducerContract(unittest.TestCase):
    def _function(self):
        if importlib.util.find_spec(_PRODUCER) is None:
            self.fail(
                f"{_MARKER}: first-party readiness producer module is missing"
            )
        mod = importlib.import_module(_PRODUCER)
        fn = getattr(mod, "classify_component_full_bf16_readiness", None)
        self.assertTrue(
            callable(fn), f"{_MARKER}: frozen classifier function is missing"
        )
        return fn

    def _classify(self, policy=None, *, train_type="sd-lora", config=None, **kwargs):
        fn = self._function()
        return fn(
            _policy(train_type) if policy is None else policy,
            train_type=train_type,
            effective_config=_config() if config is None else config,
            **kwargs,
        )

    def test_b101_frozen_first_party_function_has_no_caller_authority_args(self):
        fn = self._function()
        params = inspect.signature(fn).parameters
        self.assertEqual(
            list(params),
            ["canonical_policy", "train_type", "effective_config", "require_assets"],
        )
        for name in (
            "capability", "warnings", "hard_errors", "evidence_tier",
            "decision", "can_start", "trusted_runtime_issues",
        ):
            self.assertNotIn(name, params)
        self.assertNotIn("**", str(inspect.signature(fn)))
        self.assertEqual(params["require_assets"].default, False)

    def test_b102_pending_certificate_only_is_warning_for_audited_sd1(self):
        pending = execution.ExecutionFeatureQualification(
            "pending", "Test-only missing GPU release record", None
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sd-lora": pending},
        ):
            report = self._classify()
        self.assertIsInstance(report, ExecutionReadiness)
        self.assertEqual(report.capability, "implemented")
        self.assertEqual(report.decision, "warn")
        self.assertTrue(report.can_start)
        self.assertEqual(report.hard_errors, ())
        self.assertEqual(report.evidence_tier, "unverified")
        self.assertIn(
            "qualification.backend.pending",
            {x.code for x in report.warnings},
        )

    def test_b103_fake_qualified_without_evidence_cannot_promote_tier(self):
        row = execution.ExecutionFeatureQualification(
            "qualified", "test-only false release", None
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sd-lora": row},
        ):
            report = self._classify()
        self.assertEqual(report.capability, "implemented")
        self.assertEqual(report.decision, "warn")
        self.assertEqual(report.evidence_tier, "unverified")
        self.assertIn(
            "qualification.backend.missing_evidence",
            {x.code for x in report.warnings},
        )

    def test_b104_actual_precision_error_beats_forged_warning_and_can_start(self):
        res = self._classify(config=_config(
            mixed_precision="fp16",
            can_start=True,
            decision="allow",
            claimed_capability="implemented",
            qg_claim={
                "hard_errors": [],
                "warnings": [{"code": "runtime.dtype.mismatch", "source": "R03"}],
            },
        ))
        self.assertEqual(res.decision, "block")
        self.assertFalse(res.can_start)
        self.assertIn(
            "runtime.bf16.precision_mismatch",
            {x.code for x in res.hard_errors},
        )

    def test_b105_unknown_backend_stays_blocked_despite_fake_release_row(self):
        forged = execution.ExecutionFeatureQualification(
            "qualified", "Worker wrote this row", "fake:evidence-id"
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"unknown-future-backend": forged},
        ):
            res = self._classify(
                policy=_policy(),
                train_type="unknown-future-backend",
                config=_config(capability="implemented", can_start=True),
            )
        self.assertEqual(res.decision, "block")
        self.assertFalse(res.can_start)
        self.assertIn(
            "capability.backend.unknown", {x.code for x in res.hard_errors}
        )
        self.assertNotEqual(res.evidence_tier, "formally_qualified")

    def test_b106_planned_optimizer_not_granted_by_fake_qualification(self):
        forged = execution.ExecutionFeatureQualification(
            "qualified", "fake", "fake:came-evidence"
        )
        with patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"pytorch_optimizer.CAME": forged},
        ):
            res = self._classify(
                policy=_policy(optimizer="pytorch_optimizer.CAME"),
                config=_config(can_start=True),
            )
        self.assertEqual(res.decision, "block")
        self.assertFalse(res.can_start)
        self.assertIn(
            "capability.optimizer.not_implemented",
            {x.code for x in res.hard_errors},
        )

    def test_b107_quoted_false_must_never_be_treated_as_inactive_standard(self):
        res = self._classify(config=_config(full_bf16="false"))
        self.assertIsNotNone(res)
        self.assertEqual(res.decision, "block")
        self.assertFalse(res.can_start)
        self.assertIn("runtime.bool.invalid", {x.code for x in res.hard_errors})

    def test_b108_real_sdxl_gradient_checkpointing_risk_stays_hard(self):
        res = self._classify(
            policy=_policy("sdxl-lora"),
            train_type="sdxl-lora",
            config=_config(
                gradient_checkpointing=True,
                qg_claim={"status": "formally_qualified", "decision": "allow"},
            ),
        )
        self.assertEqual(res.decision, "block")
        self.assertFalse(res.can_start)
        self.assertIn(
            "runtime.sdxl.gradient_checkpointing_unreviewed",
            {x.code for x in res.hard_errors},
        )

    def test_b109_sd3_full_bf16_unreviewed_capability_not_promoted(self):
        fake = execution.ExecutionFeatureQualification(
            "qualified", "fake outside actual audit", "fake:sd3-physical"
        )
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sd3-lora": fake},
        ):
            res = self._classify(
                policy=_policy("sd3-lora"),
                train_type="sd3-lora",
                config=_config(
                    can_start=True,
                    claimed_capability="implemented",
                    qg_claim={"runtime.dtype.mismatch": "warning"},
                ),
            )
        self.assertEqual(res.decision, "block")
        self.assertFalse(res.can_start)
        self.assertIn(
            "capability.backend.audit_required",
            {x.code for x in res.hard_errors},
        )
        self.assertNotEqual(res.evidence_tier, "formally_qualified")

    def test_b110_unreferenced_optimizers_do_not_change_tier_or_permissions(self):
        base = self._classify()
        extras = self._classify(policy=_policy(
            extra_profiles={
                "unused_pending_lion": {"type": "Lion", "args": {}},
            },
        ))
        self.assertEqual(extras.decision, base.decision)
        self.assertEqual(extras.evidence_tier, base.evidence_tier)
        self.assertEqual(extras.capability, base.capability)
        with self.subTest("referenced pending unsupported audited optimizer"):
            changed = self._classify(policy=_policy(fallback="AdamW8bit"))
            self.assertEqual(changed.decision, "block")
            self.assertFalse(changed.can_start)
            self.assertIn(
                "capability.optimizer.audit_required",
                {x.code for x in changed.hard_errors},
            )

    def test_b111_source_release_rows_and_identity_never_mutated(self):
        policy = _policy()
        cfg = _config()
        before_policy, before_config = deepcopy(policy), deepcopy(cfg)
        backend_rows = deepcopy(execution.FULL_BF16_BACKEND_QUALIFICATIONS)
        optimizer_rows = deepcopy(execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS)
        contract = execution.build_parameter_policy_execution_contract(
            train_type="sd-lora", effective_config=cfg
        )
        identity, signature = contract.execution_identity(), contract.execution_signature()
        with patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"sd-lora": execution.ExecutionFeatureQualification(
                "pending", "test changed coverage", None
            )},
        ):
            result = self._classify(policy=policy, config=cfg)
            self.assertEqual(result.evidence_tier, "unverified")
            self.assertEqual(contract.execution_signature(), signature)
            self.assertEqual(contract.execution_identity(), identity)
        self.assertEqual(policy, before_policy)
        self.assertEqual(cfg, before_config)
        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS, backend_rows
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS, optimizer_rows
        )

    def test_b112_inactive_strict_noop_and_refuses_untrusted_keywords(self):
        fn = self._function()
        self.assertIsNone(self._classify(config=_config(full_bf16=False)))
        with self.assertRaises(TypeError):
            fn(
                _policy(), train_type="sd-lora", effective_config=_config(),
                hard_errors=[], warnings=[], capability="implemented",
                evidence_tier="formally_qualified", can_start=True,
            )
        report = self._classify()
        public = report.to_diagnostics()
        self.assertEqual(
            set(public),
            {"decision", "capability", "hard_errors", "warnings", "evidence_tier"},
        )
        for secret in ("physical_cuda_qualified", "worker_complete", "may_issue"):
            self.assertNotIn(secret, json.dumps(public))


if __name__ == "__main__":
    unittest.main()
