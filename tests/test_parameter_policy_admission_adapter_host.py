"""QG-1B3-A independent Host adapter invariants, without runtime gate changes."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest import TestCase
from unittest.mock import patch

from mikazuki.parameter_policy_admission import (
    ComponentAdmission,
    assess_component_training_admission,
)
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
    FULL_BF16_OPTIMIZER_QUALIFICATIONS,
)
from tests.test_parameter_policy_admission_red import _config, _policy


def _admit(
    *, purpose="preview", config=None, ack=False, backend="sd-lora",
    policy=None,
):
    return assess_component_training_admission(
        _policy(backend) if policy is None else policy,
        train_type=backend,
        effective_config=_config() if config is None else config,
        purpose=purpose,
        experimental_ack=ack,
    )


def _pending():
    return ExecutionFeatureQualification(
        "pending", "test: evidence not released", None
    )


class QG1B3AHostAdapterIndependentTests(TestCase):
    def test_pending_preview_export_cannot_use_true_ack_as_permission(self):
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            for purpose in ("preview", "export"):
                with self.subTest(purpose=purpose):
                    report = _admit(purpose=purpose, ack=True)
                    self.assertEqual(report.status, "ack_required")
                    self.assertFalse(report.host_admissible)
                    self.assertTrue(report.requires_experimental_ack)

    def test_missing_backend_row_is_not_legacy_approved(self):
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": None}):
            report = _admit()
            self.assertEqual(report.status, "ack_required")
            self.assertEqual(report.readiness.evidence_tier, "unverified")
            self.assertIn(
                "qualification.backend.missing",
                {w.code for w in report.readiness.warnings},
            )
            self.assertEqual(
                _admit(purpose="start", ack=True).status,
                "experimental_host_candidate",
            )

    def test_pending_optimizer_does_not_inherit_legacy_exemption(self):
        with patch.dict(FULL_BF16_OPTIMIZER_QUALIFICATIONS, {"AdamW": _pending()}):
            report = _admit(purpose="start")
            self.assertEqual(report.status, "ack_required")
            self.assertIn(
                "qualification.optimizer.pending",
                {w.code for w in report.readiness.warnings},
            )
            with_ack = _admit(purpose="direct_trainer", ack=True)
            self.assertEqual(with_ack.status, "experimental_host_candidate")
            self.assertTrue(with_ack.host_admissible)

    def test_wrong_case_id_qualified_row_is_not_a_source_proof(self):
        fake = ExecutionFeatureQualification(
            "qualified", "test: wrongly claimed", "some:wrong-id"
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": fake}):
            report = _admit()
            self.assertEqual(report.status, "ack_required")
            self.assertIn(
                "qualification.backend.missing_evidence",
                {x.code for x in report.readiness.warnings},
            )
            candidate = _admit(purpose="start", ack=True)
            self.assertNotEqual(candidate.status, "legacy_approved")
            self.assertEqual(candidate.readiness.evidence_tier, "unverified")

    def test_explicit_two_processes_are_blocked_even_when_legacy_qualified(self):
        report = _admit(config=_config(num_processes=2), purpose="start")
        self.assertEqual(report.status, "blocked")
        self.assertFalse(report.host_admissible)
        self.assertIn("single-process", " ".join(report.runtime_blockers))

    def test_explicit_ddp_and_malformed_process_count_are_blocked(self):
        for field in (
            {"distributed_type": "MULTI_GPU"},
            {"num_processes": True},
            {"num_processes": "1"},
        ):
            with self.subTest(field=field):
                report = _admit(config=_config(**field), purpose="direct_trainer")
                self.assertEqual(report.status, "blocked")
                self.assertFalse(report.host_admissible)
                self.assertTrue(report.runtime_blockers)

    def test_source_semantic_hard_blockers_are_not_swallowed_by_ack(self):
        report = _admit(
            config=_config(deepspeed=True),
            purpose="start",
            ack=True,
        )
        self.assertEqual(report.status, "blocked")
        self.assertFalse(report.host_admissible)
        self.assertTrue(report.readiness.hard_errors)

    def test_worker_claimed_status_never_changes_pending_ack_requirement(self):
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            report = _admit(
                config=_config(
                    can_start=True,
                    capability="implemented",
                    decision="allow",
                    evidence_tier="formally_qualified",
                    launch_assets_checked=True,
                ),
                purpose="start",
            )
            self.assertEqual(report.status, "ack_required")
            self.assertFalse(report.host_admissible)

    def test_host_path_performs_no_file_open_or_signing(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
            report = _admit(purpose="start")
        self.assertEqual(report.status, "legacy_approved")
        self.assertEqual(report.readiness.evidence_tier, "unverified")

    def test_admission_result_is_frozen_with_typed_immutable_messages(self):
        report = _admit(purpose="preview")
        self.assertIsInstance(report, ComponentAdmission)
        self.assertIsInstance(report.runtime_blockers, tuple)
        self.assertIsInstance(report.warnings, tuple)
        with self.assertRaises(FrozenInstanceError):
            report.status = "blocked"
        with self.assertRaises(FrozenInstanceError):
            report.host_admissible = False

    def test_unknown_backend_stays_blocked_without_qualification(self):
        report = assess_component_training_admission(
            _policy(), train_type="not-a-backend",
            effective_config=_config(
                capability="implemented", decision="allow",
                can_start=True, evidence_tier="formally_qualified",
            ), purpose="start", experimental_ack=True,
        )
        self.assertEqual(report.status, "blocked")
        self.assertFalse(report.host_admissible)

    def test_inactive_full_bf16_is_strict_noop_for_policy_object(self):
        report = assess_component_training_admission(
            object(), train_type="sd-lora",
            effective_config={"full_bf16": False, "mixed_precision": "bf16"},
            purpose="preview",
        )
        self.assertEqual(report.status, "inactive")
        self.assertIsNone(report.readiness)
        self.assertTrue(report.host_admissible)
        self.assertFalse(report.runtime_blockers)


if __name__ == "__main__":
    import unittest
    unittest.main()
