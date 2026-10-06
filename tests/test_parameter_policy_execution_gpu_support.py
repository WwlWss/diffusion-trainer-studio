from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from mikazuki import parameter_policy_execution as execution
from mikazuki.optimizer_profiles import MUON_ARGUMENTS
from tools.parameter_policy_execution_gpu_support import (
    ADAMW_FULL_BF16_CASE_IDS,
    ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
    EXECUTION_GPU_CASE_PHASES,
    ExecutionGpuMatrixError,
    SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID,
    MUON_FULL_BF16_ARGUMENT_FAMILY,
    MUON_FULL_BF16_CASE_IDS,
    MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
    MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS,
    MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID,
    summarize_adamw_full_bf16_bundle,
    summarize_muon_full_bf16_bundle,
    summarize_muon_adamw_fallback_full_bf16_bundle,
    summarize_shared_full_bf16_promotion,
    summarize_shared_full_bf16_regression,
    temporary_execution_qualification,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_GPU_RUNNER = ROOT / "tools" / "run_parameter_policy_execution_gpu_matrix.py"


class ExecutionGpuCaseProtocolTests(unittest.TestCase):
    def test_adamw_evidence_bundle_and_phases_are_stable(self):
        self.assertEqual(
            ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
            "phase-c:adamw-full-bf16:v1",
        )
        self.assertEqual(
            EXECUTION_GPU_CASE_PHASES["optimizer:adamw:full-bf16:accum1:v1"],
            ("train_save", "resume_second_step"),
        )
        self.assertEqual(
            EXECUTION_GPU_CASE_PHASES["optimizer:adamw:full-bf16:accum2:v1"],
            ("train_save", "resume_second_step"),
        )
        self.assertEqual(
            EXECUTION_GPU_CASE_PHASES["infra:cuda-bf16-capability:v1"],
            ("probe",),
        )
        self.assertEqual(
            EXECUTION_GPU_CASE_PHASES["infra:full-bf16-session:v1"],
            ("probe",),
        )

    def test_muon_evidence_bundle_and_phases_are_stable(self):
        self.assertEqual(
            MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
            "phase-c:muon-full-bf16:v1",
        )
        for case_id in MUON_FULL_BF16_CASE_IDS:
            self.assertEqual(
                EXECUTION_GPU_CASE_PHASES[case_id],
                ("train_save", "resume_second_step"),
            )

    def test_c4_explicit_fallback_bundle_and_phases_are_stable(self):
        self.assertEqual(
            MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID,
            "phase-c:muon-adamw-explicit-fallback-full-bf16:v1",
        )
        for case_id in MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS:
            self.assertEqual(
                EXECUTION_GPU_CASE_PHASES[case_id],
                ("train_save", "resume_second_step"),
            )

    def test_muon_qualification_argument_family_tracks_production_surface(self):
        self.assertEqual(
            frozenset(MUON_FULL_BF16_ARGUMENT_FAMILY),
            MUON_ARGUMENTS,
        )

    def test_resumed_muon_accumulation_preserves_existing_intermediate_state(self):
        source = EXECUTION_GPU_RUNNER.read_text(encoding="utf-8")
        self.assertIn("path_before = _muon_path_evidence(session)", source)
        self.assertIn("if path_after != path_before:", source)
        self.assertIn(
            "Intermediate Muon accumulation microstep changed optimizer ",
            source,
        )
        self.assertNotIn(
            "Intermediate Muon accumulation microstep created optimizer state.",
            source,
        )


class AdamWFullBf16BundleTests(unittest.TestCase):
    def test_bundle_pass_requires_both_cases_to_pass(self):
        rows = [
            {"case_id": case_id, "status": "pass"}
            for case_id in ADAMW_FULL_BF16_CASE_IDS
        ]
        bundle = summarize_adamw_full_bf16_bundle(rows)
        self.assertEqual(bundle["status"], "pass")
        self.assertEqual(bundle["missing_cases"], [])
        self.assertTrue(bundle["optimizer_evidence_complete"])
        self.assertFalse(bundle["promotion_eligible"])
        self.assertFalse(bundle["optimizer_qualification_eligible"])

    def test_bundle_is_incomplete_when_only_one_case_is_present(self):
        bundle = summarize_adamw_full_bf16_bundle(
            [{"case_id": ADAMW_FULL_BF16_CASE_IDS[0], "status": "pass"}]
        )
        self.assertEqual(bundle["status"], "incomplete")
        self.assertEqual(
            bundle["missing_cases"],
            [ADAMW_FULL_BF16_CASE_IDS[1]],
        )
        self.assertFalse(bundle["optimizer_evidence_complete"])
        self.assertFalse(bundle["promotion_eligible"])
        self.assertFalse(bundle["optimizer_qualification_eligible"])

    def test_bundle_fails_when_required_case_fails(self):
        bundle = summarize_adamw_full_bf16_bundle(
            [
                {"case_id": ADAMW_FULL_BF16_CASE_IDS[0], "status": "pass"},
                {"case_id": ADAMW_FULL_BF16_CASE_IDS[1], "status": "fail"},
            ]
        )
        self.assertEqual(bundle["status"], "fail")
        self.assertFalse(bundle["optimizer_evidence_complete"])
        self.assertFalse(bundle["optimizer_qualification_eligible"])

    def test_bundle_is_absent_without_adamw_cases(self):
        self.assertIsNone(
            summarize_adamw_full_bf16_bundle(
                [{"case_id": "infra:cuda-bf16-capability:v1", "status": "pass"}]
            )
        )


class MuonFullBf16BundleTests(unittest.TestCase):
    def test_bundle_pass_requires_both_cases_to_pass(self):
        rows = [
            {"case_id": case_id, "status": "pass"}
            for case_id in MUON_FULL_BF16_CASE_IDS
        ]
        bundle = summarize_muon_full_bf16_bundle(rows)
        self.assertEqual(bundle["status"], "pass")
        self.assertEqual(bundle["missing_cases"], [])
        self.assertTrue(bundle["optimizer_evidence_complete"])
        self.assertFalse(bundle["promotion_eligible"])
        self.assertFalse(bundle["optimizer_qualification_eligible"])

    def test_bundle_is_incomplete_when_only_one_case_is_present(self):
        bundle = summarize_muon_full_bf16_bundle(
            [{"case_id": MUON_FULL_BF16_CASE_IDS[0], "status": "pass"}]
        )
        self.assertEqual(bundle["status"], "incomplete")
        self.assertEqual(
            bundle["missing_cases"],
            [MUON_FULL_BF16_CASE_IDS[1]],
        )
        self.assertFalse(bundle["optimizer_evidence_complete"])

    def test_bundle_fails_when_required_case_fails(self):
        bundle = summarize_muon_full_bf16_bundle(
            [
                {"case_id": MUON_FULL_BF16_CASE_IDS[0], "status": "pass"},
                {"case_id": MUON_FULL_BF16_CASE_IDS[1], "status": "fail"},
            ]
        )
        self.assertEqual(bundle["status"], "fail")
        self.assertFalse(bundle["optimizer_evidence_complete"])
        self.assertFalse(bundle["optimizer_qualification_eligible"])

    def test_bundle_is_absent_without_muon_cases(self):
        self.assertIsNone(
            summarize_muon_full_bf16_bundle(
                [{"case_id": "infra:cuda-bf16-capability:v1", "status": "pass"}]
            )
        )


class MuonAdamWExplicitFallbackFullBf16BundleTests(unittest.TestCase):
    def test_bundle_pass_requires_both_cases(self):
        rows = [
            {"case_id": case_id, "status": "pass"}
            for case_id in MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS
        ]
        bundle = summarize_muon_adamw_fallback_full_bf16_bundle(rows)
        self.assertEqual(bundle["status"], "pass")
        self.assertEqual(bundle["scope"], "optimizer_topology")
        self.assertEqual(
            bundle["topology"],
            "muon_primary_adamw_explicit_fallback",
        )
        self.assertEqual(bundle["optimizers"], ["Muon", "AdamW"])
        self.assertEqual(
            bundle["prerequisite_bundles"],
            [
                ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
                MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
            ],
        )
        self.assertTrue(bundle["topology_evidence_complete"])
        self.assertFalse(bundle["promotion_eligible"])
        self.assertFalse(bundle["optimizer_qualification_eligible"])

    def test_bundle_is_incomplete_when_only_one_case_is_present(self):
        bundle = summarize_muon_adamw_fallback_full_bf16_bundle(
            [
                {
                    "case_id": MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS[0],
                    "status": "pass",
                }
            ]
        )
        self.assertEqual(bundle["status"], "incomplete")
        self.assertEqual(
            bundle["missing_cases"],
            [MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS[1]],
        )
        self.assertFalse(bundle["topology_evidence_complete"])

    def test_bundle_fails_when_required_case_fails(self):
        bundle = summarize_muon_adamw_fallback_full_bf16_bundle(
            [
                {
                    "case_id": MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS[0],
                    "status": "pass",
                },
                {
                    "case_id": MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS[1],
                    "status": "fail",
                },
            ]
        )
        self.assertEqual(bundle["status"], "fail")
        self.assertFalse(bundle["topology_evidence_complete"])

    def test_bundle_is_absent_without_c4_cases(self):
        self.assertIsNone(
            summarize_muon_adamw_fallback_full_bf16_bundle(
                [{"case_id": MUON_FULL_BF16_CASE_IDS[0], "status": "pass"}]
            )
        )


class SharedFullBf16PromotionTests(unittest.TestCase):
    def _snapshot(self):
        return {
            "backends": {
                name: {
                    "status": row.status,
                    "reason": row.reason,
                    "evidence_case_id": row.evidence_case_id,
                }
                for name, row in execution.FULL_BF16_BACKEND_QUALIFICATIONS.items()
            },
            "optimizers": {
                name: {
                    "status": row.status,
                    "reason": row.reason,
                    "evidence_case_id": row.evidence_case_id,
                }
                for name, row in execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS.items()
            },
        }

    def _passing_rows(self):
        return [
            {"case_id": case_id, "status": "pass"}
            for case_id in EXECUTION_GPU_CASE_PHASES
        ]

    def test_complete_c1_c4_matrix_promotes_candidate_rows(self):
        summary = summarize_shared_full_bf16_promotion(
            self._passing_rows(),
            self._snapshot(),
        )
        self.assertEqual(
            summary["id"],
            SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["promotion_eligible"])
        self.assertTrue(summary["optimizer_qualification_eligible"])
        self.assertTrue(summary["infra_complete"])
        self.assertTrue(summary["target_rows_match"])
        self.assertEqual(summary["unexpected_backend_promotions"], [])
        self.assertEqual(summary["unexpected_optimizer_promotions"], [])

    def test_missing_or_failed_case_blocks_promotion(self):
        rows = self._passing_rows()
        missing = rows[:-1]
        summary = summarize_shared_full_bf16_promotion(
            missing,
            self._snapshot(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["promotion_eligible"])
        self.assertTrue(summary["missing_cases"])

        failed = self._passing_rows()
        failed[0] = dict(failed[0], status="fail")
        summary = summarize_shared_full_bf16_promotion(
            failed,
            self._snapshot(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["failed_cases"])

    def test_wrong_candidate_metadata_blocks_promotion(self):
        snapshot = self._snapshot()
        snapshot["optimizers"]["AdamW"]["evidence_case_id"] = "wrong:evidence"
        summary = summarize_shared_full_bf16_promotion(
            self._passing_rows(),
            snapshot,
        )
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["target_rows_match"])

    def test_unexpected_backend_or_optimizer_promotion_blocks_d0(self):
        snapshot = self._snapshot()
        snapshot["backends"]["flux-finetune"] = {
            "status": "qualified",
            "reason": "unexpected",
            "evidence_case_id": "unexpected:backend",
        }
        snapshot["optimizers"]["Lion"] = {
            "status": "qualified",
            "reason": "unexpected",
            "evidence_case_id": "unexpected:optimizer",
        }
        summary = summarize_shared_full_bf16_promotion(
            self._passing_rows(),
            snapshot,
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["unexpected_backend_promotions"], ["flux-finetune"])
        self.assertEqual(summary["unexpected_optimizer_promotions"], ["Lion"])


class SharedFullBf16RegressionTests(SharedFullBf16PromotionTests):
    def test_regression_allows_candidate_backend_qualification(self):
        snapshot = self._snapshot()
        snapshot["backends"]["flux-finetune"] = {
            "status": "qualified",
            "reason": "D1 candidate",
            "evidence_case_id": "phase-d:backend:flux-finetune:full-bf16:v1",
        }
        summary = summarize_shared_full_bf16_regression(
            self._passing_rows(),
            snapshot,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["scope"], "shared_optimizer_regression")
        self.assertFalse(summary["promotion_eligible"])
        self.assertFalse(summary["optimizer_qualification_eligible"])

    def test_regression_still_requires_every_c1_c4_case(self):
        rows = self._passing_rows()
        rows[-1] = dict(rows[-1], status="fail")
        summary = summarize_shared_full_bf16_regression(
            rows,
            self._snapshot(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["failed_cases"])

    def test_regression_requires_d0_optimizer_authority(self):
        for optimizer_name in ("AdamW", "Muon"):
            with self.subTest(optimizer=optimizer_name):
                snapshot = self._snapshot()
                snapshot["optimizers"][optimizer_name]["evidence_case_id"] = "wrong"
                summary = summarize_shared_full_bf16_regression(
                    self._passing_rows(),
                    snapshot,
                )
                self.assertEqual(summary["status"], "fail")
                self.assertFalse(summary["target_rows_match"])


class TemporaryExecutionQualificationTests(unittest.TestCase):
    def test_candidate_adamw_is_not_leased_while_pending_backend_is(self):
        backend_before = execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"]
        optimizer_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"]
        unrelated_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"]

        with temporary_execution_qualification(
            backend="flux-finetune",
            optimizers=("AdamW",),
            evidence_case_id="test:lease",
        ) as lease:
            self.assertEqual(
                execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"].status,
                "qualified",
            )
            self.assertEqual(
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"].status,
                "qualified",
            )
            self.assertEqual(
                [row["lease_applied"] for row in lease["records"]],
                [True, False],
            )
            self.assertEqual(
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"],
                unrelated_before,
            )

        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
            backend_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"],
            optimizer_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"],
            unrelated_before,
        )

    def test_candidate_muon_is_not_leased_while_pending_backend_is(self):
        backend_before = execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"]
        muon_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"]
        adamw_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"]

        with temporary_execution_qualification(
            backend="flux-finetune",
            optimizers=("Muon",),
            evidence_case_id="test:muon-lease",
        ) as lease:
            self.assertEqual(
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"].status,
                "qualified",
            )
            self.assertEqual(
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"],
                adamw_before,
            )
            self.assertEqual(
                [row["lease_applied"] for row in lease["records"]],
                [True, False],
            )

        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
            backend_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"],
            muon_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"],
            adamw_before,
        )

    def test_candidate_muon_adamw_are_not_leased_with_pending_backend(self):
        backend_before = execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"]
        muon_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"]
        adamw_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"]

        with temporary_execution_qualification(
            backend="flux-finetune",
            optimizers=("Muon", "AdamW"),
            evidence_case_id="test:c4-lease",
        ) as lease:
            self.assertEqual(
                execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"].status,
                "qualified",
            )
            self.assertEqual(
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"].status,
                "qualified",
            )
            self.assertEqual(
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"].status,
                "qualified",
            )
            self.assertEqual(
                [row["name"] for row in lease["records"]],
                ["flux-finetune", "Muon", "AdamW"],
            )
            self.assertEqual(
                [row["lease_applied"] for row in lease["records"]],
                [True, False, False],
            )

        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
            backend_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"],
            muon_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"],
            adamw_before,
        )

    def test_exception_inside_lease_restores_rows(self):
        backend_before = execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"]
        optimizer_before = execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"]

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with temporary_execution_qualification(
                backend="flux-finetune",
                optimizers=("AdamW",),
                evidence_case_id="test:lease",
            ):
                raise RuntimeError("boom")

        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
            backend_before,
        )
        self.assertEqual(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"],
            optimizer_before,
        )

    def test_released_rows_with_evidence_are_not_overwritten(self):
        released_backend = execution.ExecutionFeatureQualification(
            "qualified",
            "released",
            "release:backend",
        )
        released_optimizer = execution.ExecutionFeatureQualification(
            "qualified",
            "released",
            "release:optimizer",
        )
        with mock.patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"flux-finetune": released_backend},
        ), mock.patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": released_optimizer},
        ):
            with temporary_execution_qualification(
                backend="flux-finetune",
                optimizers=("AdamW",),
                evidence_case_id="test:lease",
            ) as lease:
                self.assertEqual(
                    execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
                    released_backend,
                )
                self.assertEqual(
                    execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"],
                    released_optimizer,
                )
                self.assertTrue(
                    all(not row["lease_applied"] for row in lease["records"])
                )

    def test_malformed_qualified_row_fails_closed_and_restores_prior_rows(self):
        malformed = execution.ExecutionFeatureQualification(
            "qualified",
            "missing evidence",
            None,
        )
        backend_before = execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"]
        with mock.patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": malformed},
        ):
            with self.assertRaisesRegex(
                ExecutionGpuMatrixError,
                "qualified without an evidence_case_id",
            ):
                with temporary_execution_qualification(
                    backend="flux-finetune",
                    optimizers=("AdamW",),
                    evidence_case_id="test:lease",
                ):
                    pass
            self.assertEqual(
                execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
                backend_before,
            )

    def test_unsupported_and_invalid_rows_fail_closed(self):
        with self.assertRaisesRegex(ExecutionGpuMatrixError, "explicitly unsupported"):
            with temporary_execution_qualification(
                backend="sd-dreambooth",
                optimizers=("AdamW",),
                evidence_case_id="test:lease",
            ):
                pass

        invalid = execution.ExecutionFeatureQualification(
            "future-status",
            "invalid",
            None,
        )
        with mock.patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": invalid},
        ), self.assertRaisesRegex(ExecutionGpuMatrixError, "invalid qualification status"):
            with temporary_execution_qualification(
                backend="flux-finetune",
                optimizers=("AdamW",),
                evidence_case_id="test:lease",
            ):
                pass

    def test_empty_evidence_case_id_is_rejected_before_mutation(self):
        backend_before = execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"]
        with self.assertRaisesRegex(ExecutionGpuMatrixError, "non-empty evidence_case_id"):
            with temporary_execution_qualification(
                backend="flux-finetune",
                optimizers=("AdamW",),
                evidence_case_id="",
            ):
                pass
        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS["flux-finetune"],
            backend_before,
        )


if __name__ == "__main__":
    unittest.main()
