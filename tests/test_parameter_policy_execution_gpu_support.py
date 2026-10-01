from __future__ import annotations

import unittest
from unittest import mock

from mikazuki import parameter_policy_execution as execution
from tools.parameter_policy_execution_gpu_support import (
    ADAMW_FULL_BF16_CASE_IDS,
    ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
    EXECUTION_GPU_CASE_PHASES,
    ExecutionGpuMatrixError,
    summarize_adamw_full_bf16_bundle,
    temporary_execution_qualification,
)


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


class TemporaryExecutionQualificationTests(unittest.TestCase):
    def test_pending_rows_are_leased_then_restored_exactly(self):
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
            self.assertTrue(all(row["lease_applied"] for row in lease["records"]))
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
