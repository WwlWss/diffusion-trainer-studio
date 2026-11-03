"""C4-A Tests-only RED: pure Host timeline checks, NO CUDA qualification.

These are metadata guards, not evidence that Accelerator.save_state/load_state ran.
A Host test that already passes on the old C3 code is a legitimate baseline
PASS, not an invented RED. No Torch or optional GPU imports are permitted.
"""
from __future__ import annotations

import copy
import unittest

from tools import sd3_full_bf16_evidence_support as host
from tests.test_sd3_full_bf16_worker import R1, R2, event
from tests.test_sd3_full_bf16_worker_c3 import r2_step


def _fresh_r2():
    return [
        *r2_step(1, step=1, phase="fresh"),
        *r2_step(9, step=2, phase="fresh"),
        event("train_completed", 17, case=R2, step=2),
    ]


def _resume_r2():
    loaded = event("checkpoint_loaded", 1, case=R2, phase="resume", step=1)
    loaded["checkpoint_id"] = "a" * 32
    return [
        loaded,
        *r2_step(2, step=2, phase="resume"),
        event("train_completed", 10, case=R2, phase="resume", step=2),
    ]


def _renumber(events):
    for index, item in enumerate(events, start=1):
        item["event_seq"] = index
    return events


class C4AHostMetadataContract(unittest.TestCase):
    def test_h01_fresh_two_step_r2_metadata_does_not_claim_cuda(self):
        result = host.validate_worker_event_stream(
            case_id=R2, phase="fresh", events=_fresh_r2()
        )
        self.assertEqual(result["status"], "timeline_metadata_valid")
        self.assertEqual(result["observed_saved_steps"], [1, 2])
        self.assertIs(result["physical_cuda_qualified"], False)
        self.assertNotIn("worker_complete", result)

    def test_h02_resume_load_must_be_first_and_real_step_two_only(self):
        good = _resume_r2()
        result = host.validate_worker_event_stream(
            case_id=R2, phase="resume", events=good
        )
        self.assertEqual(result["observed_saved_steps"], [2])
        self.assertIs(result["physical_cuda_qualified"], False)
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="resume", events=_renumber(copy.deepcopy(good[1:]))
            )
        late_load = copy.deepcopy(good)
        late_load[0], late_load[1] = late_load[1], late_load[0]
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="resume", events=_renumber(late_load)
            )

    def test_h03_duplicate_step_save_never_certifies_completion(self):
        rows = _fresh_r2()
        duplicate = copy.deepcopy(rows[7])
        duplicate["checkpoint_id"] = "b" * 32
        rows.insert(8, duplicate)
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="fresh", events=_renumber(rows)
            )

    def test_h04_false_completion_before_checkpoint_or_final_step_rejected(self):
        rows = [
            *r2_step(1, step=1, phase="fresh", save=False),
            event("train_completed", 8, case=R2, step=1),
        ]
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="fresh", events=rows
            )
        rows2 = _fresh_r2()
        rows2.pop(15)
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="fresh", events=_renumber(rows2)
            )

    def test_h05_cross_run_phase_source_and_case_identity_rejected(self):
        for field, invalid in (
            ("run_id", "run-wrong-0001"),
            ("phase_id", "resume-wrong-0001"),
            ("source_commit", "b" * 40),
            ("case_id", R1),
        ):
            rows = _fresh_r2()
            rows[3][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                host.validate_worker_event_stream(
                    case_id=R2, phase="fresh", events=rows
                )

    def test_h06_r2_non_sync_child_and_early_scheduler_rejected(self):
        for invalid_kind in ("physical_child_stepped", "physical_scheduler_stepped"):
            rows = _fresh_r2()
            early = event(invalid_kind, 3, case=R2, step=1, profile="muon_primary")
            if invalid_kind == "physical_scheduler_stepped":
                early.pop("optimizer_profile", None)
            rows.insert(2, early)
            with self.subTest(kind=invalid_kind), self.assertRaises(ValueError):
                host.validate_worker_event_stream(
                    case_id=R2, phase="fresh", events=_renumber(rows)
                )

    def test_h07_r2_repeated_microbatch_sample_or_false_sync_rejected(self):
        for field, invalid in (
            ("sample_identity", "dts_c2b_sample_0001"),
            ("sync_gradients", False),
        ):
            rows = _fresh_r2()
            rows[2][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                host.validate_worker_event_stream(
                    case_id=R2, phase="fresh", events=rows
                )

    def test_h08_last_tail_or_post_completion_is_not_authoritative_step(self):
        rows = _fresh_r2()
        extra = event("checkpoint_saved", 18, case=R2, step=2)
        extra["state_dir"] = "fresh/dts_c4a-state"  # ancillary last, not Step-2
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="fresh", events=[*rows, extra]
            )
        # A completed metadata timeline itself proves neither real saved bytes
        # nor process ancestry. C4-B/C/D and Fix D must certify those.
        valid = host.validate_worker_event_stream(
            case_id=R2, phase="fresh", events=rows
        )
        self.assertEqual(valid["status"], "timeline_metadata_valid")
        self.assertFalse(valid["physical_cuda_qualified"])


if __name__ == "__main__":
    unittest.main()
