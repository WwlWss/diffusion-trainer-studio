"""C4-C pure Host Resume chronology / accepted-parent-id metadata tests.

No Torch, no CUDA, no untrusted pickle loading. A timeline validated here
is NEVER physical evidence or a Runner-accepted Fresh Parent. The new binding
API must only compare the loaded-event metadata to an *externally supplied*
already accepted parent ID; provenance/bytes belong to independent Fix D.
"""
from __future__ import annotations

import copy
import unittest

from tools import sd3_full_bf16_evidence_support as host
from tests.test_sd3_full_bf16_worker import R1, R2, event
from tests.test_sd3_full_bf16_worker_c4a_host import (
    _resume_r2, _fresh_r2, _renumber,
)

GOOD_PARENT = "a" * 32
FOREIGN_PARENT = "b" * 32
SOURCE = "a" * 40


def _binding(test):
    found = getattr(host, "validate_c4c_resume_load_binding", None)
    test.assertTrue(
        callable(found),
        "C4-C HOST CONTRACT ABSENT: loaded event must bind to the "
        "independently accepted Fresh Step-1 checkpoint ID without "
        "turning Host metadata into physical CUDA qualification",
    )
    return found


class C4CHostParentTimelineRED(unittest.TestCase):
    def test_h01_r2_correct_resume_metadata_is_never_physical_qualification(self):
        summary = host.validate_worker_event_stream(
            case_id=R2, phase="resume", events=_resume_r2(),
        )
        self.assertEqual(summary["observed_saved_steps"], [2])
        self.assertEqual(summary["status"], "timeline_metadata_valid")
        self.assertIs(summary["physical_cuda_qualified"], False)

    def test_h02_missing_load_first_or_load_after_batch_fails(self):
        good = _resume_r2()
        for bad in (
            good[1:],
            [good[1], good[0], *good[2:]],
        ):
            with self.subTest(kind=bad[0]["event_type"]):
                with self.assertRaises(ValueError):
                    host.validate_worker_event_stream(
                        case_id=R2, phase="resume",
                        events=_renumber(copy.deepcopy(bad)),
                    )

    def test_h03_duplicate_checkpoint_loaded_before_training_fails(self):
        good = _resume_r2()
        twice = copy.deepcopy(good)
        twice.insert(1, copy.deepcopy(good[0]))
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="resume", events=_renumber(twice),
            )

    def test_h04_cross_source_or_cross_phase_event_fails(self):
        for key, value in (
            ("source_commit", "b" * 40),
            ("phase_id", "fresh-instead-of-resume-0001"),
            ("run_id", "wrong-run-0001"),
            ("case_id", R1),
        ):
            bad = copy.deepcopy(_resume_r2())
            bad[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                host.validate_worker_event_stream(
                    case_id=R2, phase="resume", events=bad,
                )

    def test_h05_forged_loaded_checkpoint_id_must_not_match_accepted_parent(self):
        # A fully formed timeline accepts any 32-hex checkpoint ID today;
        # the independent accepted-parent ID must be checked separately.
        events = copy.deepcopy(_resume_r2())
        events[0]["checkpoint_id"] = FOREIGN_PARENT
        self.assertEqual(host.validate_worker_event_stream(
            case_id=R2, phase="resume", events=events,
        )["status"], "timeline_metadata_valid")
        binder = _binding(self)
        with self.assertRaises(ValueError):
            binder(
                case_id=R2, events=events,
                accepted_parent_checkpoint_id=GOOD_PARENT,
                expected_source_commit=SOURCE,
            )

    def test_h06_resume_identity_cannot_be_cross_source_parent_metadata(self):
        binder = _binding(self)
        with self.assertRaises(ValueError):
            binder(
                case_id=R2, events=_resume_r2(),
                accepted_parent_checkpoint_id=GOOD_PARENT,
                expected_source_commit="b" * 40,
            )

    def test_h07_last_tail_and_fresh_loaded_events_cannot_create_lineage(self):
        rows = _fresh_r2()
        fake = event("checkpoint_loaded", 18, case=R2, step=1)
        fake["checkpoint_id"] = GOOD_PARENT
        rows.append(fake)
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(
                case_id=R2, phase="fresh", events=rows,
            )
        # A normal Fresh two-step source cannot claim a pre-Batch Load.
        self.assertIs(host.validate_worker_event_stream(
            case_id=R2, phase="fresh", events=_fresh_r2(),
        )["physical_cuda_qualified"], False)

    def test_h08_correct_accepted_parent_id_is_still_metadata_only(self):
        binder = _binding(self)
        value = binder(
            case_id=R2, events=_resume_r2(),
            accepted_parent_checkpoint_id=GOOD_PARENT,
            expected_source_commit=SOURCE,
        )
        self.assertEqual(value["status"], "timeline_parent_id_bound_not_qualified")
        self.assertIs(value["physical_cuda_qualified"], False)
        self.assertNotIn("worker_complete", value)


if __name__ == "__main__":
    unittest.main()
