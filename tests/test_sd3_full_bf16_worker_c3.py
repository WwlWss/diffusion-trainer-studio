"""C3 pure Host timeline RED->GREEN regression, not physical CUDA proof."""
from __future__ import annotations
import copy
import unittest

from tools import sd3_full_bf16_evidence_support as host
from tests.test_sd3_full_bf16_worker import event, R1, R2


def r2_step(start=1, *, step=1, phase="fresh", save=True):
    sample1 = "dts_c2b_sample_0001"
    sample2 = "dts_c2b_sample_0002"
    rows = [
        event("batch_started", start, case=R2, phase=phase, step=step, batch=1, sample=sample1, sync=False),
        event("backward_completed", start+1, case=R2, phase=phase, step=step, batch=1, sample=sample1, sync=False),
        event("batch_started", start+2, case=R2, phase=phase, step=step, batch=2, sample=sample2, sync=True),
        event("backward_completed", start+3, case=R2, phase=phase, step=step, batch=2, sample=sample2, sync=True),
        event("physical_child_stepped", start+4, case=R2, phase=phase, step=step, profile="muon_primary"),
        event("physical_child_stepped", start+5, case=R2, phase=phase, step=step, profile="adamw_fallback"),
        event("physical_scheduler_stepped", start+6, case=R2, phase=phase, step=step),
    ]
    if save:
        saved = event("checkpoint_saved", start+7, case=R2, phase=phase, step=step)
        saved["checkpoint_id"] = f"{step:x}" * 32
        rows.append(saved)
    return rows


class C3HostEventContract(unittest.TestCase):
    def test_fresh_two_physical_steps_with_actual_r2_accumulation_geometry(self):
        rows = [
            *r2_step(1, step=1),
            *r2_step(9, step=2),
            event("train_completed", 17, case=R2, step=2),
        ]
        result = host.validate_worker_event_stream(case_id=R2, phase="fresh", events=rows)
        self.assertEqual(result["status"], "timeline_metadata_valid")
        self.assertEqual(result["observed_saved_steps"], [1, 2])
        self.assertFalse(result["physical_cuda_qualified"])

    def test_resume_requires_load_then_step_two(self):
        loaded = event("checkpoint_loaded", 1, case=R2, phase="resume", step=1)
        loaded["checkpoint_id"] = "a" * 32
        rows = [loaded, *r2_step(2, step=2, phase="resume"),
                event("train_completed", 10, case=R2, phase="resume", step=2)]
        valid = host.validate_worker_event_stream(case_id=R2, phase="resume", events=rows)
        self.assertEqual(valid["observed_saved_steps"], [2])
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(case_id=R2, phase="resume", events=rows[1:])

    def test_missing_fallback_child_rejected(self):
        rows = [*r2_step(), event("train_completed", 9, case=R2)]
        rows.pop(5)
        for n, e in enumerate(rows, 1):
            e["event_seq"] = n
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(case_id=R2, phase="fresh", events=rows)

    def test_early_scheduler_and_repeated_muon_child_rejected(self):
        for index in (5, 4):
            rows = [*r2_step(), event("train_completed", 9, case=R2)]
            rows[index] = dict(rows[index])
            if index == 5:
                rows[index]["optimizer_profile"] = "muon_primary"
            else:
                rows[index]["event_type"] = "physical_scheduler_stepped"
                rows[index].pop("optimizer_profile", None)
            with self.subTest(index=index), self.assertRaises(ValueError):
                host.validate_worker_event_stream(case_id=R2, phase="fresh", events=rows)

    def test_false_non_sync_flag_rejected(self):
        rows = [*r2_step(), event("train_completed", 9, case=R2)]
        rows[0] = dict(rows[0], sync_gradients=True)
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(case_id=R2, phase="fresh", events=rows)

    def test_mixed_run_and_source_identity_rejected(self):
        rows = [*r2_step(), event("train_completed", 9, case=R2)]
        for field, value in (("run_id","other-run-0001"), ("source_commit","b"*40)):
            changed = copy.deepcopy(rows)
            changed[3][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                host.validate_worker_event_stream(case_id=R2, phase="fresh", events=changed)

    def test_missing_checkpoint_or_event_after_completion_rejected(self):
        stream = [*r2_step(save=False), event("train_completed", 8, case=R2)]
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(case_id=R2, phase="fresh", events=stream)
        stream = [*r2_step(), event("train_completed", 9, case=R2),
                  event("batch_started", 10, case=R2, step=2, batch=1,
                        sample="dts_c2b_sample_0001", sync=False)]
        with self.assertRaises(ValueError):
            host.validate_worker_event_stream(case_id=R2, phase="fresh", events=stream)


if __name__ == "__main__":
    unittest.main()
