"""C4-B-G1 independent pinned real-CPU Save authorization hardening RED.

The tests use a real pinned Accelerator, genuine completed BF16 Step and
the original production-style save pre-hook. No fake Save implementation,
no CPU production receipt, and no changes to the locked C4-B 12 tests.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.test_sd3_full_bf16_worker_c4a_cpu import _r1_real_step
from tests.test_sd3_full_bf16_worker_c4b_cpu import (
    _bound_adapter, _diagnostic_pre_save_hook,
)


class C4BG1IndependentRealCpu(unittest.TestCase):
    def test_g01_failed_original_save_cannot_retry_same_step(self):
        bundle = _r1_real_step(self)
        accelerator, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            calls, handle = _diagnostic_pre_save_hook(accelerator, reject=True)
            self.addCleanup(handle.remove)
            path = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            with self.assertRaisesRegex(RuntimeError, "injected pre-save failure"):
                accelerator.save_state(str(path), safe_serialization=True)
            self.assertEqual(len(calls), 1)
            try:
                accelerator.save_state(str(path), safe_serialization=True)
            except ValueError as error:
                self.assertRegex(str(error), "already attempted|retry|consumed|duplicate")
            except RuntimeError:
                self.fail(
                    "C4-B-G1 CONTRACT ABSENT: the original pre-save hook"
                    " ran a second time after failed Step Save"
                )
            else:
                self.fail(
                    "C4-B-G1 CONTRACT ABSENT: retry falsely accepted for"
                    " already attempted physical Step"
                )
            self.assertEqual(len(calls), 1, "Unauthorized retry touched original")
            self.assertFalse(any(
                row["event_type"] == "checkpoint_saved" for row in probe.events
            ))

    def test_g02_symlink_alias_to_exact_step_directory_fails_before_io(self):
        bundle = _r1_real_step(self)
        accelerator, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            actual = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            actual.mkdir(parents=True)
            alias = root / "fresh" / "alias-step00000001-state"
            alias.symlink_to(actual, target_is_directory=True)
            calls, handle = _diagnostic_pre_save_hook(accelerator)
            self.addCleanup(handle.remove)
            try:
                accelerator.save_state(str(alias), safe_serialization=True)
            except ValueError as error:
                self.assertRegex(str(error), "C4-B|symlink|alias|unreviewed|linked")
            else:
                self.fail(
                    "C4-B-G1 CONTRACT ABSENT: symlink alias resolved to"
                    " authorized Step-N path and original Save ran"
                )
            self.assertEqual(calls, [], "Alias rejection occurred after pre-hook")
            self.assertEqual(list(actual.iterdir()), [], "Alias wrote real state files")
            self.assertFalse(any(
                row["event_type"] == "checkpoint_saved" for row in probe.events
            ))

    def test_g03_fresh_step_two_cannot_save_without_verified_step_one(self):
        # BOTH real physical optimizer Steps have run, but no real Step-1
        # original Save or full verified receipt exists. A Step-2 candidate
        # must not advance a missing Fresh Step-1 checkpoint lineage.
        bundle = _r1_real_step(self, count=2)
        accelerator, probe = bundle[1], bundle[5]
        self.assertEqual(probe._last_completed_step, 2)
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=2)
            calls, handle = _diagnostic_pre_save_hook(accelerator)
            self.addCleanup(handle.remove)
            path = root / "fresh" / "dts_c4a_cpu-step00000002-state"
            try:
                accelerator.save_state(str(path), safe_serialization=True)
            except ValueError as error:
                self.assertRegex(
                    str(error), "Step-1|Fresh Step-2|prerequisite|verified",
                )
            else:
                self.fail(
                    "C4-B-G2 CONTRACT ABSENT: physical Fresh Step-2 saved "
                    "without a verified original Step-1 checkpoint"
                )
            self.assertEqual(calls, [], "Step-2 refusal must precede original")
            self.assertFalse(path.exists())
            self.assertFalse(any(
                event["event_type"] == "checkpoint_saved"
                for event in probe.events
            ))



if __name__ == "__main__":
    unittest.main()
