"""C4-C Tests-only RED: original Accelerate Resume admission and true CPU mechanics.

Pinned real CPU Torch 2.7.0 / Accelerate 1.6.0 / genuine Composite
Optimizer+Scheduler and BF16 LoRAModule. A child Python process proves an
actual Accelerator.load_state from a previously written real Step-1.
These CPU files are intentionally four-file DIAGNOSTICS, not a full
789-Tensor/nine-file CUDA parent, Host acceptance or worker_complete.

Contract absence must be a targeted AssertionError and never an
ImportError/AttributeError, fixture ERROR or SKIP.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from tests.test_sd3_full_bf16_worker_c3_f8_cpu import _make_r1_cpu_probe
from tests.test_sd3_full_bf16_worker_c3_f9_cpu import _lora_bundle
from tests.test_sd3_full_bf16_worker_c4a_cpu import (
    _r1_real_step, _r2_real_step, _manifest, _invocation,
)
from tests.test_sd3_full_bf16_worker_c4b_cpu import ACCELERATE_CPU_FILES
from tools import sd3_full_bf16_worker_observer as observer
from tools.sd3_full_bf16_worker_step_adapter import C4StepAdapter


def _snap(bundle):
    model, acc, _, _, facade, probe = bundle
    return {
        "model": observer.fingerprint_state_tree(model.state_dict()),
        "optimizer": observer.fingerprint_state_tree(facade.state_dict()),
        "scheduler": observer.fingerprint_state_tree(probe.scheduler.state_dict()),
    }


def _child_restored(case, source):
    # Run in a separate OS PID with an independently constructed prepared
    # Accelerator and genuine model/optimizer/scheduler, not an object alias.
    bundle = _make_r1_cpu_probe() if case == "r1" else _lora_bundle()
    acc = bundle[1]
    acc.load_state(str(source))
    return {"restored": _snap(bundle), "pid": __import__("os").getpid()}


def _new_process_restore(test, bundle, case, root):
    model, acc, _, _, _, probe = bundle
    state = root / "original-fresh-step1"
    before = _snap(bundle)
    acc.save_state(str(state), safe_serialization=True)
    test.assertEqual(
        {p.name for p in state.iterdir()}, ACCELERATE_CPU_FILES,
    )
    completed = subprocess.run(
        [
            sys.executable, "-m",
            "tests.test_sd3_full_bf16_worker_c4c_cpu",
            "--child-restore", case, str(state),
        ],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=90, check=False,
    )
    test.assertEqual(
        completed.returncode, 0,
        "Original new-process Accelerator.load_state failed: "
        + completed.stderr[-1700:],
    )
    outputs = [
        row[len("C4C_CHILD_RESULT="):]
        for row in completed.stdout.splitlines()
        if row.startswith("C4C_CHILD_RESULT=")
    ]
    test.assertEqual(len(outputs), 1, "No exact child-process Save/Load evidence")
    payload = json.loads(outputs[0])
    test.assertNotEqual(payload["pid"], __import__("os").getpid())
    test.assertEqual(payload["restored"], before)
    test.assertEqual(probe._last_completed_step, 1)
    return payload


def _resume_adapter(test, root):
    bundle = _make_r1_cpu_probe()
    probe, acc = bundle[5], bundle[1]
    # Diagnostic CPU Fixture: same real C3 class, a fresh independent Process
    # identity, and the genuine Resume geometry start_step=1. No accepted
    # production parent is simulated or authorized here.
    probe.phase = "resume"
    probe.phase_id = "resume-c4c-cpu-0001"
    probe.start_step = 1
    probe.install()
    test.addCleanup(probe.close)
    adapter = C4StepAdapter(
        probe=probe, accelerator=acc, network=bundle[0],
        scheduler=probe.scheduler, invocation=_invocation(probe),
        manifest_payload=_manifest(probe, root),
    )
    adapter.install()
    test.addCleanup(adapter.close)
    return bundle, adapter


def _require(test, value, label):
    test.assertTrue(
        callable(value),
        "C4-C CONTRACT ABSENT: " + label,
    )
    return value


class C4CRealCPULoadContractRED(unittest.TestCase):
    def test_c01_r1_real_fresh_cpu_step1_saved_and_new_process_loads_state(self):
        fresh = _r1_real_step(self)
        with tempfile.TemporaryDirectory() as td:
            _new_process_restore(self, fresh, "r1", Path(td))

    def test_c02_r2_real_muon_adamw_accum2_new_process_loads_both_children(self):
        fresh = _r2_real_step(self)
        self.assertEqual(fresh[5].sync_sequence, (False, True))
        with tempfile.TemporaryDirectory() as td:
            _new_process_restore(self, fresh, "r2", Path(td))

    def test_c03_resume_installs_managed_load_on_same_accelerator(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            acc = bundle[1]
            current = vars(acc).get("load_state")
            _require(
                self, current,
                "same prepared Accelerator.load_state must be wrapped "
                "before the production Trainer begins Restore",
            )
            original = adapter._original_accelerator_slots["load_state"][1]
            self.assertIsNot(current, original)

    def test_c04_unapproved_parent_does_not_enter_original_load_prehooks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = _r1_real_step(self)
            unauthorized = root / "unapproved-parent"
            fresh[1].save_state(str(unauthorized), safe_serialization=True)
            bundle, _adapter = _resume_adapter(self, root)
            acc = bundle[1]
            seen = []
            handle = acc.register_load_state_pre_hook(
                lambda models, input_dir: seen.append(str(input_dir))
            )
            self.addCleanup(handle.remove)
            try:
                acc.load_state(str(unauthorized))
            except ValueError:
                pass
            except Exception as exc:
                self.fail("C4-C CONTRACT ABSENT: unauthorized original "
                          "Load raised unexpected " + type(exc).__name__)
            self.assertFalse(
                seen,
                "C4-C CONTRACT ABSENT: unauthorized Fresh parent reached "
                "original Accelerator pre-load hooks",
            )

    def test_c05_fresh_phase_cannot_load_resume_state(self):
        bundle = _r1_real_step(self)
        acc = bundle[1]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            genuine = root / "original-snapshot"
            acc.save_state(str(genuine), safe_serialization=True)
            adapter = C4StepAdapter(
                probe=bundle[5], accelerator=acc, network=bundle[0],
                scheduler=bundle[5].scheduler,
                invocation=_invocation(bundle[5]),
                manifest_payload=_manifest(bundle[5], root),
            )
            adapter.install()
            self.addCleanup(adapter.close)
            try:
                acc.load_state(str(genuine))
            except ValueError:
                pass
            else:
                self.fail("C4-C CONTRACT ABSENT: Fresh Phase called original "
                          "Accelerator.load_state without a Resume Ticket")

    def test_c06_load_observation_cannot_be_claimed_before_real_original_return(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            read = _require(
                self, getattr(adapter, "resume_load_observation", None),
                "real original load attempt observer missing",
            )
            with self.assertRaises(ValueError):
                read()
            self.assertFalse(bundle[5].events)

    def test_c07_authentic_checkpoint_loaded_requires_pre_batch_event_api(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            probe = bundle[5]
            recorder = _require(
                self, getattr(probe, "append_authenticated_checkpoint_loaded", None),
                "pre-Batch checkpoint_loaded recorder missing",
            )
            with self.assertRaises(ValueError):
                recorder()
            self.assertIsNone(probe._current_step)
            self.assertFalse(probe.events)

    def test_c08_no_verified_resume_receipt_without_authorized_parent(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            verified = _require(
                self, getattr(adapter, "verified_resume_load", None),
                "source-verified original Load receipt accessor missing",
            )
            with self.assertRaises(ValueError):
                verified()

    def test_c09_real_post_load_source_state_has_a_readonly_verifier(self):
        from tools import sd3_full_bf16_worker_save_verifier as save_verifier
        verify = _require(
            self, getattr(save_verifier, "verify_c4c_restored_state", None),
            "actual loaded BF16 Model/Child Optimizer/Scheduler/RNG "
            "must be checked against original Fresh Step-1 bytes",
        )
        with self.assertRaises(ValueError):
            verify()

    def test_c10_missing_optimizer_state_never_creates_a_load_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = _r1_real_step(self)
            state = root / "diagnostic-parent"
            fresh[1].save_state(str(state), safe_serialization=True)
            (state / "optimizer.bin").unlink()
            bundle, _adapter = _resume_adapter(self, root)
            with self.assertRaises((ValueError, OSError, RuntimeError, KeyError)):
                bundle[1].load_state(str(state))
            self.assertFalse(any(
                event["event_type"] == "checkpoint_loaded"
                for event in bundle[5].events
            ))

    def test_c11_no_checkpoint_loaded_event_before_any_actual_load(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, _adapter = _resume_adapter(self, Path(td))
            self.assertEqual(bundle[5].events, [])
            self.assertIsNone(bundle[5]._current_step)
            self.assertEqual(bundle[5].start_step, 1)

    def test_c12_resume_step2_anchor_still_fail_closed_until_true_parent_load(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            with self.assertRaises(ValueError):
                adapter.completed_step_anchor(logical_step=2)
            self.assertFalse(any(
                event["event_type"] == "checkpoint_loaded"
                for event in bundle[5].events
            ))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child-restore":
        result = _child_restored(sys.argv[2], Path(sys.argv[3]))
        print("C4C_CHILD_RESULT=" + json.dumps(result, sort_keys=True))
    else:
        unittest.main()
