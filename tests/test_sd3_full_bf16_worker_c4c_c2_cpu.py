"""C4-C2 Tests-only RED: Runner-origin Fresh Step-1 parent before original Load.

The CPU suite deliberately has NO trusted FullBF16 CUDA parent. A matching
Worker Ticket/path/Checkpoint ID, an invented 'physical_binding_valid' dict,
and CPU diagnostic Save files must all fail before the first original
Accelerator.load_state prehook. A positive original CPU Save -> new OS
process Load is tested separately, without an Adapter authorization.

RED is only a tagged contract AssertionError, never fixture/import ERROR
or self-issued accepted_parent evidence. Do not add fake CUDA admission.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.test_sd3_full_bf16_worker_c3_f8_cpu import _make_r1_cpu_probe
from tests.test_sd3_full_bf16_worker_c4a_cpu import _r1_real_step
from tests.test_sd3_full_bf16_worker_c4c_cpu import (
    _new_process_restore, _resume_adapter,
)
from tests.test_sd3_full_bf16_worker_c4c_c1_cpu import _real_prepared_resume
from tools import sd3_full_bf16_evidence_support as host


_SOURCE = "a" * 40
_FOREIGN_SOURCE = "b" * 40
_CHECKPOINT = "a" * 32
_OTHER_CHECKPOINT = "b" * 32
_TICKET_NONCE = "c" * 64


def _untrusted_report(*, case_id="r1-adamw-accum1",
                      source_commit=_SOURCE, checkpoint_id=_CHECKPOINT):
    """Worker-supplied lookalike, NOT an independently accepted Runner proof."""
    return {
        "status": "physical_binding_valid",
        "case_id": case_id,
        "phase": "fresh",
        "logical_step": 1,
        "source_commit": source_commit,
        "checkpoint_id": checkpoint_id,
        "ticket_nonce": _TICKET_NONCE,
        "case_manifest_sha256": "d" * 64,
        "input_identity_sha256": "e" * 64,
        "policy_hash": "f" * 64,
        "adapter_file_sha256": "0" * 64,
        "checkpoint_file_sha256": {
            k: "1" * 64 for k in sorted(host._SD3_STATE_FILES)
        },
        "claimed_issuer": "independent-runner",
    }


def _binder(test, adapter):
    binder = getattr(adapter, "bind_runner_accepted_fresh_parent", None)
    test.assertTrue(
        callable(binder),
        "C4-C2 CONTRACT ABSENT: independently Runner-authenticated "
        "Fresh Step-1 parent bind entrypoint is missing",
    )
    return binder


def _reject_untrusted(test, adapter, report, *, binder=None):
    if binder is None:
        binder = _binder(test, adapter)
    try:
        binder(parent_admission=report)
    except ValueError:
        return
    test.fail(
        "C4-C2 CONTRACT ABSENT: a Worker-constructible parent record was "
        "accepted as original Accelerator Load authority",
    )


class C4C2RunnerParentAuthorityCpuRED(unittest.TestCase):
    def test_c201_parent_binding_requires_a_separate_trusted_entrypoint(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            _binder(self, adapter)

    def test_c202_worker_supplied_physical_binding_valid_is_not_authority(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            _reject_untrusted(self, adapter, _untrusted_report())

    def test_c203_self_signed_file_hashes_cannot_mint_runner_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            asserted = _untrusted_report()
            asserted["claimed_issuer"] = "signed-runner"
            asserted["signature"] = "9" * 128
            _reject_untrusted(self, adapter, asserted)

    def test_c204_foreign_case_or_source_cannot_bind_fresh_parent(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            binder = _binder(self, adapter)
            for wrong in (
                _untrusted_report(case_id="r2-muon-adamw-fallback-accum2"),
                _untrusted_report(source_commit=_FOREIGN_SOURCE),
            ):
                with self.subTest(case=wrong["case_id"], sha=wrong["source_commit"]):
                    _reject_untrusted(self, adapter, wrong, binder=binder)

    def test_c205_checkpoint_id_from_ticket_and_progress_is_not_runner_proof(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            asserted = _untrusted_report(checkpoint_id=_CHECKPOINT)
            asserted["expected_resume_checkpoint_id"] = _CHECKPOINT
            asserted["resume_source_checkpoint_id"] = _CHECKPOINT
            asserted["progress_checkpoint_id"] = _CHECKPOINT
            _reject_untrusted(self, adapter, asserted)

    def test_c206_real_four_file_cpu_parent_must_never_gain_cuda_admission(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = _r1_real_step(self)
            state = root / "authentic-but-cpu-only-step1"
            fresh[1].save_state(str(state), safe_serialization=True)
            self.assertEqual({p.name for p in state.iterdir()}, {
                "model.safetensors", "optimizer.bin", "scheduler.bin",
                "random_states_0.pkl",
            })
            _bundle, adapter = _real_prepared_resume(
                self, root, advertised_parent=state,
            )
            asserted = _untrusted_report()
            asserted["checkpoint_dir"] = str(state)
            asserted["checkpoint_file_sha256"] = {
                p.name: host._sha256_file(p) for p in state.iterdir()
            }
            _reject_untrusted(self, adapter, asserted)

    def test_c207_even_matching_declared_parent_cannot_reach_original_prehook(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = _make_r1_cpu_probe()
            state = root / "declared-fresh-parent"
            fresh[1].save_state(str(state), safe_serialization=True)
            bundle, _adapter = _real_prepared_resume(
                self, root, advertised_parent=state,
            )
            seen = []
            hook = bundle[1].register_load_state_pre_hook(
                lambda models, input_dir: seen.append(str(input_dir))
            )
            self.addCleanup(hook.remove)
            with self.assertRaises(ValueError):
                bundle[1].load_state(str(state))
            self.assertEqual(seen, [])
            self.assertEqual(bundle[5].events, [])

    def test_c208_rejected_original_load_has_zero_original_calls_and_no_event(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            with self.assertRaises(ValueError):
                bundle[1].load_state(str(Path(td) / "unauthorized"))
            attempt = adapter.resume_load_observation()
            self.assertEqual(attempt.original_call_count, 0)
            self.assertFalse(attempt.original_returned)
            self.assertFalse(attempt.qualified)
            self.assertFalse(bundle[5].events)

    def test_c209_repeated_untrusted_load_denied_without_original_call(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = _make_r1_cpu_probe()
            state = root / "one-shot"
            fresh[1].save_state(str(state), safe_serialization=True)
            bundle, adapter = _real_prepared_resume(
                self, root, advertised_parent=state,
            )
            for _ in range(2):
                with self.assertRaises(ValueError):
                    bundle[1].load_state(str(state))
            self.assertEqual(adapter._load_original_calls, 0)
            self.assertFalse(bundle[5].events)

    def test_c210_host_source_claim_must_not_be_promoted_to_private_capability(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            asserted = _untrusted_report()
            asserted["accepted_fresh_evidence"] = _untrusted_report()
            asserted["runner_status"] = "accepted"
            _reject_untrusted(self, adapter, asserted)

    def test_c211_delegation_requires_a_reviewed_original_call_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            _bundle, adapter = _resume_adapter(self, Path(td))
            gate = getattr(adapter, "_delegate_verified_original_load_once", None)
            self.assertTrue(
                callable(gate),
                "C4-C2 CONTRACT ABSENT: no class-owned, exactly-once "
                "original Load delegate behind Runner-accepted parent",
            )
            # A missing authoritative admission cannot enter the real
            # Accelerator prehooks even through the internal dispatcher.
            with self.assertRaises(ValueError):
                gate()

    def test_c212_genuine_cpu_new_process_restore_remains_a_diagnostic(self):
        fresh = _r1_real_step(self)
        with tempfile.TemporaryDirectory() as td:
            outcome = _new_process_restore(self, fresh, "r1", Path(td))
            self.assertEqual(set(outcome["restored"]), {
                "model", "optimizer", "scheduler",
            })


if __name__ == "__main__":
    unittest.main()
