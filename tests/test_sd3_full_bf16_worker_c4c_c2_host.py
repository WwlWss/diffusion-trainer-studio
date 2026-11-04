"""C4-C2 pure Host RED: independent Fresh-parent admission ≠ Worker report.

No Torch/GPU and no pickle load. All positive-looking report objects in this
suite are Worker-constructible, hence NEVER genuine Runner acceptances.
The source-authenticating Host/Runner bridge is missing on this RED HEAD.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.test_sd3_full_bf16_worker import R1, R2
from tests.test_sd3_full_bf16_worker_c4a_host import _resume_r2
from tools import sd3_full_bf16_evidence_support as host

SOURCE = "a" * 40
CHECKPOINT = "b" * 32
NONCE = "c" * 64
MANIFEST = "d" * 64
INPUT = "e" * 64


def _pretend_acceptance(**overrides):
    claim = {
        "status": "physical_binding_valid",
        "claimed_issuer": "independent-runner",
        "phase": "fresh",
        "case_id": R2,
        "source_commit": SOURCE,
        "logical_step": 1,
        "checkpoint_id": CHECKPOINT,
        "expected_resume_checkpoint_id": CHECKPOINT,
        "ticket_nonce": NONCE,
        "case_manifest_sha256": MANIFEST,
        "input_identity_sha256": INPUT,
        "policy_hash": "f" * 64,
        "checkpoint_file_sha256": {
            key: "1" * 64 for key in sorted(host._SD3_STATE_FILES)
        },
        "adapter_file_sha256": "2" * 64,
    }
    claim.update(overrides)
    return claim


def _verifier(test):
    verify = getattr(host, "verify_c4c_runner_parent_admission", None)
    test.assertTrue(
        callable(verify),
        "C4-C2 HOST CONTRACT ABSENT: no reviewed independent Runner "
        "parent-admission source verifier",
    )
    return verify


def _reject(test, claim, *, verify=None):
    if verify is None:
        verify = _verifier(test)
    try:
        verify(
            parent_admission=claim,
            case_id=R2,
            source_commit=SOURCE,
            expected_checkpoint_id=CHECKPOINT,
            ticket_nonce=NONCE,
            case_manifest_sha256=MANIFEST,
            input_identity_sha256=INPUT,
        )
    except ValueError:
        return
    test.fail(
        "C4-C2 HOST CONTRACT ABSENT: self-reported acceptance was "
        "promoted to independent Runner parent authority",
    )


class C4C2IndependentParentHostRED(unittest.TestCase):
    def test_h201_runner_admission_origin_verifier_is_required(self):
        _verifier(self)

    def test_h202_worker_physical_binding_valid_flag_cannot_admit_parent(self):
        _reject(self, _pretend_acceptance())

    def test_h203_fake_runner_signature_and_issuer_name_are_not_trust(self):
        _reject(self, _pretend_acceptance(
            signature="0" * 128,
            claimed_issuer="independent-runner",
            claimed_runner_pid=12345,
        ))

    def test_h204_foreign_case_source_or_step_cannot_admit_parent(self):
        verify = _verifier(self)
        for changed in (
            {"case_id": R1},
            {"source_commit": "b" * 40},
            {"logical_step": 2},
        ):
            with self.subTest(changed=changed):
                _reject(self, _pretend_acceptance(**changed), verify=verify)

    def test_h205_matching_worker_ticket_id_does_not_substitute_runner(self):
        _reject(self, _pretend_acceptance(
            checkpoint_id=CHECKPOINT,
            expected_resume_checkpoint_id=CHECKPOINT,
            progress_checkpoint_id=CHECKPOINT,
        ))

    def test_h206_replayed_or_self_claimed_nonce_does_not_substitute_origin(self):
        _reject(self, _pretend_acceptance(
            claimed_replay_protected=True,
            consumed_ticket_nonce=NONCE,
        ))

    def test_h207_nine_arbitrary_files_with_real_hashes_are_not_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "fabricated-fresh-step1"
            root.mkdir()
            for name in host._SD3_STATE_FILES:
                (root / name).write_bytes(
                    b"not-an-original-Accelerate-save-" + name.encode()
                )
            # The inventory helper hashes names/bytes, NOT the trusted writer
            # provenance or a valid nine-file SD3 CUDA state.
            actual = host._checkpoint_file_inventory(root)
            self.assertEqual(set(actual), host._SD3_STATE_FILES)
            _reject(self, _pretend_acceptance(
                checkpoint_dir=str(root),
                checkpoint_file_sha256=actual,
            ))

    def test_h208_real_cpu_four_file_diagnostic_is_not_nine_file_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in (
                "model.safetensors", "optimizer.bin", "scheduler.bin",
                "random_states_0.pkl",
            ):
                (root / name).write_bytes(b"cpu-diagnostic")
            with self.assertRaises(ValueError):
                host._checkpoint_file_inventory(root)

    def test_h209_resume_event_timeline_cannot_claim_cuda_or_runner_auth(self):
        summary = host.validate_worker_event_stream(
            case_id=R2, phase="resume", events=_resume_r2()
        )
        self.assertEqual(summary["status"], "timeline_metadata_valid")
        self.assertFalse(summary["physical_cuda_qualified"])
        self.assertNotIn("parent_admission_issued", summary)

    def test_h210_resume_phase_or_wrong_manifest_self_report_is_not_fresh(self):
        verify = _verifier(self)
        for changed in (
            {"phase": "resume"},
            {"case_manifest_sha256": "0" * 64},
            {"input_identity_sha256": "0" * 64},
        ):
            with self.subTest(changed=changed):
                _reject(self, _pretend_acceptance(**changed), verify=verify)


if __name__ == "__main__":
    unittest.main()
