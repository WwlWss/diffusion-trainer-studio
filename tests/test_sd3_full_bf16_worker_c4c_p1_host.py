"""C4-C2-P1 Tests-only RED: Runner signature mechanism, not CUDA admission.

Only ephemeral TEST Ed25519 private keys are created inside tests; none are
written to files or passed to a Worker. The expected verifier must use an
independently pinned public key, NEVER an envelope or Worker-provided key.
A valid TEST signature only means "signature_valid_unqualified": no genuine
Fresh Step-1, original Load, checkpoint_loaded, or worker_complete.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from tests.test_sd3_full_bf16_worker import R2
from tests.test_sd3_full_bf16_worker_c4a_host import _resume_r2
from tools import sd3_full_bf16_evidence_support as host


DOMAIN = b"DTS-C4C2-P1-ED25519-V1\x00"
AUDIENCE = "dts.sd3c2b.c4c2.resume-load"
SCHEMA = "dts.sd3c2b.c4c2p1.provisional-step1-v1"
SIGNED_SCHEMA = "dts.sd3c2b.c4c2p1.signed-provisional-step1-v1"
NOW = 1_800_000_000


def _payload():
    return {
        "schema": SCHEMA,
        "version": 1,
        "proof_scope": "provisional_fresh_step1_only",
        "audience": AUDIENCE,
        "case_id": R2,
        "phase": "fresh",
        "logical_step": 1,
        "run_id": "run-p1-test-0001",
        "fresh_phase_id": "fresh-p1-test-0001",
        "resume_phase_id": "resume-p1-test-0001",
        "source_commit": "a" * 40,
        "case_manifest_sha256": "b" * 64,
        "input_identity_sha256": "c" * 64,
        "reference_policy_sha256": "d" * 64,
        "reference_target_spec_sha256": "e" * 64,
        "resume_ticket_sha256": "f" * 64,
        "resume_ticket_nonce": "0" * 64,
        "checkpoint_id": "1" * 32,
        "checkpoint_dir": "/external-only/never-a-real-cuda-parent",
        "checkpoint_file_sha256": {
            name: "2" * 64 for name in sorted(host._SD3_STATE_FILES)
        },
        "adapter_path": "/external-only/never-a-real-cuda-parent.safetensors",
        "adapter_file_sha256": "3" * 64,
        "physical_step_anchor_sha256": "4" * 64,
        "checkpoint_event_prefix_sha256": "5" * 64,
        "issued_at_unix": NOW - 5,
        "expires_at_unix": NOW + 55,
        "one_use_nonce": "6" * 64,
    }


def _keys():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private, public


def _envelope(private, payload=None):
    body = _payload() if payload is None else payload
    signature = private.sign(
        DOMAIN + host._canonical_json_bytes(body)
    )
    return {
        "schema": SIGNED_SCHEMA,
        "payload": body,
        "signature_ed25519": base64.b64encode(signature).decode("ascii"),
    }


def _context():
    p = _payload()
    return {
        k: p[k] for k in (
            "audience", "case_id", "run_id", "fresh_phase_id",
            "resume_phase_id", "source_commit", "checkpoint_id",
            "resume_ticket_sha256", "resume_ticket_nonce",
            "case_manifest_sha256", "input_identity_sha256",
            "reference_policy_sha256", "reference_target_spec_sha256",
        )
    }


def _verifier(test):
    name = "tools.sd3_full_bf16_runner_trust"
    spec = importlib.util.find_spec(name)
    test.assertIsNotNone(
        spec,
        "C4-C2-P1 HOST CONTRACT ABSENT: independent Runner Ed25519 "
        "verifier module is missing",
    )
    module = importlib.import_module(name)
    verify = getattr(module, "verify_provisional_step1_signature", None)
    test.assertTrue(
        callable(verify),
        "C4-C2-P1 HOST CONTRACT ABSENT: no reviewed independent "
        "Runner provisional Step-1 signature verifier",
    )
    return verify


def _verify(verify, envelope, key, *, now=NOW, seen=None, context=None):
    return verify(
        envelope=envelope,
        pinned_public_key=key,
        expected_context=_context() if context is None else context,
        now_unix=now,
        seen_nonces=set() if seen is None else seen,
    )


def _must_reject(test, verify, envelope, key, **kwargs):
    try:
        _verify(verify, envelope, key, **kwargs)
    except ValueError:
        return
    test.fail(
        "C4-C2-P1 HOST CONTRACT ABSENT: forged/expired/replayed "
        "Runner signature was accepted as evidence"
    )


class C4C2P1RunnerSignatureHostRED(unittest.TestCase):
    def test_p101_ephemeral_test_ed25519_signer_is_real(self):
        private, public = _keys()
        message = DOMAIN + host._canonical_json_bytes(_payload())
        private.public_key().verify(private.sign(message), message)
        self.assertEqual(len(public), 32)

    def test_p102_existing_host_four_file_snapshot_is_not_full_parent(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            for name in (
                "model.safetensors", "optimizer.bin", "scheduler.bin",
                "random_states_0.pkl",
            ):
                (root / name).write_bytes(b"test-only")
            with self.assertRaises(ValueError):
                host._checkpoint_file_inventory(root)

    def test_p103_existing_resume_timeline_is_metadata_only(self):
        summary = host.validate_worker_event_stream(
            case_id=R2, phase="resume", events=_resume_r2(),
        )
        self.assertEqual(summary["status"], "timeline_metadata_valid")
        self.assertFalse(summary["physical_cuda_qualified"])

    def test_p104_genuine_test_signature_is_valid_but_not_cuda_admission(self):
        verify = _verifier(self)
        private, public = _keys()
        result = _verify(verify, _envelope(private), public)
        self.assertEqual(result["status"], "signature_valid_unqualified")
        self.assertIs(result["physical_cuda_qualified"], False)
        self.assertNotIn("worker_complete", result)
        self.assertNotIn("physical_binding_valid", str(result))

    def test_p105_modified_original_signed_checkpoint_id_rejects(self):
        verify = _verifier(self)
        private, public = _keys()
        forged = _envelope(private)
        forged["payload"]["checkpoint_id"] = "9" * 32
        _must_reject(self, verify, forged, public)

    def test_p106_modified_signature_bytes_rejects(self):
        verify = _verifier(self)
        private, public = _keys()
        forged = _envelope(private)
        sig = bytearray(base64.b64decode(forged["signature_ed25519"]))
        sig[0] ^= 1
        forged["signature_ed25519"] = base64.b64encode(sig).decode()
        _must_reject(self, verify, forged, public)

    def test_p107_self_selected_issuer_key_cannot_replace_pinned_root(self):
        verify = _verifier(self)
        _trusted_private, trusted_pub = _keys()
        attacker_private, attacker_pub = _keys()
        forged = _envelope(attacker_private)
        forged["claimed_issuer"] = "independent-runner"
        forged["embedded_public_key"] = base64.b64encode(
            attacker_pub
        ).decode("ascii")
        _must_reject(self, verify, forged, trusted_pub)

    def test_p108_wrong_audience_or_cross_source_rejects(self):
        verify = _verifier(self)
        private, public = _keys()
        for field, value in (
            ("audience", "dts.other.service"),
            ("source_commit", "f" * 40),
            ("case_id", "r1-adamw-accum1"),
            ("resume_ticket_sha256", "b" * 64),
        ):
            with self.subTest(field=field):
                p = _payload()
                p[field] = value
                _must_reject(self, verify, _envelope(private, p), public)

    def test_p109_expired_or_not_yet_valid_signed_proof_rejects(self):
        verify = _verifier(self)
        private, public = _keys()
        for changed in (
            {"expires_at_unix": NOW - 1},
            {"issued_at_unix": NOW + 1},
            {"issued_at_unix": NOW - 500, "expires_at_unix": NOW + 500},
        ):
            with self.subTest(changed=changed):
                p = _payload()
                p.update(changed)
                _must_reject(self, verify, _envelope(private, p), public)

    def test_p110_replay_of_once_verified_nonce_rejects(self):
        verify = _verifier(self)
        private, public = _keys()
        nonce_registry = set()
        original = _envelope(private)
        first = _verify(verify, original, public, seen=nonce_registry)
        self.assertEqual(first["status"], "signature_valid_unqualified")
        self.assertIn(original["payload"]["one_use_nonce"], nonce_registry)
        _must_reject(self, verify, original, public, seen=nonce_registry)

    def test_p111_wrong_phase_or_physical_step_rejects(self):
        verify = _verifier(self)
        private, public = _keys()
        for field, value in (
            ("phase", "resume"),
            ("logical_step", 2),
            ("proof_scope", "worker_complete"),
            ("checkpoint_file_sha256", {"optimizer.bin": "2" * 64}),
        ):
            with self.subTest(field=field):
                p = _payload()
                p[field] = value
                _must_reject(self, verify, _envelope(private, p), public)

    def test_p112_signature_cannot_trust_extra_fields_or_noncanonical_wire(self):
        verify = _verifier(self)
        private, public = _keys()
        base = _envelope(private)
        for fake in (
            {**base, "status": "physical_binding_valid"},
            {**base, "runner_public_key": "z" * 64},
            {**base, "worker_complete": True},
        ):
            with self.subTest(extra=list(set(fake) - set(base))):
                _must_reject(self, verify, fake, public)


if __name__ == "__main__":
    unittest.main()
