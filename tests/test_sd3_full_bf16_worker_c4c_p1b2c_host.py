"""C4-C2-P1-B2-C: production Runner security principal Tests-only RED.

Unprivileged two-OS Host checks only. No OS privilege is forged, no private
production key is supplied, no physical Fresh parent or CUDA accepted.
The nine missing authority APIs must fail as tagged CONTRACT ABSENT, rather
than import ERROR, TypeError, or an unrelated authorization failure.
"""
from __future__ import annotations

from dataclasses import fields
import importlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_sd3_full_bf16_worker_c4c_p1_host import (
    NOW, _context, _envelope, _keys, _payload,
)
from tools.sd3_full_bf16_evidence_worker import WorkerInvocation
from tools.sd3_full_bf16_independent_runner import (
    _probe_unprovisioned, get_pinned_runner_trust_root,
    issue_provisional_fresh_step1_attestation,
)
from tools.sd3_full_bf16_runner_replay import RunnerNonceLedger
from tools.sd3_full_bf16_runner_trust import verify_provisional_step1_signature


_MISSING = "C4-C2-P1-B2-C HOST CONTRACT ABSENT"
_AUTHORITY = "tools.sd3_full_bf16_runner_authority"
_AUDIENCE = "dts.sd3c2b.c4c2.resume-load"


def _api(test: unittest.TestCase, name: str):
    spec = importlib.util.find_spec(_AUTHORITY)
    test.assertIsNotNone(
        spec,
        f"{_MISSING}: no OS-protected production authority source "
        f"for {_AUTHORITY}.{name}",
    )
    module = importlib.import_module(_AUTHORITY)
    function = getattr(module, name, None)
    test.assertTrue(
        callable(function),
        f"{_MISSING}: no reviewed production authority entrypoint "
        f"{_AUTHORITY}.{name}",
    )
    return function


class C4C2P1B2CProtectedIssuerHostRED(unittest.TestCase):
    def test_c201_baseline_test_signature_remains_unqualified(self):
        private, public = _keys()
        result = verify_provisional_step1_signature(
            envelope=_envelope(private),
            pinned_public_key=public,
            expected_context=_context(),
            now_unix=NOW,
            seen_nonces=set(),
        )
        self.assertEqual(result, {
            "status": "signature_valid_unqualified",
            "physical_cuda_qualified": False,
        })

    def test_c202_baseline_durable_test_nonce_does_not_become_authority(self):
        with tempfile.TemporaryDirectory() as td:
            with RunnerNonceLedger(Path(td) / "test-only.sqlite3") as ledger:
                claim = {
                    "issuer_id": "runner-test-only",
                    "audience": _AUDIENCE,
                    "ticket_sha256": "a" * 64,
                    "one_use_nonce": "b" * 64,
                    "token_id": "c" * 64,
                    "issued_at_unix": NOW - 5,
                    "expires_at_unix": NOW + 55,
                    "now_unix": NOW,
                }
                first = ledger.consume_once(**claim)
                self.assertEqual(first, {
                    "status": "nonce_consumed_unqualified",
                    "physical_cuda_qualified": False,
                })
                with self.assertRaises(ValueError):
                    ledger.consume_once(**claim)

    def test_c203_baseline_unprovisioned_runner_and_worker_still_closed(self):
        self.assertEqual(_probe_unprovisioned(), {
            "schema": "dts.sd3c2b.c4c2p1.runner-probe-v1",
            "status": "unprovisioned",
            "may_issue": False,
            "physical_cuda_qualified": False,
        })
        with self.assertRaises(ValueError):
            get_pinned_runner_trust_root()
        with self.assertRaises(ValueError):
            issue_provisional_fresh_step1_attestation(_payload())
        forbidden = {"runner_private_key", "signing_seed", "issuer_secret"}
        self.assertFalse(forbidden & {f.name for f in fields(WorkerInvocation)})

    def test_c204_claimed_uid_sid_pid_cannot_prove_distinct_os_principal(self):
        check = _api(self, "require_distinct_runner_security_principal")
        with self.assertRaises(ValueError):
            check(
                worker_identity={"pid": 111, "uid": 1000, "sid": "worker"},
                runner_identity={"pid": 222, "uid": 1001, "sid": "runner"},
            )
        with self.assertRaises(ValueError):
            check(
                worker_identity={"pid": os.getpid(), "uid": 999},
                runner_identity={"pid": os.getpid() + 1, "uid": 1000},
            )

    def test_c205_worker_inherited_key_root_issuer_env_cannot_provision(self):
        load = _api(self, "load_protected_runner_configuration")
        marker = "P1B2C-WORKER-CONTROLLED-SECRET-MUST-FAIL"
        with patch.dict(os.environ, {
            "DTS_RUNNER_PRIVATE_KEY": marker,
            "DTS_RUNNER_SIGNING_SEED": marker,
            "DTS_RUNNER_PUBLIC_KEY": "f" * 64,
            "DTS_RUNNER_PUBLIC_KEY_PATH": marker,
            "DTS_RUNNER_TRUST_ROOT": marker,
            "DTS_RUNNER_ISSUER": "test-issuer-not-authority",
            "DTS_RUNNER_LEDGER_PATH": marker,
        }):
            with self.assertRaises(ValueError):
                load()

    def test_c206_self_declared_ipc_peer_is_not_an_os_authenticated_peer(self):
        authenticate = _api(self, "authenticate_runner_ipc_peer")
        with self.assertRaises(ValueError):
            authenticate(transport={
                "pid": os.getpid(),
                "claimed_uid": 99,
                "claimed_sid": "S-1-5-18",
                "named_pipe": r"\\.\pipe\fake-runner",
                "authenticated": True,
            })

    def test_c207_worker_private_key_and_selected_path_cannot_open_signer(self):
        signer = _api(self, "open_protected_runner_signer")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                signer(
                    private_key=b"x" * 32,  # hostile synthetic bytes, NOT a key
                    key_path=str(Path(td) / "worker-own-key.bin"),
                )

    def test_c208_worker_selected_ledger_cannot_be_production_ledger(self):
        open_ledger = _api(self, "open_protected_runner_ledger")
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "worker-test-ledger.sqlite3"
            with RunnerNonceLedger(db) as ledger:
                with self.assertRaises(ValueError):
                    open_ledger(
                        ledger=ledger, ledger_path=str(db), work_root=td,
                    )

    def test_c209_test_signature_and_caller_key_not_production_proof(self):
        verify = _api(self, "verify_production_runner_proof")
        private, public = _keys()
        with self.assertRaises(ValueError):
            verify(
                envelope=_envelope(private),
                pinned_public_key=public,
                expected_context=_context(),
                now_unix=NOW,
                seen_nonces=set(),
            )

    def test_c210_worker_claimed_physical_parent_never_mints_sign_capability(self):
        authorize = _api(self, "authorize_provisional_fresh_step1_issue")
        with self.assertRaises(ValueError):
            authorize(
                worker_candidate={
                    "status": "physical_binding_valid",
                    "payload": _payload(),
                    "runner_pid": os.getpid() + 1,
                    "source": "worker",
                },
                runner_label="trusted-independent-runner",
            )

    def test_c211_self_reported_file_acl_ownership_cannot_protect_secret(self):
        inspect = _api(self, "inspect_runner_credential_acl")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                inspect(
                    metadata={
                        "owner": "SYSTEM",
                        "mode": "0600",
                        "dacl_protected": True,
                        "reparse_free": True,
                    },
                    key_path=str(Path(td) / "fake-protected-key"),
                )

    def test_c212_unprivileged_ci_must_not_report_real_issuer_ready(self):
        probe = _api(self, "probe_runner_provisioning_security")
        status = probe()
        self.assertEqual(status, {
            "status": "unprovisioned",
            "may_issue": False,
            "physical_cuda_qualified": False,
        })


if __name__ == "__main__":
    unittest.main()
