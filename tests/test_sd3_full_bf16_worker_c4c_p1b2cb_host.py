"""C4-C2-P1-B2-C-B B1: OS-protected Runner boundary Tests-only RED.

Host-only two-OS negative contracts. A different PID, user-declared SID/UID,
fake IPC transport, injected signer/root or same-account SQLite DB cannot
confer protected production Runner authority. No signing or CUDA capability.

B1 frozen RED: O301-O303 3 baseline PASS; O304-O312 9 tagged missing
source FAIL; 0 ERROR/SKIP/xfail/xpass. Genuine protected integration is later.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.sd3_full_bf16_independent_runner import (
    get_pinned_runner_trust_root,
    issue_provisional_fresh_step1_attestation,
    verify_original_provisional_step1_source,
)
from tools.sd3_full_bf16_runner_authority import (
    probe_runner_provisioning_security,
    require_distinct_runner_security_principal,
    open_protected_runner_signer,
    verify_production_runner_proof,
)
from tools.sd3_full_bf16_runner_replay import RunnerNonceLedger


_OS_MODULE = "tools.sd3_full_bf16_runner_os_boundary"
_MISSING = "C4-C2-P1-B2-C-B OS BOUNDARY CONTRACT ABSENT"
_PROBE = {
    "status": "unprovisioned",
    "os_security_qualified": False,
    "may_issue": False,
    "physical_cuda_qualified": False,
}


def _api(test: unittest.TestCase, function_name: str):
    spec = importlib.util.find_spec(_OS_MODULE)
    test.assertIsNotNone(
        spec, f"{_MISSING}: missing separately reviewed source {_OS_MODULE}",
    )
    module = importlib.import_module(_OS_MODULE)
    fn = getattr(module, function_name, None)
    test.assertTrue(
        callable(fn), f"{_MISSING}: missing {function_name} entrypoint"
    )
    return fn


class C4C2P1B2CBProtectedOsBoundaryRED(unittest.TestCase):
    def test_o301_baseline_production_authority_remains_unprovisioned(self):
        self.assertEqual(probe_runner_provisioning_security(), {
            "status": "unprovisioned",
            "may_issue": False,
            "physical_cuda_qualified": False,
        })
        with self.assertRaises(ValueError):
            require_distinct_runner_security_principal()
        with self.assertRaises(ValueError):
            open_protected_runner_signer()
        with self.assertRaises(ValueError):
            verify_production_runner_proof()

    def test_o302_baseline_independent_runner_cannot_sign_or_load(self):
        for fn in (
            get_pinned_runner_trust_root,
            issue_provisional_fresh_step1_attestation,
            verify_original_provisional_step1_source,
        ):
            with self.subTest(function=fn.__name__):
                with self.assertRaises(ValueError):
                    fn()

    def test_o303_baseline_injected_nonce_ledger_stays_unqualified(self):
        with tempfile.TemporaryDirectory() as td:
            with RunnerNonceLedger(Path(td) / "test-only.sqlite3") as ledger:
                args = {
                    "issuer_id": "runner-host-test",
                    "audience": "dts.sd3c2b.c4c2.resume-load",
                    "ticket_sha256": "a" * 64,
                    "one_use_nonce": "b" * 64,
                    "token_id": "c" * 64,
                    "issued_at_unix": 1700000000,
                    "expires_at_unix": 1700000060,
                    "now_unix": 1700000005,
                }
                claimed = ledger.consume_once(**args)
                self.assertEqual(claimed, {
                    "status": "nonce_consumed_unqualified",
                    "physical_cuda_qualified": False,
                })
                with self.assertRaises(ValueError):
                    ledger.consume_once(**args)

    def test_o304_fake_principal_and_pid_must_not_establish_os_separation(self):
        fn = _api(self, "require_protected_service_principal")
        with self.assertRaises(ValueError):
            fn(
                worker_identity={"uid": 1000, "sid": "S-1-5-21-WORKER", "pid": 100},
                runner_identity={"uid": 1001, "sid": "S-1-5-18", "pid": 200},
            )
        with self.assertRaises(ValueError):
            fn()

    def test_o305_worker_selected_executable_and_hash_not_trusted(self):
        fn = _api(self, "require_protected_executable_origin")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "user-supplied-runner.py"
            path.write_text("pass\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                fn(runtime_path=sys.executable, source_path=str(path),
                   expected_sha256="f" * 64, claimed_owner="TrustedInstaller")
        with self.assertRaises(ValueError):
            fn()

    def test_o306_inherited_launch_environment_cannot_provision_service(self):
        fn = _api(self, "require_sanitized_service_startup")
        with patch.dict(os.environ, {
            "PYTHONPATH": "WORKER_SHADOW_PYTHONPATH",
            "PYTHONHOME": "WORKER_SHADOW_PYTHONHOME",
            "PYTHONUSERBASE": "WORKER_SHADOW_USERSITE",
            "LD_PRELOAD": "WORKER_SHADOW_NATIVE",
            "LD_LIBRARY_PATH": "WORKER_SHADOW_NATIVE_PATH",
            "PATH": "WORKER_SHADOW_DLL_PATH",
            "DTS_RUNNER_PRIVATE_KEY": "WORKER_SHADOW_KEY",
        }):
            with self.assertRaises(ValueError):
                fn()
            with self.assertRaises(ValueError):
                fn(service_manager={"authenticated": True},
                   import_path="WORKER_SHADOW_PYTHONPATH")

    def test_o307_named_pipe_or_unix_path_without_real_handle_not_server_proof(self):
        fn = _api(self, "require_authenticated_ipc_service")
        with self.assertRaises(ValueError):
            fn(transport={"socket_path": "/run/trusted.sock",
                          "pipe_name": r"\\.\pipe\dts-protected-runner",
                          "service_pid": 200, "authenticated": True})

    def test_o308_kernel_peer_required_and_same_user_socket_not_accepted(self):
        fn = _api(self, "require_kernel_authenticated_peer")
        if sys.platform.startswith("linux"):
            # Real AF_UNIX kernel credentials; same current process UID is
            # never an independent protected Runner service principal.
            left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            with left, right:
                with self.assertRaises(ValueError):
                    fn(connected_socket=left, claimed_peer={
                        "uid": 1001, "pid": 9999, "authorized": True,
                    })
        else:
            # Windows Host CI is deliberately not a privileged named-pipe
            # service. A fabricated dict or a fake handle MUST be refused.
            with self.assertRaises(ValueError):
                fn(connected_pipe=object(), claimed_sid="S-1-5-18")

    def test_o309_worker_supplied_private_bytes_or_key_path_not_service_key(self):
        fn = _api(self, "require_service_owned_signing_key")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                fn(private_key=b"x" * 32,
                   key_path=str(Path(td) / "worker-owned-private-key"))
        with self.assertRaises(ValueError):
            fn()

    def test_o310_worker_selected_public_ed25519_key_not_pinned_root(self):
        fn = _api(self, "require_protected_verifier_root")
        with tempfile.TemporaryDirectory() as td:
            root_path = Path(td) / "worker-root.pub"
            root_path.write_bytes(b"x" * 32)
            with self.assertRaises(ValueError):
                fn(pinned_public_key=b"x" * 32, root_path=str(root_path),
                   claimed_pinned=True)

    def test_o311_worker_sqlite_ledger_does_not_confer_service_authority(self):
        fn = _api(self, "require_service_owned_replay_ledger")
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "worker-nonces.sqlite3"
            with RunnerNonceLedger(db) as ledger:
                with self.assertRaises(ValueError):
                    fn(ledger=ledger, ledger_path=str(db), work_root=td)

    def test_o312_unprovisioned_os_probe_must_not_claim_privileged_service(self):
        fn = _api(self, "probe_runner_os_protection")
        self.assertEqual(
            fn(), _PROBE,
            "unprovisioned Host cannot advertise protected OS service or CUDA",
        )


if __name__ == "__main__":
    unittest.main()
