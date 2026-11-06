"""B2-A Linux kernel/OS preflight/read-only service: exact 12 Host tests.

A1 intentionally fails L104-L112 with CONTRACT ABSENT (3 baseline PASS).
A2-A4 must deliver 12 PASS using real Linux SO_PEERCRED. This is NOT
independently provisioned OS service authority or physical CUDA acceptance.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.sd3_full_bf16_independent_runner import (
    get_pinned_runner_trust_root,
    issue_provisional_fresh_step1_attestation,
)
from tools.sd3_full_bf16_runner_authority import (
    probe_runner_provisioning_security, open_protected_runner_signer,
)
from tools.sd3_full_bf16_runner_replay import RunnerNonceLedger


_MISSING = "C4-C2-P1-B2-C-B2-A LINUX CONTRACT ABSENT"
_LINUX_MODULE = "tools.sd3_full_bf16_runner_linux_boundary"
_SERVICE_MODULE = "tools.sd3_full_bf16_runner_linux_service"
_GENERIC_MODULE = "tools.sd3_full_bf16_runner_os_boundary"
_PROBE = {
    "schema": "dts.sd3c2b.c4c2b2a.linux-service-v1",
    "status": "unprovisioned",
    "os_security_qualified": False,
    "may_issue": False,
    "physical_cuda_qualified": False,
}


def _api(test: unittest.TestCase, name: str, *, service: bool = False):
    module_name = _SERVICE_MODULE if service else _LINUX_MODULE
    test.assertIsNotNone(
        importlib.util.find_spec(module_name),
        f"{_MISSING}: missing isolated {module_name}.{name}",
    )
    module = importlib.import_module(module_name)
    target = getattr(module, name, None)
    test.assertTrue(
        callable(target), f"{_MISSING}: missing {module_name}.{name}"
    )
    return target


class C4C2P1B2ALinuxHost(unittest.TestCase):
    def test_l101_baseline_generic_b1_os_module_remains_absent(self):
        self.assertIsNone(importlib.util.find_spec(_GENERIC_MODULE))

    def test_l102_baseline_signer_and_load_authority_still_closed(self):
        self.assertEqual(probe_runner_provisioning_security(), {
            "status": "unprovisioned",
            "may_issue": False,
            "physical_cuda_qualified": False,
        })
        for fn in (get_pinned_runner_trust_root,
                   issue_provisional_fresh_step1_attestation,
                   open_protected_runner_signer):
            with self.subTest(name=fn.__name__):
                with self.assertRaises(ValueError):
                    fn()

    def test_l103_baseline_test_ledger_is_not_protected_authority(self):
        with tempfile.TemporaryDirectory() as td:
            with RunnerNonceLedger(Path(td) / "test-nonce.sqlite3") as ledger:
                result = ledger.consume_once(
                    issuer_id="test-only",
                    audience="dts.sd3c2b.c4c2.resume-load",
                    ticket_sha256="a" * 64,
                    one_use_nonce="b" * 64,
                    token_id="c" * 64,
                    issued_at_unix=1700000000,
                    expires_at_unix=1700000060,
                    now_unix=1700000001,
                )
                self.assertEqual(result, {
                    "status": "nonce_consumed_unqualified",
                    "physical_cuda_qualified": False,
                })

    def test_l104_real_so_peercred_reports_actual_kernel_identity(self):
        read = _api(self, "read_linux_connected_peer_credentials")
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            peer = read(left)
            self.assertEqual((peer.pid, peer.uid, peer.gid),
                             (os.getpid(), os.geteuid(), os.getegid()))
            self.assertIs(type(peer.pid), int)
            self.assertFalse(hasattr(peer, "may_issue"))

    def test_l105_caller_supplied_claims_cannot_override_kernel_peer(self):
        read = _api(self, "read_linux_connected_peer_credentials")
        self.assertEqual(list(inspect.signature(read).parameters),
                         ["connected_socket"])
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            with self.assertRaises((ValueError, TypeError)):
                read(left, peer_uid=0, peer_pid=42)
            observed = read(left)
            self.assertEqual(observed.uid, os.geteuid())
            self.assertEqual(observed.pid, os.getpid())

    def test_l106_invalid_or_unconnected_socket_fails_closed(self):
        read = _api(self, "read_linux_connected_peer_credentials")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unconnected:
            with self.assertRaises(ValueError):
                read(unconnected)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as inet:
            with self.assertRaises(ValueError):
                read(inet)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as datagram:
            with self.assertRaises(ValueError):
                read(datagram)
        with self.assertRaises(ValueError):
            read(object())

    def test_l107_same_uid_connected_unix_peer_cannot_authorize(self):
        read = _api(self, "read_linux_connected_peer_credentials")
        auth = _api(self, "require_linux_authenticated_peer")
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            observed = read(left)
            self.assertEqual(observed.uid, os.geteuid())
            with self.assertRaises(ValueError):
                auth(left, claimed_uid=12345, claimed_pid=99999,
                     authorized=True)

    def test_l108_unprovisioned_service_identity_never_proves_auth(self):
        observe = _api(self, "observe_linux_service_identity")
        preflight = _api(self, "require_linux_protected_service_preflight")
        current = observe()
        self.assertEqual(current.euid, os.geteuid())
        self.assertEqual(current.egid, os.getegid())
        self.assertEqual(current.ruid, os.getuid())
        self.assertEqual(current.rgid, os.getgid())
        with self.assertRaises(ValueError):
            preflight()

    def test_l109_inherited_python_and_loader_env_rejected(self):
        check = _api(self, "require_linux_sanitized_service_startup")
        with patch.dict(os.environ, {
            "PYTHONPATH": "/tmp/worker-shadow",
            "PYTHONHOME": "/tmp/worker-sys",
            "PYTHONUSERBASE": "/tmp/worker-site",
            "LD_PRELOAD": "/tmp/worker-preload.so",
            "LD_LIBRARY_PATH": "/tmp/worker-native",
            "DTS_RUNNER_PRIVATE_KEY": "WORKER-CONTROLLED-SECRET",
        }):
            with self.assertRaises(ValueError):
                check()
        # Merely clearing inherited strings does not confer privileged status.
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                check()

    def test_l110_worker_writable_code_and_symlinks_rejected(self):
        audit = _api(self, "audit_linux_protected_code_path")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            user_module = root / "worker_module.py"
            user_module.write_text("trusted=True\n", encoding="utf-8")
            user_module.chmod(0o666)
            link = root / "worker_alias.py"
            link.symlink_to(user_module)
            with self.assertRaises(ValueError):
                audit(user_module)
            with self.assertRaises(ValueError):
                audit(link)

    def test_l111_worker_selected_key_root_and_ledger_not_trusted(self):
        req = _api(self, "require_linux_protected_credentials")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                req(
                    signing_key_path=str(Path(td) / "private-key"),
                    trust_root_path=str(Path(td) / "root.pub"),
                    ledger_path=str(Path(td) / "nonces.sqlite3"),
                    claimed_controller_uid=0,
                )
        with self.assertRaises(ValueError):
            req()

    def test_l112_service_probe_and_denied_only_diagnostic(self):
        probe = _api(self, "probe_linux_service", service=True)
        handler = _api(self, "handle_linux_diagnostic_connection", service=True)
        self.assertEqual(probe(), _PROBE)
        finished = subprocess.run(
            [sys.executable, "-m", _SERVICE_MODULE, "--probe-unprovisioned"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(json.loads(finished.stdout), _PROBE)
        with tempfile.TemporaryDirectory() as td:
            socket_path = Path(td) / "should-not-create.sock"
            denied = subprocess.run(
                [sys.executable, "-m", _SERVICE_MODULE,
                 "--serve", "--socket-path", str(socket_path)],
                capture_output=True, text=True, timeout=10, check=False,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertFalse(socket_path.exists())
        for request, expected_status in (
            (b'{"op":"status"}\n', "unprovisioned"),
            (b'{"op":"sign"}\n', "denied"),
        ):
            left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            with right:
                right.settimeout(2)
                right.sendall(request)
                handler(left)  # Service closes its accepted connection.
                received = right.recv(4096)
            report = json.loads(received)
            self.assertEqual(report["status"], expected_status)
            self.assertIs(report["may_issue"], False)
            self.assertIs(report["os_security_qualified"], False)
            self.assertIs(report["physical_cuda_qualified"], False)


if __name__ == "__main__":
    unittest.main()
