"""C4-C2-P1-B2-C-A-SR1: independent Host regression for fail-closed authority.

SCOPED HOST tests ONLY. Never provision a signer, protected OS principal,
privileged IPC or physical CUDA Fresh Step-1. The current production authority
rejects all eight operations while unprovisioned.

Frozen Tests-only RED expected at parent df7d359f:
  SR101-SR106 PASS (existing correct behavior)
  SR107-SR108 targeted FAIL (mutable diagnostic and stale contract)
"""
from __future__ import annotations

import builtins
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import unittest
from unittest.mock import patch

from tools import sd3_full_bf16_runner_authority as authority


_MISSING = "C4-C2-P1-B2-C-SR1 CONTRACT MISMATCH"
_EIGHT = (
    "require_distinct_runner_security_principal",
    "load_protected_runner_configuration",
    "authenticate_runner_ipc_peer",
    "open_protected_runner_signer",
    "open_protected_runner_ledger",
    "verify_production_runner_proof",
    "authorize_provisional_fresh_step1_issue",
    "inspect_runner_credential_acl",
)
_PROBE = {
    "status": "unprovisioned",
    "may_issue": False,
    "physical_cuda_qualified": False,
}


class _HostileObject:
    def __getattribute__(self, name):
        raise AssertionError("CALLER_OBJECT_ATTR_ACCESSED")
    def __str__(self):
        raise AssertionError("CALLER_OBJECT_STRINGIFIED")
    def __repr__(self):
        raise AssertionError("CALLER_OBJECT_REPR_CALLED")
    def __iter__(self):
        raise AssertionError("CALLER_OBJECT_ITERATED")
    def __bool__(self):
        raise AssertionError("CALLER_OBJECT_BOOL_CALLED")
    def __fspath__(self):
        raise AssertionError("CALLER_OBJECT_PATH_CALLED")


class C4C2P1B2CSr1Host(unittest.TestCase):
    def test_sr101_all_eight_authority_apis_refuse_zero_argument_calls(self):
        for name in _EIGHT:
            with self.subTest(name=name):
                function = getattr(authority, name)
                with self.assertRaisesRegex(
                    ValueError, "requires independent OS provisioning"
                ):
                    function()

    def test_sr102_alternative_argument_forms_stay_denied(self):
        for name in _EIGHT:
            function = getattr(authority, name)
            with self.subTest(name=name, kind="positionals"):
                with self.assertRaises(ValueError):
                    function("runner", 42, None)
            with self.subTest(name=name, kind="kwargs"):
                with self.assertRaises(ValueError):
                    function(
                        issuer="worker-controlled",
                        proof={"status": "physical_binding_valid"},
                        allow_sign=True, trusted_root="/tmp/not-trusted",
                        claimed_uid=1, claimed_sid="S-1-5-18",
                    )

    def test_sr103_hostile_object_is_not_inspected_before_denial(self):
        evil = _HostileObject()
        for name in _EIGHT:
            function = getattr(authority, name)
            with self.subTest(name=name, kind="positional"):
                with self.assertRaises(ValueError):
                    function(evil)
            with self.subTest(name=name, kind="keyword"):
                with self.assertRaises(ValueError):
                    function(peer=evil, file_path=evil, status=evil)

    def test_sr104_rejects_do_not_log_or_echo_secret_markers(self):
        marker = "SR1-CANNOT-LEAK-PRIVATE-KEY-" + "Z" * 40
        for name in _EIGHT:
            function = getattr(authority, name)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.subTest(name=name):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(ValueError) as denied:
                        function(marker, private_key=marker, key_path=marker)
                self.assertNotIn(marker, str(denied.exception))
                self.assertNotIn(marker, stdout.getvalue())
                self.assertNotIn(marker, stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), "")

    def test_sr105_denial_occurs_without_file_sqlite_ipc_or_subprocess_io(self):
        # Imports and patch setup occur before blocking any I/O.
        calls = (
            patch.object(builtins, "open", side_effect=AssertionError("FILE_IO")),
            patch.object(os, "open", side_effect=AssertionError("OS_FILE_IO")),
            patch.object(Path, "open", side_effect=AssertionError("PATH_IO")),
            patch.object(sqlite3, "connect", side_effect=AssertionError("DB_IO")),
            patch.object(socket, "socket", side_effect=AssertionError("IPC_IO")),
            patch.object(subprocess, "Popen", side_effect=AssertionError("PROCESS_IO")),
        )
        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in calls]
            for name in _EIGHT:
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        getattr(authority, name)(source="/tmp/untrusted")
            for mock in mocks:
                mock.assert_not_called()

    def test_sr106_worker_environment_pollution_does_not_enable_any_gate(self):
        with patch.dict(os.environ, {
            "DTS_RUNNER_PRIVATE_KEY": "WORKER_MARKER_SIGNING_KEY",
            "DTS_RUNNER_SIGNING_SEED": "WORKER_MARKER_SIGNING_SEED",
            "DTS_RUNNER_TRUST_ROOT": "WORKER_MARKER_TRUST_ROOT",
            "DTS_RUNNER_LEDGER_PATH": "WORKER_MARKER_LEDGER",
            "DTS_RUNNER_ISSUER": "WORKER_FORGED_ISSUER",
            "PYTHONPATH": "/tmp/worker-controlled-imports",
            "PYTHONHOME": "/tmp/worker-controlled-install",
        }):
            for name in _EIGHT:
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        getattr(authority, name)()
            self.assertEqual(authority.probe_runner_provisioning_security(), _PROBE)

    def test_sr107_probe_ignores_spoofed_legacy_module_status(self):
        # Parent mutable constant must fail; literal source after fix passes.
        with patch.object(
            authority, "_UNPROVISIONED_STATUS", "worker-forged-ready", create=True
        ):
            report = authority.probe_runner_provisioning_security()
        self.assertEqual(
            report, _PROBE,
            f"{_MISSING}: diagnostic must not read mutable legacy status",
        )

    def test_sr108_contract_tracks_current_green_and_freezes_protected_install(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs/d1-sd3c2b-c4c-p1b2c-protected-issuer-contract.md"
        )
        document = path.read_text(encoding="utf-8")
        for anchor in (
            "**Current stage:** P1-B2-C-A-SR1 scoped Host fail-closed",
            "## Historical P1-B2-C Tests-only RED baseline",
            "## P1-B2-C-B: protected executable and import integrity",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(
                    anchor, document,
                    f"{_MISSING}: current contract lacks {anchor}",
                )


if __name__ == "__main__":
    unittest.main()
