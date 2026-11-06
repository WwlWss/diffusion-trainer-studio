"""B2-B Windows native TokenUser / named pipe negative Tests-only RED.

Exactly 12 Windows Host tests: W101-W103 existing PASS; W104-W112 tagged
missing Windows source FAIL at B2-B1. No actual protected service or signer.
The real named-pipe tests use Win32 connected same-user handles, not mocks;
their success proves only kernel-token *observation*, NEVER service authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

from tools.sd3_full_bf16_runner_authority import (
    probe_runner_provisioning_security,
    open_protected_runner_signer,
)
from tools.sd3_full_bf16_independent_runner import (
    get_pinned_runner_trust_root,
    issue_provisional_fresh_step1_attestation,
)
from tools.sd3_full_bf16_runner_replay import RunnerNonceLedger


_MISSING = "C4-C2-P1-B2-C-B2-B WINDOWS CONTRACT ABSENT"
_WINDOWS = "tools.sd3_full_bf16_runner_windows_boundary"
_SERVICE = "tools.sd3_full_bf16_runner_windows_service"
_GENERIC = "tools.sd3_full_bf16_runner_os_boundary"
_PROBE = {
    "schema": "dts.sd3c2b.c4c2b2b.windows-service-v1",
    "status": "unprovisioned",
    "os_security_qualified": False,
    "may_issue": False,
    "physical_cuda_qualified": False,
}


def _api(test: unittest.TestCase, name: str, *, service: bool = False):
    module = _SERVICE if service else _WINDOWS
    test.assertIsNotNone(
        importlib.util.find_spec(module),
        f"{_MISSING}: missing Windows-only {module}.{name}",
    )
    fn = getattr(importlib.import_module(module), name, None)
    test.assertTrue(callable(fn), f"{_MISSING}: missing {module}.{name}")
    return fn


@contextmanager
def _connected_test_pipe(*, client_impersonation_level=0x00020000):
    """Test-only same-user Win32 pipe; controllable client impersonation QoS."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    H = wintypes.HANDLE
    kernel32.CreateNamedPipeW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    )
    kernel32.CreateNamedPipeW.restype = H
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, H,
    )
    kernel32.CreateFileW.restype = H
    kernel32.ConnectNamedPipe.argtypes = (H, ctypes.c_void_p)
    kernel32.ConnectNamedPipe.restype = wintypes.BOOL
    kernel32.WriteFile.argtypes = (
        H, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    )
    kernel32.WriteFile.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = (
        H, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    )
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (H,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    path = r"\\.\pipe\dts-windows-host-test-" + uuid.uuid4().hex
    invalid = ctypes.c_void_p(-1).value
    server = kernel32.CreateNamedPipeW(
        path, 0x00000003,  # duplex
        0x00000008,        # byte mode, reject remote clients
        1, 512, 512, 1000, None,
    )
    if int(server or 0) in (0, invalid):
        raise AssertionError("CreateNamedPipeW test fixture failed")
    client = None
    try:
        client = kernel32.CreateFileW(
            path, 0xC0000000, 0, None, 3,
            0x00100000 | client_impersonation_level, None,  # explicit client QoS
        )
        if int(client or 0) in (0, invalid):
            raise AssertionError("CreateFileW test fixture failed")
        if not kernel32.ConnectNamedPipe(server, None):
            if ctypes.get_last_error() != 535:  # ERROR_PIPE_CONNECTED
                raise AssertionError("ConnectNamedPipe test fixture failed")
        payload = ctypes.create_string_buffer(b"whoami\n")
        written = wintypes.DWORD(0)
        if not kernel32.WriteFile(
            client, payload, len(payload.value), ctypes.byref(written), None
        ):
            raise AssertionError("WriteFile test fixture failed")
        request = ctypes.create_string_buffer(128)
        nread = wintypes.DWORD(0)
        if not kernel32.ReadFile(
            server, request, 128, ctypes.byref(nread), None
        ):
            raise AssertionError("ReadFile test fixture failed")
        if request.raw[:nread.value] != b"whoami\n":
            raise AssertionError("ReadFile test fixture wrong bytes")
        yield int(server), int(client)
    finally:
        if client not in (None, 0, invalid):
            kernel32.CloseHandle(client)
        kernel32.CloseHandle(server)


class C4C2P1B2BWindowsHost(unittest.TestCase):
    def test_w101_baseline_cross_platform_os_aggregate_remains_absent(self):
        self.assertIsNone(importlib.util.find_spec(_GENERIC))

    def test_w102_baseline_runner_and_authority_unprovisioned(self):
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

    def test_w103_baseline_test_nonce_ledger_not_signer(self):
        with tempfile.TemporaryDirectory() as td:
            with RunnerNonceLedger(Path(td) / "test.sqlite3") as ledger:
                result = ledger.consume_once(
                    issuer_id="test-only", audience="dts.sd3c2b.c4c2.resume-load",
                    ticket_sha256="a" * 64, one_use_nonce="b" * 64,
                    token_id="c" * 64, issued_at_unix=1700000000,
                    expires_at_unix=1700000060, now_unix=1700000001,
                )
                self.assertEqual(result, {
                    "status": "nonce_consumed_unqualified",
                    "physical_cuda_qualified": False,
                })

    def test_w104_real_win32_process_token_user_sid(self):
        read = _api(self, "observe_windows_process_token_user")
        observed = read()
        self.assertRegex(observed.sid, r"^S-1-\d+")
        self.assertIs(type(observed.sid), str)
        self.assertFalse(hasattr(observed, "may_issue"))
        self.assertEqual(observed.provenance, "windows_process_token_unqualified")

    def test_w105_worker_supplied_sid_pid_cannot_override_native_token(self):
        read = _api(self, "observe_windows_process_token_user")
        self.assertEqual(list(inspect.signature(read).parameters), [])
        current = read()
        with self.assertRaises((TypeError, ValueError)):
            read(claimed_sid="S-1-5-18", claimed_pid=os.getpid() + 55)
        self.assertEqual(read().sid, current.sid)

    def test_w106_real_named_pipe_impersonation_token_user_sid(self):
        read = _api(self, "observe_windows_named_pipe_client_token")
        process = _api(self, "observe_windows_process_token_user")
        with _connected_test_pipe() as (server, _client):
            observed = read(server)
        self.assertEqual(observed.sid, process().sid)
        self.assertEqual(observed.provenance, "windows_pipe_client_token_unqualified")
        self.assertFalse(hasattr(observed, "may_issue"))

    def test_w107_real_same_sid_client_never_authorized(self):
        authorize = _api(self, "require_windows_authenticated_peer")
        with _connected_test_pipe() as (server, _client):
            with self.assertRaises(ValueError):
                authorize(server, claimed_sid="S-1-5-18",
                          alleged_service_process_id=123, authorized=True)

    def test_w108_invalid_fake_and_unread_pipe_refused(self):
        read = _api(self, "observe_windows_named_pipe_client_token")
        for fake in (0, -1, None, object(), {"handle": 12345}):
            with self.subTest(fake_type=type(fake).__name__):
                with self.assertRaises(ValueError):
                    read(fake)
        with self.assertRaises(ValueError):
            read(0x7FFFFFFF)

    def test_w109_poisoned_or_clean_startup_never_grants_authority(self):
        fn = _api(self, "require_windows_sanitized_service_startup")
        with patch.dict(os.environ, {
            "PYTHONPATH": "WORKER_SHADOW", "PYTHONHOME": "WORKER_SITE",
            "PYTHONUSERBASE": "WORKER_USER_SITE",
            "DTS_RUNNER_PRIVATE_KEY": "SHOULD_NOT_LOG",
            "DTS_RUNNER_TRUST_ROOT": "WORKER_ROOT",
        }):
            with self.assertRaises(ValueError):
                fn()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                fn()

    def test_w110_worker_writable_executable_not_protected_acl(self):
        audit = _api(self, "audit_windows_protected_code_path")
        with tempfile.TemporaryDirectory() as td:
            code = Path(td) / "worker_entry.py"
            code.write_text("raise RuntimeError('worker owned')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit(code)
            with self.assertRaises(ValueError):
                audit(Path(td) / "nonexistent.py")

    def test_w111_worker_chosen_key_root_ledger_not_protected(self):
        fn = _api(self, "require_windows_protected_credentials")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                fn(private_key=str(Path(td) / "test-key"),
                   root_path=str(Path(td) / "test-root"),
                   ledger_path=str(Path(td) / "test-ledger"),
                   claimed_dacl={"protected": True, "owner": "SYSTEM"})
        with self.assertRaises(ValueError):
            fn()

    def test_w112_service_unprovisioned_and_diagnostic_only(self):
        probe = _api(self, "probe_windows_service", service=True)
        denied = _api(self, "deny_windows_diagnostic_request", service=True)
        self.assertEqual(probe(), _PROBE)
        done = subprocess.run(
            [sys.executable, "-m", _SERVICE, "--probe-unprovisioned"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), _PROBE)
        with tempfile.TemporaryDirectory() as td:
            name = "dts-windows-should-not-listen-" + uuid.uuid4().hex
            command = subprocess.run(
                [sys.executable, "-m", _SERVICE, "--serve",
                 "--pipe-name", name, "--key-path", str(Path(td) / "key")],
                capture_output=True, text=True, timeout=10, check=False,
            )
            self.assertNotEqual(command.returncode, 0)
        for request, status in (
            (b'{"op":"status"}\n', "unprovisioned"),
            (b'{"op":"sign"}\n', "denied"),
            (b'{"op":"authorize_resume"}\n', "denied"),
            (b'{"op":"status","trusted":true}\n', "denied"),
            (b"x" * 512, "denied"),
        ):
            with self.subTest(data=request[:48]):
                report = denied(request)
                self.assertEqual(report["status"], status)
                self.assertIs(report["os_security_qualified"], False)
                self.assertIs(report["may_issue"], False)
                self.assertIs(report["physical_cuda_qualified"], False)


if __name__ == "__main__":
    unittest.main()
