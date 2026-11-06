"""B2-B A5: native Windows low-impersonation / thread-revert Strict Review.

S201 initially RED: SecurityIdentification client level is insufficient
for a privileged local service impersonation boundary. Observe only actual
TokenImpersonation with native impersonation level >= SecurityImpersonation.
S202-S204 frozen existing-safe baseline PASS, without production signing.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_sd3_full_bf16_worker_c4c_p1b2cb_windows_host import (
    _connected_test_pipe,
)
from tools.sd3_full_bf16_runner_windows_boundary import (
    observe_windows_named_pipe_client_token,
    observe_windows_process_token_user,
)
from tools.sd3_full_bf16_runner_windows_service import (
    deny_windows_diagnostic_request,
    probe_windows_service,
)


_MARKER = "B2-B-A5 LOW_IMPERSONATION_LEVEL MUST BE REJECTED"


class C4C2B2BWindowsStrict(unittest.TestCase):
    def test_s201_client_security_identification_level_must_not_pass(self):
        # Windows SECURITY_IDENTIFICATION: SID readable, but no normal
        # local impersonation privilege. Do not accept this as an
        # impersonated client identity eligible for future IPC admission.
        with _connected_test_pipe(
            client_impersonation_level=0x00010000
        ) as (server, _client):
            try:
                observed = observe_windows_named_pipe_client_token(server)
            except ValueError:
                return
            self.fail(
                f"{_MARKER}: accepted {observed.provenance!r}"
            )

    def test_s202_valid_client_token_observation_restores_native_thread(self):
        with _connected_test_pipe() as (server, _client):
            observed = observe_windows_named_pipe_client_token(server)
        self.assertEqual(observed.sid, observe_windows_process_token_user().sid)
        # Check actual calling thread: no leaked impersonation token.
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi.OpenThreadToken.argtypes = (
            wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi.OpenThreadToken.restype = wintypes.BOOL
        kernel.GetCurrentThread.argtypes = ()
        kernel.GetCurrentThread.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        token = wintypes.HANDLE()
        got = advapi.OpenThreadToken(
            kernel.GetCurrentThread(), 0x0008, True, ctypes.byref(token)
        )
        if got:
            kernel.CloseHandle(token)
        self.assertFalse(bool(got), "impersonation token leaked after return")
        self.assertEqual(ctypes.get_last_error(), 1008)  # ERROR_NO_TOKEN

    def test_s203_client_end_and_invalid_pipe_handles_not_trusted(self):
        with _connected_test_pipe() as (_server, client):
            with self.assertRaises(ValueError):
                observe_windows_named_pipe_client_token(client)
        for invalid in (0, -1, 0x7FFFFFFF, object(), {"handle": 1}):
            with self.subTest(input_type=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    observe_windows_named_pipe_client_token(invalid)

    def test_s204_unprovisioned_command_and_untrusted_payload_stay_denied(self):
        marker = "B2-B-SHOULD-NEVER-ECHO-SIGNER-KEY"
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "tools.sd3_full_bf16_runner_windows_service",
                    "--serve", "--pipe-name", "worker-pipe",
                    "--private-key", marker,
                ],
                capture_output=True, text=True, timeout=10, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(marker, result.stdout + result.stderr)
        self.assertIs(probe_windows_service()["may_issue"], False)
        for payload in (
            b'{"op":"sign"}\n',
            b'{"op":"status","trusted":true}\n',
            marker.encode(),
            b"x" * 1024,
        ):
            with self.subTest(payload=payload[:30]):
                response = deny_windows_diagnostic_request(payload)
                self.assertEqual(response["status"], "denied")
                self.assertFalse(response["may_issue"])
                self.assertFalse(response["os_security_qualified"])
                self.assertFalse(response["physical_cuda_qualified"])
                self.assertNotIn(marker, json.dumps(response))


if __name__ == "__main__":
    unittest.main()
