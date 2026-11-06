"""Independent B2-A-A5 strict kernel-identity and denied-service regression.

One defect-specific Tests-only RED at parent 89934635:
S201 must reject a spoofed subclass that overrides getsockopt/getpeername.
The genuine socketpair and denied-only service baselines must still PASS.
No privileged OS service / key / CUDA can be qualified by these tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

from tools.sd3_full_bf16_runner_linux_boundary import (
    read_linux_connected_peer_credentials,
)
from tools.sd3_full_bf16_runner_linux_service import (
    handle_linux_diagnostic_connection, probe_linux_service,
)


_MARKER = "B2-A-A5 FORGED_SOCKET_CLASS MUST BE REJECTED"


class _ForgedKernelCredentialSocket(socket.socket):
    """Unconnected native socket with Python-overridden system-call methods."""

    def __init__(self):
        super().__init__(socket.AF_UNIX, socket.SOCK_STREAM)

    def getpeername(self):
        return "/tmp/claimed-connected-socket"

    def getsockopt(self, *args):
        return struct.pack("=3i", 12345, 0, 0)


class C4C2B2ALinuxStrictReview(unittest.TestCase):
    def test_s201_forged_socket_subclass_cannot_supply_kernel_identity(self):
        with _ForgedKernelCredentialSocket() as fabricated:
            try:
                observed = read_linux_connected_peer_credentials(fabricated)
            except ValueError:
                return
            self.fail(
                f"{_MARKER}: observed={observed!r} despite fake getsockopt"
            )

    def test_s202_actual_native_socketpair_still_reports_real_identity(self):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            identity = read_linux_connected_peer_credentials(left)
            self.assertEqual(
                (identity.pid, identity.uid, identity.gid),
                (os.getpid(), os.geteuid(), os.getegid()),
            )
            self.assertFalse(hasattr(identity, "physical_cuda_qualified"))

    def test_s203_malformed_and_oversized_requests_only_denied(self):
        self.assertIs(probe_linux_service()["may_issue"], False)
        for data in (
            b'{"op":"status","issuer":"worker"}\n',
            b'{"op":"authorize_resume"}\n',
            b'{"op":"sign"}\n',
            b"Z" * 512,
            b"",
        ):
            with self.subTest(kind=data[:36]):
                left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                with right:
                    right.settimeout(3)
                    if data:
                        right.sendall(data)
                    handle_linux_diagnostic_connection(left)
                    report = json.loads(right.recv(2048))
                self.assertEqual(report["status"], "denied")
                self.assertIs(report["os_security_qualified"], False)
                self.assertIs(report["may_issue"], False)
                self.assertIs(report["physical_cuda_qualified"], False)

    def test_s204_unprovisioned_serve_refuses_before_socket_creation(self):
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "never-create.sock"
            secret = "NOT-TRUSTED-PRIVATE-KEY-DO-NOT-PRINT"
            finished = subprocess.run(
                [
                    sys.executable, "-m",
                    "tools.sd3_full_bf16_runner_linux_service",
                    "--serve", "--socket-path", str(sock_path),
                    "--private-key", secret,
                ],
                capture_output=True, text=True, timeout=10, check=False,
            )
            self.assertNotEqual(finished.returncode, 0)
            self.assertFalse(sock_path.exists())
            self.assertNotIn(secret, finished.stdout + finished.stderr)


if __name__ == "__main__":
    unittest.main()
