"""DTS C4-C2-P1-B2-C-B2-A: unprovisioned Linux read-only service shell.

This program has NO signing service. Its --serve entrypoint deliberately
rejects BEFORE any socket bind/listen, disk write or credential lookup until
real independently provisioned service-manager trust is implemented/reviewed.

The public socketpair diagnostic handler is for bounded HOST negative tests.
It never creates a listener, authenticates a protected service or confers
signing, parent checkpoint, CUDA or original Load authority.
"""
from __future__ import annotations

import json
import socket
import sys
from typing import Any

from tools.sd3_full_bf16_runner_linux_boundary import (
    read_linux_connected_peer_credentials,
    require_linux_protected_service_preflight,
)


_SCHEMA = "dts.sd3c2b.c4c2b2a.linux-service-v1"
_MAX_REQUEST_BYTES = 256


def probe_linux_service() -> dict[str, Any]:
    """Untrusted public probe. Never a protected service entitlement."""
    return {
        "schema": _SCHEMA,
        "status": "unprovisioned",
        "os_security_qualified": False,
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


def _denied_result(*, is_status_request: bool) -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "status": "unprovisioned" if is_status_request else "denied",
        "os_security_qualified": False,
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


def handle_linux_diagnostic_connection(
    connected_socket: socket.socket,
) -> None:
    """One bounded read-only, **never authorized** AF_UNIX diagnostic.

    A Python caller possessing its own socketpair can call this function,
    therefore its return/reply is explicitly **not** a service attestation.
    Unrecognized and signing requests are always denied, never executed.
    """
    try:
        read_linux_connected_peer_credentials(connected_socket)
        connected_socket.settimeout(2.0)
        try:
            request = connected_socket.recv(_MAX_REQUEST_BYTES + 1)
        except (OSError, TimeoutError):
            request = b""
        reply = _denied_result(
            is_status_request=request == b'{"op":"status"}\n'
        )
        payload = (json.dumps(
            reply, separators=(",", ":"), sort_keys=True, ensure_ascii=True
        ) + "\n").encode("ascii")
        connected_socket.sendall(payload)
    finally:
        # The caller owns no post-response authorization or live handle.
        connected_socket.close()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--probe-unprovisioned"]:
        print(json.dumps(probe_linux_service(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True))
        return 0
    if args and args[0] == "--serve":
        try:
            # IMPORTANT: check before even parsing a Worker-proposed socket
            # path, importing a signer, constructing a listener or mkdir.
            require_linux_protected_service_preflight()
        except ValueError:
            print("Linux protected Runner service is not provisioned",
                  file=sys.stderr)
            return 2
        # Unreachable until independently reviewed OS privilege deployment.
        return 2
    print("Linux diagnostic service: unsupported or unauthorized command",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
