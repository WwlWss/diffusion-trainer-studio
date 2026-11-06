"""DTS C4-C2-P1-B2-C-B2-B: denied-only Windows Runner service shell.

NO CreateNamedPipeW listener, SCM service account, private key, sign/Load or
CUDA path exists. --serve fails before any key/pipe/open or Worker path use.
Host-only bounded diagnostics never confer protected service authority.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from tools.sd3_full_bf16_runner_windows_boundary import (
    require_windows_protected_service_preflight,
)


_SCHEMA = "dts.sd3c2b.c4c2b2b.windows-service-v1"
_MAX_REQUEST_BYTES = 256
_STATUS_REQUEST = b'{"op":"status"}\n'


def probe_windows_service() -> dict[str, Any]:
    """Unsigned public status only, not an OS identity or trusted receipt."""
    return {
        "schema": _SCHEMA,
        "status": "unprovisioned",
        "os_security_qualified": False,
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


def deny_windows_diagnostic_request(payload: bytes) -> dict[str, Any]:
    """Offline formatter; never a network/pipe listener or signing handler."""
    return {
        "schema": _SCHEMA,
        "status": (
            "unprovisioned"
            if type(payload) is bytes
            and len(payload) <= _MAX_REQUEST_BYTES
            and payload == _STATUS_REQUEST
            else "denied"
        ),
        "os_security_qualified": False,
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--probe-unprovisioned"]:
        print(json.dumps(
            probe_windows_service(),
            separators=(",", ":"), sort_keys=True, ensure_ascii=True,
        ))
        return 0
    if args and args[0] == "--serve":
        try:
            # No command-line/Worker socket path is interpreted before the
            # independently managed protected Windows service preflight.
            require_windows_protected_service_preflight()
        except ValueError:
            print("Windows protected Runner service is not provisioned",
                  file=sys.stderr)
            return 2
        # No listener or signer even if a future preflight accidentally
        # becomes permissive; opening these requires a separate review.
        return 2
    print("Windows Runner: unsupported or unauthorized command",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
