"""C4-C2-P1-B: fail-closed Runner EXTERNAL-PRINCIPAL boundary (NO CUDA).

This executable is intentionally unprovisioned. A different Python process,
alone, has no independent signing authority. OS-protected root/issuer,
independent physical source and protected IPC must be reviewed separately.

No TEST key is embedded, inherited Worker environment cannot provision
authority, and no signature or Load qualification can be minted here.
"""
from __future__ import annotations

import json
import sys
from typing import Any


_PROBE_SCHEMA = "dts.sd3c2b.c4c2p1.runner-probe-v1"


def get_pinned_runner_trust_root() -> bytes:
    """Require a future controller-owned, OS-protected root (absent now).

    Do not read env/CLI/Ticket/work_root/repository/Worker-selected paths.
    This function may be opened only with a separately reviewed provisioner.
    """
    raise ValueError(
        "C4-C2-P1-B Runner trust root not externally provisioned"
    )


def verify_original_provisional_step1_source(*args: Any, **kwargs: Any) -> None:
    """No independently observed nine-file/789-BF16 evidence exists yet."""
    raise ValueError(
        "C4-C2-P1-C physical Fresh Step-1 source not independently verified"
    )


def issue_provisional_fresh_step1_attestation(
    *args: Any, **kwargs: Any
) -> None:
    """Never sign a Worker-provided dict or a self-presented candidate."""
    raise ValueError(
        "C4-C2-P1-B protected independent Runner issuer not provisioned"
    )


def _probe_unprovisioned() -> dict[str, Any]:
    return {
        "schema": _PROBE_SCHEMA,
        "status": "unprovisioned",
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["--probe-unprovisioned"]:
        print("C4-C2-P1-B Runner: no production issuer command provisioned",
              file=sys.stderr)
        return 2
    print(json.dumps(_probe_unprovisioned(), sort_keys=True,
                     separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
