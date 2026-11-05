"""DTS C4-C2-P1-B2-C-A: unprovisioned production Runner authority boundary.

SCOPED HOST FAIL-CLOSED GREEN ONLY. There is currently no provisioned
OS-separated service principal, kernel-authenticated IPC, protected signer,
pinned production trust root, or protected replay ledger.

None of these names grants authority. An arbitrary Worker/caller can supply
Python dicts, signatures, public keys, paths, fake peer credentials, file
metadata, environment variables and test-only SQLite ledgers. No such data
can enable a production issuer or authorize original Accelerator.load_state.

Real OS-protected lifecycle, positive service-principal integration tests,
and physical Fresh Step-1 verification are separate later stages. This
module must NOT be called a production signer merely because Host tests pass.
"""
from __future__ import annotations

from typing import Any, NoReturn


# Constant status is intentionally independent of environment, CLI options,
# Python module globals, Ticket, manifest, Worker configuration and file paths.
# No key generation, signing backend, filesystem or credential loading exists.
_UNPROVISIONED_STATUS = "unprovisioned"


def _deny_unprovisioned(operation: str) -> NoReturn:
    """Raise without interpolating caller input or exposing key material."""
    raise ValueError(
        "C4-C2-P1-B2-C protected Runner authority unavailable: "
        + operation + " requires independent OS provisioning"
    )


def require_distinct_runner_security_principal(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """A caller's UID/SID/PID maps cannot attest a distinct OS principal."""
    _deny_unprovisioned("distinct Runner principal")


def load_protected_runner_configuration(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """Never accept Worker env/CLI paths or synthesize default credentials."""
    _deny_unprovisioned("protected service configuration")


def authenticate_runner_ipc_peer(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """Do not trust peer PID/SID/UID dictionaries or claimed named pipes."""
    _deny_unprovisioned("kernel-authenticated IPC peer")


def open_protected_runner_signer(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """No signer is provisioned; injected keys and key paths are untrusted."""
    _deny_unprovisioned("protected signing key")


def open_protected_runner_ledger(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """A caller-selected Host SQLite ledger is not production authority."""
    _deny_unprovisioned("protected one-use replay ledger")


def verify_production_runner_proof(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """A TEST-signed envelope and caller key cannot grant Load admission."""
    _deny_unprovisioned("production verifier trust root and replay gate")


def authorize_provisional_fresh_step1_issue(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """Worker candidate data cannot mint protected Fresh Step-1 provenance."""
    _deny_unprovisioned("independently verified physical parent authority")


def inspect_runner_credential_acl(
    *args: Any, **kwargs: Any
) -> NoReturn:
    """Self-declared owner/mode/DACL/reparse flags cannot prove key safety."""
    _deny_unprovisioned("kernel-verified credential ownership and ACL")


def probe_runner_provisioning_security() -> dict[str, Any]:
    """Read-only unprovisioned diagnostic. NOT a signed authority receipt."""
    return {
        "status": _UNPROVISIONED_STATUS,
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


__all__ = [
    "require_distinct_runner_security_principal",
    "load_protected_runner_configuration",
    "authenticate_runner_ipc_peer",
    "open_protected_runner_signer",
    "open_protected_runner_ledger",
    "verify_production_runner_proof",
    "authorize_provisional_fresh_step1_issue",
    "inspect_runner_credential_acl",
    "probe_runner_provisioning_security",
]
