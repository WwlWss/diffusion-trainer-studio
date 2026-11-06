"""DTS C4-C2-P1-B2-C-B2-A: Linux kernel observations and denied preflight.

UNQUALIFIED HOST MECHANISM ONLY. This module never authorizes a service,
attestation, signing key, trusted root, replay ledger, CUDA or original Load.
SO_PEERCRED is a genuine kernel observation for connected AF_UNIX sockets;
it is NOT an authenticated protected service or permission to sign.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import stat
import struct
import sys
from typing import Any, NoReturn


_UCRED = struct.Struct("=3i")
_UNSAFE_INHERITED = frozenset({
    "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE",
    "PYTHONSTARTUP", "PYTHONINSPECT",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
    "DTS_RUNNER_PRIVATE_KEY", "DTS_RUNNER_SIGNING_SEED",
    "DTS_RUNNER_TRUST_ROOT", "DTS_RUNNER_LEDGER_PATH",
    "DTS_RUNNER_KEY_PATH", "DTS_RUNNER_ISSUER",
})


def _deny(reason: str) -> NoReturn:
    # Fixed reason names; never interpolate Worker data, key or path.
    raise ValueError("C4-C2 B2-A Linux unqualified: " + reason)


def _linux_required() -> None:
    if not sys.platform.startswith("linux"):
        _deny("Linux kernel interface unavailable")


@dataclass(frozen=True, slots=True)
class LinuxPeerCredentials:
    """Connection-time kernel peer observation; NEVER signing authority."""
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True, slots=True)
class LinuxProcessIdentity:
    """Actual local process identity observed from OS; NOT service proof."""
    ruid: int
    euid: int
    suid: int
    rgid: int
    egid: int
    sgid: int
    groups: tuple[int, ...]


def read_linux_connected_peer_credentials(
    connected_socket: socket.socket,
) -> LinuxPeerCredentials:
    """Read real SO_PEERCRED from a connected AF_UNIX stream socket only."""
    _linux_required()
    if not isinstance(connected_socket, socket.socket):
        _deny("native socket handle required")
    if (connected_socket.family != socket.AF_UNIX
            or (connected_socket.type & socket.SOCK_STREAM)
            != socket.SOCK_STREAM):
        _deny("connected AF_UNIX stream socket required")
    try:
        # A connected socketpair can return an empty address; no string
        # address or caller-chosen socket path is used as peer evidence.
        connected_socket.getpeername()
        raw = connected_socket.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size,
        )
    except (OSError, AttributeError, TypeError) as exc:
        _deny("kernel peer observation failed: " + type(exc).__name__)
    if len(raw) != _UCRED.size:
        _deny("invalid kernel peer credential length")
    pid, uid, gid = _UCRED.unpack(raw)
    if pid <= 0 or uid < 0 or gid < 0:
        _deny("invalid kernel peer credential values")
    return LinuxPeerCredentials(pid=pid, uid=uid, gid=gid)


def observe_linux_service_identity() -> LinuxProcessIdentity:
    """Observe current real/effective/saved IDs; not service-manager origin."""
    _linux_required()
    try:
        ruid, euid, suid = os.getresuid()
        rgid, egid, sgid = os.getresgid()
        groups = tuple(os.getgroups())
    except (OSError, AttributeError) as exc:
        _deny("process identity unavailable: " + type(exc).__name__)
    return LinuxProcessIdentity(
        ruid=ruid, euid=euid, suid=suid,
        rgid=rgid, egid=egid, sgid=sgid, groups=groups,
    )


def require_linux_authenticated_peer(
    connected_socket: socket.socket,
    **worker_claims: Any,
) -> NoReturn:
    """Read true peer IDs but NEVER authorize absent protected service policy.

    Worker claims have no effect; this function has no affirmative branch.
    Future positive admission must require an independently reviewed
    service-only protected policy and actual distinct OS account.
    """
    peer = read_linux_connected_peer_credentials(connected_socket)
    service = observe_linux_service_identity()
    if peer.uid == service.euid:
        _deny("same-UID peer is not a separate protected service principal")
    _deny("externally pinned service-side peer policy not provisioned")


def require_linux_protected_service_preflight(
    *args: Any, **kwargs: Any,
) -> NoReturn:
    """No out-of-band controller configuration exists; ALWAYS deny startup."""
    _linux_required()
    _deny("independently managed service principal and install unavailable")


def require_linux_sanitized_service_startup(
    *args: Any, **kwargs: Any,
) -> NoReturn:
    """No Worker inherited env can create a privileged Runner service."""
    _linux_required()
    contaminated = sorted(name for name in _UNSAFE_INHERITED if name in os.environ)
    if contaminated:
        # Only fixed known variable names are safe to report; not values.
        _deny("inherited startup environment is untrusted")
    _deny("independent service-manager provenance not provisioned")


def audit_linux_protected_code_path(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Diagnostic conservative path-shape scan, NOT a race-safe authority.

    Reject Worker-owned and group/other-writable code/ancestors, symlinks,
    unsafe targets and explicit aliases. Even an all-root-owned path remains
    unqualified because ACL, mount namespace and TOCTOU were not proven.
    """
    _linux_required()
    try:
        raw = os.fspath(path)
    except (TypeError, ValueError):
        _deny("invalid code path")
    if not isinstance(raw, str) or not raw.startswith("/") or "\x00" in raw:
        _deny("absolute code path required")
    if any(part in (".", "..", "") for part in raw.split("/")[1:-1]):
        _deny("path alias or repeated separator")
    if raw.endswith("/") or raw in ("/", ""):
        _deny("regular code file required")
    p = Path(raw)
    # lstat each component; do NOT follow symlink shortcuts via Path.resolve.
    components = list(reversed(p.parents)) + [p]
    for i, component in enumerate(components):
        try:
            info = component.lstat()
        except (OSError, ValueError):
            _deny("code component could not be inspected")
        if stat.S_ISLNK(info.st_mode):
            _deny("symbolic link in code path")
        if info.st_uid != 0 or info.st_mode & 0o022:
            _deny("code path not protected from Worker mutation")
        if i < len(components) - 1:
            if not stat.S_ISDIR(info.st_mode):
                _deny("code ancestor not a directory")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _deny("code entry is not one regular file")
    return {
        "status": "path_shape_checked_unqualified",
        "os_security_qualified": False,
        "may_issue": False,
        "physical_cuda_qualified": False,
    }


def require_linux_protected_credentials(
    *args: Any, **kwargs: Any,
) -> NoReturn:
    """Injected key/root/ledger paths are not service provisioning."""
    _linux_required()
    _deny("signing key, pinned root and replay ledger not protected")


__all__ = [
    "LinuxPeerCredentials", "LinuxProcessIdentity",
    "read_linux_connected_peer_credentials",
    "observe_linux_service_identity",
    "require_linux_authenticated_peer",
    "require_linux_protected_service_preflight",
    "require_linux_sanitized_service_startup",
    "audit_linux_protected_code_path",
    "require_linux_protected_credentials",
]
