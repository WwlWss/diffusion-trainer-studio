"""DTS C4-C2-P1-B2-C-B2-B: real Win32 token observations; deny-only.

A Win32 TokenUser returned by the kernel is an UNQUALIFIED observation:
the same-user CI process is not a protected SCM service principal.
Nothing here provides a signing key, root, replay ledger, policy,
Checkpoint Load, CUDA qualification, or worker_complete permission.
All protected service authorization APIs have *no positive branch*.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import stat
import sys
from typing import Any, NoReturn


_TOKEN_QUERY = 0x0008
_TOKEN_USER_CLASS = 1
_TOKEN_TYPE_CLASS = 8
_TOKEN_IMPERSONATION_LEVEL_CLASS = 9
_TOKEN_IMPERSONATION_TYPE = 2
_SECURITY_IMPERSONATION_LEVEL = 2
_ERROR_INSUFFICIENT_BUFFER = 122
_FILE_TYPE_PIPE = 3
_PIPE_SERVER_END = 0x00000001
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_UNTRUSTED_ENV = frozenset({
    "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE",
    "PYTHONSTARTUP", "PYTHONINSPECT",
    "DTS_RUNNER_PRIVATE_KEY", "DTS_RUNNER_SIGNING_SEED",
    "DTS_RUNNER_TRUST_ROOT", "DTS_RUNNER_LEDGER_PATH",
    "DTS_RUNNER_KEY_PATH", "DTS_RUNNER_ISSUER",
})


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", wintypes.DWORD),
    ]


class _TOKEN_USER(ctypes.Structure):
    _fields_ = [("User", _SID_AND_ATTRIBUTES)]


@dataclass(frozen=True, slots=True)
class WindowsTokenObservation:
    """Native token SID text; neither an ACL check nor a signer capability."""
    sid: str
    provenance: str


def _deny(reason: str) -> NoReturn:
    # Only static reason codes, no interpolated Worker input or SID.
    raise ValueError("C4-C2 B2-B Windows unqualified: " + reason)


def _windows_required() -> None:
    if sys.platform != "win32":
        _deny("native Windows token interface unavailable")


def _native():
    """Configure real Win32 function signatures; no injected API objects."""
    _windows_required()
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    a = ctypes.WinDLL("advapi32", use_last_error=True)
    H, D = wintypes.HANDLE, wintypes.DWORD

    k.GetCurrentProcess.argtypes = ()
    k.GetCurrentProcess.restype = H
    k.GetCurrentThread.argtypes = ()
    k.GetCurrentThread.restype = H
    k.CloseHandle.argtypes = (H,)
    k.CloseHandle.restype = wintypes.BOOL
    k.GetFileType.argtypes = (H,)
    k.GetFileType.restype = D
    k.GetNamedPipeInfo.argtypes = (
        H, ctypes.POINTER(D), ctypes.POINTER(D),
        ctypes.POINTER(D), ctypes.POINTER(D),
    )
    k.GetNamedPipeInfo.restype = wintypes.BOOL
    k.LocalFree.argtypes = (ctypes.c_void_p,)
    k.LocalFree.restype = ctypes.c_void_p

    a.OpenProcessToken.argtypes = (H, D, ctypes.POINTER(H))
    a.OpenProcessToken.restype = wintypes.BOOL
    a.ImpersonateNamedPipeClient.argtypes = (H,)
    a.ImpersonateNamedPipeClient.restype = wintypes.BOOL
    a.OpenThreadToken.argtypes = (H, D, wintypes.BOOL, ctypes.POINTER(H))
    a.OpenThreadToken.restype = wintypes.BOOL
    a.GetTokenInformation.argtypes = (
        H, D, ctypes.c_void_p, D, ctypes.POINTER(D),
    )
    a.GetTokenInformation.restype = wintypes.BOOL
    a.IsValidSid.argtypes = (ctypes.c_void_p,)
    a.IsValidSid.restype = wintypes.BOOL
    a.ConvertSidToStringSidW.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p),
    )
    a.ConvertSidToStringSidW.restype = wintypes.BOOL
    a.RevertToSelf.argtypes = ()
    a.RevertToSelf.restype = wintypes.BOOL
    return k, a


def _sid_from_native_token(k, a, handle) -> str:
    """Read one actual TokenUser; free Win32-local SID string in all cases."""
    size = wintypes.DWORD(0)
    a.GetTokenInformation(handle, _TOKEN_USER_CLASS, None, 0, ctypes.byref(size))
    if (ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER
            or size.value < ctypes.sizeof(_TOKEN_USER)
            or size.value > 65536):
        _deny("TokenUser buffer sizing failed")
    data = ctypes.create_string_buffer(size.value)
    if not a.GetTokenInformation(
        handle, _TOKEN_USER_CLASS, data, size.value, ctypes.byref(size)
    ):
        _deny("TokenUser read failed")
    user = ctypes.cast(data, ctypes.POINTER(_TOKEN_USER)).contents
    if not user.User.Sid or not a.IsValidSid(user.User.Sid):
        _deny("invalid native TokenUser SID")
    native_string = ctypes.c_wchar_p()
    if not a.ConvertSidToStringSidW(user.User.Sid, ctypes.byref(native_string)):
        _deny("TokenUser SID conversion failed")
    try:
        value = native_string.value
        if not value or not value.startswith("S-1-"):
            _deny("invalid native SID format")
        return value
    finally:
        if native_string:
            k.LocalFree(ctypes.cast(native_string, ctypes.c_void_p))


def observe_windows_process_token_user() -> WindowsTokenObservation:
    """Real current-process TokenUser, not a service-manager identity proof."""
    k, a = _native()
    token = wintypes.HANDLE()
    if not a.OpenProcessToken(
        k.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
    ):
        _deny("current process token unavailable")
    try:
        sid = _sid_from_native_token(k, a, token)
    finally:
        k.CloseHandle(token)
    return WindowsTokenObservation(
        sid=sid, provenance="windows_process_token_unqualified"
    )


def observe_windows_named_pipe_client_token(
    server_pipe_handle: int,
) -> WindowsTokenObservation:
    """Impersonate *last read* native pipe client and revert unconditionally.

    The server must call ReadFile on a connected server pipe first. No
    caller-supplied claimed SID/pipe name/PID is accepted. An observation
    returned here is NOT authorization to sign or grant Load access.
    """
    _windows_required()
    if (type(server_pipe_handle) is not int
            or server_pipe_handle <= 0
            or server_pipe_handle == _INVALID_HANDLE):
        _deny("exact native server pipe HANDLE integer required")
    k, a = _native()
    handle = wintypes.HANDLE(server_pipe_handle)
    if k.GetFileType(handle) != _FILE_TYPE_PIPE:
        _deny("HANDLE not a native pipe")
    flags = wintypes.DWORD()
    if not k.GetNamedPipeInfo(handle, ctypes.byref(flags), None, None, None):
        _deny("named-pipe server handle unavailable")
    if not flags.value & _PIPE_SERVER_END:
        _deny("only actual named-pipe server handles can observe clients")
    # The OS ties the impersonation to the last message read by the server.
    # A client process ID, claimed user SID or pipe name is never enough.
    if not a.ImpersonateNamedPipeClient(handle):
        _deny("native client impersonation failed")
    try:
        token = wintypes.HANDLE()
        if not a.OpenThreadToken(
            k.GetCurrentThread(), _TOKEN_QUERY, True, ctypes.byref(token)
        ):
            _deny("impersonated client thread token unavailable")
        try:
            # A low-privilege SecurityIdentification token can reveal SID
            # text without being sufficient for local impersonation. Require
            # actual TokenImpersonation and SecurityImpersonation+ before
            # reporting this as a genuine authenticated client observation.
            token_type = wintypes.DWORD()
            level = wintypes.DWORD()
            obtained = wintypes.DWORD()
            if not a.GetTokenInformation(
                token, _TOKEN_TYPE_CLASS, ctypes.byref(token_type),
                ctypes.sizeof(token_type), ctypes.byref(obtained)
            ) or token_type.value != _TOKEN_IMPERSONATION_TYPE:
                _deny("native client token is not an impersonation token")
            if not a.GetTokenInformation(
                token, _TOKEN_IMPERSONATION_LEVEL_CLASS, ctypes.byref(level),
                ctypes.sizeof(level), ctypes.byref(obtained)
            ) or level.value < _SECURITY_IMPERSONATION_LEVEL:
                _deny("client impersonation level too low")
            sid = _sid_from_native_token(k, a, token)
        finally:
            k.CloseHandle(token)
    finally:
        if not a.RevertToSelf():
            # A future privileged service must terminate this worker thread
            # or whole process on an unrecoverable native revert failure.
            # Here, no protected service is ever provisioned or started.
            _deny("fatal failure to revert impersonation")
    return WindowsTokenObservation(
        sid=sid, provenance="windows_pipe_client_token_unqualified"
    )


def require_windows_authenticated_peer(
    server_pipe_handle: int, **worker_claims: Any,
) -> NoReturn:
    """Observe real peer, then ALWAYS refuse absent protected service policy."""
    peer = observe_windows_named_pipe_client_token(server_pipe_handle)
    process = observe_windows_process_token_user()
    if peer.sid == process.sid:
        _deny("same-SID client is not a distinct service authority")
    _deny("protected independently provisioned peer policy unavailable")


def require_windows_protected_service_preflight(
    *args: Any, **kwargs: Any,
) -> NoReturn:
    """No SCM-protected separate service exists. ALWAYS deny startup."""
    _windows_required()
    _deny("independently managed Windows service principal unavailable")


def require_windows_sanitized_service_startup(
    *args: Any, **kwargs: Any,
) -> NoReturn:
    """Even a clean inherited environment cannot prove SCM service origin."""
    _windows_required()
    if any(key in os.environ for key in _UNTRUSTED_ENV):
        _deny("inherited Python or Runner configuration not trusted")
    _deny("protected Windows SCM service startup not provisioned")


def audit_windows_protected_code_path(
    path: str | os.PathLike[str],
) -> NoReturn:
    """Read-only negative source preflight, NEVER proof of effective DACL."""
    _windows_required()
    try:
        raw = os.fspath(path)
    except (TypeError, ValueError):
        _deny("invalid source path")
    if type(raw) is not str or not raw or "\x00" in raw:
        _deny("untrusted source path")
    p = Path(raw)
    if not p.is_absolute() or ".." in p.parts:
        _deny("untrusted or nonabsolute code path")
    try:
        info = p.lstat()
    except (OSError, ValueError):
        _deny("source cannot be opened as protected")
    if (stat.S_ISLNK(info.st_mode)
            or (getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))):
        _deny("reparse or symlink code path")
    # Even a regular file's existence/digest/owner is NOT a DACL proof.
    _deny("independently protected binary/DLL/ACL deployment unavailable")


def require_windows_protected_credentials(
    *args: Any, **kwargs: Any,
) -> NoReturn:
    _windows_required()
    _deny("protected signer key, pinned verifier root and ledger unavailable")


__all__ = [
    "WindowsTokenObservation",
    "observe_windows_process_token_user",
    "observe_windows_named_pipe_client_token",
    "require_windows_authenticated_peer",
    "require_windows_protected_service_preflight",
    "require_windows_sanitized_service_startup",
    "audit_windows_protected_code_path",
    "require_windows_protected_credentials",
]
