"""C4-C2-P1-A: pure Host Ed25519 proof signature MECHANISM only.

This module does not sign, issue, authenticate a physical Fresh parent, or
authorize Accelerator.load_state. pinned_public_key and seen_nonces are
INJECTED MECHANISM-TEST inputs, never a production Worker trust root or a
durable replay store. A verified signature is explicitly UNQUALIFIED.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tools import sd3_full_bf16_evidence_support as host


_DOMAIN = b"DTS-C4C2-P1-ED25519-V1\x00"
_PAYLOAD_SCHEMA = "dts.sd3c2b.c4c2p1.provisional-step1-v1"
_ENVELOPE_SCHEMA = "dts.sd3c2b.c4c2p1.signed-provisional-step1-v1"
_AUDIENCE = "dts.sd3c2b.c4c2.resume-load"
_PROOF_SCOPE = "provisional_fresh_step1_only"
_MAX_LIFETIME = 60
_MAX_CANONICAL_BYTES = 16384
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CASES = frozenset((
    "r1-adamw-accum1", "r2-muon-adamw-fallback-accum2",
))
_ENVELOPE_FIELDS = frozenset((
    "schema", "payload", "signature_ed25519",
))
_PAYLOAD_FIELDS = frozenset((
    "schema", "version", "proof_scope", "audience", "case_id",
    "phase", "logical_step", "run_id", "fresh_phase_id",
    "resume_phase_id", "source_commit", "case_manifest_sha256",
    "input_identity_sha256", "reference_policy_sha256",
    "reference_target_spec_sha256", "resume_ticket_sha256",
    "resume_ticket_nonce", "checkpoint_id", "checkpoint_dir",
    "checkpoint_file_sha256", "adapter_path", "adapter_file_sha256",
    "physical_step_anchor_sha256", "checkpoint_event_prefix_sha256",
    "issued_at_unix", "expires_at_unix", "one_use_nonce",
))
_CONTEXT_FIELDS = frozenset((
    "audience", "case_id", "run_id", "fresh_phase_id",
    "resume_phase_id", "source_commit", "checkpoint_id",
    "resume_ticket_sha256", "resume_ticket_nonce",
    "case_manifest_sha256", "input_identity_sha256",
    "reference_policy_sha256", "reference_target_spec_sha256",
))
_HEX64_FIELDS = frozenset((
    "case_manifest_sha256", "input_identity_sha256",
    "reference_policy_sha256", "reference_target_spec_sha256",
    "resume_ticket_sha256", "resume_ticket_nonce",
    "adapter_file_sha256", "physical_step_anchor_sha256",
    "checkpoint_event_prefix_sha256", "one_use_nonce",
))
_ID_FIELDS = frozenset(("run_id", "fresh_phase_id", "resume_phase_id"))


def _deny(reason: str) -> None:
    raise ValueError("C4-C2-P1 unqualified signed proof: " + reason)


def _valid_text(value: Any, *, field: str, limit: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > limit:
        _deny("invalid text field " + field)
    try:
        value.encode("utf-8", "strict")
    except UnicodeError:
        _deny("invalid unicode field " + field)
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        _deny("control character in " + field)
    return value


def _hex(value: Any, *, field: str, size: int = 64) -> str:
    v = _valid_text(value, field=field, limit=size)
    expression = {32: _HEX32, 40: _HEX40, 64: _HEX64}[size]
    if expression.fullmatch(v) is None:
        _deny("invalid lowercase digest " + field)
    return v


def _external_path(value: Any, *, field: str) -> str:
    path = _valid_text(value, field=field)
    if not (PurePosixPath(path).is_absolute()
            or PureWindowsPath(path).is_absolute()):
        _deny("non-absolute path " + field)
    # Raw segments matter: PurePath would normalize dot aliases away.
    parts = re.split(r"[/\\]", path)
    if any(part in (".", "..") for part in parts):
        _deny("noncanonical path component " + field)
    return path


def _canonical_payload(payload: Any) -> bytes:
    if type(payload) is not dict or frozenset(payload) != _PAYLOAD_FIELDS:
        _deny("extra/missing provisional Step-1 fields")
    if (payload["schema"] != _PAYLOAD_SCHEMA
            or type(payload["version"]) is not int
            or payload["version"] != 1
            or payload["proof_scope"] != _PROOF_SCOPE
            or payload["audience"] != _AUDIENCE
            or payload["phase"] != "fresh"
            or type(payload["logical_step"]) is not int
            or payload["logical_step"] != 1
            or payload["case_id"] not in _CASES):
        _deny("wrong schema/audience/scope/Case/physical Step")
    for name in _ID_FIELDS:
        v = _valid_text(payload[name], field=name, limit=128)
        if _ID.fullmatch(v) is None:
            _deny("invalid run/Phase identity " + name)
    _hex(payload["source_commit"], field="source_commit", size=40)
    _hex(payload["checkpoint_id"], field="checkpoint_id", size=32)
    for name in _HEX64_FIELDS:
        _hex(payload[name], field=name)
    _external_path(payload["checkpoint_dir"], field="checkpoint_dir")
    adapter_path = _external_path(payload["adapter_path"], field="adapter_path")
    if not adapter_path.endswith(".safetensors"):
        _deny("external original LoRA path is not safetensors")
    inventory = payload["checkpoint_file_sha256"]
    if (type(inventory) is not dict
            or frozenset(inventory) != host._SD3_STATE_FILES
            or len(inventory) != 9):
        _deny("invalid exact original nine-file inventory")
    for name, digest in inventory.items():
        _hex(digest, field="checkpoint_file_sha256." + name)
    issued = payload["issued_at_unix"]
    expires = payload["expires_at_unix"]
    if (type(issued) is not int or type(expires) is not int
            or issued < 0 or expires <= issued
            or expires - issued > _MAX_LIFETIME):
        _deny("invalid provisional lifetime")
    try:
        canonical = host._canonical_json_bytes(payload)
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        _deny("noncanonical signed payload: " + type(exc).__name__)
    if len(canonical) > _MAX_CANONICAL_BYTES:
        _deny("oversized signed payload")
    return canonical


def verify_provisional_step1_signature(
    *,
    envelope: Mapping[str, Any],
    pinned_public_key: bytes,
    expected_context: Mapping[str, Any],
    now_unix: int,
    seen_nonces: set[str],
) -> dict[str, Any]:
    """Verify a TEST-injected Ed25519 signature; NEVER qualify a parent.

    A real production caller must obtain a pinned issuer key and an atomic,
    durable replay ledger outside the Worker and its writable work_root.
    This pure function has no such provisioning, so no returned result
    is an Accelerator Load capability or a valid CUDA checkpoint.
    """
    if type(envelope) is not dict or frozenset(envelope) != _ENVELOPE_FIELDS:
        _deny("extra/missing envelope fields or unparsed raw JSON")
    if envelope["schema"] != _ENVELOPE_SCHEMA:
        _deny("invalid signed envelope schema")
    if type(pinned_public_key) is not bytes or len(pinned_public_key) != 32:
        _deny("no exact pre-pinned Ed25519 public key")
    if (type(expected_context) is not dict
            or frozenset(expected_context) != _CONTEXT_FIELDS):
        _deny("invalid exact trusted context")
    if type(now_unix) is not int or now_unix < 0:
        _deny("invalid independent wall-clock")
    if type(seen_nonces) is not set:
        _deny("invalid nonce registry")
    payload = envelope["payload"]
    canonical = _canonical_payload(payload)
    for name in _CONTEXT_FIELDS:
        if type(expected_context[name]) is not str:
            _deny("untyped expected context " + name)
        if payload[name] != expected_context[name]:
            _deny("source/Ticket/Case/audience context mismatch " + name)
    issued = payload["issued_at_unix"]
    expires = payload["expires_at_unix"]
    if issued > now_unix or expires <= now_unix:
        _deny("provisional signature expired or not yet issued")
    signature_text = _valid_text(
        envelope["signature_ed25519"], field="signature_ed25519", limit=88,
    )
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except (ValueError, binascii.Error):
        _deny("noncanonical Ed25519 signature base64")
    if (len(signature) != 64
            or base64.b64encode(signature).decode("ascii") != signature_text):
        _deny("invalid Ed25519 signature wire encoding")
    try:
        Ed25519PublicKey.from_public_bytes(pinned_public_key).verify(
            signature, _DOMAIN + canonical,
        )
    except (InvalidSignature, ValueError, TypeError):
        _deny("wrong signed content or independent key")
    nonce = payload["one_use_nonce"]
    if nonce in seen_nonces:
        _deny("duplicate provisional signature nonce")
    # Test-only, in-process mechanism. NOT durable/atomic across processes.
    seen_nonces.add(nonce)
    return {
        "status": "signature_valid_unqualified",
        "physical_cuda_qualified": False,
    }


__all__ = ["verify_provisional_step1_signature"]
