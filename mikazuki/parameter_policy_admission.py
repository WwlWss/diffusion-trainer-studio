"""First-party *host-only* QG-1B3-A Component Full-BF16 admission adapter.

This module evaluates a validated DTS Parameter Policy and its *effective*
configuration.  It does NOT change Preview/Export/Start/Direct Trainer gates,
validate file assets, authorize a launch, mint real-GPU evidence, or implement
the per-job GUI/CLI acknowledgement transport.  A positive host result only
means that a later, independently audited runtime check may be attempted.

The new QG-1B readiness producer owns implementation capability and hard
errors.  Pinned, exact source evidence identifiers only preserve existing
D1 admission compatibility; they do not prove a new job's model/input SHA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import hashlib
import json
import math
from typing import Any, Literal

from mikazuki.optimizer_profiles import get_optimizer_capability
from mikazuki.parameter_policy import (
    load_parameter_policy_sidecar,
    validate_parameter_policy,
)
from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
    FULL_BF16_OPTIMIZER_QUALIFICATIONS,
    FULL_BF16_SD_LORA_EVIDENCE_ID,
    FULL_BF16_SDXL_LORA_EVIDENCE_ID,
    FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID,
    _referenced_optimizer_profiles,
    build_parameter_policy_execution_contract,
    parameter_policy_execution_environment_blockers,
)
from mikazuki.parameter_policy_readiness import ExecutionReadiness
from mikazuki.parameter_policy_readiness_producer import (
    classify_component_full_bf16_readiness,
)

AdmissionPurpose = Literal["preview", "export", "start", "direct_trainer"]
AdmissionStatus = Literal[
    "inactive",
    "legacy_approved",
    "ack_required",
    "experimental_host_candidate",
    "blocked",
]

_PURPOSES = frozenset({"preview", "export", "start", "direct_trainer"})
_STATUSES = frozenset({
    "inactive", "legacy_approved", "ack_required",
    "experimental_host_candidate", "blocked",
})
# Compile-time source constants: release records are not a source of truth
# about *implementation*.  A Worker-forged "qualified" row with another
# case ID must not inherit the existing D1 no-new-ack compatibility route.
_LEGACY_BACKEND_CASE_IDS = {
    "sd-lora": FULL_BF16_SD_LORA_EVIDENCE_ID,
    "sdxl-lora": FULL_BF16_SDXL_LORA_EVIDENCE_ID,
}


@dataclass(frozen=True, slots=True)
class ComponentAdmission:
    """Immutable host diagnostic, never a credential or Start permission."""

    status: AdmissionStatus
    purpose: AdmissionPurpose
    readiness: ExecutionReadiness | None
    runtime_blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    requires_experimental_ack: bool
    host_admissible: bool

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _STATUSES:
            raise ValueError("Unknown Component admission status.")
        if type(self.purpose) is not str or self.purpose not in _PURPOSES:
            raise ValueError("Unknown Component admission purpose.")
        if type(self.requires_experimental_ack) is not bool:
            raise ValueError("Component admission acknowledgement flag must be boolean.")
        if type(self.host_admissible) is not bool:
            raise ValueError("Component host admissibility must be boolean.")
        if self.readiness is not None and not isinstance(
            self.readiness, ExecutionReadiness
        ):
            raise ValueError("Component admission requires typed readiness.")

        for attr in ("runtime_blockers", "warnings"):
            value = getattr(self, attr)
            if (
                isinstance(value, (str, bytes, bytearray))
                or not isinstance(value, (tuple, list))
                or any(type(item) is not str or not item.strip() for item in value)
            ):
                raise ValueError(f"Component admission {attr} must be a string tuple.")
            object.__setattr__(self, attr, tuple(value))

        inactive = self.status == "inactive"
        if inactive:
            if (self.readiness is not None or self.runtime_blockers
                    or self.warnings or self.requires_experimental_ack
                    or not self.host_admissible):
                raise ValueError("Inactive admission must be a strict no-op.")
            return

        if self.readiness is None:
            raise ValueError("Active admission requires first-party readiness.")
        if self.status == "blocked":
            if (self.host_admissible or self.requires_experimental_ack
                    or not self.runtime_blockers):
                raise ValueError("Blocked admission must contain hard reasons.")
        elif self.status == "ack_required":
            if (self.host_admissible or not self.requires_experimental_ack
                    or not self.runtime_blockers or not self.warnings
                    or self.readiness.hard_errors):
                raise ValueError("Acknowledgement is advisory-only and not admissible.")
        elif self.status in {"legacy_approved", "experimental_host_candidate"}:
            if (
                not self.host_admissible or self.requires_experimental_ack
                or self.runtime_blockers or self.readiness.hard_errors
                or self.readiness.capability != "implemented"
            ):
                raise ValueError("Host-admissible states require verified source capability.")
            if self.status == "experimental_host_candidate" and not self.warnings:
                raise ValueError("Experimental host admission requires an advisory.")
            if self.status == "legacy_approved" and self.readiness.decision == "block":
                raise ValueError("Legacy admission cannot override a hard runtime error.")


def _canonical_effective_config_value(value: object, active: set[int]) -> object:
    """Build a detached JSON tree; reject aliases that form reference cycles."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Effective training config cannot contain NaN or Infinity.")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("Cyclic effective training config mapping.")
        active.add(identity)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("Effective training config keys must be strings.")
                if key == "experimental_full_bf16_ack":
                    raise ValueError("Experimental acknowledgement cannot enter training config.")
                result[key] = _canonical_effective_config_value(item, active)
            return result
        finally:
            active.remove(identity)
    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            raise ValueError("Cyclic effective training config sequence.")
        active.add(identity)
        try:
            return [_canonical_effective_config_value(item, active) for item in value]
        finally:
            active.remove(identity)
    raise TypeError(f"Unsupported effective training config value: {type(value).__name__}.")


def fingerprint_effective_training_config(config: Mapping[str, Any]) -> str:
    """Full stable SHA-256 of a strict canonical effective-config snapshot."""
    if not isinstance(config, Mapping):
        raise TypeError("Effective training config must be a Mapping.")
    detached = _canonical_effective_config_value(config, set())
    canonical = json.dumps(
        detached,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DirectTrainerAdmissionSnapshot:
    """Identity-only Host admission snapshot; never a launch credential."""

    policy_hash: str
    effective_config_fingerprint: str
    train_type: str
    experimental_ack: bool
    admission: ComponentAdmission

    def __post_init__(self) -> None:
        for field_name in ("policy_hash", "effective_config_fingerprint"):
            digest = getattr(self, field_name)
            if (type(digest) is not str or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError(f"{field_name} must be a full lowercase SHA-256.")
        if type(self.train_type) is not str or not self.train_type.strip():
            raise ValueError("Direct Trainer snapshot needs a nonempty train type.")
        if self.train_type != self.train_type.strip().lower():
            raise ValueError("Direct Trainer snapshot train type must be normalized.")
        if type(self.experimental_ack) is not bool:
            raise ValueError("Direct Trainer snapshot acknowledgement must be bool.")
        if not isinstance(self.admission, ComponentAdmission):
            raise TypeError("Direct Trainer snapshot needs first-party admission.")
        if self.admission.purpose != "direct_trainer":
            raise ValueError("Direct Trainer snapshot requires direct_trainer admission.")

    def assert_matches(
        self, *, policy_hash: str, effective_config_fingerprint: str,
        train_type: str, experimental_ack: bool,
    ) -> None:
        """Fail closed on any identity/intent drift; do not grant permission."""
        if (
            type(policy_hash) is not str or policy_hash != self.policy_hash
            or type(effective_config_fingerprint) is not str
            or effective_config_fingerprint != self.effective_config_fingerprint
            or type(train_type) is not str or train_type != self.train_type
            or type(experimental_ack) is not bool
            or experimental_ack != self.experimental_ack
        ):
            raise ValueError("Direct Trainer admission snapshot identity mismatch.")


def prepare_direct_trainer_admission(
    policy_path: str | os.PathLike[str], *,
    train_type: str,
    effective_config: Mapping[str, Any],
    admission_intent: DirectTrainerAdmissionIntent,
) -> DirectTrainerAdmissionSnapshot:
    """Create a Host-only snapshot, detached from all caller-owned objects."""
    if not isinstance(admission_intent, DirectTrainerAdmissionIntent):
        raise TypeError("Direct Trainer requires a typed one-shot admission intent.")
    if type(train_type) is not str or not train_type.strip():
        raise ValueError("Direct Trainer requires a nonempty backend identifier.")
    backend = train_type.strip().lower()
    detached_config = _canonical_effective_config_value(effective_config, set())
    if not isinstance(detached_config, dict):
        raise TypeError("Effective training config must be a JSON mapping.")
    policy, policy_hash = load_parameter_policy_sidecar(policy_path)
    config_hash = fingerprint_effective_training_config(detached_config)
    acknowledgement = admission_intent.experimental_full_bf16_ack
    if type(acknowledgement) is not bool:
        raise ValueError("Direct Trainer acknowledgement must be strict bool.")
    admission = assess_component_training_admission(
        policy,
        train_type=backend,
        effective_config=detached_config,
        purpose="direct_trainer",
        experimental_ack=acknowledgement,
    )
    return DirectTrainerAdmissionSnapshot(
        policy_hash=policy_hash,
        effective_config_fingerprint=config_hash,
        train_type=backend,
        experimental_ack=acknowledgement,
        admission=admission,
    )


def _source_release_matches(row: object, expected_case: str) -> bool:
    """Exact first-party source ID, not the row's self-reported qualified flag."""
    return (
        isinstance(row, ExecutionFeatureQualification)
        and row.status == "qualified"
        and type(row.evidence_case_id) is str
        and row.evidence_case_id == expected_case
    )


def _original_d1_compatibility(
    canonical_policy: Mapping[str, Any],
    backend: str,
) -> bool:
    """Maintain prior D1 admission without falsely certifying this job.

    This is not an independent proof of today's pretrained model, resolution,
    network geometry, physical GPU or Checkpoint. Those are separately audited.
    """
    expected_backend = _LEGACY_BACKEND_CASE_IDS.get(backend)
    if expected_backend is None or not _source_release_matches(
        FULL_BF16_BACKEND_QUALIFICATIONS.get(backend), expected_backend
    ):
        return False

    profiles = canonical_policy["optimizer_profiles"]
    used = _referenced_optimizer_profiles(canonical_policy)
    if not used:
        return False
    for profile_name in used:
        profile = profiles[profile_name]
        optimizer = get_optimizer_capability(profile["type"])
        if not _source_release_matches(
            FULL_BF16_OPTIMIZER_QUALIFICATIONS.get(optimizer.name),
            FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID,
        ):
            return False
    return True


def _explicit_environment_blockers(
    effective_config: Mapping[str, Any], *, train_type: str,
) -> tuple[str, ...]:
    """Reject explicit multi-process/DDP hints; no claim about live Accelerate.

    A missing process/distribution field is NOT evidence of a single-process
    actual runtime.  Live distributed_type/device checks remain in Trainer.
    """
    if "num_processes" not in effective_config and (
        "distributed_type" not in effective_config
    ):
        return ()
    try:
        contract = build_parameter_policy_execution_contract(
            train_type=train_type, effective_config=effective_config,
        )
        if contract is None:
            return ()
        return tuple(parameter_policy_execution_environment_blockers(
            contract,
            num_processes=effective_config.get("num_processes", 1),
            distributed_type=effective_config.get("distributed_type", "NO"),
        ))
    except ValueError as exc:
        return (str(exc),)


def _diagnostic_warnings(readiness: ExecutionReadiness) -> tuple[str, ...]:
    return tuple(
        f"{issue.code}: {issue.message}" for issue in readiness.warnings
    )


def assess_component_training_admission(
    canonical_policy: Mapping[str, Any],
    *,
    train_type: str,
    effective_config: Mapping[str, Any],
    purpose: AdmissionPurpose,
    experimental_ack: bool = False,
) -> ComponentAdmission:
    """Classify *host-only* eligibility from real source, not caller claims.

    Preview/Export NEVER consume a positive acknowledgement as authorization.
    Start/Direct may only become an experimental HOST candidate with explicit
    strict True. QG-2 must still implement trustworthy opt-in transport and
    run real launch-time assets, dtype, root, optimizer and checkpoint guards.
    """
    if type(purpose) is not str or purpose not in _PURPOSES:
        raise ValueError("Unknown Component admission purpose.")
    if type(experimental_ack) is not bool:
        raise ValueError("Experimental Full-BF16 acknowledgement must be a real boolean.")
    if not isinstance(effective_config, Mapping):
        raise ValueError("Component admission requires an Effective Config mapping.")
    if type(train_type) is not str or not train_type.strip():
        raise ValueError("Component admission requires a nonempty backend string.")

    # This producer validates the canonical policy for any *active* feature;
    # inactive mixed BF16 must stay a no-op, not a forced Full-BF16 gate.
    readiness = classify_component_full_bf16_readiness(
        canonical_policy,
        train_type=train_type,
        effective_config=effective_config,
        require_assets=False,
    )
    if readiness is None:
        return ComponentAdmission(
            status="inactive",
            purpose=purpose,
            readiness=None,
            runtime_blockers=(),
            warnings=(),
            requires_experimental_ack=False,
            host_admissible=True,
        )

    warnings = _diagnostic_warnings(readiness)

    # Real source incompatibilities cannot become warnings through an ack or
    # a Worker-supplied "can_start", "decision", "capability" or evidence tier.
    if readiness.hard_errors or readiness.capability != "implemented":
        errors = tuple(
            f"{issue.code}: {issue.message}" for issue in readiness.hard_errors
        )
        if not errors:
            errors = ("The source-owned Component feature is not implemented.",)
        return ComponentAdmission(
            status="blocked", purpose=purpose, readiness=readiness,
            runtime_blockers=errors, warnings=warnings,
            requires_experimental_ack=False, host_admissible=False,
        )

    backend = train_type.strip().lower()
    environment_errors = _explicit_environment_blockers(
        effective_config, train_type=backend,
    )
    if environment_errors:
        return ComponentAdmission(
            status="blocked", purpose=purpose, readiness=readiness,
            runtime_blockers=environment_errors, warnings=warnings,
            requires_experimental_ack=False, host_admissible=False,
        )

    # Revalidate Policy to pin the existing D1 compatibility exception to
    # the actual referenced optimizers, not arbitrary user-supplied rows.
    canonical = validate_parameter_policy(canonical_policy)
    if _original_d1_compatibility(canonical, backend):
        return ComponentAdmission(
            status="legacy_approved", purpose=purpose, readiness=readiness,
            runtime_blockers=(), warnings=warnings,
            requires_experimental_ack=False, host_admissible=True,
        )

    # No evidence/missing-evidence/ID mismatch can grant a new legacy pass.
    # An otherwise audited configuration may become a host-only candidate
    # after the actual Start/Direct UI/CLI capture is wired and reviewed.
    if purpose in {"start", "direct_trainer"} and experimental_ack is True:
        return ComponentAdmission(
            status="experimental_host_candidate",
            purpose=purpose, readiness=readiness,
            runtime_blockers=(), warnings=warnings or (
                "Full-BF16 local experimental use is not formally qualified.",
            ),
            requires_experimental_ack=False, host_admissible=True,
        )

    return ComponentAdmission(
        status="ack_required", purpose=purpose, readiness=readiness,
        runtime_blockers=(
            "Experimental Full-BF16 needs an explicit per-job "
            "experimental_full_bf16_ack=True before the real Start/Direct "
            "Trainer runtime preflight.",
        ),
        warnings=warnings or (
            "Full-BF16 local experimental use is not formally qualified.",
        ),
        requires_experimental_ack=True, host_admissible=False,
    )


__all__ = [
    "AdmissionPurpose", "AdmissionStatus", "ComponentAdmission",
    "assess_component_training_admission",
    "fingerprint_effective_training_config",
    "DirectTrainerAdmissionSnapshot",
    "prepare_direct_trainer_admission",
]
