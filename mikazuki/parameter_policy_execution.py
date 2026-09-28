"""Host-side execution-feature qualification for Parameter Policy.

This module is intentionally dormant unless an execution feature is explicitly
active in the effective trainer configuration.  It is torch-free, model-free,
I/O-free, and must not mutate policy/config state.

Phase A introduces only the qualification contract for Component full BF16.
All backend and optimizer qualifications start fail-closed; later phases may
promote individual entries only after exact-head CUDA evidence exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mikazuki.optimizer_profiles import get_optimizer_capability


QualificationStatus = Literal["pending", "qualified", "unsupported"]


@dataclass(frozen=True)
class ExecutionFeatureQualification:
    status: QualificationStatus
    reason: str
    evidence_case_id: str | None = None


FULL_BF16_BACKEND_QUALIFICATIONS: dict[str, ExecutionFeatureQualification] = {
    "sd-lora": ExecutionFeatureQualification(
        "pending",
        "SD LoRA full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "sdxl-lora": ExecutionFeatureQualification(
        "pending",
        "SDXL LoRA full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "sd-dreambooth": ExecutionFeatureQualification(
        "unsupported",
        "SD DreamBooth does not currently expose an equivalent full-BF16 training-model cast contract.",
    ),
    "sdxl-finetune": ExecutionFeatureQualification(
        "pending",
        "SDXL Full full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "sd3-lora": ExecutionFeatureQualification(
        "pending",
        "SD3 LoRA full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "flux-lora": ExecutionFeatureQualification(
        "pending",
        "Flux LoRA full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "chroma-lora": ExecutionFeatureQualification(
        "pending",
        "Chroma LoRA full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "flux-finetune": ExecutionFeatureQualification(
        "pending",
        "Flux Full full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "anima-lora": ExecutionFeatureQualification(
        "pending",
        "Anima LoRA full BF16 has not completed Component exact-head CUDA qualification.",
    ),
    "anima-finetune": ExecutionFeatureQualification(
        "pending",
        "Anima Full full BF16 has not completed Component exact-head CUDA qualification.",
    ),
}


FULL_BF16_OPTIMIZER_QUALIFICATIONS: dict[str, ExecutionFeatureQualification] = {
    "AdamW": ExecutionFeatureQualification(
        "pending",
        "AdamW true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "AdamW8bit": ExecutionFeatureQualification(
        "pending",
        "AdamW8bit true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "PagedAdamW8bit": ExecutionFeatureQualification(
        "pending",
        "PagedAdamW8bit true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "PagedAdamW": ExecutionFeatureQualification(
        "pending",
        "PagedAdamW true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "PagedAdamW32bit": ExecutionFeatureQualification(
        "pending",
        "PagedAdamW32bit true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "Lion": ExecutionFeatureQualification(
        "pending",
        "Lion true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "Lion8bit": ExecutionFeatureQualification(
        "pending",
        "Lion8bit true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "PagedLion8bit": ExecutionFeatureQualification(
        "pending",
        "PagedLion8bit true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "SGDNesterov": ExecutionFeatureQualification(
        "pending",
        "SGDNesterov true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "SGDNesterov8bit": ExecutionFeatureQualification(
        "pending",
        "SGDNesterov8bit true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "RAdamScheduleFree": ExecutionFeatureQualification(
        "pending",
        "RAdamScheduleFree true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "AdamWScheduleFree": ExecutionFeatureQualification(
        "pending",
        "AdamWScheduleFree true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "SGDScheduleFree": ExecutionFeatureQualification(
        "pending",
        "SGDScheduleFree true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
    "Muon": ExecutionFeatureQualification(
        "pending",
        "Muon true-BF16 Parameter Policy execution has not completed shared CUDA qualification.",
    ),
}


_FULL_BF16_ADAMW_ARGUMENTS = frozenset({"betas", "eps", "weight_decay"})


def _strict_bool(value: object, *, field: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(
            f"Parameter Policy execution: {field} must be true/false or 1/0."
        )
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(
            f"Parameter Policy execution: {field} cannot parse boolean value {value!r}."
        )
    raise ValueError(
        f"Parameter Policy execution: {field} cannot parse boolean value {value!r}."
    )


def active_parameter_policy_execution_features(
    effective_config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the explicitly active Parameter Policy execution features."""

    if not isinstance(effective_config, Mapping):
        raise ValueError("Parameter Policy execution: effective_config must be a mapping.")

    if _strict_bool(effective_config.get("full_bf16"), field="full_bf16"):
        return ("full_bf16",)
    return ()


def _qualification_is_released(
    qualification: ExecutionFeatureQualification | None,
) -> bool:
    return bool(
        qualification is not None
        and qualification.status == "qualified"
        and str(qualification.evidence_case_id or "").strip()
    )


def _qualification_blocker(
    *,
    subject: str,
    qualification: ExecutionFeatureQualification | None,
) -> str | None:
    if qualification is None:
        return f"{subject} has no qualification record; execution remains fail-closed."
    if _qualification_is_released(qualification):
        return None
    if qualification.status == "qualified":
        return (
            f"{subject} is marked qualified but has no evidence_case_id; "
            "execution remains fail-closed."
        )
    return f"{subject} is {qualification.status}: {qualification.reason}"


def _referenced_optimizer_profiles(
    canonical_policy: Mapping[str, Any],
) -> tuple[str, ...]:
    components = canonical_policy.get("components")
    profiles = canonical_policy.get("optimizer_profiles")
    if not isinstance(components, Mapping) or not isinstance(profiles, Mapping):
        raise ValueError(
            "Parameter Policy execution requires a canonical policy with components "
            "and optimizer_profiles mappings."
        )

    referenced: set[str] = set()
    for route in components.values():
        if not isinstance(route, Mapping) or not bool(route.get("train")):
            continue
        primary = route.get("optimizer_profile")
        if primary not in (None, ""):
            referenced.add(str(primary))
        fallback = route.get("fallback_optimizer_profile")
        if fallback not in (None, ""):
            referenced.add(str(fallback))
    return tuple(sorted(referenced, key=str.casefold))


def _full_bf16_optimizer_variant_blocker(
    *,
    optimizer_type: str,
    arguments: Mapping[str, Any],
) -> str | None:
    if optimizer_type != "AdamW":
        return None

    unsupported = sorted(set(arguments).difference(_FULL_BF16_ADAMW_ARGUMENTS))
    if not unsupported:
        return None
    return (
        "Optimizer AdamW full-BF16 qualification does not cover explicit "
        "execution/state variants in args: "
        + ", ".join(unsupported)
        + "."
    )


def parameter_policy_execution_blockers(
    canonical_policy: Mapping[str, Any],
    *,
    train_type: str,
    effective_config: Mapping[str, Any],
) -> list[str]:
    """Return execution-feature blockers for one canonical Component policy.

    Inactive features are a strict no-op: when full_bf16 is not enabled, this
    function returns before consulting backend/optimizer qualification metadata.
    """

    features = active_parameter_policy_execution_features(effective_config)
    if "full_bf16" not in features:
        return []

    mixed_precision = str(effective_config.get("mixed_precision") or "no").strip().lower()
    if mixed_precision != "bf16":
        return [
            "Component full_bf16 requires mixed_precision='bf16'; "
            f"received {mixed_precision!r}."
        ]

    normalized_train_type = str(train_type or "").strip().lower()
    backend_qualification = FULL_BF16_BACKEND_QUALIFICATIONS.get(
        normalized_train_type
    )
    backend_blocker = _qualification_blocker(
        subject=f"backend={normalized_train_type!r} full_bf16",
        qualification=backend_qualification,
    )
    if backend_blocker is not None:
        return [backend_blocker]

    profiles = canonical_policy.get("optimizer_profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError(
            "Parameter Policy execution requires optimizer_profiles to be a mapping."
        )

    blockers: list[str] = []
    for profile_name in _referenced_optimizer_profiles(canonical_policy):
        profile = profiles.get(profile_name)
        if not isinstance(profile, Mapping):
            raise ValueError(
                f"Parameter Policy execution referenced missing/invalid Optimizer "
                f"Profile {profile_name!r}."
            )

        capability = get_optimizer_capability(str(profile.get("type") or ""))
        if capability.component_support != "supported":
            # Baseline Component support owns this blocker.  Do not emit a
            # second feature-specific message for an already-unrunnable profile.
            continue

        qualification = FULL_BF16_OPTIMIZER_QUALIFICATIONS.get(capability.name)
        blocker = _qualification_blocker(
            subject=f"optimizer={capability.name!r} full_bf16",
            qualification=qualification,
        )
        if blocker is not None:
            blockers.append(blocker)
            continue

        arguments = profile.get("args") or {}
        if not isinstance(arguments, Mapping):
            raise ValueError(
                f"Parameter Policy execution Optimizer Profile {profile_name!r}.args "
                "must be a mapping."
            )
        variant_blocker = _full_bf16_optimizer_variant_blocker(
            optimizer_type=capability.name,
            arguments=arguments,
        )
        if variant_blocker is not None:
            blockers.append(variant_blocker)

    return list(dict.fromkeys(blockers))


__all__ = [
    "ExecutionFeatureQualification",
    "FULL_BF16_BACKEND_QUALIFICATIONS",
    "FULL_BF16_OPTIMIZER_QUALIFICATIONS",
    "active_parameter_policy_execution_features",
    "parameter_policy_execution_blockers",
]
