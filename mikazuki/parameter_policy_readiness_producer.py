"""First-party host preflight for Component true-Full-BF16 (QG-1B2).

This is a source-audited *diagnostic producer*, not a Start permission, GPU
qualification, evidence signer, or live parameter/checkpoint audit.  Its
capability and severity facts are computed from DTS-owned policy, backend,
optimizer and semantic registries, never a supplied claimed decision.
The legacy training gates are deliberately NOT wired to this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mikazuki.optimizer_profiles import get_optimizer_capability
from mikazuki.parameter_policy_evidence_coverage import (
    match_full_bf16_evidence_coverage,
)
from mikazuki.parameter_policy import (
    _non_anima_policy_structure_and_target_blockers,
    _non_anima_text_encoder_cache_blockers,
    validate_parameter_policy,
)
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers
from mikazuki.parameter_policy_execution import (
    FULL_BF16_BACKEND_QUALIFICATIONS,
    FULL_BF16_OPTIMIZER_QUALIFICATIONS,
    FULL_BF16_SD_LORA_EVIDENCE_ID,
    FULL_BF16_SDXL_LORA_EVIDENCE_ID,
    FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID,
    _full_bf16_optimizer_variant_blocker,
    _referenced_optimizer_profiles,
    build_parameter_policy_execution_contract,
)
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.parameter_policy_readiness import (
    ExecutionReadiness,
    ReadinessIssue,
    evaluate_execution_readiness,
)
from mikazuki.parameter_policy_runtime_bools import (
    component_runtime_boolean_type_blockers,
    strict_component_runtime_bool,
)


# This is a reviewed host *capability* scope, not a table of GPU certificates.
# New backends/variants need their own implementation audit before joining it.
_AUDITED_BACKENDS = frozenset({"sd-lora", "sdxl-lora"})
_AUDITED_OPTIMIZERS = frozenset({"AdamW", "Muon"})
_BACKEND_EVIDENCE_IDS = {
    "sd-lora": FULL_BF16_SD_LORA_EVIDENCE_ID,
    "sdxl-lora": FULL_BF16_SDXL_LORA_EVIDENCE_ID,
}


def _issue(code: str, message: str, source: str) -> ReadinessIssue:
    return ReadinessIssue(code=code, message=message, source=source)


def _registry_issue(
    *,
    subject: str,
    row: object,
    required_id: str,
) -> ReadinessIssue | None:
    """Read existing release metadata; it is never an execution authority."""
    if row is None:
        return _issue(
            f"qualification.{subject}.missing",
            f"No formal {subject} Full-BF16 evidence record is available.",
            "E02" if subject == "backend" else "E04",
        )
    status = getattr(row, "status", None)
    case_id = getattr(row, "evidence_case_id", None)
    if status == "qualified" and type(case_id) is str and case_id == required_id:
        return None
    if status == "qualified":
        return _issue(
            f"qualification.{subject}.missing_evidence",
            f"The {subject} Full-BF16 release row lacks the expected source evidence ID.",
            "E02" if subject == "backend" else "E04",
        )
    if status == "pending":
        return _issue(
            f"qualification.{subject}.pending",
            f"The {subject} Full-BF16 formal CUDA release evidence is pending.",
            "E02" if subject == "backend" else "E04",
        )
    return _issue(
        f"qualification.{subject}.unverified",
        f"No recognized qualified {subject} Full-BF16 release record is present.",
        "E02" if subject == "backend" else "E04",
    )


_CAPABILITY_PRIORITY = {
    "implemented": 0,
    "experimental_candidate": 1,
    "not_implemented": 2,
    "known_incompatible": 3,
}


def _combine_capability(current: str, incoming: str) -> str:
    """An actual unsupported/incompatible fact beats a review-only HOLD."""
    return max((current, incoming), key=_CAPABILITY_PRIORITY.__getitem__)


def _aggregate_issues(
    issues: list[ReadinessIssue],
) -> tuple[ReadinessIssue, ...]:
    """Preserve every distinct profile-specific reason under unique codes."""
    by_code: dict[str, set[tuple[str, str]]] = {}
    for issue in issues:
        by_code.setdefault(issue.code, set()).add((issue.source, issue.message))

    result: list[ReadinessIssue] = []
    for code in sorted(by_code):
        details = sorted(by_code[code])
        if len(details) == 1:
            source, message = details[0]
        else:
            source = "QG1B"
            message = "; ".join(
                f"[{item_source}] {item_message}"
                for item_source, item_message in details
            )
        result.append(_issue(code, message, source))
    return tuple(result)


def classify_component_full_bf16_readiness(
    canonical_policy: Mapping[str, Any],
    *,
    train_type: str,
    effective_config: Mapping[str, Any],
    require_assets: bool = False,
) -> ExecutionReadiness | None:
    """Construct host-only readiness from first-party facts, not caller claims.

    None means the Full-BF16 feature is strictly inactive.  The result's
    can_start is *advisory*: Preview/Start/Trainer do not consume it yet and
    real Torch dtype/device/optimizer/Checkpoint checks remain separate.
    require_assets=True raises until an actual asset validator is connected;
    it never silently pretends that model/encoder assets were inspected.
    """
    if not isinstance(effective_config, Mapping):
        raise ValueError("Component Full-BF16 readiness needs an Effective Config mapping.")
    if type(require_assets) is not bool:
        raise ValueError("Component Full-BF16 require_assets must be boolean.")

    # The stock trainer treats every quoted nonempty boolean string as True.
    # A malformed full_bf16 value MUST NOT be converted to inactive mode.
    try:
        active = strict_component_runtime_bool(
            effective_config.get("full_bf16"), field="full_bf16"
        )
    except ValueError:
        return evaluate_execution_readiness(
            capability="known_incompatible",
            hard_errors=(
                _issue(
                    "runtime.bool.invalid",
                    "Component full_bf16 must be a real boolean, never a quoted string.",
                    "C01",
                ),
            ),
        )
    if not active:
        return None
    if require_assets:
        raise ValueError(
            "QG-1B Host asset preflight is not implemented; "
            "use the existing launch asset validator."
        )

    # The explicit argument names exclude any Worker-supplied claimed
    # capability, severity bucket, evidence tier, or can_start permission.
    canonical = validate_parameter_policy(canonical_policy)
    normalized_backend = str(train_type or "").strip().lower()
    if normalized_backend not in PARAMETER_POLICY_RUNTIME_TRAIN_TYPES:
        return evaluate_execution_readiness(
            capability="not_implemented",
            hard_errors=(
                _issue(
                    "capability.backend.unknown",
                    "The requested backend has no registered Component trainer.",
                    "G01",
                ),
            ),
        )

    hard: list[ReadinessIssue] = []
    capability_status = "implemented"

    # Actual BF16 execution semantics are checked *before* evidence metadata.
    try:
        build_parameter_policy_execution_contract(
            train_type=normalized_backend,
            effective_config=effective_config,
        )
    except ValueError:
        hard.append(_issue(
            "runtime.bf16.precision_mismatch",
            "Component full_bf16 requires mixed_precision='bf16'.",
            "E09",
        ))

    # Validate all stock trainer boolean flags before checking later feature
    # semantics; an invalid bool can otherwise be truthy in real execution.
    bool_failures = component_runtime_boolean_type_blockers(
        effective_config, train_type=normalized_backend,
    )
    if bool_failures:
        hard.append(_issue(
            "runtime.bool.invalid",
            "; ".join(bool_failures),
            "C01",
        ))

    # Keep the current source-of-truth semantic and target/TE cache blockers.
    # Do not rewrite known fused, distributed/offload or LoRA+ conflicts as
    # mere missing qualification warnings.
    if not bool_failures:
        semantic = parameter_policy_v1_semantic_blockers(
            effective_config, normalized_backend,
        )
        if semantic:
            hard.append(_issue(
                "runtime.component.semantic_incompatible",
                "; ".join(semantic),
                "C06",
            ))
        if normalized_backend not in {"anima-lora", "anima-finetune"}:
            target = _non_anima_policy_structure_and_target_blockers(
                canonical, train_type=normalized_backend,
                effective_config=effective_config,
            )
            if target:
                hard.append(_issue(
                    "runtime.component.target_incompatible",
                    "; ".join(target),
                    "C21",
                ))
            cache = _non_anima_text_encoder_cache_blockers(
                canonical, train_type=normalized_backend,
                effective_config=effective_config,
            )
            if cache:
                hard.append(_issue(
                    "runtime.text_encoder.cache_conflict",
                    "; ".join(cache),
                    "C22",
                ))

    # Known source-level hazards are independent of exact-head CUDA coverage.
    if normalized_backend == "sdxl-lora":
        try:
            grad_ckpt = strict_component_runtime_bool(
                effective_config.get("gradient_checkpointing"),
                field="gradient_checkpointing",
            )
        except ValueError:
            grad_ckpt = False  # already diagnosed as runtime.bool.invalid above
        if grad_ckpt:
            hard.append(_issue(
                "runtime.sdxl.gradient_checkpointing_unreviewed",
                "SDXL Full-BF16 gradient checkpointing may train base Text "
                "Encoder embeddings outside the Component optimizer.",
                "E08",
            ))
            capability_status = _combine_capability(
                capability_status, "known_incompatible"
            )

    if normalized_backend == "sd-lora":
        try:
            sd_v2 = strict_component_runtime_bool(
                effective_config.get("v2"), field="v2",
            )
        except ValueError:
            sd_v2 = False  # already diagnosed
        if sd_v2:
            hard.append(_issue(
                "capability.backend.audit_required",
                "SD2.x true Full-BF16 Component execution needs a distinct "
                "implementation/ownership audit.",
                "E07",
            ))
            capability_status = _combine_capability(
                capability_status, "experimental_candidate"
            )

    # Source-audited production Host path is intentionally narrower than
    # registered baseline Component trainers.  An unreviewed true Full-BF16
    # backend is held for implementation audit regardless of release rows.
    if normalized_backend not in _AUDITED_BACKENDS:
        hard.append(_issue(
            "capability.backend.audit_required",
            "This backend true Full-BF16 Component execution contract has "
            "not completed a source/optimizer-ownership audit.",
            "E03",
        ))
        capability_status = _combine_capability(
            capability_status, "experimental_candidate"
        )

    referenced_types: set[str] = set()
    referenced_profiles_by_type: dict[str, list[str]] = {}
    profiles = canonical["optimizer_profiles"]
    for profile_name in _referenced_optimizer_profiles(canonical):
        profile = profiles[profile_name]
        optimizer = get_optimizer_capability(profile["type"])
        name = optimizer.name
        referenced_profiles_by_type.setdefault(name, []).append(profile_name)
        if optimizer.component_support == "planned":
            hard.append(_issue(
                "capability.optimizer.not_implemented",
                f"Optimizer Profile {profile_name!r} ({name}) is registered "
                "but not implemented for Component v1 execution.",
                "O07",
            ))
            capability_status = _combine_capability(
                capability_status, "not_implemented"
            )
        elif optimizer.component_support != "supported":
            hard.append(_issue(
                "capability.optimizer.audit_required",
                f"Optimizer Profile {profile_name!r} ({name}) needs a "
                "Component-specific optimizer/learning-rate ownership audit.",
                "O06",
            ))
            capability_status = _combine_capability(
                capability_status, "experimental_candidate"
            )
        elif name not in _AUDITED_OPTIMIZERS:
            hard.append(_issue(
                "capability.optimizer.audit_required",
                f"Optimizer Profile {profile_name!r} ({name}) does not yet "
                "have this source-audited true Full-BF16 execution contract.",
                "O02",
            ))
            capability_status = _combine_capability(
                capability_status, "experimental_candidate"
            )
        else:
            # The existing AdamW variant gate describes known D0 proof scope.
            # Extra explicit state variants need an actual constructor/Resume
            # review; this is a capability-audit hold, not merely no certificate.
            variant = _full_bf16_optimizer_variant_blocker(
                optimizer_type=name, arguments=profile["args"],
            )
            if variant:
                hard.append(_issue(
                    "capability.optimizer.audit_required",
                    f"Optimizer Profile {profile_name!r} ({name}): {variant}",
                    "E06",
                ))
                capability_status = _combine_capability(
                    capability_status, "experimental_candidate"
                )
        referenced_types.add(name)

    # Multiple referenced Profiles can share an issue code. Preserve every
    # reason in a deterministic aggregate without weakening any hard error.
    unique_hard = _aggregate_issues(hard)

    # Unreviewed or incompatible host semantics must never display a formal
    # tier based solely on a release row, even if that row is manipulated.
    if unique_hard or capability_status != "implemented":
        return evaluate_execution_readiness(
            capability=capability_status,
            hard_errors=unique_hard,
            evidence_tier="unverified",
        )

    warnings: list[ReadinessIssue] = []
    backend_warning = _registry_issue(
        subject="backend",
        row=FULL_BF16_BACKEND_QUALIFICATIONS.get(normalized_backend),
        required_id=_BACKEND_EVIDENCE_IDS[normalized_backend],
    )
    if backend_warning:
        warnings.append(backend_warning)

    # Do not convert the backend's broad release row into a certificate for
    # arbitrary training settings. D1 cases bind exact input model/policy
    # identities; this Host-only producer cannot verify those identities.
    coverage = match_full_bf16_evidence_coverage(
        train_type=normalized_backend,
        canonical_policy=canonical,
        effective_config=effective_config,
    )
    if coverage.covered:
        warnings.append(_issue(
            "qualification.variant.identity_unverified",
            f"D1 reference shape {coverage.case_id} matched, but this Host "
            "has not verified the model/input SHA or checkpoint provenance.",
            "E13",
        ))
    else:
        warnings.append(_issue(
            "qualification.variant.outside_scope",
            "Host config is outside or insufficient to establish the D1 "
            "reference-shape scope: " + ", ".join(coverage.reasons),
            "E13",
        ))

    # Only referenced primary/fallback Profiles influence evidence. Each
    # source Profile is retained in the diagnostics even when types and codes
    # repeat. Unused stored Profiles do not enter the current optimizer.
    for name in sorted(referenced_types):
        optimizer_warning = _registry_issue(
            subject="optimizer",
            row=FULL_BF16_OPTIMIZER_QUALIFICATIONS.get(name),
            required_id=FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID,
        )
        if optimizer_warning:
            owners = ", ".join(sorted(referenced_profiles_by_type[name]))
            warnings.append(_issue(
                optimizer_warning.code,
                f"Optimizer {name} via Profiles {owners}: "
                f"{optimizer_warning.message}",
                optimizer_warning.source,
            ))

    return evaluate_execution_readiness(
        capability="implemented",
        warnings=_aggregate_issues(warnings),
        # Qualified case-family rows do not authenticate THIS job's model
        # files. The Host cannot claim per-run formal proof without that bind.
        evidence_tier="unverified",
    )


__all__ = ["classify_component_full_bf16_readiness"]
