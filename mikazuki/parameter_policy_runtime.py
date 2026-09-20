from __future__ import annotations

"""Deterministic, torch-free runtime topology for Parameter Training Policy."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from operator import index as operator_index
from typing import Any, Literal, Mapping

from mikazuki.model_component_profiles import ParameterClass
from mikazuki.optimizer_profiles import (
    OptimizerCapability,
    require_unrestricted_component_optimizer,
)
from mikazuki.parameter_policy import validate_parameter_policy
from mikazuki.parameter_routing import RoutingAssignment, RoutingPlan


PARAMETER_POLICY_RUNTIME_SPEC_VERSION = 1
RuntimeTrainRouteKind = Literal["primary", "fallback"]


@dataclass(frozen=True)
class RuntimeSpecIssue:
    code: str
    message: str
    profile_name: str | None = None
    component_id: str | None = None
    parameter_name: str | None = None


class ParameterPolicyRuntimeSpecError(ValueError):
    def __init__(self, issues):
        normalized = tuple(sorted(tuple(issues), key=_issue_sort_key))
        if not normalized:
            raise ValueError("ParameterPolicyRuntimeSpecError requires at least one issue.")
        self.issues = normalized
        super().__init__(
            "; ".join(f"{item.code}: {item.message}" for item in normalized)
        )


@dataclass(frozen=True)
class RuntimeParameterSpec:
    parameter: Any = field(repr=False, compare=False)
    parameter_id: int = field(repr=False, compare=False)
    canonical_name: str
    shape: tuple[int, ...]
    numel: int
    component_id: str
    route_kind: RuntimeTrainRouteKind
    parameter_class: ParameterClass


@dataclass(frozen=True)
class OptimizerGroupSpec:
    learning_rate: float
    parameters: tuple[RuntimeParameterSpec, ...]

    @property
    def learning_rate_key(self) -> str:
        return self.learning_rate.hex()

    @property
    def tensor_count(self) -> int:
        return len(self.parameters)

    @property
    def numel(self) -> int:
        return sum(item.numel for item in self.parameters)


@dataclass(frozen=True)
class OptimizerInstanceSpec:
    profile_name: str
    optimizer_type: str
    optimizer_arguments_json: str
    supports_group_lr: bool
    uses_external_scheduler: bool
    lr_semantics: str
    groups: tuple[OptimizerGroupSpec, ...]
    topology_fingerprint: str

    @property
    def optimizer_arguments(self) -> dict[str, Any]:
        value = json.loads(self.optimizer_arguments_json)
        if not isinstance(value, dict):
            raise RuntimeError("Optimizer arguments JSON must decode to an object.")
        return value

    @property
    def tensor_count(self) -> int:
        return sum(group.tensor_count for group in self.groups)

    @property
    def numel(self) -> int:
        return sum(group.numel for group in self.groups)


@dataclass(frozen=True)
class ParameterPolicyRuntimeSpec:
    version: int
    optimizers: tuple[OptimizerInstanceSpec, ...]
    topology_fingerprint: str

    @property
    def tensor_count(self) -> int:
        return sum(item.tensor_count for item in self.optimizers)

    @property
    def numel(self) -> int:
        return sum(item.numel for item in self.optimizers)


def _issue_sort_key(issue: RuntimeSpecIssue):
    return (
        issue.code,
        issue.profile_name or "",
        issue.component_id or "",
        issue.parameter_name or "",
        issue.message,
    )


def _raise_issues(issues):
    if issues:
        raise ParameterPolicyRuntimeSpecError(issues)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _profile_sort_key(name: str):
    return (name.casefold(), name)


def _extract_shape(parameter: Any, *, parameter_name: str):
    try:
        raw_shape = parameter.shape
        values = tuple(raw_shape)
    except Exception as exc:
        raise ParameterPolicyRuntimeSpecError(
            [
                RuntimeSpecIssue(
                    code="parameter_shape_unavailable",
                    message=f"Cannot read shape for parameter {parameter_name!r}: {exc}",
                    parameter_name=parameter_name,
                )
            ]
        ) from exc

    shape = []
    for raw_dim in values:
        if isinstance(raw_dim, bool):
            raise ParameterPolicyRuntimeSpecError(
                [
                    RuntimeSpecIssue(
                        code="parameter_shape_invalid",
                        message=f"Parameter {parameter_name!r} shape contains boolean dimension.",
                        parameter_name=parameter_name,
                    )
                ]
            )
        try:
            dim = operator_index(raw_dim)
        except Exception as exc:
            raise ParameterPolicyRuntimeSpecError(
                [
                    RuntimeSpecIssue(
                        code="parameter_shape_invalid",
                        message=(
                            f"Parameter {parameter_name!r} shape contains non-integer "
                            f"dimension {raw_dim!r}."
                        ),
                        parameter_name=parameter_name,
                    )
                ]
            ) from exc
        if dim < 0:
            raise ParameterPolicyRuntimeSpecError(
                [
                    RuntimeSpecIssue(
                        code="parameter_shape_invalid",
                        message=f"Parameter {parameter_name!r} shape contains negative dimension {dim}.",
                        parameter_name=parameter_name,
                    )
                ]
            )
        shape.append(dim)

    frozen_shape = tuple(shape)
    return frozen_shape, math.prod(frozen_shape) if frozen_shape else 1


def _routing_error_issue(plan: RoutingPlan) -> RuntimeSpecIssue:
    codes = sorted(
        {
            issue.code
            for issue in plan.issues
            if issue.severity == "error"
        }
    )
    return RuntimeSpecIssue(
        code="invalid_routing_plan",
        message=(
            "Runtime Spec requires a valid RoutingPlan"
            + (f"; routing errors: {', '.join(codes)}." if codes else ".")
        ),
    )


def _validate_assignment_structure(
    assignment: RoutingAssignment,
    *,
    seen_ids,
    seen_names,
    issues,
) -> None:
    name = assignment.canonical_name
    if not isinstance(name, str) or not name.strip():
        issues.append(
            RuntimeSpecIssue(
                code="invalid_canonical_parameter_name",
                message="Routing assignment canonical_name must be a non-empty string.",
                component_id=assignment.component_id,
            )
        )
        return

    actual_id = id(assignment.parameter)
    if assignment.parameter_id != actual_id:
        issues.append(
            RuntimeSpecIssue(
                code="parameter_identity_mismatch",
                message=(
                    f"Routing assignment for {name!r} carries parameter_id="
                    f"{assignment.parameter_id}, but id(parameter)={actual_id}."
                ),
                component_id=assignment.component_id,
                parameter_name=name,
            )
        )

    previous = seen_ids.get(actual_id)
    if previous is not None:
        issues.append(
            RuntimeSpecIssue(
                code="duplicate_physical_parameter",
                message=(
                    f"Physical parameter {name!r} is assigned more than once "
                    f"({previous.route_kind}/{previous.component_id} and "
                    f"{assignment.route_kind}/{assignment.component_id})."
                ),
                component_id=assignment.component_id,
                parameter_name=name,
            )
        )
    else:
        seen_ids[actual_id] = assignment

    previous_name = seen_names.get(name)
    if previous_name is not None and id(previous_name.parameter) != actual_id:
        issues.append(
            RuntimeSpecIssue(
                code="duplicate_canonical_parameter_name",
                message=f"Different physical parameters share canonical name {name!r}.",
                component_id=assignment.component_id,
                parameter_name=name,
            )
        )
    else:
        seen_names[name] = assignment

    if assignment.route_kind in {"frozen", "unavailable"}:
        if assignment.optimizer_profile is not None or assignment.learning_rate is not None:
            issues.append(
                RuntimeSpecIssue(
                    code="inactive_route_has_optimizer_metadata",
                    message=(
                        f"{assignment.route_kind} assignment {name!r} must not carry "
                        "optimizer profile or learning rate."
                    ),
                    component_id=assignment.component_id,
                    parameter_name=name,
                )
            )
        return

    if assignment.route_kind not in {"primary", "fallback"}:
        issues.append(
            RuntimeSpecIssue(
                code="unknown_route_kind",
                message=f"Routing assignment {name!r} has unknown route_kind {assignment.route_kind!r}.",
                component_id=assignment.component_id,
                parameter_name=name,
            )
        )
        return

    if not isinstance(assignment.optimizer_profile, str) or not assignment.optimizer_profile:
        issues.append(
            RuntimeSpecIssue(
                code="trainable_route_missing_profile",
                message=f"Trainable assignment {name!r} has no optimizer profile.",
                component_id=assignment.component_id,
                parameter_name=name,
            )
        )

    lr = assignment.learning_rate
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(float(lr))
        or float(lr) <= 0
    ):
        issues.append(
            RuntimeSpecIssue(
                code="trainable_route_invalid_learning_rate",
                message=f"Trainable assignment {name!r} has invalid learning rate {lr!r}.",
                profile_name=assignment.optimizer_profile,
                component_id=assignment.component_id,
                parameter_name=name,
            )
        )


def _validate_policy_coherence(
    assignment: RoutingAssignment,
    *,
    canonical_policy: Mapping[str, Any],
    issues,
) -> None:
    if assignment.route_kind not in {"primary", "fallback"}:
        return

    component_id = assignment.component_id
    name = assignment.canonical_name
    route = canonical_policy["components"].get(component_id)
    if route is None:
        issues.append(
            RuntimeSpecIssue(
                code="assignment_component_missing_from_policy",
                message=f"Assignment references Component {component_id!r}, absent from current policy.",
                profile_name=assignment.optimizer_profile,
                component_id=component_id,
                parameter_name=name,
            )
        )
        return
    if not route["train"]:
        issues.append(
            RuntimeSpecIssue(
                code="assignment_component_frozen_in_policy",
                message=f"Assignment references Component {component_id!r}, frozen in current policy.",
                profile_name=assignment.optimizer_profile,
                component_id=component_id,
                parameter_name=name,
            )
        )
        return

    if assignment.route_kind == "primary":
        expected_profile = route["optimizer_profile"]
        expected_lr = route["learning_rate"]
    else:
        expected_profile = route.get("fallback_optimizer_profile")
        expected_lr = route.get("fallback_learning_rate", route["learning_rate"])

    if expected_profile is None:
        issues.append(
            RuntimeSpecIssue(
                code="assignment_route_missing_from_policy",
                message=(
                    f"{assignment.route_kind} assignment {name!r} has no "
                    "corresponding optimizer route in current policy."
                ),
                profile_name=assignment.optimizer_profile,
                component_id=component_id,
                parameter_name=name,
            )
        )
        return

    if assignment.optimizer_profile != expected_profile:
        issues.append(
            RuntimeSpecIssue(
                code="assignment_profile_mismatch",
                message=(
                    f"Assignment {name!r} uses Profile {assignment.optimizer_profile!r}; "
                    f"current policy requires {expected_profile!r}."
                ),
                profile_name=assignment.optimizer_profile,
                component_id=component_id,
                parameter_name=name,
            )
        )

    if assignment.learning_rate != expected_lr:
        issues.append(
            RuntimeSpecIssue(
                code="assignment_learning_rate_mismatch",
                message=(
                    f"Assignment {name!r} uses learning rate {assignment.learning_rate!r}; "
                    f"current policy requires {expected_lr!r}."
                ),
                profile_name=assignment.optimizer_profile,
                component_id=component_id,
                parameter_name=name,
            )
        )


def _runtime_parameter_spec(assignment: RoutingAssignment) -> RuntimeParameterSpec:
    shape, numel = _extract_shape(
        assignment.parameter,
        parameter_name=assignment.canonical_name,
    )
    return RuntimeParameterSpec(
        parameter=assignment.parameter,
        parameter_id=assignment.parameter_id,
        canonical_name=assignment.canonical_name,
        shape=shape,
        numel=numel,
        component_id=assignment.component_id,
        route_kind=assignment.route_kind,
        parameter_class=assignment.parameter_class,
    )


def _optimizer_topology_payload(
    *,
    profile_name: str,
    optimizer_type: str,
    optimizer_arguments_json: str,
    capability: OptimizerCapability,
    groups: tuple[OptimizerGroupSpec, ...],
):
    return {
        "schema": "dts.parameter-policy.optimizer-topology",
        "version": PARAMETER_POLICY_RUNTIME_SPEC_VERSION,
        "profile_name": profile_name,
        "optimizer_type": optimizer_type,
        "optimizer_arguments_json": optimizer_arguments_json,
        "uses_external_scheduler": capability.uses_external_scheduler,
        "lr_semantics": capability.lr_semantics,
        "groups": [
            {
                "learning_rate_hex": group.learning_rate.hex(),
                "parameters": [
                    {
                        "canonical_name": item.canonical_name,
                        "shape": list(item.shape),
                    }
                    for item in group.parameters
                ],
            }
            for group in groups
        ],
    }


def _runtime_topology_payload(optimizers):
    return {
        "schema": "dts.parameter-policy.runtime-topology",
        "version": PARAMETER_POLICY_RUNTIME_SPEC_VERSION,
        "optimizers": [
            {
                "profile_name": item.profile_name,
                "topology_fingerprint": item.topology_fingerprint,
            }
            for item in optimizers
        ],
    }


def compile_parameter_policy_runtime_spec(
    policy: Mapping[str, Any],
    routing_plan: RoutingPlan,
) -> ParameterPolicyRuntimeSpec:
    try:
        canonical_policy = validate_parameter_policy(policy)
    except ValueError as exc:
        raise ParameterPolicyRuntimeSpecError(
            [RuntimeSpecIssue(code="invalid_parameter_policy", message=str(exc))]
        ) from exc

    if not isinstance(routing_plan, RoutingPlan):
        raise ParameterPolicyRuntimeSpecError(
            [
                RuntimeSpecIssue(
                    code="invalid_routing_plan_type",
                    message="routing_plan must be a RoutingPlan instance.",
                )
            ]
        )
    if not routing_plan.is_valid:
        raise ParameterPolicyRuntimeSpecError([_routing_error_issue(routing_plan)])

    issues = []
    seen_ids = {}
    seen_names = {}
    for assignment in routing_plan.assignments:
        _validate_assignment_structure(
            assignment,
            seen_ids=seen_ids,
            seen_names=seen_names,
            issues=issues,
        )
        _validate_policy_coherence(
            assignment,
            canonical_policy=canonical_policy,
            issues=issues,
        )
    _raise_issues(issues)

    trainable = [
        item
        for item in routing_plan.assignments
        if item.route_kind in {"primary", "fallback"}
    ]
    if not trainable:
        raise ParameterPolicyRuntimeSpecError(
            [
                RuntimeSpecIssue(
                    code="no_trainable_parameter_assignments",
                    message="RoutingPlan contains no primary/fallback assignments for optimizer runtime.",
                )
            ]
        )

    buckets = {}
    for assignment in trainable:
        profile_name = assignment.optimizer_profile
        if profile_name is None:
            raise AssertionError("Trainable assignment passed validation without profile.")
        lr = float(assignment.learning_rate)
        buckets.setdefault(profile_name, {}).setdefault(lr, []).append(assignment)

    optimizer_specs = []
    runtime_issues = []
    for profile_name in sorted(buckets, key=_profile_sort_key):
        profile = canonical_policy["optimizer_profiles"].get(profile_name)
        if profile is None:
            runtime_issues.append(
                RuntimeSpecIssue(
                    code="assignment_profile_missing_from_policy",
                    message=f"Used Optimizer Profile {profile_name!r} is absent from current policy.",
                    profile_name=profile_name,
                )
            )
            continue

        try:
            capability = require_unrestricted_component_optimizer(profile["type"])
        except ValueError as exc:
            runtime_issues.append(
                RuntimeSpecIssue(
                    code="optimizer_profile_not_runnable",
                    message=str(exc),
                    profile_name=profile_name,
                )
            )
            continue

        lr_buckets = buckets[profile_name]
        if len(lr_buckets) > 1 and not capability.supports_group_lr:
            runtime_issues.append(
                RuntimeSpecIssue(
                    code="optimizer_profile_multiple_lrs_unsupported",
                    message=(
                        f"Optimizer Profile {profile_name!r} ({capability.name}) has "
                        f"{len(lr_buckets)} learning rates but does not support group LR."
                    ),
                    profile_name=profile_name,
                )
            )
            continue

        args_json = _canonical_json(profile["args"])
        groups = []
        for lr in sorted(lr_buckets):
            ordered = sorted(lr_buckets[lr], key=lambda item: item.canonical_name)
            groups.append(
                OptimizerGroupSpec(
                    learning_rate=float(lr),
                    parameters=tuple(_runtime_parameter_spec(item) for item in ordered),
                )
            )
        group_tuple = tuple(groups)
        payload = _optimizer_topology_payload(
            profile_name=profile_name,
            optimizer_type=capability.name,
            optimizer_arguments_json=args_json,
            capability=capability,
            groups=group_tuple,
        )
        optimizer_specs.append(
            OptimizerInstanceSpec(
                profile_name=profile_name,
                optimizer_type=capability.name,
                optimizer_arguments_json=args_json,
                supports_group_lr=capability.supports_group_lr,
                uses_external_scheduler=capability.uses_external_scheduler,
                lr_semantics=capability.lr_semantics,
                groups=group_tuple,
                topology_fingerprint=_sha256_json(payload),
            )
        )

    _raise_issues(runtime_issues)
    optimizers = tuple(optimizer_specs)
    return ParameterPolicyRuntimeSpec(
        version=PARAMETER_POLICY_RUNTIME_SPEC_VERSION,
        optimizers=optimizers,
        topology_fingerprint=_sha256_json(_runtime_topology_payload(optimizers)),
    )


__all__ = [
    "PARAMETER_POLICY_RUNTIME_SPEC_VERSION",
    "OptimizerGroupSpec",
    "OptimizerInstanceSpec",
    "ParameterPolicyRuntimeSpec",
    "ParameterPolicyRuntimeSpecError",
    "RuntimeParameterSpec",
    "RuntimeSpecIssue",
    "compile_parameter_policy_runtime_spec",
]
