from __future__ import annotations

"""Dependency-light parameter scanning primitives for Parameter Policy.

This module deliberately does not import PyTorch. It only reads structural
metadata exposed by module/parameter-like objects and preserves alias identity
so later routing can fail closed on shared/tied parameters.

Commit 3A/3B scope ends at deterministic parameter descriptors. Optimizer
routing is added by later Commit 3 stages.
"""

from dataclasses import dataclass
from operator import index as operator_index
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from mikazuki.model_component_profiles import (
    ModelComponentProfile,
    ParameterClass,
    TrainingTargetProfile,
    get_model_component_profile,
    resolve_training_target_profile,
)
from mikazuki.optimizer_profiles import get_optimizer_capability
from mikazuki.parameter_policy import validate_parameter_policy


class ParameterScanError(ValueError):
    """Raised when the scanner cannot build a trustworthy parameter inventory."""


@dataclass(frozen=True)
class AdapterTargetMetadata:
    root: str
    module_path: str
    module_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root.strip():
            raise ValueError("AdapterTargetMetadata.root must be a non-empty string.")
        if not isinstance(self.module_path, str):
            raise ValueError("AdapterTargetMetadata.module_path must be a string.")
        if not isinstance(self.module_type, str) or not self.module_type.strip():
            raise ValueError("AdapterTargetMetadata.module_type must be a non-empty string.")

        object.__setattr__(self, "root", self.root.strip().casefold())
        object.__setattr__(self, "module_path", self.module_path.strip("."))
        object.__setattr__(self, "module_type", self.module_type.strip())


@dataclass(frozen=True)
class ParameterAlias:
    root: str
    full_name: str
    module_path: str
    module_type: str
    module_class: str
    ancestor_module_types: tuple[str, ...]
    ancestor_module_classes: tuple[str, ...]
    parameter_role: str
    parameter_class: ParameterClass
    adapter_target: AdapterTargetMetadata | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.root}.{self.full_name}"


@dataclass(frozen=True)
class ParameterDescriptor:
    parameter: Any
    parameter_id: int
    aliases: tuple[ParameterAlias, ...]
    shape: tuple[int, ...]
    ndim: int
    numel: int
    dtype: str
    requires_grad: bool

    @property
    def canonical_alias(self) -> ParameterAlias:
        if not self.aliases:
            raise RuntimeError("ParameterDescriptor must contain at least one alias.")
        return self.aliases[0]

    @property
    def canonical_name(self) -> str:
        return self.canonical_alias.qualified_name


RouteKind = Literal["primary", "fallback", "frozen", "unavailable"]
IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class RoutingAssignment:
    parameter: Any
    parameter_id: int
    canonical_name: str
    parameter_class: ParameterClass
    component_id: str
    route_kind: RouteKind
    optimizer_profile: str | None
    learning_rate: float | None
    reason: str


@dataclass(frozen=True)
class RoutingIssue:
    severity: IssueSeverity
    code: str
    message: str
    component_id: str | None = None
    parameter_id: int | None = None
    count: int = 1
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterStat:
    tensors: int
    numel: int


@dataclass(frozen=True)
class RoutingStats:
    total: ParameterStat
    assigned: ParameterStat
    unassigned: ParameterStat
    conflicts: ParameterStat
    unroutable: ParameterStat
    by_component: Mapping[str, ParameterStat]
    by_route: Mapping[str, ParameterStat]
    by_parameter_class: Mapping[str, ParameterStat]


@dataclass(frozen=True)
class RoutingPlan:
    assignments: tuple[RoutingAssignment, ...]
    issues: tuple[RoutingIssue, ...]
    stats: RoutingStats

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class _TrainableOwnership:
    descriptor: ParameterDescriptor
    component_id: str
    parameter_class: ParameterClass
    route: Mapping[str, Any]


@dataclass(frozen=True)
class _OwnershipPass:
    policy: Mapping[str, Any]
    profile: ModelComponentProfile
    target: TrainingTargetProfile
    assignments: tuple[RoutingAssignment, ...]
    trainable: tuple[_TrainableOwnership, ...]
    issues: tuple[RoutingIssue, ...]
    observed_components: frozenset[str]
    unassigned: tuple[ParameterDescriptor, ...]
    conflicts: tuple[ParameterDescriptor, ...]


@dataclass(frozen=True)
class _OptimizerRoutingPass:
    assignments: tuple[RoutingAssignment, ...]
    issues: tuple[RoutingIssue, ...]
    unroutable: tuple[ParameterDescriptor, ...]
    fallback_usage: Mapping[str, int]


@dataclass
class _ParameterAggregate:
    parameter: Any
    shape: tuple[int, ...]
    ndim: int
    numel: int
    dtype: str
    requires_grad: bool
    aliases: set[ParameterAlias]


def _normalize_root_id(raw_root: object) -> str:
    if not isinstance(raw_root, str):
        raise ParameterScanError(
            f"Parameter root IDs must be strings, got {type(raw_root).__name__}."
        )
    root = raw_root.strip().casefold()
    if not root:
        raise ParameterScanError("Parameter root ID must not be empty.")
    return root


def _module_path(raw_path: object, *, root: str) -> str:
    if not isinstance(raw_path, str):
        raise ParameterScanError(
            f"Root {root!r} named_modules() returned a non-string module path."
        )
    path = raw_path.strip(".")
    if ".." in path:
        raise ParameterScanError(
            f"Root {root!r} returned invalid module path {raw_path!r}."
        )
    return path


def _module_type(module: Any) -> tuple[str, str]:
    cls = type(module)
    short = str(getattr(cls, "__name__", "") or "")
    qualname = str(getattr(cls, "__qualname__", short) or short)
    module_name = str(getattr(cls, "__module__", "") or "")
    if not short:
        short = qualname.rsplit(".", 1)[-1] or "UnknownModule"
    qualified = f"{module_name}.{qualname}" if module_name else qualname
    return qualified, short


def _named_modules(module: Any, *, root: str) -> list[tuple[str, Any]]:
    method = getattr(module, "named_modules", None)
    if not callable(method):
        raise ParameterScanError(
            f"Root {root!r} does not provide named_modules(remove_duplicate=False)."
        )

    try:
        raw_entries = list(method(remove_duplicate=False))
    except TypeError as exc:
        raise ParameterScanError(
            f"Root {root!r} must support named_modules(remove_duplicate=False); "
            "DTS will not fall back to duplicate-removing traversal."
        ) from exc
    except Exception as exc:
        raise ParameterScanError(
            f"Root {root!r} failed while enumerating named_modules(remove_duplicate=False): {exc}"
        ) from exc

    entries: dict[str, Any] = {}
    for item in raw_entries:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ParameterScanError(
                f"Root {root!r} named_modules() yielded an invalid entry {item!r}."
            )
        raw_path, child = item
        path = _module_path(raw_path, root=root)
        previous = entries.get(path)
        if previous is not None and previous is not child:
            raise ParameterScanError(
                f"Root {root!r} produced duplicate module path {path!r} "
                "for different module objects."
            )
        entries[path] = child

    if "" not in entries:
        raise ParameterScanError(
            f"Root {root!r} named_modules(remove_duplicate=False) did not expose the root module."
        )
    if entries[""] is not module:
        raise ParameterScanError(
            f"Root {root!r} named_modules() mapped the empty path to a different module object."
        )

    for path in entries:
        if not path:
            continue
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        if parent_path not in entries:
            raise ParameterScanError(
                f"Root {root!r} module path {path!r} is missing parent {parent_path!r}."
            )

    return sorted(entries.items(), key=lambda item: item[0])


def _named_parameters_local(
    module: Any,
    *,
    root: str,
    module_path: str,
) -> list[tuple[str, Any]]:
    method = getattr(module, "named_parameters", None)
    if not callable(method):
        raise ParameterScanError(
            f"Module {root}.{module_path or '<root>'} does not provide "
            "named_parameters(recurse=False, remove_duplicate=False)."
        )

    try:
        raw_entries = list(method(recurse=False, remove_duplicate=False))
    except TypeError as exc:
        raise ParameterScanError(
            f"Module {root}.{module_path or '<root>'} must support "
            "named_parameters(recurse=False, remove_duplicate=False); "
            "DTS will not degrade to recursive or duplicate-removing traversal."
        ) from exc
    except Exception as exc:
        raise ParameterScanError(
            f"Module {root}.{module_path or '<root>'} failed while enumerating "
            f"local parameters: {exc}"
        ) from exc

    entries: dict[str, Any] = {}
    for item in raw_entries:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ParameterScanError(
                f"Module {root}.{module_path or '<root>'} yielded invalid parameter entry {item!r}."
            )
        raw_name, parameter = item
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ParameterScanError(
                f"Module {root}.{module_path or '<root>'} yielded an invalid parameter name."
            )
        local_name = raw_name.strip()
        if "." in local_name:
            raise ParameterScanError(
                f"Module {root}.{module_path or '<root>'} returned non-local parameter "
                f"name {local_name!r} despite recurse=False."
            )
        previous = entries.get(local_name)
        if previous is not None and previous is not parameter:
            raise ParameterScanError(
                f"Module {root}.{module_path or '<root>'} produced duplicate local "
                f"parameter name {local_name!r} for different objects."
            )
        entries[local_name] = parameter

    return sorted(entries.items(), key=lambda item: item[0])


def _ancestor_paths(module_path: str) -> tuple[str, ...]:
    if not module_path:
        return ()
    parts = module_path.split(".")
    return tuple("" if index == 0 else ".".join(parts[:index]) for index in range(len(parts)))


def _module_chain(
    module_path: str,
    module_table: Mapping[str, Any],
) -> tuple[Any, ...]:
    paths = _ancestor_paths(module_path) + (module_path,)
    return tuple(module_table[path] for path in paths)


def _resolve_adapter_target(
    *,
    root: str,
    full_name: str,
    module_path: str,
    module_table: Mapping[str, Any],
    adapter_targets: Mapping[int, AdapterTargetMetadata],
) -> AdapterTargetMetadata | None:
    registrations: list[AdapterTargetMetadata] = []
    for module in _module_chain(module_path, module_table):
        metadata = adapter_targets.get(id(module))
        if metadata is not None:
            registrations.append(metadata)

    if not registrations:
        return None

    first = registrations[0]
    if any(metadata != first for metadata in registrations[1:]):
        distinct = sorted(
            {
                (metadata.root, metadata.module_path, metadata.module_type)
                for metadata in registrations
            }
        )
        raise ParameterScanError(
            f"Parameter {root}.{full_name} inherits conflicting nested adapter target metadata: "
            + ", ".join(repr(item) for item in distinct)
        )

    # Metadata values agree. The last entry is the nearest registered ancestor.
    return registrations[-1]


def _dimension(raw: object, *, context: str) -> int:
    if isinstance(raw, bool):
        raise ParameterScanError(f"{context} contains boolean shape/ndim metadata.")
    try:
        value = operator_index(raw)
    except Exception as exc:
        raise ParameterScanError(
            f"{context} must contain integer shape/ndim metadata, got {raw!r}."
        ) from exc
    if value < 0:
        raise ParameterScanError(f"{context} contains negative dimension {value}.")
    return value


def _read_parameter_metadata(
    parameter: Any,
    *,
    qualified_name: str,
) -> tuple[tuple[int, ...], int, int, str, bool]:
    if parameter is None:
        raise ParameterScanError(f"Parameter {qualified_name} is None.")

    try:
        raw_shape = getattr(parameter, "shape")
    except Exception as exc:
        raise ParameterScanError(f"Parameter {qualified_name} has unreadable shape metadata.") from exc

    try:
        shape = tuple(
            _dimension(item, context=f"Parameter {qualified_name} shape")
            for item in tuple(raw_shape)
        )
    except TypeError as exc:
        raise ParameterScanError(
            f"Parameter {qualified_name} shape metadata is not iterable."
        ) from exc

    try:
        raw_ndim = getattr(parameter, "ndim")
    except Exception as exc:
        raise ParameterScanError(f"Parameter {qualified_name} has unreadable ndim metadata.") from exc
    ndim = _dimension(raw_ndim, context=f"Parameter {qualified_name} ndim")
    if ndim != len(shape):
        raise ParameterScanError(
            f"Parameter {qualified_name} reports ndim={ndim} but shape has {len(shape)} dimensions."
        )

    try:
        raw_dtype = getattr(parameter, "dtype")
    except Exception as exc:
        raise ParameterScanError(f"Parameter {qualified_name} has unreadable dtype metadata.") from exc
    dtype = str(raw_dtype).strip()
    if not dtype:
        raise ParameterScanError(f"Parameter {qualified_name} has empty dtype metadata.")

    try:
        requires_grad = getattr(parameter, "requires_grad")
    except Exception as exc:
        raise ParameterScanError(
            f"Parameter {qualified_name} has unreadable requires_grad metadata."
        ) from exc
    if not isinstance(requires_grad, bool):
        raise ParameterScanError(
            f"Parameter {qualified_name} requires_grad must be bool, got "
            f"{type(requires_grad).__name__}."
        )

    numel = 1
    for dim in shape:
        numel *= dim

    return shape, ndim, numel, dtype, requires_grad


def _is_bias_role(role: str) -> bool:
    lowered = role.casefold()
    return lowered == "bias" or lowered.startswith("bias_") or lowered.endswith("_bias")


def _is_normalization_module(module_class: str) -> bool:
    lowered = module_class.casefold()
    if lowered in {
        "layernorm",
        "groupnorm",
        "rmsnorm",
        "qknorm",
        "llmadapterrmsnorm",
    }:
        return True
    if lowered.endswith("norm"):
        return True
    return lowered.startswith("batchnorm") or lowered.startswith("instancenorm")


def _is_embedding_module(module_class: str) -> bool:
    return module_class.casefold() in {"embedding", "embeddingbag"}


def _is_convolution_module(module_class: str) -> bool:
    lowered = module_class.casefold()
    suffixes = (
        "conv1d",
        "conv2d",
        "conv3d",
        "convtranspose1d",
        "convtranspose2d",
        "convtranspose3d",
        "convnd",
    )
    return any(lowered.endswith(suffix) for suffix in suffixes)


def classify_parameter_class(
    *,
    parameter_role: str,
    module_class: str,
    ndim: int,
) -> ParameterClass:
    if _is_bias_role(parameter_role):
        return "bias"
    if _is_normalization_module(module_class):
        return "norm_weight"
    if _is_embedding_module(module_class):
        return "embedding_weight"
    if _is_convolution_module(module_class):
        return "conv_weight"
    if parameter_role.casefold() == "weight" and ndim == 2:
        return "matrix_weight"
    return "other"


def _alias_sort_key(alias: ParameterAlias) -> tuple[object, ...]:
    target = alias.adapter_target
    target_key = (
        target.root,
        target.module_path,
        target.module_type,
    ) if target is not None else ("", "", "")
    return (
        alias.qualified_name,
        alias.module_type,
        alias.parameter_role,
        alias.parameter_class,
        target_key,
        alias.ancestor_module_types,
    )


def _descriptor_sort_key(descriptor: ParameterDescriptor) -> tuple[object, ...]:
    return (
        descriptor.canonical_name,
        tuple(alias.qualified_name for alias in descriptor.aliases),
        descriptor.shape,
        descriptor.dtype,
    )


def _normalize_adapter_targets(
    raw: Mapping[int, AdapterTargetMetadata] | None,
) -> dict[int, AdapterTargetMetadata]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ParameterScanError("adapter_targets must be a mapping keyed by id(adapter_module).")

    normalized: dict[int, AdapterTargetMetadata] = {}
    for key, metadata in raw.items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise ParameterScanError(
                "adapter_targets keys must be integer object IDs returned by id(module)."
            )
        if not isinstance(metadata, AdapterTargetMetadata):
            raise ParameterScanError(
                "adapter_targets values must be AdapterTargetMetadata instances."
            )
        normalized[key] = metadata
    return normalized


def scan_parameter_roots(
    roots: Mapping[str, Any],
    *,
    adapter_targets: Mapping[int, AdapterTargetMetadata] | None = None,
) -> tuple[ParameterDescriptor, ...]:
    """Scan semantic model roots while preserving every parameter alias.

    The scan is metadata-only and never calls tensor/device/copy helpers.
    """

    if not isinstance(roots, Mapping) or not roots:
        raise ParameterScanError("scan_parameter_roots() requires at least one semantic root.")

    normalized_roots: dict[str, Any] = {}
    for raw_root, module in roots.items():
        root = _normalize_root_id(raw_root)
        if root in normalized_roots:
            raise ParameterScanError(
                f"Parameter root ID {root!r} is duplicated after normalization."
            )
        if module is None:
            raise ParameterScanError(f"Parameter root {root!r} is None.")
        normalized_roots[root] = module

    targets = _normalize_adapter_targets(adapter_targets)
    aggregates: dict[int, _ParameterAggregate] = {}
    semantic_names: dict[str, int] = {}

    for root in sorted(normalized_roots):
        root_module = normalized_roots[root]
        module_entries = _named_modules(root_module, root=root)
        module_table = dict(module_entries)

        type_table: dict[str, tuple[str, str]] = {
            path: _module_type(module)
            for path, module in module_entries
        }

        for module_path, module in module_entries:
            ancestors = _ancestor_paths(module_path)
            ancestor_types = tuple(type_table[path][0] for path in ancestors)
            ancestor_classes = tuple(type_table[path][1] for path in ancestors)
            module_type, module_class = type_table[module_path]

            for parameter_role, parameter in _named_parameters_local(
                module,
                root=root,
                module_path=module_path,
            ):
                full_name = (
                    f"{module_path}.{parameter_role}"
                    if module_path
                    else parameter_role
                )
                qualified_name = f"{root}.{full_name}"
                parameter_id = id(parameter)

                existing_semantic = semantic_names.get(qualified_name)
                if existing_semantic is not None and existing_semantic != parameter_id:
                    raise ParameterScanError(
                        f"Semantic parameter name {qualified_name!r} refers to different "
                        "parameter objects."
                    )
                semantic_names[qualified_name] = parameter_id

                aggregate = aggregates.get(parameter_id)
                if aggregate is None:
                    shape, ndim, numel, dtype, requires_grad = _read_parameter_metadata(
                        parameter,
                        qualified_name=qualified_name,
                    )
                    aggregate = _ParameterAggregate(
                        parameter=parameter,
                        shape=shape,
                        ndim=ndim,
                        numel=numel,
                        dtype=dtype,
                        requires_grad=requires_grad,
                        aliases=set(),
                    )
                    aggregates[parameter_id] = aggregate
                elif aggregate.parameter is not parameter:
                    # id() collision cannot occur while both objects are live, but keep the
                    # ownership invariant explicit rather than assuming it.
                    raise ParameterScanError(
                        f"Object identity collision detected for parameter {qualified_name}."
                    )

                parameter_class = classify_parameter_class(
                    parameter_role=parameter_role,
                    module_class=module_class,
                    ndim=aggregate.ndim,
                )
                adapter_target = _resolve_adapter_target(
                    root=root,
                    full_name=full_name,
                    module_path=module_path,
                    module_table=module_table,
                    adapter_targets=targets,
                )

                aggregate.aliases.add(
                    ParameterAlias(
                        root=root,
                        full_name=full_name,
                        module_path=module_path,
                        module_type=module_type,
                        module_class=module_class,
                        ancestor_module_types=ancestor_types,
                        ancestor_module_classes=ancestor_classes,
                        parameter_role=parameter_role,
                        parameter_class=parameter_class,
                        adapter_target=adapter_target,
                    )
                )

    descriptors: list[ParameterDescriptor] = []
    for parameter_id, aggregate in aggregates.items():
        aliases = tuple(sorted(aggregate.aliases, key=_alias_sort_key))
        if not aliases:
            raise ParameterScanError(
                f"Parameter object id={parameter_id} was scanned without any semantic alias."
            )
        descriptors.append(
            ParameterDescriptor(
                parameter=aggregate.parameter,
                parameter_id=parameter_id,
                aliases=aliases,
                shape=aggregate.shape,
                ndim=aggregate.ndim,
                numel=aggregate.numel,
                dtype=aggregate.dtype,
                requires_grad=aggregate.requires_grad,
            )
        )

    return tuple(sorted(descriptors, key=_descriptor_sort_key))


def _issue_sort_key(issue: RoutingIssue) -> tuple[object, ...]:
    return (
        0 if issue.severity == "error" else 1,
        issue.code,
        issue.component_id or "",
        issue.examples,
        issue.message,
    )


def _assignment_sort_key(assignment: RoutingAssignment) -> tuple[object, ...]:
    return (
        assignment.canonical_name,
        assignment.component_id,
        assignment.route_kind,
    )


def _descriptor_identity_issues(
    descriptors: Sequence[ParameterDescriptor],
) -> tuple[set[int], list[RoutingIssue]]:
    by_parameter_id: dict[int, list[ParameterDescriptor]] = {}
    issues: list[RoutingIssue] = []
    conflict_ids: set[int] = set()

    for descriptor in descriptors:
        if not isinstance(descriptor, ParameterDescriptor):
            raise TypeError(
                "Parameter routing requires ParameterDescriptor instances from "
                "scan_parameter_roots()."
            )
        by_parameter_id.setdefault(descriptor.parameter_id, []).append(descriptor)
        if id(descriptor.parameter) != descriptor.parameter_id:
            conflict_ids.add(descriptor.parameter_id)
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="descriptor_identity_mismatch",
                    message=(
                        f"Descriptor {descriptor.canonical_name!r} carries "
                        "parameter_id that does not match id(parameter)."
                    ),
                    parameter_id=descriptor.parameter_id,
                    examples=(descriptor.canonical_name,),
                )
            )

    for parameter_id, group in by_parameter_id.items():
        if len(group) <= 1:
            continue
        conflict_ids.add(parameter_id)
        examples = tuple(sorted({item.canonical_name for item in group})[:5])
        issues.append(
            RoutingIssue(
                severity="error",
                code="duplicate_descriptor_identity",
                message=(
                    "Routing input contains multiple descriptors for one physical "
                    "parameter identity."
                ),
                parameter_id=parameter_id,
                count=len(group),
                examples=examples,
            )
        )

    return conflict_ids, issues


def _effective_root(alias: ParameterAlias) -> str:
    if alias.adapter_target is not None:
        return alias.adapter_target.root
    return alias.root


def _routing_assignment(
    descriptor: ParameterDescriptor,
    *,
    component_id: str,
    parameter_class: ParameterClass,
    route_kind: RouteKind,
    reason: str,
    optimizer_profile: str | None = None,
    learning_rate: float | None = None,
) -> RoutingAssignment:
    return RoutingAssignment(
        parameter=descriptor.parameter,
        parameter_id=descriptor.parameter_id,
        canonical_name=descriptor.canonical_name,
        parameter_class=parameter_class,
        component_id=component_id,
        route_kind=route_kind,
        optimizer_profile=optimizer_profile,
        learning_rate=learning_rate,
        reason=reason,
    )


def _aggregate_issue(
    *,
    code: str,
    message: str,
    descriptors: Sequence[ParameterDescriptor],
    component_id: str | None = None,
) -> RoutingIssue:
    ordered = sorted(descriptors, key=_descriptor_sort_key)
    return RoutingIssue(
        severity="error",
        code=code,
        message=message,
        component_id=component_id,
        count=len(ordered),
        examples=tuple(item.canonical_name for item in ordered[:5]),
    )


def _unique_descriptors(
    descriptors: Sequence[ParameterDescriptor],
) -> list[ParameterDescriptor]:
    seen: set[int] = set()
    result: list[ParameterDescriptor] = []
    for descriptor in descriptors:
        marker = id(descriptor)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(descriptor)
    return result


def _resolve_parameter_ownership_pass(
    policy: Mapping[str, Any],
    *,
    train_type: str,
    effective_config: Mapping[str, Any],
    descriptors: Sequence[ParameterDescriptor],
) -> _OwnershipPass:
    """Resolve model ownership and higher-level target semantics.

    This is the Commit 3C pass. It intentionally stops before optimizer
    eligibility/fallback routing. Target-available Train=true descriptors are
    returned as trainable candidates for Commit 3D.
    """

    canonical_policy = validate_parameter_policy(policy)
    profile = get_model_component_profile(train_type)
    target = resolve_training_target_profile(train_type, effective_config)

    ordered_descriptors = tuple(sorted(descriptors, key=_descriptor_sort_key))
    identity_conflicts, identity_issues = _descriptor_identity_issues(
        ordered_descriptors
    )

    issues: list[RoutingIssue] = list(identity_issues)
    assignments: list[RoutingAssignment] = []
    trainable: list[_TrainableOwnership] = []
    conflicts: list[ParameterDescriptor] = []
    unassigned: list[ParameterDescriptor] = []
    observed_components: set[str] = set()

    policy_components = canonical_policy["components"]

    for component_id in sorted(policy_components):
        route = policy_components[component_id]
        if component_id not in profile.components:
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="unknown_policy_component",
                    message=(
                        f"Parameter Policy Component {component_id!r} does not exist "
                        f"in Model Component Profile {profile.train_type!r}."
                    ),
                    component_id=component_id,
                )
            )
            continue
        if route["train"] and not target.is_available(component_id):
            reason = target.unavailable_reasons.get(
                component_id,
                "Disabled by the current trainer target/effective configuration.",
            )
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="target_unavailable_train_enabled",
                    message=(
                        f"Component {component_id!r} has Train=true but the current "
                        f"trainer target marks it unavailable: {reason}"
                    ),
                    component_id=component_id,
                )
            )

    unknown_descriptors: list[ParameterDescriptor] = []
    missing_policy: dict[str, list[ParameterDescriptor]] = {}

    for descriptor in ordered_descriptors:
        if descriptor.parameter_id in identity_conflicts:
            conflicts.append(descriptor)
            continue

        if not descriptor.aliases:
            conflicts.append(descriptor)
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="descriptor_without_alias",
                    message="Parameter descriptor contains no aliases.",
                    parameter_id=descriptor.parameter_id,
                )
            )
            continue

        parameter_classes = {alias.parameter_class for alias in descriptor.aliases}
        adapter_targets = {alias.adapter_target for alias in descriptor.aliases}
        descriptor_has_conflict = False

        if len(parameter_classes) != 1:
            descriptor_has_conflict = True
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="alias_parameter_class_conflict",
                    message=(
                        f"Aliases for {descriptor.canonical_name!r} disagree on "
                        "structural ParameterClass."
                    ),
                    parameter_id=descriptor.parameter_id,
                    count=len(descriptor.aliases),
                    examples=tuple(
                        alias.qualified_name
                        for alias in sorted(descriptor.aliases, key=_alias_sort_key)[:5]
                    ),
                )
            )

        if len(adapter_targets) != 1:
            descriptor_has_conflict = True
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="alias_adapter_target_conflict",
                    message=(
                        f"Aliases for {descriptor.canonical_name!r} disagree on "
                        "AdapterTargetMetadata."
                    ),
                    parameter_id=descriptor.parameter_id,
                    count=len(descriptor.aliases),
                    examples=tuple(
                        alias.qualified_name
                        for alias in sorted(descriptor.aliases, key=_alias_sort_key)[:5]
                    ),
                )
            )

        classified: list[tuple[ParameterAlias, str | None]] = []
        classifier_failed = False
        for alias in descriptor.aliases:
            try:
                component_id = profile.classify_alias(alias)
            except Exception as exc:
                classifier_failed = True
                descriptor_has_conflict = True
                issues.append(
                    RoutingIssue(
                        severity="error",
                        code="component_classifier_error",
                        message=(
                            f"Model Component Profile {profile.train_type!r} failed "
                            f"while classifying alias {alias.qualified_name!r}: {exc}"
                        ),
                        parameter_id=descriptor.parameter_id,
                        examples=(alias.qualified_name,),
                    )
                )
                continue

            classified.append((alias, component_id))
            if component_id is not None and component_id in profile.components:
                observed_components.add(component_id)

        if classifier_failed:
            conflicts.append(descriptor)
            continue

        component_values = [component_id for _, component_id in classified]
        non_null_components = {item for item in component_values if item is not None}

        for component_id in sorted(non_null_components):
            if component_id not in profile.components:
                descriptor_has_conflict = True
                issues.append(
                    RoutingIssue(
                        severity="error",
                        code="classifier_unknown_component",
                        message=(
                            f"Model Component Profile {profile.train_type!r} returned "
                            f"unknown Component {component_id!r} for "
                            f"{descriptor.canonical_name!r}."
                        ),
                        component_id=component_id,
                        parameter_id=descriptor.parameter_id,
                        examples=(descriptor.canonical_name,),
                    )
                )

        if descriptor_has_conflict:
            conflicts.append(descriptor)
            continue

        if not non_null_components:
            unknown_descriptors.append(descriptor)
            unassigned.append(descriptor)
            continue

        if len(non_null_components) != 1 or any(item is None for item in component_values):
            conflicts.append(descriptor)
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="alias_component_conflict",
                    message=(
                        f"Aliases for {descriptor.canonical_name!r} do not agree on "
                        "one Model Component."
                    ),
                    parameter_id=descriptor.parameter_id,
                    count=len(descriptor.aliases),
                    examples=tuple(
                        alias.qualified_name
                        for alias in sorted(descriptor.aliases, key=_alias_sort_key)[:5]
                    ),
                )
            )
            continue

        component_id = next(iter(non_null_components))
        definition = profile.components.get(component_id)
        if definition is None:
            conflicts.append(descriptor)
            continue

        invalid_roots = sorted(
            {
                _effective_root(alias)
                for alias, _ in classified
                if _effective_root(alias) not in definition.roots
            }
        )
        if invalid_roots:
            conflicts.append(descriptor)
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="alias_component_root_conflict",
                    message=(
                        f"Aliases for {descriptor.canonical_name!r} resolve to "
                        f"Component {component_id!r} from invalid architectural "
                        f"root(s): {', '.join(invalid_roots)}."
                    ),
                    component_id=component_id,
                    parameter_id=descriptor.parameter_id,
                    examples=tuple(
                        alias.qualified_name
                        for alias in sorted(descriptor.aliases, key=_alias_sort_key)[:5]
                    ),
                )
            )
            continue

        parameter_class = next(iter(parameter_classes))

        if not target.is_available(component_id):
            assignments.append(
                _routing_assignment(
                    descriptor,
                    component_id=component_id,
                    parameter_class=parameter_class,
                    route_kind="unavailable",
                    reason=target.unavailable_reasons.get(
                        component_id,
                        "Disabled by the current trainer target/effective configuration.",
                    ),
                )
            )
            continue

        route = policy_components.get(component_id)
        if route is None:
            unassigned.append(descriptor)
            missing_policy.setdefault(component_id, []).append(descriptor)
            continue

        if not route["train"]:
            assignments.append(
                _routing_assignment(
                    descriptor,
                    component_id=component_id,
                    parameter_class=parameter_class,
                    route_kind="frozen",
                    reason="Component policy sets Train=false.",
                )
            )
            continue

        trainable.append(
            _TrainableOwnership(
                descriptor=descriptor,
                component_id=component_id,
                parameter_class=parameter_class,
                route=MappingProxyType(dict(route)),
            )
        )

    if unknown_descriptors:
        issues.append(
            _aggregate_issue(
                code="unassigned_parameter",
                message=(
                    "One or more real parameters could not be classified into any "
                    f"Component for Model Component Profile {profile.train_type!r}."
                ),
                descriptors=unknown_descriptors,
            )
        )

    for component_id in sorted(missing_policy):
        descriptors_for_component = missing_policy[component_id]
        issues.append(
            _aggregate_issue(
                code="missing_component_policy",
                message=(
                    f"Target-available Component {component_id!r} has real parameters "
                    "but no Parameter Policy row."
                ),
                component_id=component_id,
                descriptors=descriptors_for_component,
            )
        )

    return _OwnershipPass(
        policy=MappingProxyType(canonical_policy),
        profile=profile,
        target=target,
        assignments=tuple(sorted(assignments, key=_assignment_sort_key)),
        trainable=tuple(
            sorted(trainable, key=lambda item: _descriptor_sort_key(item.descriptor))
        ),
        issues=tuple(sorted(issues, key=_issue_sort_key)),
        observed_components=frozenset(observed_components),
        unassigned=tuple(
            sorted(_unique_descriptors(unassigned), key=_descriptor_sort_key)
        ),
        conflicts=tuple(
            sorted(_unique_descriptors(conflicts), key=_descriptor_sort_key)
        ),
    )


def _route_trainable_ownership(
    ownership: _OwnershipPass,
) -> _OptimizerRoutingPass:
    """Route Commit 3C trainable candidates to primary/fallback profiles.

    This is Commit 3D only. Component presence, unused-fallback warnings,
    statistics, and the public final RoutingPlan are completed in Commit 3E.
    """

    assignments: list[RoutingAssignment] = []
    issues: list[RoutingIssue] = []
    unroutable: list[ParameterDescriptor] = []
    missing_fallback: dict[str, list[ParameterDescriptor]] = {}
    fallback_usage: dict[str, int] = {}

    optimizer_profiles = ownership.policy["optimizer_profiles"]

    for candidate in ownership.trainable:
        descriptor = candidate.descriptor
        component_id = candidate.component_id
        route = candidate.route

        primary_name = route["optimizer_profile"]
        primary_profile = optimizer_profiles[primary_name]
        capability = get_optimizer_capability(primary_profile["type"])

        eligibility_policy = capability.eligibility_policy
        if eligibility_policy is None:
            assignments.append(
                _routing_assignment(
                    descriptor,
                    component_id=component_id,
                    parameter_class=candidate.parameter_class,
                    route_kind="primary",
                    optimizer_profile=primary_name,
                    learning_rate=route["learning_rate"],
                    reason=(
                        f"Optimizer Profile {primary_name!r} does not require "
                        "parameter eligibility routing."
                    ),
                )
            )
            continue

        decisions: list[tuple[ParameterAlias, bool, str]] = []
        for alias in descriptor.aliases:
            decision = ownership.profile.eligibility_check(
                eligibility_policy,
                alias,
                component_id,
            )
            if (
                not isinstance(decision, tuple)
                or len(decision) != 2
                or not isinstance(decision[0], bool)
                or not isinstance(decision[1], str)
            ):
                raise ValueError(
                    f"Model Component Profile {ownership.profile.train_type!r} "
                    f"returned invalid eligibility decision for "
                    f"{alias.qualified_name!r}."
                )
            decisions.append((alias, decision[0], decision[1]))

        eligibility_values = {eligible for _, eligible, _ in decisions}
        if len(eligibility_values) != 1:
            unroutable.append(descriptor)
            examples = tuple(
                f"{alias.qualified_name} => "
                f"{'eligible' if eligible else 'ineligible'}"
                for alias, eligible, _ in sorted(
                    decisions,
                    key=lambda item: _alias_sort_key(item[0]),
                )[:5]
            )
            issues.append(
                RoutingIssue(
                    severity="error",
                    code="alias_eligibility_conflict",
                    message=(
                        f"Aliases for {descriptor.canonical_name!r} disagree on "
                        f"optimizer eligibility policy {eligibility_policy!r}."
                    ),
                    component_id=component_id,
                    parameter_id=descriptor.parameter_id,
                    count=len(decisions),
                    examples=examples,
                )
            )
            continue

        eligible = next(iter(eligibility_values))
        reasons = sorted({reason for _, _, reason in decisions})
        eligibility_reason = "; ".join(reasons)

        if eligible:
            assignments.append(
                _routing_assignment(
                    descriptor,
                    component_id=component_id,
                    parameter_class=candidate.parameter_class,
                    route_kind="primary",
                    optimizer_profile=primary_name,
                    learning_rate=route["learning_rate"],
                    reason=(
                        f"Eligible for {capability.name} via "
                        f"{eligibility_policy}: {eligibility_reason}"
                    ),
                )
            )
            continue

        fallback_name = route.get("fallback_optimizer_profile")
        if fallback_name:
            fallback_profile = optimizer_profiles[fallback_name]
            fallback_capability = get_optimizer_capability(fallback_profile["type"])
            if fallback_capability.requires_parameter_eligibility:
                raise ValueError(
                    f"Canonical Parameter Policy unexpectedly routed Component "
                    f"{component_id!r} to fallback Profile {fallback_name!r}, "
                    "which still requires parameter eligibility."
                )

            fallback_lr = route.get(
                "fallback_learning_rate",
                route["learning_rate"],
            )
            assignments.append(
                _routing_assignment(
                    descriptor,
                    component_id=component_id,
                    parameter_class=candidate.parameter_class,
                    route_kind="fallback",
                    optimizer_profile=fallback_name,
                    learning_rate=fallback_lr,
                    reason=(
                        f"Ineligible for primary Profile {primary_name!r} "
                        f"({capability.name}) via {eligibility_policy}: "
                        f"{eligibility_reason} Routed to fallback "
                        f"Profile {fallback_name!r}."
                    ),
                )
            )
            fallback_usage[component_id] = fallback_usage.get(component_id, 0) + 1
            continue

        unroutable.append(descriptor)
        missing_fallback.setdefault(component_id, []).append(descriptor)

    for component_id in sorted(missing_fallback):
        descriptors = missing_fallback[component_id]
        route = next(
            candidate.route
            for candidate in ownership.trainable
            if candidate.component_id == component_id
        )
        primary_name = route["optimizer_profile"]
        primary_profile = optimizer_profiles[primary_name]
        capability = get_optimizer_capability(primary_profile["type"])
        issues.append(
            _aggregate_issue(
                code="missing_fallback",
                message=(
                    f"Component {component_id!r} uses eligibility-constrained "
                    f"Optimizer Profile {primary_name!r} ({capability.name}), but "
                    "real ineligible parameters were found and no fallback "
                    "Optimizer Profile is configured."
                ),
                component_id=component_id,
                descriptors=descriptors,
            )
        )

    return _OptimizerRoutingPass(
        assignments=tuple(sorted(assignments, key=_assignment_sort_key)),
        issues=tuple(sorted(issues, key=_issue_sort_key)),
        unroutable=tuple(
            sorted(_unique_descriptors(unroutable), key=_descriptor_sort_key)
        ),
        fallback_usage=MappingProxyType(
            {
                component_id: fallback_usage[component_id]
                for component_id in sorted(fallback_usage)
            }
        ),
    )



__all__ = [
    "AdapterTargetMetadata",
    "ParameterAlias",
    "ParameterDescriptor",
    "ParameterScanError",
    "ParameterStat",
    "RoutingAssignment",
    "RoutingIssue",
    "RoutingPlan",
    "RoutingStats",
    "classify_parameter_class",
    "scan_parameter_roots",
]
