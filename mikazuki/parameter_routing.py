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
from typing import Any, Mapping

from mikazuki.model_component_profiles import ParameterClass


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


__all__ = [
    "AdapterTargetMetadata",
    "ParameterAlias",
    "ParameterDescriptor",
    "ParameterScanError",
    "classify_parameter_class",
    "scan_parameter_roots",
]
