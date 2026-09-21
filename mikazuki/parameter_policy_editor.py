"""Host-side GUI/editor contract for Parameter Policy Step 7.

This module is intentionally dependency-light. It derives public editor metadata
from the existing Model Component Profile and Optimizer Capability registries,
normalizes legacy Schemastery string values, and provides an exact-or-fail
Standard -> Component bootstrap. It does not construct optimizers, load models,
or alter trainer/runtime behavior.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.optimizer_profiles import list_optimizer_capabilities
from mikazuki.parameter_policy import (
    PARAMETER_POLICY_GUI_KEYS,
    PARAMETER_POLICY_VERSION,
    canonicalize_parameter_policy,
    rehydrate_parameter_policy,
    validate_parameter_policy,
)
from mikazuki.parameter_policy_bootstrap import bootstrap_parameter_policy_from_standard
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.training_config import PAGE_BACKEND_MAP


_COMPONENT_MODE_ALIASES = {"component", "component-wise", "componentwise"}
_NUMERIC_LITERAL = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


def _backend_train_type(train_type: str | None) -> str:
    raw = str(train_type or "").strip()
    backend = str(PAGE_BACKEND_MAP.get(raw, raw))
    if backend not in PARAMETER_POLICY_RUNTIME_TRAIN_TYPES:
        raise ValueError(
            f"Parameter Policy editor does not support training backend {raw or train_type!r}."
        )
    get_model_component_profile(backend)
    return backend


def parameter_policy_editor_metadata(train_type: str) -> dict[str, Any]:
    """Return deterministic editor metadata derived only from runtime registries."""

    backend = _backend_train_type(train_type)
    profile = get_model_component_profile(backend)

    optimizer_types = []
    for capability in list_optimizer_capabilities():
        if capability.component_support != "supported":
            continue
        optimizer_types.append(
            {
                "type": capability.name,
                "supports_group_lr": capability.supports_group_lr,
                "uses_external_scheduler": capability.uses_external_scheduler,
                "lr_semantics": capability.lr_semantics,
                "dependency": capability.dependency,
                "eligibility_policy": capability.eligibility_policy,
                "requires_parameter_eligibility": capability.requires_parameter_eligibility,
            }
        )

    components = []
    for component_id in sorted(profile.components):
        component = profile.components[component_id]
        components.append(
            {
                "id": component.component_id,
                "label": component.display_name,
                "description": component.description,
                "roots": sorted(component.roots),
            }
        )

    return {
        "version": PARAMETER_POLICY_VERSION,
        "train_type": backend,
        "optimizer_types": optimizer_types,
        "components": components,
    }


def _parse_editor_literal(value: Any, *, field: str) -> Any:
    """Convert a Schemastery dictionary value into deterministic JSON-ish data.

    Bare text remains text so values such as Muon's quintic preset stay
    ergonomic. Values that visibly claim to be literals are parsed strictly;
    malformed structured literals fail closed instead of silently reaching an
    optimizer.
    """

    if not isinstance(value, str):
        return deepcopy(value)

    text = value.strip()
    if text == "":
        return ""

    lower = text.casefold()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None

    literal_like = (
        bool(_NUMERIC_LITERAL.fullmatch(text))
        or text[0] in {'"', "'", "[", "{", "("}
    )
    if not literal_like:
        return text

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f"Parameter Policy editor: {field} is not a valid literal: {value!r}."
        ) from exc


def normalize_parameter_policy_editor_state(config: dict) -> None:
    """Normalize active Component editor values in-place before canonicalization.

    Standard mode is a strict no-op, including stale hidden policy state.
    """

    mode = str(config.get("optimization_mode") or "standard").strip().lower()
    if mode not in _COMPONENT_MODE_ALIASES:
        return

    profiles = config.get("parameter_policy_profiles")
    if profiles is None:
        return
    if not isinstance(profiles, Mapping):
        raise ValueError("Parameter Policy editor: Optimizer Profiles must be an object.")

    normalized_profiles: dict[Any, Any] = {}
    for raw_name, raw_profile in profiles.items():
        if not isinstance(raw_profile, Mapping):
            raise ValueError(
                f"Parameter Policy editor: Optimizer Profile {raw_name!r} must be an object."
            )
        profile = deepcopy(dict(raw_profile))
        raw_args = profile.get("args")
        if raw_args is not None:
            if not isinstance(raw_args, Mapping):
                raise ValueError(
                    f"Parameter Policy editor: Optimizer Profile {raw_name!r}.args must be an object."
                )
            profile["args"] = {
                key: _parse_editor_literal(
                    value,
                    field=f"parameter_policy_profiles.{raw_name}.args.{key}",
                )
                for key, value in raw_args.items()
            }
        normalized_profiles[raw_name] = profile

    config["parameter_policy_profiles"] = normalized_profiles


def _validate_policy_backend_components(
    policy: Mapping[str, Any],
    train_type: str,
) -> None:
    backend = _backend_train_type(train_type)
    expected = set(get_model_component_profile(backend).components)
    actual = set(policy.get("components", {}))
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("extra=" + ", ".join(extra))
        raise ValueError(
            f"Parameter Policy editor: Component set does not match backend {backend!r}: "
            + "; ".join(details)
        )


def _existing_component_editor_policy(raw_config: Mapping[str, Any]) -> dict[str, Any] | None:
    mode = str(raw_config.get("optimization_mode") or "standard").strip().lower()
    if mode not in _COMPONENT_MODE_ALIASES:
        return None

    has_profiles = "parameter_policy_profiles" in raw_config
    has_components = "parameter_policy_components" in raw_config
    if not has_profiles and not has_components:
        return None
    if not (has_profiles and has_components):
        raise ValueError(
            "Parameter Policy editor: existing Component policy is incomplete; "
            "refusing to overwrite partial editor state with Standard bootstrap."
        )

    candidate = deepcopy(dict(raw_config))
    normalize_parameter_policy_editor_state(candidate)
    return canonicalize_parameter_policy(
        {
            "optimizer_profiles": candidate.get("parameter_policy_profiles"),
            "components": candidate.get("parameter_policy_components"),
        }
    )


def bootstrap_parameter_policy_editor(
    raw_config: Mapping[str, Any],
    page_train_type: str,
    *,
    resolve_backend: Callable | None = None,
) -> dict[str, Any]:
    """Return Component GUI state without overwriting an existing policy."""

    if not isinstance(raw_config, Mapping):
        raise ValueError("Parameter Policy editor bootstrap: config must be a mapping.")

    _backend_train_type(page_train_type)

    existing = _existing_component_editor_policy(raw_config)
    if existing is not None:
        _validate_policy_backend_components(existing, page_train_type)
        return rehydrate_parameter_policy(existing)

    candidate = deepcopy(dict(raw_config))
    for key in PARAMETER_POLICY_GUI_KEYS:
        candidate.pop(key, None)
    candidate.pop("parameter_policy_config", None)
    candidate["optimization_mode"] = "standard"

    if resolve_backend is None:
        import mikazuki.app.api as legacy_api

        resolve_backend = legacy_api.resolve_training_backend

    policy = bootstrap_parameter_policy_from_standard(
        candidate,
        page_train_type,
        resolve_backend=resolve_backend,
    )
    return rehydrate_parameter_policy(policy)


def parameter_policy_editor_preview(
    policy: Mapping[str, Any],
    train_type: str,
    runtime_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a model-free component-level preview for the future GUI surface."""

    canonical = validate_parameter_policy(policy)
    _validate_policy_backend_components(canonical, train_type)
    metadata = parameter_policy_editor_metadata(train_type)
    component_meta = {row["id"]: row for row in metadata["components"]}

    profiles = [
        {
            "name": name,
            "type": profile["type"],
            "args": deepcopy(profile["args"]),
        }
        for name, profile in canonical["optimizer_profiles"].items()
    ]

    components = []
    for component_id, route in canonical["components"].items():
        meta = component_meta.get(component_id)
        components.append(
            {
                "id": component_id,
                "label": meta["label"] if meta else component_id,
                "description": meta["description"] if meta else "",
                **deepcopy(route),
            }
        )

    blockers = list(runtime_blockers)
    return {
        "version": canonical["version"],
        "train_type": metadata["train_type"],
        "runtime_ready": not bool(blockers),
        "runtime_blockers": blockers,
        "profiles": profiles,
        "components": components,
    }


__all__ = [
    "bootstrap_parameter_policy_editor",
    "normalize_parameter_policy_editor_state",
    "parameter_policy_editor_metadata",
    "parameter_policy_editor_preview",
]
