from __future__ import annotations

"""Optimizer capability metadata for the future Parameter Training Policy.

This module is intentionally host-side and opt-in.  Importing it does not alter
legacy optimizer selection, trainer arguments, device placement, or PyTorch.
The existing Standard training path must continue to use sd-scripts'
get_optimizer() until a Parameter Policy is explicitly enabled.
"""

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping


ComponentSupport = Literal["supported", "restricted", "planned"]


@dataclass(frozen=True)
class OptimizerCapability:
    name: str
    component_support: ComponentSupport
    supports_group_lr: bool
    uses_external_scheduler: bool
    lr_semantics: Literal["normal", "adaptive", "optimizer_managed"]
    dependency: str | None = None
    requires_parameter_eligibility: bool = False
    restriction: str | None = None


# Keep this registry independent from the legacy Standard optimizer dropdown.
# Muon deliberately appears here first: exposing it through the old global
# optimizer_type would incorrectly route non-hidden/non-2D parameters to Muon.
_CAPABILITIES: tuple[OptimizerCapability, ...] = (
    OptimizerCapability("AdamW", "supported", True, True, "normal"),
    OptimizerCapability("AdamW8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("PagedAdamW8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("Lion", "supported", True, True, "normal", dependency="lion-pytorch"),
    OptimizerCapability("Lion8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("PagedLion8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("SGDNesterov", "supported", True, True, "normal"),
    OptimizerCapability("SGDNesterov8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability(
        "RAdamScheduleFree",
        "supported",
        True,
        False,
        "optimizer_managed",
        dependency="schedulefree",
    ),
    OptimizerCapability(
        "DAdaptation",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdam",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdaGrad",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdanIP",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptLion",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptSGD",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "Prodigy",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="prodigyopt",
        restriction="Component mode must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "AdaFactor",
        "restricted",
        True,
        True,
        "normal",
        dependency="transformers",
        restriction="Component mode v1 requires relative_step=False.",
    ),
    OptimizerCapability(
        "prodigyplus.ProdigyPlusScheduleFree",
        "restricted",
        False,
        False,
        "adaptive",
        dependency="prodigy-plus-schedule-free",
        restriction="Needs dedicated adaptive/schedule-free profile handling before training integration.",
    ),
    OptimizerCapability(
        "pytorch_optimizer.CAME",
        "planned",
        True,
        True,
        "normal",
        dependency="pytorch-optimizer",
        restriction="Enable only after dedicated GPU smoke coverage.",
    ),
    OptimizerCapability(
        "Muon",
        "supported",
        True,
        True,
        "normal",
        dependency="torch.optim.Muon",
        requires_parameter_eligibility=True,
        restriction="Only model-profile-approved hidden-layer 2D weights are eligible; other parameters require fallback routing.",
    ),
)

_BY_NAME = {cap.name.casefold(): cap for cap in _CAPABILITIES}

MUON_ARGUMENTS = frozenset(
    {
        "momentum",
        "weight_decay",
        "nesterov",
        "ns_coefficients",
        "eps",
        "ns_steps",
        "adjust_lr_fn",
    }
)
MUON_ADJUST_LR_MODES = frozenset({"original", "match_rms_adamw", "spectral_unclamped"})
_PROFILE_RESERVED_ARGUMENTS = frozenset({"params", "lr"})


def list_optimizer_capabilities() -> tuple[OptimizerCapability, ...]:
    return _CAPABILITIES


def get_optimizer_capability(optimizer_type: str) -> OptimizerCapability:
    key = str(optimizer_type or "").strip().casefold()
    if not key:
        raise ValueError("Optimizer Profile requires an optimizer type.")
    capability = _BY_NAME.get(key)
    if capability is None:
        raise ValueError(
            f"Optimizer {optimizer_type!r} is not registered for Parameter Training Policy."
        )
    return capability


def canonical_optimizer_type(optimizer_type: str) -> str:
    return get_optimizer_capability(optimizer_type).name


def _finite_number(value: Any, *, field: str, minimum: float | None = None, strict_minimum: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Muon {field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Muon {field} must be a finite number.")
    if minimum is not None:
        invalid = number <= minimum if strict_minimum else number < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"Muon {field} must be {relation} {minimum}.")
    return number


def validate_muon_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate only arguments accepted by the native torch.optim.Muon API.

    Learning rate intentionally does not belong here.  In Parameter Training
    Policy it belongs to each Component/param-group so one Muon profile can
    serve multiple components with different LRs.
    """

    raw = dict(arguments or {})
    reserved = _PROFILE_RESERVED_ARGUMENTS.intersection(raw)
    if reserved:
        raise ValueError(
            "Optimizer Profile args may not define "
            + ", ".join(sorted(reserved))
            + "; Parameter Policy owns parameter routing and LR."
        )

    unknown = set(raw).difference(MUON_ARGUMENTS)
    if unknown:
        raise ValueError(
            "Unsupported Muon argument(s): " + ", ".join(sorted(unknown))
        )

    result: dict[str, Any] = {}

    if "momentum" in raw:
        momentum = _finite_number(raw["momentum"], field="momentum", minimum=0.0)
        if momentum >= 1.0:
            raise ValueError("Muon momentum must be < 1.0.")
        result["momentum"] = momentum

    if "weight_decay" in raw:
        result["weight_decay"] = _finite_number(
            raw["weight_decay"], field="weight_decay", minimum=0.0
        )

    if "nesterov" in raw:
        if not isinstance(raw["nesterov"], bool):
            raise ValueError("Muon nesterov must be boolean.")
        result["nesterov"] = raw["nesterov"]

    if "ns_coefficients" in raw:
        coeffs = raw["ns_coefficients"]
        if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 3:
            raise ValueError("Muon ns_coefficients must contain exactly three numbers.")
        result["ns_coefficients"] = tuple(
            _finite_number(value, field="ns_coefficients") for value in coeffs
        )

    if "eps" in raw:
        result["eps"] = _finite_number(
            raw["eps"], field="eps", minimum=0.0, strict_minimum=True
        )

    if "ns_steps" in raw:
        value = raw["ns_steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Muon ns_steps must be an integer >= 1.")
        result["ns_steps"] = value

    if "adjust_lr_fn" in raw:
        value = raw["adjust_lr_fn"]
        if value is not None and value not in MUON_ADJUST_LR_MODES:
            raise ValueError(
                "Muon adjust_lr_fn must be one of: "
                + ", ".join(sorted(MUON_ADJUST_LR_MODES))
                + ", or null."
            )
        result["adjust_lr_fn"] = value

    return result


def normalize_optimizer_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical optimizer profile without constructing an optimizer."""

    if not isinstance(profile, Mapping):
        raise ValueError("Optimizer Profile must be an object.")

    optimizer_type = canonical_optimizer_type(str(profile.get("type") or ""))
    capability = get_optimizer_capability(optimizer_type)

    raw_args = profile.get("args") or {}
    if not isinstance(raw_args, Mapping):
        raise ValueError("Optimizer Profile args must be an object.")

    reserved = _PROFILE_RESERVED_ARGUMENTS.intersection(raw_args)
    if reserved:
        raise ValueError(
            "Optimizer Profile args may not define "
            + ", ".join(sorted(reserved))
            + "; Parameter Policy owns parameter routing and LR."
        )

    if optimizer_type == "Muon":
        args = validate_muon_arguments(raw_args)
    else:
        args = dict(raw_args)

    return {
        "type": capability.name,
        "args": args,
    }


def require_component_optimizer_support(optimizer_type: str) -> OptimizerCapability:
    capability = get_optimizer_capability(optimizer_type)
    if capability.component_support == "planned":
        raise ValueError(
            f"Optimizer {capability.name} is registered but not enabled for Component-wise training yet: "
            f"{capability.restriction or 'GPU validation is still required.'}"
        )
    return capability


def resolve_native_muon_class(torch_module: Any | None = None) -> type:
    """Resolve torch.optim.Muon lazily and fail closed when unavailable.

    This function never installs packages, changes PyTorch, changes device
    placement, or falls back to CPU/another optimizer.
    """

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    optim = getattr(torch_module, "optim", None)
    muon = getattr(optim, "Muon", None) if optim is not None else None
    if muon is None:
        version = getattr(torch_module, "__version__", "unknown")
        raise RuntimeError(
            "Muon requires a PyTorch build that provides torch.optim.Muon "
            f"(current torch={version}). DTS will not auto-upgrade PyTorch, "
            "fall back to another optimizer, or move training to CPU."
        )
    return muon
