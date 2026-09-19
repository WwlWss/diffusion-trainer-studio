from __future__ import annotations

"""Optimizer capability metadata for the future Parameter Training Policy.

This module is deliberately host-side and opt-in. Importing it does not alter
legacy optimizer selection, trainer arguments, device placement, dependencies,
or PyTorch. Standard training must continue to use sd-scripts' get_optimizer()
until a Parameter Policy is explicitly enabled.
"""

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping


ComponentSupport = Literal["supported", "restricted", "planned"]
ParameterEligibilityPolicy = Literal["model_hidden_2d_weight"]


@dataclass(frozen=True)
class OptimizerCapability:
    name: str
    component_support: ComponentSupport
    supports_group_lr: bool
    uses_external_scheduler: bool
    lr_semantics: Literal["normal", "adaptive", "optimizer_managed"]
    dependency: str | None = None
    implementation: str | None = None
    eligibility_policy: ParameterEligibilityPolicy | None = None
    restriction: str | None = None

    @property
    def requires_parameter_eligibility(self) -> bool:
        """Backward-compatible Step 2 view of the explicit eligibility contract."""

        return self.eligibility_policy is not None


# "supported" means intended for Component v1 without an extra semantic gate.
# "restricted" means intended for v1 only after its optimizer-specific contract
# is validated. "planned" is known to DTS but deliberately unavailable in v1.
_CAPABILITIES: tuple[OptimizerCapability, ...] = (
    OptimizerCapability("AdamW", "supported", True, True, "normal", implementation="torch.optim.AdamW"),
    OptimizerCapability("AdamW8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("PagedAdamW8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("PagedAdamW", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("PagedAdamW32bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("Lion", "supported", True, True, "normal", dependency="lion-pytorch"),
    OptimizerCapability("Lion8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("PagedLion8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability("SGDNesterov", "supported", True, True, "normal", implementation="torch.optim.SGD"),
    OptimizerCapability("SGDNesterov8bit", "supported", True, True, "normal", dependency="bitsandbytes"),
    OptimizerCapability(
        "RAdamScheduleFree",
        "restricted",
        True,
        False,
        "optimizer_managed",
        dependency="schedulefree",
        restriction="Requires schedule-free train/eval lifecycle handling and no external LR scheduler.",
    ),
    OptimizerCapability(
        "AdamWScheduleFree",
        "restricted",
        True,
        False,
        "optimizer_managed",
        dependency="schedulefree",
        restriction="Requires schedule-free train/eval lifecycle handling and no external LR scheduler.",
    ),
    OptimizerCapability(
        "SGDScheduleFree",
        "restricted",
        True,
        False,
        "optimizer_managed",
        dependency="schedulefree",
        restriction="Requires schedule-free train/eval lifecycle handling and no external LR scheduler.",
    ),
    OptimizerCapability(
        "DAdaptation",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdamPreprint",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdam",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdaGrad",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdan",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptAdanIP",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptLion",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "DAdaptSGD",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="dadaptation",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "Prodigy",
        "restricted",
        False,
        True,
        "adaptive",
        dependency="prodigyopt",
        restriction="Component v1 must use profile-owned adaptive LR semantics.",
    ),
    OptimizerCapability(
        "AdaFactor",
        "restricted",
        True,
        True,
        "normal",
        dependency="transformers",
        restriction="Component v1 requires relative_step=False.",
    ),
    OptimizerCapability(
        "prodigyplus.ProdigyPlusScheduleFree",
        "restricted",
        False,
        False,
        "adaptive",
        dependency="prodigy-plus-schedule-free",
        restriction="Needs dedicated adaptive and schedule-free profile handling.",
    ),
    OptimizerCapability(
        "pytorch_optimizer.CAME",
        "planned",
        True,
        True,
        "normal",
        dependency="pytorch-optimizer==3.10.0",
        restriction="Enable only after dedicated GPU smoke coverage.",
    ),
    OptimizerCapability(
        "Custom",
        "planned",
        False,
        True,
        "normal",
        restriction="Arbitrary custom optimizers need an explicit capability contract before Component-wise use.",
    ),
    OptimizerCapability(
        "Muon",
        "supported",
        True,
        True,
        "normal",
        dependency="pytorch-optimizer==3.10.0",
        implementation="pytorch_optimizer.Muon",
        eligibility_policy="model_hidden_2d_weight",
        restriction=(
            "Only model-profile-approved hidden-layer 2D weights are eligible; "
            "other parameters require fallback routing."
        ),
    ),
)

_BY_NAME = {cap.name.casefold(): cap for cap in _CAPABILITIES}

# DTS pins pytorch-optimizer 3.10.0. Use that Muon implementation instead of
# requiring a newer torch build. Do not expose the class's internal AdamW
# fallback controls: Parameter Policy owns fallback routing as a separate
# Optimizer Profile.
MUON_ARGUMENTS = frozenset(
    {
        "momentum",
        "weight_decay",
        "weight_decouple",
        "nesterov",
        "ns_steps",
        "ns_coeffs",
        "use_adjusted_lr",
    }
)
MUON_NS_PRESETS = frozenset({"original", "quintic", "polar_express", "polar_express_safer"})
_PROFILE_RESERVED_ARGUMENTS = frozenset({"params", "lr", "use_muon", "adamw_lr", "adamw_betas", "adamw_wd", "adamw_eps"})


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


def require_component_optimizer_candidate(optimizer_type: str) -> OptimizerCapability:
    """Reject v1-deferred optimizers but preserve restricted ones for explicit validators."""

    capability = get_optimizer_capability(optimizer_type)
    if capability.component_support == "planned":
        raise ValueError(
            f"Optimizer {capability.name} is registered but not enabled for Component-wise v1: "
            f"{capability.restriction or 'dedicated validation is still required.'}"
        )
    return capability


def require_unrestricted_component_optimizer(optimizer_type: str) -> OptimizerCapability:
    """Return only optimizers that need no additional semantic gate."""

    capability = require_component_optimizer_candidate(optimizer_type)
    if capability.component_support != "supported":
        raise ValueError(
            f"Optimizer {capability.name} requires optimizer-specific Component-wise validation: "
            f"{capability.restriction or 'restricted capability.'}"
        )
    return capability


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
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
    if maximum is not None:
        invalid = number >= maximum if strict_maximum else number > maximum
        if invalid:
            relation = "<" if strict_maximum else "<="
            raise ValueError(f"Muon {field} must be {relation} {maximum}.")
    return number


def _canonical_json_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Optimizer Profile {field} must not contain NaN or infinity.")
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item, field=field) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Optimizer Profile {field} object keys must be strings.")
            result[key] = _canonical_json_value(item, field=field)
        return result
    raise ValueError(
        f"Optimizer Profile {field} contains unsupported non-JSON value {type(value).__name__}."
    )


def validate_muon_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the DTS-supported subset of pytorch_optimizer.Muon arguments."""

    raw = dict(arguments or {})
    reserved = _PROFILE_RESERVED_ARGUMENTS.intersection(raw)
    if reserved:
        raise ValueError(
            "Optimizer Profile args may not define "
            + ", ".join(sorted(reserved))
            + "; Parameter Policy owns parameter routing, fallback routing, and LR."
        )

    unknown = set(raw).difference(MUON_ARGUMENTS)
    if unknown:
        raise ValueError("Unsupported Muon argument(s): " + ", ".join(sorted(unknown)))

    result: dict[str, Any] = {}

    if "momentum" in raw:
        result["momentum"] = _finite_number(
            raw["momentum"], field="momentum", minimum=0.0, maximum=1.0, strict_maximum=True
        )

    if "weight_decay" in raw:
        result["weight_decay"] = _finite_number(
            raw["weight_decay"], field="weight_decay", minimum=0.0
        )

    for field in ("weight_decouple", "nesterov", "use_adjusted_lr"):
        if field in raw:
            if not isinstance(raw[field], bool):
                raise ValueError(f"Muon {field} must be boolean.")
            result[field] = raw[field]

    if "ns_steps" in raw:
        value = raw["ns_steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Muon ns_steps must be an integer >= 1.")
        result["ns_steps"] = value

    if "ns_coeffs" in raw:
        value = raw["ns_coeffs"]
        if not isinstance(value, str) or value not in MUON_NS_PRESETS:
            raise ValueError(
                "Muon ns_coeffs must be one of: " + ", ".join(sorted(MUON_NS_PRESETS)) + "."
            )
        result["ns_coeffs"] = value

    return result


def normalize_optimizer_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a profile into JSON-safe host configuration.

    This does not construct an optimizer and does not make a restricted
    optimizer valid; policy-level validators must apply the capability's
    restriction before training.
    """

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
            + "; Parameter Policy owns parameter routing, fallback routing, and LR."
        )

    if optimizer_type == "Muon":
        args = validate_muon_arguments(raw_args)
    else:
        args = _canonical_json_value(dict(raw_args), field="args")

    return {"type": capability.name, "args": args}


def resolve_muon_class(pytorch_optimizer_module: Any | None = None) -> type:
    """Resolve the pinned pytorch-optimizer Muon lazily and fail closed.

    The resolver never installs packages, upgrades PyTorch, changes CUDA,
    changes device placement, falls back to another optimizer, or moves
    training to CPU.
    """

    if pytorch_optimizer_module is None:
        try:
            import pytorch_optimizer as pytorch_optimizer_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "Muon requires DTS's pinned pytorch-optimizer dependency. "
                "DTS will not auto-install packages, upgrade PyTorch, fall back "
                "to another optimizer, or move training to CPU."
            ) from exc

    muon = getattr(pytorch_optimizer_module, "Muon", None)
    if muon is None:
        version = getattr(pytorch_optimizer_module, "__version__", "unknown")
        raise RuntimeError(
            "The installed pytorch-optimizer build does not provide Muon "
            f"(current pytorch-optimizer={version}). DTS will not auto-upgrade "
            "dependencies, fall back to another optimizer, or move training to CPU."
        )
    return muon
