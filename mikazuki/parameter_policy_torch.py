from __future__ import annotations

"""Torch optimizer runtime for Parameter Training Policy.

This module is runtime-only and intentionally imported only by trainer-side
integration. It constructs one child optimizer per OptimizerInstanceSpec and
exposes a single real torch.optim.Optimizer facade for Accelerate.

Step 5B intentionally does not construct LR schedulers, mutate requires_grad,
or integrate trainers.
"""

from dataclasses import dataclass
import importlib
from typing import Any, Mapping

import torch

from mikazuki.optimizer_profiles import resolve_muon_class
from mikazuki.parameter_policy_runtime import (
    PARAMETER_POLICY_RUNTIME_SPEC_VERSION,
    OptimizerInstanceSpec,
    ParameterPolicyRuntimeSpec,
)


COMPOSITE_OPTIMIZER_STATE_KIND = "dts_parameter_policy_composite_optimizer"
COMPOSITE_OPTIMIZER_STATE_VERSION = 1


class ParameterPolicyTorchRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerRuntimeEntry:
    profile_name: str
    optimizer_type: str
    topology_fingerprint: str
    optimizer: torch.optim.Optimizer


def _import_dependency(module_name: str, *, profile_name: str, optimizer_type: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ParameterPolicyTorchRuntimeError(
            f"Optimizer Profile {profile_name!r} ({optimizer_type}) requires "
            f"dependency {module_name!r}; DTS will not auto-install packages, "
            "change PyTorch/CUDA, fall back to another optimizer, or move training to CPU."
        ) from exc


def _parameter_groups(
    spec: OptimizerInstanceSpec,
    *,
    muon: bool = False,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group in spec.groups:
        params = [item.parameter for item in group.parameters]
        if not params:
            raise ParameterPolicyTorchRuntimeError(
                f"Optimizer Profile {spec.profile_name!r} contains an empty parameter group."
            )
        payload: dict[str, Any] = {
            "params": params,
            "lr": group.learning_rate,
        }
        if muon:
            payload["use_muon"] = True
        groups.append(payload)
    if not groups:
        raise ParameterPolicyTorchRuntimeError(
            f"Optimizer Profile {spec.profile_name!r} contains no parameter groups."
        )
    return groups


def _build_sgd_nesterov(
    spec: OptimizerInstanceSpec,
    *,
    optimizer_class,
):
    kwargs = spec.optimizer_arguments
    requested_nesterov = kwargs.pop("nesterov", True)
    if requested_nesterov is not True:
        raise ParameterPolicyTorchRuntimeError(
            f"Optimizer Profile {spec.profile_name!r} is SGDNesterov but explicitly "
            "sets nesterov=False."
        )
    kwargs.setdefault("momentum", 0.9)
    return optimizer_class(
        _parameter_groups(spec),
        nesterov=True,
        **kwargs,
    )


def _build_bitsandbytes_optimizer(spec: OptimizerInstanceSpec):
    module = _import_dependency(
        "bitsandbytes",
        profile_name=spec.profile_name,
        optimizer_type=spec.optimizer_type,
    )
    optim = getattr(module, "optim", None)
    if optim is None:
        raise ParameterPolicyTorchRuntimeError(
            f"bitsandbytes for Profile {spec.profile_name!r} does not expose optim."
        )

    mapping = {
        "AdamW8bit": "AdamW8bit",
        "PagedAdamW8bit": "PagedAdamW8bit",
        "PagedAdamW": "PagedAdamW",
        "PagedAdamW32bit": "PagedAdamW32bit",
        "Lion8bit": "Lion8bit",
        "PagedLion8bit": "PagedLion8bit",
    }
    class_name = mapping.get(spec.optimizer_type)
    if class_name is None:
        raise ParameterPolicyTorchRuntimeError(
            f"Unsupported bitsandbytes optimizer type {spec.optimizer_type!r}."
        )
    optimizer_class = getattr(optim, class_name, None)
    if optimizer_class is None:
        raise ParameterPolicyTorchRuntimeError(
            f"Installed bitsandbytes does not provide {class_name} for "
            f"Profile {spec.profile_name!r}."
        )
    return optimizer_class(
        _parameter_groups(spec),
        **spec.optimizer_arguments,
    )


def build_child_optimizer(spec: OptimizerInstanceSpec) -> torch.optim.Optimizer:
    """Construct one child optimizer from one deterministic profile spec."""

    optimizer_type = spec.optimizer_type

    if optimizer_type == "AdamW":
        return torch.optim.AdamW(
            _parameter_groups(spec),
            **spec.optimizer_arguments,
        )

    if optimizer_type == "SGDNesterov":
        return _build_sgd_nesterov(
            spec,
            optimizer_class=torch.optim.SGD,
        )

    if optimizer_type == "SGDNesterov8bit":
        module = _import_dependency(
            "bitsandbytes",
            profile_name=spec.profile_name,
            optimizer_type=optimizer_type,
        )
        optim = getattr(module, "optim", None)
        optimizer_class = getattr(optim, "SGD8bit", None) if optim is not None else None
        if optimizer_class is None:
            raise ParameterPolicyTorchRuntimeError(
                f"Installed bitsandbytes does not provide SGD8bit for "
                f"Profile {spec.profile_name!r}."
            )
        return _build_sgd_nesterov(spec, optimizer_class=optimizer_class)

    if optimizer_type in {
        "AdamW8bit",
        "PagedAdamW8bit",
        "PagedAdamW",
        "PagedAdamW32bit",
        "Lion8bit",
        "PagedLion8bit",
    }:
        return _build_bitsandbytes_optimizer(spec)

    if optimizer_type == "Lion":
        module = _import_dependency(
            "lion_pytorch",
            profile_name=spec.profile_name,
            optimizer_type=optimizer_type,
        )
        optimizer_class = getattr(module, "Lion", None)
        if optimizer_class is None:
            raise ParameterPolicyTorchRuntimeError(
                f"Installed lion_pytorch does not provide Lion for "
                f"Profile {spec.profile_name!r}."
            )
        return optimizer_class(
            _parameter_groups(spec),
            **spec.optimizer_arguments,
        )

    if optimizer_type == "Muon":
        optimizer_class = resolve_muon_class()
        groups = _parameter_groups(spec, muon=True)
        if any(group.get("use_muon") is not True for group in groups):
            raise ParameterPolicyTorchRuntimeError(
                f"Muon Profile {spec.profile_name!r} contains a non-Muon group; "
                "fallback parameters must be routed to a separate Optimizer Profile."
            )
        return optimizer_class(
            groups,
            **spec.optimizer_arguments,
        )

    raise ParameterPolicyTorchRuntimeError(
        f"Optimizer Profile {spec.profile_name!r} uses unsupported runtime optimizer "
        f"{optimizer_type!r}."
    )


def _audit_runtime_spec_for_torch(spec: ParameterPolicyRuntimeSpec) -> None:
    if not isinstance(spec, ParameterPolicyRuntimeSpec):
        raise ParameterPolicyTorchRuntimeError(
            "build_parameter_policy_optimizer requires ParameterPolicyRuntimeSpec."
        )
    if spec.version != PARAMETER_POLICY_RUNTIME_SPEC_VERSION:
        raise ParameterPolicyTorchRuntimeError(
            f"Unsupported Parameter Policy Runtime Spec version {spec.version!r}."
        )
    if not spec.optimizers:
        raise ParameterPolicyTorchRuntimeError(
            "Parameter Policy Runtime Spec contains no optimizer instances."
        )

    seen_profiles: set[str] = set()
    seen_parameters: dict[int, str] = {}
    for optimizer_spec in spec.optimizers:
        folded = optimizer_spec.profile_name.casefold()
        if folded in seen_profiles:
            raise ParameterPolicyTorchRuntimeError(
                f"Runtime Spec contains duplicate Optimizer Profile "
                f"{optimizer_spec.profile_name!r}."
            )
        seen_profiles.add(folded)

        if not optimizer_spec.groups:
            raise ParameterPolicyTorchRuntimeError(
                f"Optimizer Profile {optimizer_spec.profile_name!r} contains no groups."
            )
        for group in optimizer_spec.groups:
            if not group.parameters:
                raise ParameterPolicyTorchRuntimeError(
                    f"Optimizer Profile {optimizer_spec.profile_name!r} contains an empty group."
                )
            for item in group.parameters:
                actual_id = id(item.parameter)
                if item.parameter_id != actual_id:
                    raise ParameterPolicyTorchRuntimeError(
                        f"Runtime parameter {item.canonical_name!r} has stale parameter identity."
                    )
                previous = seen_parameters.get(actual_id)
                if previous is not None:
                    raise ParameterPolicyTorchRuntimeError(
                        f"Physical parameter {item.canonical_name!r} is owned by both "
                        f"{previous!r} and {optimizer_spec.profile_name!r}."
                    )
                seen_parameters[actual_id] = optimizer_spec.profile_name


class CompositeOptimizer(torch.optim.Optimizer):
    """One real Optimizer facade over disjoint child optimizers."""

    def __init__(
        self,
        runtime_spec: ParameterPolicyRuntimeSpec,
        entries: tuple[OptimizerRuntimeEntry, ...],
    ):
        if not entries:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer requires at least one child optimizer."
            )

        self.runtime_spec = runtime_spec
        self.entries = tuple(entries)

        flat_parameters = [
            parameter
            for entry in self.entries
            for group in entry.optimizer.param_groups
            for parameter in group["params"]
        ]
        if not flat_parameters:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer child optimizers expose no parameters."
            )

        self._initializing_composite_base = True
        super().__init__(flat_parameters, defaults={})
        self._initializing_composite_base = False

        # This is deliberately a shallow flattening of the child optimizers'
        # actual group dicts. Accelerate/GradScaler/schedulers must observe and
        # mutate the exact same LR/group objects used by child.step().
        self.param_groups = [
            group
            for entry in self.entries
            for group in entry.optimizer.param_groups
        ]

    @property
    def child_optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        return tuple(entry.optimizer for entry in self.entries)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if getattr(self, "_initializing_composite_base", False):
            super().add_param_group(param_group)
            return
        raise ParameterPolicyTorchRuntimeError(
            "CompositeOptimizer topology is immutable after construction; "
            "add_param_group() is not supported."
        )

    def step(self, closure=None):
        if closure is not None:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer does not support closures; DTS trainers use closure-free steps."
            )
        for entry in self.entries:
            entry.optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for entry in self.entries:
            entry.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": COMPOSITE_OPTIMIZER_STATE_KIND,
            "version": COMPOSITE_OPTIMIZER_STATE_VERSION,
            "topology_fingerprint": self.runtime_spec.topology_fingerprint,
            "children": {
                entry.profile_name: {
                    "optimizer_type": entry.optimizer_type,
                    "topology_fingerprint": entry.topology_fingerprint,
                    "state_dict": entry.optimizer.state_dict(),
                }
                for entry in self.entries
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer state must be a mapping."
            )
        if state_dict.get("kind") != COMPOSITE_OPTIMIZER_STATE_KIND:
            raise ParameterPolicyTorchRuntimeError(
                "Checkpoint optimizer state is not a DTS Parameter Policy CompositeOptimizer."
            )
        version = state_dict.get("version")
        if isinstance(version, bool) or version != COMPOSITE_OPTIMIZER_STATE_VERSION:
            raise ParameterPolicyTorchRuntimeError(
                f"Unsupported CompositeOptimizer state version {version!r}."
            )
        if state_dict.get("topology_fingerprint") != self.runtime_spec.topology_fingerprint:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer topology fingerprint does not match the current runtime."
            )

        children = state_dict.get("children")
        if not isinstance(children, Mapping):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer checkpoint children must be a mapping."
            )

        expected_names = tuple(entry.profile_name for entry in self.entries)
        if set(children) != set(expected_names):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer checkpoint child Profile set does not match current runtime."
            )

        validated_payloads: list[tuple[OptimizerRuntimeEntry, Mapping[str, Any]]] = []
        for entry in self.entries:
            payload = children.get(entry.profile_name)
            if not isinstance(payload, Mapping):
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeOptimizer child state for Profile "
                    f"{entry.profile_name!r} must be a mapping."
                )
            if payload.get("optimizer_type") != entry.optimizer_type:
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeOptimizer child {entry.profile_name!r} optimizer type "
                    "does not match current runtime."
                )
            if payload.get("topology_fingerprint") != entry.topology_fingerprint:
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeOptimizer child {entry.profile_name!r} topology "
                    "does not match current runtime."
                )
            child_state = payload.get("state_dict")
            if not isinstance(child_state, Mapping):
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeOptimizer child {entry.profile_name!r} state_dict "
                    "must be a mapping."
                )
            validated_payloads.append((entry, child_state))

        # All composite-level metadata is validated before mutating any child.
        for entry, child_state in validated_payloads:
            entry.optimizer.load_state_dict(child_state)

        # Child Optimizer.load_state_dict() may replace its param_group dicts.
        # Rebuild the facade view so Accelerate/schedulers continue to share the
        # exact live child group objects after resume.
        self.param_groups = [
            group
            for entry in self.entries
            for group in entry.optimizer.param_groups
        ]


def build_parameter_policy_optimizer(
    runtime_spec: ParameterPolicyRuntimeSpec,
) -> CompositeOptimizer:
    """Build one Accelerate-facing optimizer from the deterministic Runtime Spec."""

    _audit_runtime_spec_for_torch(runtime_spec)

    entries: list[OptimizerRuntimeEntry] = []
    constructed: list[torch.optim.Optimizer] = []
    try:
        for optimizer_spec in runtime_spec.optimizers:
            optimizer = build_child_optimizer(optimizer_spec)
            constructed.append(optimizer)
            entries.append(
                OptimizerRuntimeEntry(
                    profile_name=optimizer_spec.profile_name,
                    optimizer_type=optimizer_spec.optimizer_type,
                    topology_fingerprint=optimizer_spec.topology_fingerprint,
                    optimizer=optimizer,
                )
            )
    except Exception:
        # Optimizer objects do not own external resources that need an explicit
        # close, but dropping references prevents a partially-built runtime from
        # escaping after a later constructor fails.
        constructed.clear()
        raise

    return CompositeOptimizer(runtime_spec, tuple(entries))


__all__ = [
    "COMPOSITE_OPTIMIZER_STATE_KIND",
    "COMPOSITE_OPTIMIZER_STATE_VERSION",
    "CompositeOptimizer",
    "OptimizerRuntimeEntry",
    "ParameterPolicyTorchRuntimeError",
    "build_child_optimizer",
    "build_parameter_policy_optimizer",
]
