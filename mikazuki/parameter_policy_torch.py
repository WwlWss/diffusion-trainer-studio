from __future__ import annotations

"""Torch optimizer runtime for Parameter Training Policy.

This module is runtime-only and intentionally imported only by trainer-side
integration. It constructs one child optimizer per OptimizerInstanceSpec and
exposes a single real torch.optim.Optimizer facade for Accelerate.

Step 5C adds a single real LRScheduler facade and schedule-free lifecycle
forwarding. This module still does not mutate requires_grad or integrate trainers.
"""

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Literal, Mapping

import torch

from mikazuki.optimizer_profiles import resolve_muon_class
from mikazuki.parameter_policy_runtime import (
    PARAMETER_POLICY_RUNTIME_SPEC_VERSION,
    OptimizerInstanceSpec,
    ParameterPolicyRuntimeSpec,
)


COMPOSITE_OPTIMIZER_STATE_KIND = "dts_parameter_policy_composite_optimizer"
COMPOSITE_OPTIMIZER_STATE_VERSION = 1
COMPOSITE_SCHEDULER_STATE_KIND = "dts_parameter_policy_composite_scheduler"
COMPOSITE_SCHEDULER_STATE_VERSION = 1
SchedulerMode = Literal["external", "optimizer_managed"]


class ParameterPolicyTorchRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerRuntimeEntry:
    profile_name: str
    optimizer_type: str
    topology_fingerprint: str
    optimizer: torch.optim.Optimizer


@dataclass(frozen=True)
class SchedulerRuntimeEntry:
    profile_name: str
    optimizer_type: str
    topology_fingerprint: str
    mode: SchedulerMode
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    scheduler_type: str | None


SchedulerFactory = Callable[
    [torch.optim.Optimizer],
    torch.optim.lr_scheduler.LRScheduler,
]


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

    if optimizer_type in {
        "RAdamScheduleFree",
        "AdamWScheduleFree",
        "SGDScheduleFree",
    }:
        module = _import_dependency(
            "schedulefree",
            profile_name=spec.profile_name,
            optimizer_type=optimizer_type,
        )
        optimizer_class = getattr(module, optimizer_type, None)
        if optimizer_class is None:
            raise ParameterPolicyTorchRuntimeError(
                f"Installed schedulefree does not provide {optimizer_type} for "
                f"Profile {spec.profile_name!r}."
            )
        optimizer = optimizer_class(
            _parameter_groups(spec),
            **spec.optimizer_arguments,
        )
        if not callable(getattr(optimizer, "train", None)) or not callable(
            getattr(optimizer, "eval", None)
        ):
            raise ParameterPolicyTorchRuntimeError(
                f"ScheduleFree Profile {spec.profile_name!r} does not expose "
                "the required train()/eval() lifecycle."
            )
        return optimizer

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

    def train(self):
        for entry in self.entries:
            train_fn = getattr(entry.optimizer, "train", None)
            if callable(train_fn):
                train_fn()
        return self

    def eval(self):
        for entry in self.entries:
            eval_fn = getattr(entry.optimizer, "eval", None)
            if callable(eval_fn):
                eval_fn()
        return self

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

    composite = CompositeOptimizer(runtime_spec, tuple(entries))
    composite.train()
    return composite


def _qualified_type(value: object) -> str:
    cls = value.__class__
    module_name = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", ""))
    return f"{module_name}.{qualname}" if module_name else qualname


class CompositeLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """One Accelerate-facing scheduler over per-optimizer child schedulers.

    PyTorch LRScheduler.__init__ is intentionally not called: every external
    child scheduler has already executed its own provider-defined initialization
    step, while optimizer-managed children deliberately have no scheduler.
    """

    def __init__(
        self,
        optimizer: CompositeOptimizer,
        entries: tuple[SchedulerRuntimeEntry, ...],
    ):
        if not isinstance(optimizer, CompositeOptimizer):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler requires a CompositeOptimizer."
            )
        if not entries:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler requires at least one runtime entry."
            )
        if tuple(item.profile_name for item in entries) != tuple(
            item.profile_name for item in optimizer.entries
        ):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler entry order does not match CompositeOptimizer."
            )

        self.optimizer = optimizer
        self.entries = tuple(entries)
        self.base_lrs = [
            group.get("initial_lr", group["lr"])
            for group in optimizer.param_groups
        ]

        external = [
            item.scheduler
            for item in self.entries
            if item.mode == "external"
        ]
        self._composite_step_count = self._common_external_int(
            external,
            "_step_count",
            default=0,
        )
        self.last_epoch = self._common_external_int(
            external,
            "last_epoch",
            default=-1,
        )
        self._last_lr = self._collect_last_lr()

    @staticmethod
    def _common_external_int(schedulers, attribute: str, *, default: int) -> int:
        values = []
        for scheduler in schedulers:
            if scheduler is None:
                continue
            value = getattr(scheduler, attribute, None)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ParameterPolicyTorchRuntimeError(
                    f"External scheduler {_qualified_type(scheduler)!r} does not "
                    f"expose integer {attribute}."
                )
            values.append(value)
        if not values:
            return default
        if len(set(values)) != 1:
            raise ParameterPolicyTorchRuntimeError(
                f"External child schedulers disagree on {attribute}: {values!r}."
            )
        return values[0]

    @property
    def _step_count(self) -> int:
        return self._composite_step_count

    @_step_count.setter
    def _step_count(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ParameterPolicyTorchRuntimeError(
                f"CompositeLRScheduler _step_count must be a non-negative integer, got {value!r}."
            )
        old = getattr(self, "_composite_step_count", value)
        delta = value - old
        if delta:
            for entry in getattr(self, "entries", ()):
                if entry.mode == "external" and entry.scheduler is not None:
                    child_value = getattr(entry.scheduler, "_step_count", None)
                    if isinstance(child_value, bool) or not isinstance(child_value, int):
                        raise ParameterPolicyTorchRuntimeError(
                            f"External scheduler {entry.scheduler_type!r} does not "
                            "expose integer _step_count."
                        )
                    entry.scheduler._step_count = child_value + delta
        self._composite_step_count = value

    def _collect_last_lr(self) -> list[Any]:
        values: list[Any] = []
        for entry in self.entries:
            if entry.mode == "external":
                if entry.scheduler is None:
                    raise ParameterPolicyTorchRuntimeError(
                        f"External scheduler entry {entry.profile_name!r} has no scheduler."
                    )
                child_values = list(entry.scheduler.get_last_lr())
            else:
                child_values = [
                    group["lr"]
                    for group in entry.optimizer.param_groups
                ]
            if len(child_values) != len(entry.optimizer.param_groups):
                raise ParameterPolicyTorchRuntimeError(
                    f"Scheduler LR count for Profile {entry.profile_name!r} does not "
                    "match its optimizer group count."
                )
            values.extend(child_values)
        if len(values) != len(self.optimizer.param_groups):
            raise ParameterPolicyTorchRuntimeError(
                "Composite scheduler LR count does not match CompositeOptimizer groups."
            )
        return values

    def step(self, epoch=None):
        external_schedulers = []
        for entry in self.entries:
            if entry.mode != "external":
                continue
            scheduler = entry.scheduler
            if scheduler is None:
                raise ParameterPolicyTorchRuntimeError(
                    f"External scheduler entry {entry.profile_name!r} has no scheduler."
                )
            if epoch is None:
                scheduler.step()
            else:
                scheduler.step(epoch)
            external_schedulers.append(scheduler)

        if external_schedulers:
            self._composite_step_count = self._common_external_int(
                external_schedulers,
                "_step_count",
                default=self._composite_step_count,
            )
            self.last_epoch = self._common_external_int(
                external_schedulers,
                "last_epoch",
                default=self.last_epoch,
            )
        else:
            self._composite_step_count += 1
            if epoch is None:
                self.last_epoch += 1
            else:
                if isinstance(epoch, bool) or not isinstance(epoch, int):
                    raise ParameterPolicyTorchRuntimeError(
                        f"CompositeLRScheduler epoch must be an integer, got {epoch!r}."
                    )
                self.last_epoch = epoch

        self._last_lr = self._collect_last_lr()

    def get_last_lr(self) -> list[Any]:
        self._last_lr = self._collect_last_lr()
        return list(self._last_lr)

    def get_last_lr_by_profile(self) -> dict[str, tuple[Any, ...]]:
        result: dict[str, tuple[Any, ...]] = {}
        for entry in self.entries:
            if entry.mode == "external":
                if entry.scheduler is None:
                    raise ParameterPolicyTorchRuntimeError(
                        f"External scheduler entry {entry.profile_name!r} has no scheduler."
                    )
                values = tuple(entry.scheduler.get_last_lr())
            else:
                values = tuple(group["lr"] for group in entry.optimizer.param_groups)
            result[entry.profile_name] = values
        return result

    def get_lr(self) -> list[Any]:
        return self.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": COMPOSITE_SCHEDULER_STATE_KIND,
            "version": COMPOSITE_SCHEDULER_STATE_VERSION,
            "topology_fingerprint": self.optimizer.runtime_spec.topology_fingerprint,
            "step_count": self._step_count,
            "last_epoch": self.last_epoch,
            "children": {
                entry.profile_name: {
                    "optimizer_type": entry.optimizer_type,
                    "topology_fingerprint": entry.topology_fingerprint,
                    "mode": entry.mode,
                    "scheduler_type": entry.scheduler_type,
                    **(
                        {"state_dict": entry.scheduler.state_dict()}
                        if entry.scheduler is not None
                        else {}
                    ),
                }
                for entry in self.entries
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler state must be a mapping."
            )
        if state_dict.get("kind") != COMPOSITE_SCHEDULER_STATE_KIND:
            raise ParameterPolicyTorchRuntimeError(
                "Checkpoint scheduler state is not a DTS Parameter Policy CompositeLRScheduler."
            )
        version = state_dict.get("version")
        if isinstance(version, bool) or version != COMPOSITE_SCHEDULER_STATE_VERSION:
            raise ParameterPolicyTorchRuntimeError(
                f"Unsupported CompositeLRScheduler state version {version!r}."
            )
        if (
            state_dict.get("topology_fingerprint")
            != self.optimizer.runtime_spec.topology_fingerprint
        ):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler topology fingerprint does not match current runtime."
            )

        step_count = state_dict.get("step_count")
        last_epoch = state_dict.get("last_epoch")
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
            raise ParameterPolicyTorchRuntimeError(
                f"CompositeLRScheduler checkpoint has invalid step_count {step_count!r}."
            )
        if isinstance(last_epoch, bool) or not isinstance(last_epoch, int):
            raise ParameterPolicyTorchRuntimeError(
                f"CompositeLRScheduler checkpoint has invalid last_epoch {last_epoch!r}."
            )

        children = state_dict.get("children")
        if not isinstance(children, Mapping):
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler checkpoint children must be a mapping."
            )
        expected = {entry.profile_name for entry in self.entries}
        if set(children) != expected:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeLRScheduler checkpoint child Profile set does not match current runtime."
            )

        validated_payloads = []
        for entry in self.entries:
            payload = children.get(entry.profile_name)
            if not isinstance(payload, Mapping):
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeLRScheduler child {entry.profile_name!r} must be a mapping."
                )
            if payload.get("optimizer_type") != entry.optimizer_type:
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeLRScheduler child {entry.profile_name!r} optimizer type "
                    "does not match current runtime."
                )
            if payload.get("topology_fingerprint") != entry.topology_fingerprint:
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeLRScheduler child {entry.profile_name!r} topology "
                    "does not match current runtime."
                )
            if payload.get("mode") != entry.mode:
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeLRScheduler child {entry.profile_name!r} mode "
                    "does not match current runtime."
                )
            if payload.get("scheduler_type") != entry.scheduler_type:
                raise ParameterPolicyTorchRuntimeError(
                    f"CompositeLRScheduler child {entry.profile_name!r} scheduler type "
                    "does not match current runtime."
                )

            if entry.mode == "external":
                child_state = payload.get("state_dict")
                if not isinstance(child_state, Mapping):
                    raise ParameterPolicyTorchRuntimeError(
                        f"External scheduler child {entry.profile_name!r} state_dict "
                        "must be a mapping."
                    )
                child_step = child_state.get("_step_count")
                if child_step is not None and child_step != step_count:
                    raise ParameterPolicyTorchRuntimeError(
                        f"External scheduler child {entry.profile_name!r} step count "
                        "does not match composite checkpoint."
                    )
                child_epoch = child_state.get("last_epoch")
                if child_epoch is not None and child_epoch != last_epoch:
                    raise ParameterPolicyTorchRuntimeError(
                        f"External scheduler child {entry.profile_name!r} last_epoch "
                        "does not match composite checkpoint."
                    )
                validated_payloads.append((entry, child_state))
            elif "state_dict" in payload:
                raise ParameterPolicyTorchRuntimeError(
                    f"Optimizer-managed scheduler child {entry.profile_name!r} must not "
                    "carry external scheduler state."
                )

        # Validate every composite-level and child metadata field before mutation.
        for entry, child_state in validated_payloads:
            if entry.scheduler is None:
                raise ParameterPolicyTorchRuntimeError(
                    f"External scheduler entry {entry.profile_name!r} has no scheduler."
                )
            entry.scheduler.load_state_dict(child_state)

        self._composite_step_count = step_count
        self.last_epoch = last_epoch
        self._last_lr = self._collect_last_lr()


def build_parameter_policy_scheduler(
    optimizer: CompositeOptimizer,
    scheduler_factory: SchedulerFactory | None,
) -> CompositeLRScheduler:
    """Build one Accelerate-facing scheduler over the CompositeOptimizer."""

    if not isinstance(optimizer, CompositeOptimizer):
        raise ParameterPolicyTorchRuntimeError(
            "build_parameter_policy_scheduler requires CompositeOptimizer."
        )

    specs = optimizer.runtime_spec.optimizers
    if len(specs) != len(optimizer.entries):
        raise ParameterPolicyTorchRuntimeError(
            "CompositeOptimizer runtime entries do not match Runtime Spec."
        )

    entries: list[SchedulerRuntimeEntry] = []
    for spec, optimizer_entry in zip(specs, optimizer.entries):
        if spec.profile_name != optimizer_entry.profile_name:
            raise ParameterPolicyTorchRuntimeError(
                "CompositeOptimizer Profile order does not match Runtime Spec."
            )

        if spec.uses_external_scheduler:
            if scheduler_factory is None:
                raise ParameterPolicyTorchRuntimeError(
                    f"Optimizer Profile {spec.profile_name!r} requires an external "
                    "scheduler but no scheduler_factory was supplied."
                )
            scheduler = scheduler_factory(optimizer_entry.optimizer)
            if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise ParameterPolicyTorchRuntimeError(
                    f"Scheduler factory for Profile {spec.profile_name!r} returned "
                    f"{type(scheduler).__name__}, not torch.optim.lr_scheduler.LRScheduler."
                )
            if getattr(scheduler, "optimizer", None) is not optimizer_entry.optimizer:
                raise ParameterPolicyTorchRuntimeError(
                    f"Scheduler for Profile {spec.profile_name!r} is not attached "
                    "to that child optimizer."
                )
            entries.append(
                SchedulerRuntimeEntry(
                    profile_name=spec.profile_name,
                    optimizer_type=spec.optimizer_type,
                    topology_fingerprint=spec.topology_fingerprint,
                    mode="external",
                    optimizer=optimizer_entry.optimizer,
                    scheduler=scheduler,
                    scheduler_type=_qualified_type(scheduler),
                )
            )
        else:
            if spec.lr_semantics != "optimizer_managed":
                raise ParameterPolicyTorchRuntimeError(
                    f"Optimizer Profile {spec.profile_name!r} disables external "
                    f"scheduling but has lr_semantics={spec.lr_semantics!r}."
                )
            entries.append(
                SchedulerRuntimeEntry(
                    profile_name=spec.profile_name,
                    optimizer_type=spec.optimizer_type,
                    topology_fingerprint=spec.topology_fingerprint,
                    mode="optimizer_managed",
                    optimizer=optimizer_entry.optimizer,
                    scheduler=None,
                    scheduler_type=None,
                )
            )

    return CompositeLRScheduler(optimizer, tuple(entries))


__all__ = [
    "COMPOSITE_OPTIMIZER_STATE_KIND",
    "COMPOSITE_OPTIMIZER_STATE_VERSION",
    "COMPOSITE_SCHEDULER_STATE_KIND",
    "COMPOSITE_SCHEDULER_STATE_VERSION",
    "CompositeLRScheduler",
    "CompositeOptimizer",
    "OptimizerRuntimeEntry",
    "SchedulerRuntimeEntry",
    "ParameterPolicyTorchRuntimeError",
    "build_child_optimizer",
    "build_parameter_policy_optimizer",
    "build_parameter_policy_scheduler",
]
