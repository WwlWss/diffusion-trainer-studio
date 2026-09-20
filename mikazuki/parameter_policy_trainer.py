"""Trainer-side orchestration for Parameter Training Policy v1.

Step 6A deliberately stops at a reusable integration boundary. Model-family
trainers opt into this module only when --parameter_policy_config is present;
Standard mode never imports it through the lazy trainer bridges.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mikazuki.parameter_policy import serialize_parameter_policy, validate_parameter_policy
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers
from mikazuki.parameter_policy_runtime import (
    ParameterPolicyRuntimeSpec,
    compile_parameter_policy_runtime_spec,
)
from mikazuki.parameter_policy_torch import (
    CompositeLRScheduler,
    CompositeOptimizer,
    build_parameter_policy_optimizer,
    build_parameter_policy_scheduler,
)
from mikazuki.parameter_routing import (
    ParameterDescriptor,
    RoutingPlan,
    build_parameter_routing_plan,
    scan_parameter_roots,
)


PARAMETER_POLICY_CHECKPOINT_MANIFEST = "dts_parameter_policy_manifest.json"
PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION = 1
PARAMETER_POLICY_SCHEDULER_IDENTITY_VERSION = 1


class ParameterPolicyTrainerRuntimeError(RuntimeError):
    """Raised when trainer integration cannot honor Parameter Policy exactly."""


def _effective_config(args: object) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return dict(args)
    try:
        return dict(vars(args))
    except TypeError as exc:
        raise ParameterPolicyTrainerRuntimeError(
            "Parameter Policy trainer integration requires argparse-like args or a mapping."
        ) from exc


def _policy_path(args: object) -> str:
    config = _effective_config(args)
    raw = config.get("parameter_policy_config")
    path = str(raw or "").strip()
    if not path:
        raise ParameterPolicyTrainerRuntimeError(
            "Parameter Policy trainer integration requires --parameter_policy_config."
        )
    return path


def load_parameter_policy_file(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    target = Path(path)
    if not target.is_file():
        raise ParameterPolicyTrainerRuntimeError(
            f"Parameter Policy file does not exist: {target}"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterPolicyTrainerRuntimeError(
            f"Unable to read Parameter Policy JSON: {target}: {exc}"
        ) from exc
    try:
        canonical = validate_parameter_policy(payload)
    except ValueError as exc:
        raise ParameterPolicyTrainerRuntimeError(str(exc)) from exc

    _canonical_path, canonical_text = serialize_parameter_policy(canonical)
    digest = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    return canonical, digest


def _unique_parameters(parameters: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[int] = set()
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        result.append(parameter)
    return tuple(result)


def _runtime_parameters(spec: ParameterPolicyRuntimeSpec) -> tuple[Any, ...]:
    return _unique_parameters(
        parameter_spec.parameter
        for optimizer_spec in spec.optimizers
        for group in optimizer_spec.groups
        for parameter_spec in group.parameters
    )


def _routing_error_message(plan: RoutingPlan) -> str:
    errors = [issue for issue in plan.issues if issue.severity == "error"]
    if not errors:
        return "Parameter Policy routing plan is invalid."
    detail = "\n".join(f"- {issue.code}: {issue.message}" for issue in errors)
    return "Parameter Policy routing failed:\n" + detail


def _unwrap_composite_optimizer(optimizer: object) -> CompositeOptimizer:
    if isinstance(optimizer, CompositeOptimizer):
        return optimizer
    inner = getattr(optimizer, "optimizer", None)
    if isinstance(inner, CompositeOptimizer):
        return inner
    raise ParameterPolicyTrainerRuntimeError(
        "Accelerate-facing optimizer is not backed by DTS CompositeOptimizer."
    )


def _unwrap_composite_scheduler(scheduler: object | None) -> CompositeLRScheduler | None:
    if scheduler is None:
        return None
    if isinstance(scheduler, CompositeLRScheduler):
        return scheduler
    inner = getattr(scheduler, "scheduler", None)
    if isinstance(inner, CompositeLRScheduler):
        return inner
    raise ParameterPolicyTrainerRuntimeError(
        "Accelerate-facing scheduler is not backed by DTS CompositeLRScheduler."
    )


def _canonical_scheduler_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [
            _canonical_scheduler_value(item, path=f"{path}[]")
            for item in value
        ]
    if isinstance(value, list):
        return [
            _canonical_scheduler_value(item, path=f"{path}[]")
            for item in value
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise ParameterPolicyTrainerRuntimeError(
                    f"Scheduler identity requires string mapping keys at {path}; got {key!r}."
                )
            result[key] = _canonical_scheduler_value(
                value[key],
                path=f"{path}.{key}",
            )
        return result
    raise ParameterPolicyTrainerRuntimeError(
        f"Scheduler identity cannot canonicalize {path}={value!r} "
        f"({type(value).__name__})."
    )


def _parse_scheduler_arguments(raw_args: object) -> dict[str, Any]:
    if raw_args in (None, "", []):
        return {}
    if isinstance(raw_args, str):
        items = [raw_args]
    elif isinstance(raw_args, (list, tuple)):
        items = list(raw_args)
    else:
        raise ParameterPolicyTrainerRuntimeError(
            "lr_scheduler_args must be a string or a sequence of strings."
        )

    parsed: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            raise ParameterPolicyTrainerRuntimeError(
                f"Invalid lr_scheduler_args item {item!r}; expected key=value."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ParameterPolicyTrainerRuntimeError(
                f"Invalid lr_scheduler_args item {item!r}; key cannot be empty."
            )
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError) as exc:
            raise ParameterPolicyTrainerRuntimeError(
                f"Invalid lr_scheduler_args value for {key!r}: {raw_value!r}."
            ) from exc
        parsed[key] = _canonical_scheduler_value(
            value,
            path=f"lr_scheduler_args.{key}",
        )
    return {key: parsed[key] for key in sorted(parsed)}


def _effective_step_count(value: object, *, total_steps: int, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ParameterPolicyTrainerRuntimeError(
            f"{field} must be an int, float ratio, or None."
        )
    if isinstance(value, float):
        return int(value * total_steps)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParameterPolicyTrainerRuntimeError(
            f"{field} must be an int, float ratio, or None; got {value!r}."
        ) from exc


def _scheduler_identity_from_args(
    args: object,
    *,
    num_processes: int,
) -> tuple[dict[str, Any], str]:
    config = _effective_config(args)
    try:
        max_train_steps = int(config.get("max_train_steps"))
        processes = int(num_processes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParameterPolicyTrainerRuntimeError(
            "Scheduler identity requires integer max_train_steps and num_processes."
        ) from exc
    if max_train_steps < 0 or processes <= 0:
        raise ParameterPolicyTrainerRuntimeError(
            "Scheduler identity requires max_train_steps >= 0 and num_processes > 0."
        )

    num_training_steps = max_train_steps * processes
    warmup = _effective_step_count(
        config.get("lr_warmup_steps", 0),
        total_steps=num_training_steps,
        field="lr_warmup_steps",
    )
    decay = _effective_step_count(
        config.get("lr_decay_steps", 0),
        total_steps=num_training_steps,
        field="lr_decay_steps",
    )
    if warmup is None:
        warmup = 0
    if decay is None:
        decay = 0

    identity = {
        "schema": "dts.parameter-policy.scheduler-identity",
        "version": PARAMETER_POLICY_SCHEDULER_IDENTITY_VERSION,
        "provider": "sd-scripts.get_scheduler_fix",
        "lr_scheduler": str(config.get("lr_scheduler") or "constant"),
        "lr_scheduler_type": str(config.get("lr_scheduler_type") or ""),
        "lr_scheduler_args": _parse_scheduler_arguments(
            config.get("lr_scheduler_args")
        ),
        "max_train_steps": max_train_steps,
        "num_processes": processes,
        "num_training_steps": num_training_steps,
        "num_warmup_steps": warmup,
        "num_decay_steps": decay,
        "num_stable_steps": num_training_steps - warmup - decay,
        "num_cycles": _canonical_scheduler_value(
            config.get("lr_scheduler_num_cycles", 1),
            path="lr_scheduler_num_cycles",
        ),
        "power": _canonical_scheduler_value(
            config.get("lr_scheduler_power", 1.0),
            path="lr_scheduler_power",
        ),
        "timescale": _canonical_scheduler_value(
            config.get("lr_scheduler_timescale"),
            path="lr_scheduler_timescale",
        ),
        "min_lr_ratio": _canonical_scheduler_value(
            config.get("lr_scheduler_min_lr_ratio"),
            path="lr_scheduler_min_lr_ratio",
        ),
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return identity, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegacySchedulerFactory:
    base_args: object
    get_scheduler_fix: Any
    num_processes: int
    scheduler_identity: dict[str, Any]
    scheduler_signature: str

    def __call__(self, spec, child_optimizer):
        if isinstance(self.base_args, Mapping):
            child_args = SimpleNamespace(**dict(self.base_args))
        else:
            child_args = copy.copy(self.base_args)
        setattr(child_args, "optimizer_type", spec.optimizer_type)
        return self.get_scheduler_fix(
            child_args,
            child_optimizer,
            self.num_processes,
        )


@dataclass
class ParameterPolicyTrainerSession:
    """One immutable routing/runtime topology plus mutable trainability contract."""

    train_type: str
    policy: dict[str, Any]
    policy_hash: str
    descriptors: tuple[ParameterDescriptor, ...]
    routed_descriptors: tuple[ParameterDescriptor, ...]
    routing_plan: RoutingPlan
    runtime_spec: ParameterPolicyRuntimeSpec
    optimizer: CompositeOptimizer
    structural_frozen_parameters: tuple[Any, ...]
    scheduler_identity: dict[str, Any] | None = None
    scheduler_signature: str | None = None

    @property
    def trainable_parameters(self) -> tuple[Any, ...]:
        return _runtime_parameters(self.runtime_spec)

    @property
    def trainable_parameter_ids(self) -> frozenset[int]:
        return frozenset(id(parameter) for parameter in self.trainable_parameters)

    @property
    def trainable_components(self) -> frozenset[str]:
        return frozenset(
            assignment.component_id
            for assignment in self.routing_plan.assignments
            if assignment.route_kind in {"primary", "fallback"}
        )

    @property
    def frozen_components(self) -> frozenset[str]:
        return frozenset(
            component_id
            for component_id, route in self.policy["components"].items()
            if not route["train"]
        )

    @property
    def frozen_parameters(self) -> tuple[Any, ...]:
        routed = tuple(
            assignment.parameter
            for assignment in self.routing_plan.assignments
            if assignment.route_kind in {"frozen", "unavailable"}
        )
        return _unique_parameters((*self.structural_frozen_parameters, *routed))

    def trains_component(self, component_id: str) -> bool:
        return component_id in self.trainable_components

    def trains_prefix(self, prefix: str) -> bool:
        return any(component_id.startswith(prefix) for component_id in self.trainable_components)

    def _expected_requires_grad(self) -> dict[int, bool]:
        expected: dict[int, bool] = {}
        for parameter in self.structural_frozen_parameters:
            expected[id(parameter)] = False
        for assignment in self.routing_plan.assignments:
            expected[id(assignment.parameter)] = assignment.route_kind in {"primary", "fallback"}
        return expected

    def apply_requires_grad_contract(self) -> None:
        expected = self._expected_requires_grad()
        descriptor_ids = {descriptor.parameter_id for descriptor in self.descriptors}
        missing = set(expected).difference(descriptor_ids)
        if missing:
            raise ParameterPolicyTrainerRuntimeError(
                "Parameter Policy requires-grad contract references parameters outside the scanned roots."
            )

        for descriptor in self.descriptors:
            desired = expected.get(descriptor.parameter_id)
            if desired is None:
                raise ParameterPolicyTrainerRuntimeError(
                    f"Parameter {descriptor.canonical_name!r} has no final trainability assignment."
                )
            requires_grad_ = getattr(descriptor.parameter, "requires_grad_", None)
            if not callable(requires_grad_):
                raise ParameterPolicyTrainerRuntimeError(
                    f"Parameter {descriptor.canonical_name!r} does not expose requires_grad_()."
                )
            requires_grad_(desired)

    def assert_requires_grad_contract(self) -> None:
        expected = self._expected_requires_grad()
        for descriptor in self.descriptors:
            desired = expected.get(descriptor.parameter_id)
            if desired is None:
                raise ParameterPolicyTrainerRuntimeError(
                    f"Parameter {descriptor.canonical_name!r} has no final trainability assignment."
                )
            actual = getattr(descriptor.parameter, "requires_grad", None)
            if actual is not desired:
                raise ParameterPolicyTrainerRuntimeError(
                    f"Parameter {descriptor.canonical_name!r} requires_grad={actual!r}; "
                    f"Parameter Policy requires {desired!r}."
                )

    def build_scheduler(self, scheduler_factory) -> CompositeLRScheduler:
        scheduler = build_parameter_policy_scheduler(self.optimizer, scheduler_factory)
        has_external = any(entry.mode == "external" for entry in scheduler.entries)
        if has_external:
            identity = getattr(scheduler_factory, "scheduler_identity", None)
            signature = getattr(scheduler_factory, "scheduler_signature", None)
            if not isinstance(identity, Mapping) or not isinstance(signature, str) or not signature:
                raise ParameterPolicyTrainerRuntimeError(
                    "External Parameter Policy schedulers require an audited scheduler "
                    "identity/signature factory."
                )
            self.scheduler_identity = dict(identity)
            self.scheduler_signature = signature
        else:
            identity = {
                "schema": "dts.parameter-policy.scheduler-identity",
                "version": PARAMETER_POLICY_SCHEDULER_IDENTITY_VERSION,
                "mode": "optimizer_managed",
            }
            canonical = json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.scheduler_identity = identity
            self.scheduler_signature = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
        return scheduler

    def audit_after_prepare(self, *, accelerator: object, optimizer: object) -> None:
        composite = _unwrap_composite_optimizer(optimizer)
        if composite.runtime_spec.topology_fingerprint != self.runtime_spec.topology_fingerprint:
            raise ParameterPolicyTrainerRuntimeError(
                "Prepared CompositeOptimizer topology does not match the trainer session."
            )

        actual_parameters = [
            parameter
            for group in composite.param_groups
            for parameter in group["params"]
        ]
        actual_ids = [id(parameter) for parameter in actual_parameters]
        if len(actual_ids) != len(set(actual_ids)):
            raise ParameterPolicyTrainerRuntimeError(
                "Prepared CompositeOptimizer contains duplicate physical parameters."
            )
        if set(actual_ids) != set(self.trainable_parameter_ids):
            raise ParameterPolicyTrainerRuntimeError(
                "Prepared CompositeOptimizer parameter ownership differs from Runtime Spec."
            )

        frozen_ids = {id(parameter) for parameter in self.frozen_parameters}
        leaked = frozen_ids.intersection(actual_ids)
        if leaked:
            raise ParameterPolicyTrainerRuntimeError(
                "Frozen Parameter Policy parameters leaked into the prepared optimizer."
            )

        expected_device = getattr(accelerator, "device", None)
        if expected_device is None:
            raise ParameterPolicyTrainerRuntimeError(
                "Accelerator does not expose a device for Parameter Policy device audit."
            )
        mismatches: list[str] = []
        name_by_id = {
            descriptor.parameter_id: descriptor.canonical_name
            for descriptor in self.descriptors
        }
        for parameter in self.trainable_parameters:
            device = getattr(parameter, "device", None)
            if device != expected_device:
                mismatches.append(
                    f"{name_by_id.get(id(parameter), '<parameter>')}={device}"
                )
        if mismatches:
            raise ParameterPolicyTrainerRuntimeError(
                "Parameter Policy device audit failed; trainable optimizer parameters "
                f"must be on {expected_device}: " + ", ".join(mismatches[:8])
            )

        self.assert_requires_grad_contract()

    def checkpoint_manifest(self, scheduler: object | None = None) -> dict[str, Any]:
        composite_scheduler = _unwrap_composite_scheduler(scheduler)
        if composite_scheduler is not None and (
            not isinstance(self.scheduler_identity, Mapping)
            or not isinstance(self.scheduler_signature, str)
            or not self.scheduler_signature
        ):
            raise ParameterPolicyTrainerRuntimeError(
                "Parameter Policy scheduler identity was not established before checkpointing."
            )

        manifest: dict[str, Any] = {
            "version": PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION,
            "train_type": self.train_type,
            "policy_hash": self.policy_hash,
            "runtime_topology_fingerprint": self.runtime_spec.topology_fingerprint,
            "optimizers": [
                {
                    "profile_name": spec.profile_name,
                    "optimizer_type": spec.optimizer_type,
                    "topology_fingerprint": spec.topology_fingerprint,
                }
                for spec in self.runtime_spec.optimizers
            ],
        }
        if composite_scheduler is not None:
            manifest["scheduler_identity"] = dict(self.scheduler_identity)
            manifest["scheduler_signature"] = self.scheduler_signature
            manifest["schedulers"] = [
                {
                    "profile_name": entry.profile_name,
                    "mode": entry.mode,
                    "scheduler_type": entry.scheduler_type,
                    "topology_fingerprint": entry.topology_fingerprint,
                }
                for entry in composite_scheduler.entries
            ]
        return manifest

    def validate_checkpoint_manifest(
        self,
        input_dir: str | os.PathLike[str],
        *,
        scheduler: object | None = None,
    ) -> None:
        path = Path(input_dir) / PARAMETER_POLICY_CHECKPOINT_MANIFEST
        if not path.is_file():
            raise ParameterPolicyTrainerRuntimeError(
                "Cannot resume Component-wise training from a state without "
                f"{PARAMETER_POLICY_CHECKPOINT_MANIFEST}."
            )
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ParameterPolicyTrainerRuntimeError(
                f"Invalid Parameter Policy checkpoint manifest: {path}: {exc}"
            ) from exc
        expected = self.checkpoint_manifest(scheduler)
        if actual.get("policy_hash") != expected.get("policy_hash"):
            raise ParameterPolicyTrainerRuntimeError(
                "Checkpoint Parameter Policy identity does not match the current run. "
                "Load model weights only when intentionally starting a new policy/stage."
            )
        if (
            actual.get("runtime_topology_fingerprint")
            != expected.get("runtime_topology_fingerprint")
        ):
            raise ParameterPolicyTrainerRuntimeError(
                "Checkpoint optimizer/runtime topology does not match the current run. "
                "Load model weights only when intentionally starting a new policy/stage."
            )
        if actual.get("scheduler_signature") != expected.get("scheduler_signature"):
            raise ParameterPolicyTrainerRuntimeError(
                "Checkpoint scheduler configuration does not match the current run. "
                "Load model weights only when intentionally starting a new policy/stage."
            )
        if actual != expected:
            raise ParameterPolicyTrainerRuntimeError(
                "Checkpoint Parameter Policy manifest does not match the current run. "
                "Load model weights only when intentionally starting a new policy/stage."
            )

    def register_checkpoint_manifest(
        self,
        accelerator: object,
        *,
        scheduler: object | None = None,
    ) -> None:
        expected = self.checkpoint_manifest(scheduler)

        def save_hook(models, weights, output_dir):
            del models, weights
            if not getattr(accelerator, "is_main_process", True):
                return
            path = Path(output_dir) / PARAMETER_POLICY_CHECKPOINT_MANIFEST
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(
                json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)

        def load_hook(models, input_dir):
            del models
            self.validate_checkpoint_manifest(input_dir, scheduler=scheduler)

        register_save = getattr(accelerator, "register_save_state_pre_hook", None)
        register_load = getattr(accelerator, "register_load_state_pre_hook", None)
        if not callable(register_save) or not callable(register_load):
            raise ParameterPolicyTrainerRuntimeError(
                "Accelerator does not expose save/load pre-hook registration."
            )
        register_save(save_hook)
        register_load(load_hook)


def make_legacy_scheduler_factory(
    *,
    args: object,
    get_scheduler_fix,
    num_processes: int,
) -> LegacySchedulerFactory:
    """Freeze sd-scripts scheduler construction and its resume identity together."""

    if not callable(get_scheduler_fix):
        raise ParameterPolicyTrainerRuntimeError("get_scheduler_fix must be callable.")
    if isinstance(args, Mapping):
        base_args: object = dict(args)
    else:
        base_args = copy.copy(args)
    identity, signature = _scheduler_identity_from_args(
        base_args,
        num_processes=num_processes,
    )
    return LegacySchedulerFactory(
        base_args=base_args,
        get_scheduler_fix=get_scheduler_fix,
        num_processes=num_processes,
        scheduler_identity=identity,
        scheduler_signature=signature,
    )


def create_parameter_policy_session(
    *,
    args: object,
    train_type: str,
    roots: Mapping[str, Any],
    structural_frozen_parameters: Iterable[Any] = (),
) -> ParameterPolicyTrainerSession:
    """Compile one trainer job into routing/runtime ownership exactly once."""

    effective_config = _effective_config(args)
    policy, policy_hash = load_parameter_policy_file(_policy_path(args))

    semantic_blockers = parameter_policy_v1_semantic_blockers(
        effective_config,
        train_type,
    )
    if semantic_blockers:
        raise ParameterPolicyTrainerRuntimeError(
            "Parameter Policy v1 cannot own this trainer configuration:\n- "
            + "\n- ".join(semantic_blockers)
        )

    descriptors = scan_parameter_roots(roots)
    structural = _unique_parameters(structural_frozen_parameters)
    descriptor_ids = {descriptor.parameter_id for descriptor in descriptors}
    structural_ids = {id(parameter) for parameter in structural}
    missing_structural = structural_ids.difference(descriptor_ids)
    if missing_structural:
        raise ParameterPolicyTrainerRuntimeError(
            "structural_frozen_parameters contains parameters outside the scanned roots."
        )

    routed_descriptors = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.parameter_id not in structural_ids
    )
    plan = build_parameter_routing_plan(
        policy,
        train_type=train_type,
        effective_config=effective_config,
        descriptors=routed_descriptors,
    )
    if not plan.is_valid:
        raise ParameterPolicyTrainerRuntimeError(_routing_error_message(plan))

    try:
        runtime_spec = compile_parameter_policy_runtime_spec(policy, plan)
        optimizer = build_parameter_policy_optimizer(runtime_spec)
    except (ValueError, RuntimeError) as exc:
        raise ParameterPolicyTrainerRuntimeError(str(exc)) from exc

    session = ParameterPolicyTrainerSession(
        train_type=train_type,
        policy=policy,
        policy_hash=policy_hash,
        descriptors=descriptors,
        routed_descriptors=routed_descriptors,
        routing_plan=plan,
        runtime_spec=runtime_spec,
        optimizer=optimizer,
        structural_frozen_parameters=structural,
    )
    session.apply_requires_grad_contract()
    session.assert_requires_grad_contract()
    return session


__all__ = [
    "PARAMETER_POLICY_CHECKPOINT_MANIFEST",
    "PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION",
    "LegacySchedulerFactory",
    "ParameterPolicyTrainerRuntimeError",
    "ParameterPolicyTrainerSession",
    "create_parameter_policy_session",
    "load_parameter_policy_file",
    "make_legacy_scheduler_factory",
]
