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

import torch

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
PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION = 2
PARAMETER_POLICY_SCHEDULER_IDENTITY_VERSION = 1


class ParameterPolicyTrainerRuntimeError(RuntimeError):
    """Raised when trainer integration cannot honor Parameter Policy exactly."""


def _current_cuda_device_index() -> int:
    try:
        return int(torch.cuda.current_device())
    except Exception as exc:
        raise ParameterPolicyTrainerRuntimeError(
            "Parameter Policy device audit could not resolve the current CUDA device."
        ) from exc


def _resolve_runtime_device(
    device: object,
    *,
    current_cuda_index: int | None = None,
) -> torch.device:
    try:
        normalized = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ParameterPolicyTrainerRuntimeError(
            f"Invalid runtime device for Parameter Policy audit: {device!r}."
        ) from exc

    if normalized.type != "cuda" or normalized.index is not None:
        return normalized

    if current_cuda_index is None:
        current_cuda_index = _current_cuda_device_index()

    return torch.device("cuda", current_cuda_index)


def _same_runtime_device(
    actual: object,
    expected: object,
    *,
    current_cuda_index: int | None = None,
) -> bool:
    try:
        actual_device = torch.device(actual)
        expected_device = torch.device(expected)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ParameterPolicyTrainerRuntimeError(
            "Parameter Policy device audit received an invalid device."
        ) from exc

    if actual_device == expected_device:
        return True
    if actual_device.type != expected_device.type:
        return False
    if actual_device.type != "cuda":
        return False

    return _resolve_runtime_device(
        actual_device,
        current_cuda_index=current_cuda_index,
    ) == _resolve_runtime_device(
        expected_device,
        current_cuda_index=current_cuda_index,
    )


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


@dataclass
class LegacySchedulerFactory:
    base_args: object
    get_scheduler_fix: Any
    num_processes: int
    scheduler_identity: dict[str, Any] | None = None
    scheduler_signature: str | None = None

    def _ensure_identity(self) -> None:
        if self.scheduler_identity is not None and self.scheduler_signature is not None:
            return
        identity, signature = _scheduler_identity_from_args(
            self.base_args,
            num_processes=self.num_processes,
        )
        self.scheduler_identity = identity
        self.scheduler_signature = signature

    def __call__(self, spec, child_optimizer):
        self._ensure_identity()
        if isinstance(self.base_args, Mapping):
            child_args = SimpleNamespace(**copy.deepcopy(dict(self.base_args)))
        else:
            child_args = copy.deepcopy(self.base_args)
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
    def trainable_parameter_tensor_count(self) -> int:
        return len(self.trainable_parameters)

    @property
    def trainable_parameter_element_count(self) -> int:
        total = 0
        for parameter in self.trainable_parameters:
            numel = getattr(parameter, "numel", None)
            if not callable(numel):
                raise ParameterPolicyTrainerRuntimeError(
                    "Parameter Policy diagnostics require trainable parameters to expose numel()."
                )
            total += int(numel())
        return total

    @property
    def optimizer_profiles(self) -> dict[str, str]:
        return {
            spec.profile_name: spec.optimizer_type
            for spec in self.runtime_spec.optimizers
        }

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
        mismatches: list[str] = []
        for descriptor in self.descriptors:
            desired = expected.get(descriptor.parameter_id)
            if desired is None:
                raise ParameterPolicyTrainerRuntimeError(
                    f"Parameter {descriptor.canonical_name!r} has no final trainability assignment."
                )
            actual = getattr(descriptor.parameter, "requires_grad", None)
            if actual is not desired:
                mismatches.append(
                    f"{descriptor.canonical_name}: actual={actual!r}, expected={desired!r}"
                )
        if mismatches:
            raise ParameterPolicyTrainerRuntimeError(
                "Parameter Policy requires-grad contract was mutated: "
                + "; ".join(mismatches[:8])
            )

    def model_metadata(self) -> dict[str, str]:
        metadata = {
            "ss_dts_parameter_policy_hash": self.policy_hash,
            "ss_dts_parameter_policy_topology": self.runtime_spec.topology_fingerprint,
            "ss_dts_parameter_policy_profiles": json.dumps(
                self.optimizer_profiles,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "ss_dts_parameter_policy_train_type": self.train_type,
            "ss_dts_parameter_policy_trainable_components": json.dumps(
                sorted(self.trainable_components),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "ss_dts_parameter_policy_trainable_parameter_tensors": str(
                self.trainable_parameter_tensor_count
            ),
            "ss_dts_parameter_policy_trainable_parameter_elements": str(
                self.trainable_parameter_element_count
            ),
            "ss_dts_parameter_policy_manifest_version": str(
                PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION
            ),
        }
        if self.scheduler_signature:
            metadata["ss_dts_parameter_policy_scheduler_signature"] = (
                self.scheduler_signature
            )
        return metadata

    def startup_diagnostics(self) -> dict[str, Any]:
        return {
            "train_type": self.train_type,
            "policy_hash": self.policy_hash,
            "topology_fingerprint": self.runtime_spec.topology_fingerprint,
            "trainable_components": sorted(self.trainable_components),
            "frozen_components": sorted(self.frozen_components),
            "optimizer_profiles": self.optimizer_profiles,
            "trainable_parameter_tensors": self.trainable_parameter_tensor_count,
            "trainable_parameter_elements": self.trainable_parameter_element_count,
            "scheduler_signature": self.scheduler_signature,
        }

    def log_startup_diagnostics(self, accelerator: object) -> None:
        printer = getattr(accelerator, "print", None)
        if not callable(printer):
            raise ParameterPolicyTrainerRuntimeError(
                "Accelerator does not expose print() for Parameter Policy diagnostics."
            )
        diagnostics = self.startup_diagnostics()
        profiles = ", ".join(
            f"{name}={optimizer_type}"
            for name, optimizer_type in sorted(
                diagnostics["optimizer_profiles"].items()
            )
        )
        trainable = ", ".join(diagnostics["trainable_components"]) or "<none>"
        frozen = ", ".join(diagnostics["frozen_components"]) or "<none>"
        printer(
            "\n".join(
                (
                    "[DTS Parameter Policy]",
                    f"  train_type: {diagnostics['train_type']}",
                    f"  policy_hash: {diagnostics['policy_hash']}",
                    f"  topology: {diagnostics['topology_fingerprint']}",
                    f"  trainable_components: {trainable}",
                    f"  frozen_components: {frozen}",
                    f"  optimizer_profiles: {profiles or '<none>'}",
                    "  trainable_parameters: "
                    f"{diagnostics['trainable_parameter_tensors']} tensors / "
                    f"{diagnostics['trainable_parameter_elements']} elements",
                    "  scheduler_signature: "
                    f"{diagnostics['scheduler_signature'] or '<optimizer-managed/unset>'}",
                )
            )
        )

    def component_lr_logs(self, scheduler: object) -> dict[str, Any]:
        """Return stable component-oriented LR logs from the composite runtime."""

        composite_scheduler = _unwrap_composite_scheduler(scheduler)
        if composite_scheduler is None:
            raise ParameterPolicyTrainerRuntimeError(
                "component_lr_logs requires the DTS CompositeLRScheduler."
            )

        values_by_profile = composite_scheduler.get_last_lr_by_profile()
        route_values: dict[tuple[str, str], Any] = {}
        for optimizer_spec in self.runtime_spec.optimizers:
            profile_values = values_by_profile.get(optimizer_spec.profile_name)
            if profile_values is None:
                raise ParameterPolicyTrainerRuntimeError(
                    f"Scheduler has no LR values for Optimizer Profile "
                    f"{optimizer_spec.profile_name!r}."
                )
            if len(profile_values) != len(optimizer_spec.groups):
                raise ParameterPolicyTrainerRuntimeError(
                    f"Scheduler LR group count for Optimizer Profile "
                    f"{optimizer_spec.profile_name!r} does not match Runtime Spec."
                )
            for group, lr_value in zip(optimizer_spec.groups, profile_values):
                for parameter in group.parameters:
                    key = (parameter.component_id, parameter.route_kind)
                    previous = route_values.get(key)
                    if previous is not None and previous != lr_value:
                        raise ParameterPolicyTrainerRuntimeError(
                            f"Component {parameter.component_id!r} has inconsistent "
                            f"{parameter.route_kind} scheduler LR values."
                        )
                    route_values[key] = lr_value

        logs: dict[str, Any] = {}
        components = sorted({component_id for component_id, _ in route_values})
        for component_id in components:
            primary = route_values.get((component_id, "primary"))
            fallback = route_values.get((component_id, "fallback"))
            if primary is not None:
                logs[f"lr/{component_id}"] = primary
                if fallback is not None:
                    logs[f"lr/{component_id}/fallback"] = fallback
            elif fallback is not None:
                logs[f"lr/{component_id}"] = fallback
                logs[f"lr/{component_id}/fallback"] = fallback
        return logs

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

    def _audit_prepared_optimizer_and_device(
        self,
        *,
        accelerator: object,
        optimizer: object,
    ) -> None:
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

        actual_id_set = set(actual_ids)
        expected_id_set = set(self.trainable_parameter_ids)
        frozen_ids = {id(parameter) for parameter in self.frozen_parameters}
        leaked = frozen_ids.intersection(actual_id_set)
        name_by_id = {
            descriptor.parameter_id: descriptor.canonical_name
            for descriptor in self.descriptors
        }
        if leaked:
            leaked_names = sorted(
                name_by_id.get(parameter_id, "<parameter>")
                for parameter_id in leaked
            )
            raise ParameterPolicyTrainerRuntimeError(
                "Frozen Parameter Policy parameters leaked into the prepared optimizer: "
                + ", ".join(leaked_names[:8])
            )

        if actual_id_set != expected_id_set:
            missing = sorted(
                name_by_id.get(parameter_id, "<parameter>")
                for parameter_id in expected_id_set.difference(actual_id_set)
            )
            unexpected = sorted(
                name_by_id.get(parameter_id, "<unknown parameter>")
                for parameter_id in actual_id_set.difference(expected_id_set)
            )
            raise ParameterPolicyTrainerRuntimeError(
                "Prepared CompositeOptimizer parameter ownership differs from Runtime Spec; "
                f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}."
            )

        raw_expected_device = getattr(accelerator, "device", None)
        if raw_expected_device is None:
            raise ParameterPolicyTrainerRuntimeError(
                "Accelerator does not expose a device for Parameter Policy device audit."
            )
        try:
            expected_device = torch.device(raw_expected_device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ParameterPolicyTrainerRuntimeError(
                f"Accelerator exposes an invalid device for Parameter Policy audit: "
                f"{raw_expected_device!r}."
            ) from exc

        current_cuda_index: int | None = None
        if expected_device.type == "cuda" and expected_device.index is None:
            current_cuda_index = _current_cuda_device_index()
        resolved_expected_device = _resolve_runtime_device(
            expected_device,
            current_cuda_index=current_cuda_index,
        )

        mismatches: list[str] = []
        for parameter in self.trainable_parameters:
            actual_device = getattr(parameter, "device", None)
            if actual_device is None:
                mismatches.append(
                    f"{name_by_id.get(id(parameter), '<parameter>')}=<no device>"
                )
                continue

            actual_device = torch.device(actual_device)
            if (
                current_cuda_index is None
                and actual_device.type == "cuda"
                and actual_device.index is None
            ):
                current_cuda_index = _current_cuda_device_index()

            if not _same_runtime_device(
                actual_device,
                expected_device,
                current_cuda_index=current_cuda_index,
            ):
                mismatches.append(
                    f"{name_by_id.get(id(parameter), '<parameter>')}={actual_device}"
                )
        if mismatches:
            raise ParameterPolicyTrainerRuntimeError(
                "Parameter Policy device audit failed; trainable optimizer parameters "
                f"must resolve to {resolved_expected_device} "
                f"(accelerator.device={expected_device}): "
                + ", ".join(mismatches[:8])
            )

    def assert_runtime_contract(
        self,
        *,
        phase: str,
        accelerator: object | None = None,
        optimizer: object | None = None,
    ) -> None:
        phase_name = str(phase or "").strip()
        if not phase_name:
            raise ParameterPolicyTrainerRuntimeError(
                "Parameter Policy runtime contract phase cannot be empty."
            )
        try:
            self.assert_requires_grad_contract()
            if accelerator is None and optimizer is None:
                return
            if accelerator is None or optimizer is None:
                raise ParameterPolicyTrainerRuntimeError(
                    "Runtime ownership/device audit requires both accelerator and optimizer."
                )
            self._audit_prepared_optimizer_and_device(
                accelerator=accelerator,
                optimizer=optimizer,
            )
        except ParameterPolicyTrainerRuntimeError as exc:
            raise ParameterPolicyTrainerRuntimeError(
                f"Parameter Policy runtime contract failed at {phase_name}: {exc}"
            ) from exc

    def audit_after_prepare(self, *, accelerator: object, optimizer: object) -> None:
        self.assert_runtime_contract(
            phase="post_prepare",
            accelerator=accelerator,
            optimizer=optimizer,
        )

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
            "trainable_components": sorted(self.trainable_components),
            "frozen_components": sorted(self.frozen_components),
            "trainable_parameter_tensors": self.trainable_parameter_tensor_count,
            "trainable_parameter_elements": self.trainable_parameter_element_count,
            "optimizer_profiles": self.optimizer_profiles,
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
        if not isinstance(actual, Mapping):
            raise ParameterPolicyTrainerRuntimeError(
                f"Invalid Parameter Policy checkpoint manifest object: {path}."
            )
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
        optimizer: object | None = None,
    ) -> None:
        expected = self.checkpoint_manifest(scheduler)

        def save_hook(models, weights, output_dir):
            del models, weights
            self.assert_runtime_contract(
                phase="checkpoint_save",
                accelerator=accelerator if optimizer is not None else None,
                optimizer=optimizer,
            )
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
            self.assert_runtime_contract(
                phase="checkpoint_load",
                accelerator=accelerator if optimizer is not None else None,
                optimizer=optimizer,
            )

        register_save = getattr(accelerator, "register_save_state_pre_hook", None)
        register_load = getattr(accelerator, "register_load_state_pre_hook", None)
        if not callable(register_save) or not callable(register_load):
            raise ParameterPolicyTrainerRuntimeError(
                "Accelerator does not expose save/load pre-hook registration."
            )
        register_save(save_hook)
        register_load(load_hook)

    def finalize_after_prepare(
        self,
        *,
        accelerator: object,
        optimizer: object,
        scheduler: object | None = None,
    ) -> None:
        self.assert_runtime_contract(
            phase="post_prepare",
            accelerator=accelerator,
            optimizer=optimizer,
        )
        self.register_checkpoint_manifest(
            accelerator,
            scheduler=scheduler,
            optimizer=optimizer,
        )
        self.log_startup_diagnostics(accelerator)


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
        base_args: object = copy.deepcopy(dict(args))
    else:
        base_args = copy.deepcopy(args)
    return LegacySchedulerFactory(
        base_args=base_args,
        get_scheduler_fix=get_scheduler_fix,
        num_processes=num_processes,
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
