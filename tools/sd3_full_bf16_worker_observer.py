"""C3 instance-local physical SD3 Full-BF16 training observations.

A CPU caller can exercise these algorithms but is NEVER issued physical CUDA
qualification. The fixed production admission is in sd3_full_bf16_evidence_worker;
C4 save/load fingerprint binding has not been implemented.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from mikazuki.parameter_policy_torch import CompositeOptimizer, CompositeLRScheduler

from tools import sd3_full_bf16_evidence_support as host


def _fail(message: str) -> None:
    raise host.FullBF16EvidenceError("C3 physical observer: " + message)


def fingerprint_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Actual contiguous Tensor bytes (no BF16→FP32 conversion, no GPU clone)."""
    if not isinstance(tensor, torch.Tensor):
        _fail("Tensor fingerprint requires actual torch.Tensor")
    if tensor.layout != torch.strided or tensor.is_meta:
        _fail("Unsupported sparse/meta Tensor fingerprint")
    with torch.no_grad():
        cpu = tensor.detach().contiguous().to(device="cpu")
        payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _typed_tree(value: Any) -> Any:
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("Non-finite numerical state fingerprint input")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, torch.Tensor):
        return ["tensor", fingerprint_tensor(value)]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            _fail("NumPy object arrays are not canonical state")
        return ["ndarray", {
            "dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }]
    if isinstance(value, np.generic):
        return ["numpy_scalar", str(value.dtype), _typed_tree(value.item())]
    if isinstance(value, tuple):
        return ["tuple", [_typed_tree(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_typed_tree(item) for item in value]]
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if type(key) not in (str, int, float, bool):
                _fail("Unsupported mapping key in physical state")
            typed_key = _typed_tree(key)
            items.append([typed_key, _typed_tree(item)])
        items.sort(key=lambda pair: json.dumps(
            pair[0], sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
        keys = [json.dumps(p[0], sort_keys=True, separators=(",", ":")) for p in items]
        if len(keys) != len(set(keys)):
            _fail("Duplicate canonical keys in physical state")
        return ["mapping", items]
    _fail("Unsupported physical state type: " + type(value).__name__)


def fingerprint_state_tree(value: Any) -> str:
    """Cross-process stable digest over typed values and real Tensor bytes."""
    data = json.dumps(
        _typed_tree(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def snapshot_rng_fingerprint() -> dict[str, str]:
    """Read (never mutate) Python, NumPy, torch CPU and active CUDA RNG state."""
    result = {
        "python": fingerprint_state_tree(random.getstate()),
        "numpy": fingerprint_state_tree(np.random.get_state()),
        "torch_cpu": fingerprint_state_tree(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = fingerprint_state_tree(torch.cuda.get_rng_state())
    return result


def verify_prepared_identity(*, session: Any, accelerator: Any, scheduler: Any) -> dict[str, Any]:
    """Check the actual Accelerate 1.6.0 prepared wrappers, not toy facades."""
    optimizers = getattr(accelerator, "_optimizers", None)
    schedulers = getattr(accelerator, "_schedulers", None)
    if not isinstance(optimizers, list) or len(optimizers) != 1:
        _fail("Expected exactly one prepared AcceleratedOptimizer")
    if not isinstance(schedulers, list) or len(schedulers) != 1:
        _fail("Expected exactly one prepared AcceleratedScheduler")
    opt, sch = optimizers[0], schedulers[0]
    if not isinstance(opt, AcceleratedOptimizer) or opt.optimizer is not session.optimizer:
        _fail("Prepared Optimizer wrapper does not own Session CompositeOptimizer")
    if not isinstance(sch, AcceleratedScheduler) or sch.scheduler is not scheduler:
        _fail("Prepared Scheduler wrapper does not own captured CompositeLRScheduler")
    if not isinstance(opt.optimizer, CompositeOptimizer):
        _fail("Optimizer must be real DTS CompositeOptimizer")
    if not isinstance(scheduler, CompositeLRScheduler) or scheduler.optimizer is not opt.optimizer:
        _fail("Scheduler must bind the same physical CompositeOptimizer")
    if not sch.step_with_optimizer or not accelerator.step_scheduler_with_optimizer:
        _fail("Accelerate must advance Scheduler with actual Optimizer Step")
    if getattr(accelerator.gradient_state, "adjust_scheduler", None) is not False:
        _fail("Accelerate adjust_scheduler must be False for strict R2 accumulation")
    if accelerator.num_processes != 1:
        _fail("Only one physical Accelerator process is allowed")
    return {
        "optimizer_children": len(opt.optimizer.entries),
        "scheduler_children": len(scheduler.entries),
        "device_type": str(accelerator.device.type),
        "prepared_identity": "accelerate-1.6.0-actual-wrappers-v1",
    }


def observe_prepared_ownership(
    *, session: Any, accelerator: Any, network: torch.nn.Module,
    train_dataloader: Any, case_id: str,
) -> dict[str, Any]:
    """Real B0 reference 526 BF16 CUDA Parameters; CPU cannot satisfy this."""
    spec = host.get_case_spec(case_id)
    expected_batches = spec["effective_batches_per_epoch"]
    if len(train_dataloader) != expected_batches:
        _fail("Actual prepared DataLoader has wrong reference batch count")
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        _fail("Physical reference Parameter ownership requires actual CUDA")
    if getattr(network, "split_qkv", None) is not False:
        _fail("Only non-Split-QKV SD3 LoRA ownership is reviewed")
    if getattr(network, "train_t5xxl", None) is not False:
        _fail("T5XXL adapter must be frozen")
    from mikazuki.parameter_policy_torch import CompositeOptimizer
    optimizer = session.optimizer
    if not isinstance(optimizer, CompositeOptimizer):
        _fail("Missing actual DTS composite optimizer")
    profiles = {s.profile_name: s.optimizer_type for s in session.runtime_spec.optimizers}
    expected_profiles = {name: value["type"] for name, value in
                         host.build_reference_policy(case_id)["optimizer_profiles"].items()}
    if profiles != expected_profiles or len(optimizer.entries) != len(profiles):
        _fail("Physical optimizer child Profiles differ from immutable reference")
    from pytorch_optimizer import Muon
    for entry in optimizer.entries:
        kind = profiles.get(entry.profile_name)
        if entry.optimizer_type != kind:
            _fail("Actual optimizer entry has wrong Profile/type")
        if kind == "Muon":
            if not isinstance(entry.optimizer, Muon) or not all(
                group.get("use_muon") is True for group in entry.optimizer.param_groups
            ):
                _fail("Muon must be an actual Muon with no internal AdamW fallback")
        elif kind == "AdamW":
            if type(entry.optimizer) is not torch.optim.AdamW:
                _fail("AdamW must be a real independent torch.optim.AdamW")
        else:
            _fail("Unexpected optimizer Child type")
    if len(session.runtime_spec.optimizers) != len(optimizer.entries):
        _fail("Mismatch in actual runtime Optimizer topology")
    if not hasattr(network, "named_parameters") or not hasattr(network, "named_buffers"):
        _fail("Missing prepared LoRA Network")
    named = dict(network.named_parameters())
    buffers = dict(network.named_buffers())
    target_spec = host.build_sd35_medium_lora_target_spec()
    targets = {name + suffix for name in target_spec
               for suffix in (".lora_down.weight", ".lora_up.weight")}
    alpha_keys = {name + ".alpha" for name in target_spec}
    if set(named) != targets or set(buffers) != alpha_keys:
        _fail("Prepared LoRA Parameter or Alpha set differs from 263 target modules")
    if len(named) != 526 or len(buffers) != 263:
        _fail("Expected 526 trainable LoRA weights and 263 Alpha Buffers")
    assignments = {assignment.canonical_name: assignment
                   for assignment in session.routing_plan.assignments}
    if len(assignments) != len(session.routing_plan.assignments):
        _fail("Duplicate physical routing assignment")
    descriptor_names = {descriptor.canonical_name: descriptor
                        for descriptor in session.descriptors}
    if len(descriptor_names) != len(session.descriptors):
        _fail("Duplicate canonical Session descriptor")
    by_id = {id(param): "network." + name for name, param in named.items()}
    if len(by_id) != len(named):
        _fail("Aliased physical LoRA Parameters are not a disjoint reference")
    expected_names = {"network." + name for name in targets}
    if set(descriptor_names) != expected_names or set(assignments) != expected_names:
        _fail("Session descriptor/routing ownership is not the 526 target tensors")
    runtime_names = set()
    children: dict[str, list[str]] = {}
    for runtime_spec, entry in zip(session.runtime_spec.optimizers, optimizer.entries):
        if runtime_spec.profile_name != entry.profile_name:
            _fail("Runtime Profile differs from physical Child ordering")
        spec_names = []
        for group_spec in runtime_spec.groups:
            for p in group_spec.parameters:
                if p.parameter_id != id(p.parameter):
                    _fail("RuntimeSpec contains stale Parameter ID")
                spec_names.append(p.canonical_name)
        group_names = []
        for group in entry.optimizer.param_groups:
            for p in group["params"]:
                if id(p) not in by_id:
                    _fail("Physical Child owns unknown or frozen Parameter")
                group_names.append(by_id[id(p)])
        if len(spec_names) != len(set(spec_names)) or set(spec_names) != set(group_names):
            _fail("Runtime/physical Child Parameter mismatch")
        if len(group_names) != len(set(group_names)):
            _fail("Duplicate physical Parameter within Child")
        if runtime_names.intersection(group_names):
            _fail("Optimizer Child Parameter sets overlap")
        runtime_names.update(group_names)
        children[entry.profile_name] = sorted(group_names)
    if runtime_names != expected_names:
        _fail("Physical optimizer children do not own all 526 tensors exactly once")
    rows = {}
    device = accelerator.device
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    expected_device = torch.device("cuda", device_index)
    for name, param in sorted(named.items()):
        key = "network." + name
        descriptor, route = descriptor_names[key], assignments[key]
        if (descriptor.parameter is not param or descriptor.parameter_id != id(param)
                or route.parameter is not param or route.parameter_id != id(param)
                or route.route_kind not in ("primary", "fallback")):
            _fail("Physical descriptor, route or Parameter ID does not match")
        if not param.requires_grad or param.dtype != torch.bfloat16:
            _fail("Trainable Parameter is frozen or not BF16")
        if param.device != expected_device:
            _fail("Trainable Parameter is not on reviewed CUDA device")
        profile = route.optimizer_profile
        if key not in children.get(profile, []):
            _fail("Optimizer Profile does not own its routed Parameter")
        component = route.component_id
        module = name.rsplit(".lora_", 1)[0]
        if component != target_spec.get(module, {}).get("component_id"):
            _fail("LoRA Component identity differs from real Module")
        rows[name] = {
            "canonical_name": key, "component_id": component,
            "route_kind": route.route_kind, "optimizer_profile": profile,
            "parameter_class": "matrix_weight", "dtype": "BF16",
            "shape": list(param.shape), "requires_grad": True,
        }
    for name, tensor in buffers.items():
        if (name not in alpha_keys or tensor.dtype != torch.bfloat16
                or tensor.device != expected_device or tensor.numel() != 1):
            _fail("Missing/invalid physical BF16 LoRA Alpha Buffer")
    component_counts = {}
    for row in rows.values():
        component_counts[row["component_id"]] = component_counts.get(row["component_id"], 0) + 1
    if set(component_counts) != {
        "mmdit.attention.adapter", "mmdit.mlp.adapter", "mmdit.modulation_norm.adapter",
    }:
        _fail("Missing a required trained Component")
    return {
        "source": "prepared-runtime-ownership-v1",
        "network_split_qkv": False,
        "runtime_topology_fingerprint": session.runtime_spec.topology_fingerprint,
        "parameter_rows": rows,
        "optimizer_child_parameter_names": children,
        "trainable_parameter_tensors": len(rows),
        "trained_component_counts": component_counts,
        "alpha_buffer_count": len(buffers),
    }


class PhysicalStepProbe:
    """Instance-only physical step recorder over REAL prepared torch objects.

    The CPU test path proves hook mechanics, never reference CUDA ownership.
    """
    def __init__(
        self, *, accelerator: Any, composite_optimizer: CompositeOptimizer,
        composite_scheduler: CompositeLRScheduler,
        parameter_names: Mapping[str, torch.nn.Parameter],
        case_id: str, run_id: str, phase_id: str, source_commit: str,
        phase: str, start_step: int = 0,
    ):
        spec = host.get_case_spec(case_id)
        if not isinstance(composite_optimizer, CompositeOptimizer):
            _fail("Probe requires real CompositeOptimizer")
        if not isinstance(composite_scheduler, CompositeLRScheduler):
            _fail("Probe requires real CompositeLRScheduler")
        if composite_scheduler.optimizer is not composite_optimizer:
            _fail("Probe has mismatched Optimizer/Scheduler")
        self.accelerator = accelerator
        self.optimizer = composite_optimizer
        self.scheduler = composite_scheduler
        self.params = dict(parameter_names)
        self.case_id, self.run_id, self.phase_id = case_id, run_id, phase_id
        self.source_commit, self.phase = source_commit, phase
        self.accumulation = spec["gradient_accumulation_steps"]
        self.start_step = start_step
        self.events: list[dict[str, Any]] = []
        self.sync_sequence: tuple[bool, ...] = ()
        self._installed = False
        self._patched: list[tuple[Any, str, bool, Any]] = []
        self._backwards = 0
        self._started = 0
        self._current_step: int | None = None
        self._current_micro: int | None = None
        self._current_sync: bool | None = None
        self._backward_seen = False
        self._pending_nosync_state: dict[str, str] | None = None
        self._step_children: dict[int, dict[str, int]] = {}
        self._step_scheduler: dict[int, int] = {}
        self._sample_names: dict[int, set[str]] = {}
        self._pre_step_grad: dict[str, Any] | None = None
        required_types = tuple(spec["optimizer_types"])
        physical_types = tuple(entry.optimizer_type for entry in composite_optimizer.entries)
        if sorted(physical_types) != sorted(required_types):
            _fail("Probe Child optimizer types do not match fixed R1/R2")
        if len({entry.profile_name for entry in composite_optimizer.entries}) != len(required_types):
            _fail("Duplicate Profile name in physical optimizer")
        self.child_profiles = tuple(entry.profile_name for entry in composite_optimizer.entries)
        owners = {}
        seen: set[int] = set()
        named_ids = {id(p): name for name, p in self.params.items()}
        if len(named_ids) != len(self.params):
            _fail("Aliased physical Parameters in Probe")
        for entry in composite_optimizer.entries:
            ids = [id(p) for group in entry.optimizer.param_groups for p in group["params"]]
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                _fail("Duplicate physical Child parameters")
            seen.update(ids)
            if any(i not in named_ids for i in ids):
                _fail("Physical Child owns unregistered parameters")
            owners[entry.profile_name] = [self.params[named_ids[i]] for i in ids]
        if seen != set(named_ids):
            _fail("Probe misses actual Optimizer-owned Parameters")
        self.child_params = owners

    def _record(self, event_type: str, *, profile: str | None = None,
                sample: str | None = None, sync: bool | None = None,
                details: dict[str, Any] | None = None) -> None:
        if self._current_step is None:
            _fail("Probe event outside a real training Batch")
        row: dict[str, Any] = {
            "run_id": self.run_id, "phase_id": self.phase_id,
            "case_id": self.case_id, "source_commit": self.source_commit,
            "event_seq": len(self.events) + 1, "event_type": event_type,
            "logical_step": self._current_step,
        }
        if self._current_micro is not None and event_type in (
            "batch_started", "backward_completed",
        ):
            row["microbatch_index"] = self._current_micro
        if profile is not None:
            row["optimizer_profile"] = profile
        if sample is not None:
            row["sample_identity"] = sample
        if sync is not None:
            row["sync_gradients"] = sync
        if details:
            row.update(details)
        self.events.append(row)

    def _snapshot(self) -> dict[str, str]:
        return {
            "network": fingerprint_state_tree({name: p.detach() for name,p in sorted(self.params.items())}),
            "children": fingerprint_state_tree({
                entry.profile_name: entry.optimizer.state_dict()
                for entry in self.optimizer.entries
            }),
            "scheduler": fingerprint_state_tree(self.scheduler.state_dict()),
        }

    def _gradients(self, *, require_nonzero: bool) -> dict[str, Any]:
        names = {id(p): name for name,p in self.params.items()}
        children = {}
        for entry in self.optimizer.entries:
            nonzero, count = 0, 0
            gradient_identity = hashlib.sha256()
            for p in self.child_params[entry.profile_name]:
                grad = p.grad
                if grad is None or grad.layout != torch.strided:
                    _fail("Missing or unsupported real Backward Gradient")
                if grad.dtype != torch.bfloat16 or grad.device != p.device:
                    _fail("Non-BF16 or wrong-device physical gradient")
                if not bool(torch.isfinite(grad).all().item()):
                    _fail("Non-finite physical BF16 Gradient")
                fingerprint = fingerprint_tensor(grad)
                gradient_identity.update(host._canonical_json_bytes({
                    "canonical_name": names[id(p)],
                    "dtype": fingerprint["dtype"],
                    "shape": fingerprint["shape"],
                    "sha256": fingerprint["sha256"],
                }))
                gradient_identity.update(b"\\n")
                count += 1
                nonzero += int(torch.count_nonzero(grad).item() > 0)
            if require_nonzero and not nonzero:
                _fail("Optimizer Child has no real nonzero BF16 Gradient")
            children[entry.profile_name] = {
                "tensor_count": count, "nonzero_tensor_count": nonzero,
                "gradient_sha256": gradient_identity.hexdigest(),
            }
        return {
            "child_gradients": children,
            "gradient_phase": "post_backward_pre_clip" if not require_nonzero
                              else "post_clip_pre_physical_step",
        }

    def _patch(self, obj: Any, name: str, wrapped: Callable) -> None:
        had_instance = name in vars(obj)
        original_instance = vars(obj).get(name)
        previous = getattr(obj, name)
        # torch LRScheduler checks this marker to detect overwritten optimizer
        # steps. The underlying scheduler-wrapped original still executes
        # exactly once; retaining its marker avoids a false scheduling warning.
        if name == "step" and getattr(previous, "_wrapped_by_lr_sched", False):
            wrapped._wrapped_by_lr_sched = True
        self._patched.append((obj,name,had_instance,original_instance))
        setattr(obj, name, wrapped)

    def install(self) -> None:
        if self._installed:
            _fail("Probe is already installed")
        original_backward = self.accelerator.backward
        original_optimizer_step = self.optimizer.step
        original_scheduler_step = self.scheduler.step
        self._installed = True

        def backward_wrapper(*args, **kwargs):
            if self._current_step is None or self._backward_seen:
                _fail("Backward without unique active real Batch")
            result = original_backward(*args, **kwargs)
            summary = self._gradients(require_nonzero=False)
            self._record("backward_completed", sample=self._sample,
                         sync=self._current_sync, details=summary)
            self._backward_seen = True
            self._backwards += 1
            return result

        def optimizer_wrapper(*args, **kwargs):
            if not self._backward_seen or self._current_sync is not True:
                _fail("Physical CompositeOptimizer.step before synchronized Backward")
            step = self._current_step
            if sum(self._step_children[step].values()):
                _fail("Duplicate CompositeOptimizer physical Step")
            self._pre_step_grad = self._gradients(require_nonzero=True)
            result = original_optimizer_step(*args, **kwargs)
            if set(k for k,v in self._step_children[step].items() if v) != set(self.child_profiles):
                _fail("CompositeOptimizer did not physically execute every Child once")
            return result

        def scheduler_wrapper(*args, **kwargs):
            if not self._backward_seen or self._current_sync is not True:
                _fail("Scheduler advanced without synchronized Backward")
            step = self._current_step
            if self._step_scheduler[step] != 0:
                _fail("Duplicate physical Scheduler step")
            if set(k for k,v in self._step_children[step].items() if v) != set(self.child_profiles):
                _fail("Scheduler step preceded physical Child completion")
            before = fingerprint_state_tree(self.scheduler.state_dict())
            result = original_scheduler_step(*args, **kwargs)
            after = fingerprint_state_tree(self.scheduler.state_dict())
            if after == before:
                _fail("Real CompositeLRScheduler step did not change state")
            self._step_scheduler[step] += 1
            self._record("physical_scheduler_stepped", details={
                "scheduler_before": before, "scheduler_after": after,
            })
            return result

        self._patch(self.accelerator, "backward", backward_wrapper)
        self._patch(self.optimizer, "step", optimizer_wrapper)
        self._patch(self.scheduler, "step", scheduler_wrapper)
        for entry in self.optimizer.entries:
            original_child = entry.optimizer.step
            profile = entry.profile_name

            def make_child_wrapper(original: Callable, name: str):
                def wrapped(*args, **kwargs):
                    if not self._backward_seen or self._current_sync is not True:
                        _fail("Physical Child.step outside synchronized Backward")
                    step = self._current_step
                    if self._step_children[step][name] != 0:
                        _fail("Duplicate Child physical optimizer step")
                    before_params = fingerprint_state_tree({
                        k: v.detach() for k,v in self.params.items()
                        if any(v is p for p in self.child_params[name])
                    })
                    before_state = fingerprint_state_tree(
                        next(e for e in self.optimizer.entries if e.profile_name==name)
                        .optimizer.state_dict()
                    )
                    result = original(*args, **kwargs)
                    after_params = fingerprint_state_tree({
                        k: v.detach() for k,v in self.params.items()
                        if any(v is p for p in self.child_params[name])
                    })
                    after_state = fingerprint_state_tree(
                        next(e for e in self.optimizer.entries if e.profile_name==name)
                        .optimizer.state_dict()
                    )
                    if before_state == after_state:
                        _fail("Physical Child did not advance actual optimizer state")
                    child = next(e for e in self.optimizer.entries if e.profile_name==name)
                    if child.optimizer_type == "Muon":
                        actual_steps = [
                            group.get("step")
                            for group in child.optimizer.param_groups
                        ]
                        if any(type(n) is not int or n != step for n in actual_steps):
                            _fail("Real Muon physical ParamGroup step counter disagrees with logical Step")
                    elif child.optimizer_type == "AdamW":
                        state_steps = []
                        for group in child.optimizer.param_groups:
                            for parameter in group["params"]:
                                state = child.optimizer.state.get(parameter, {})
                                actual = state.get("step")
                                if isinstance(actual, torch.Tensor):
                                    actual = actual.item()
                                state_steps.append(actual)
                        if (not state_steps
                                or any(type(n) not in (int,float) or n != step
                                       for n in state_steps)):
                            _fail("Real AdamW physical state step counter disagrees with logical Step")
                    else:
                        _fail("Unreviewed physical Optimizer Child type")
                    self._step_children[step][name] += 1
                    self._record("physical_child_stepped", profile=name, details={
                        "optimizer_state_before": before_state,
                        "optimizer_state_after": after_state,
                        "parameter_before": before_params,
                        "parameter_after": after_params,
                        "post_clip_gradient": self._pre_step_grad,
                        "physical_step": step,
                    })
                    return result
                return wrapped
            self._patch(entry.optimizer, "step", make_child_wrapper(original_child, profile))

    def begin_batch(self, sample_marker: str) -> None:
        if not self._installed:
            _fail("Cannot begin a Batch before Probe installation")
        if self._pending_nosync_state is not None:
            if self._snapshot() != self._pending_nosync_state:
                _fail("premature/non-sync accumulation changed Adapter, Optimizer or Scheduler")
            self._pending_nosync_state = None
        if self._current_step is not None and not self._backward_seen:
            _fail("Previous Batch had no real Backward")
        if self._current_step is not None and self._current_sync is True:
            self._assert_step(self._current_step)
        current_step = self.start_step + self._backwards // self.accumulation + 1
        current_micro = self._backwards % self.accumulation + 1
        sync = self.accelerator.sync_gradients
        if type(sync) is not bool or sync != (current_micro == self.accumulation):
            _fail("Incorrect real Accelerate synchronization sequence for accumulation")
        if (not isinstance(sample_marker, str)
                or not sample_marker.startswith("dts_c2b_sample_")
                or len(sample_marker) != len("dts_c2b_sample_0001")
                or not sample_marker[-4:].isdigit()):
            _fail("Real Batch must carry its recognized generated Caption marker")
        samples = self._sample_names.setdefault(current_step, set())
        if sample_marker in samples:
            _fail("Reused physical Caption marker within one logical step")
        samples.add(sample_marker)
        self._current_step, self._current_micro, self._current_sync = (
            current_step, current_micro, sync,
        )
        self._sample = sample_marker
        self._backward_seen = False
        self._started += 1
        self._step_children.setdefault(current_step, {name:0 for name in self.child_profiles})
        self._step_scheduler.setdefault(current_step, 0)
        self.sync_sequence += (sync,)
        if not sync:
            self._pending_nosync_state = self._snapshot()
        self._record("batch_started", sample=sample_marker, sync=sync)

    def _assert_step(self, step: int) -> None:
        counts = self._step_children.get(step, {})
        if set(counts) != set(self.child_profiles) or any(v != 1 for v in counts.values()):
            _fail("Physical Step missing/duplicate Muon or AdamW Child")
        if self._step_scheduler.get(step) != 1:
            _fail("Physical Step missing/duplicate real Scheduler advance")

    def assert_step_complete(self) -> None:
        if not self._installed or self._pending_nosync_state is not None:
            _fail("Incomplete accumulation or detached physical Probe")
        if self._backwards != self._started or self._backwards % self.accumulation:
            _fail("Incomplete real Backward/Microbatch pair sequence")
        if not self._step_children:
            _fail("No physical optimization step was observed")
        for step in sorted(self._step_children):
            self._assert_step(step)

    def close(self) -> None:
        for obj, name, had_instance, original_instance in reversed(self._patched):
            if had_instance:
                setattr(obj, name, original_instance)
            elif name in vars(obj):
                delattr(obj, name)
        self._patched.clear()
        self._installed = False


def __getattr__(name: str):
    if name == "ObservedSd3NetworkTrainer":
        # Lazy production import: Host tests and pure CPU probe need NOT import
        # SD3's optional GPU-only/trainer dependency graph.
        from tools.sd3_full_bf16_observed_trainer import ObservedSd3NetworkTrainer
        return ObservedSd3NetworkTrainer
    raise AttributeError(name)


__all__ = [
    "fingerprint_tensor", "fingerprint_state_tree", "snapshot_rng_fingerprint",
    "verify_prepared_identity", "observe_prepared_ownership",
    "PhysicalStepProbe", "ObservedSd3NetworkTrainer",
]
