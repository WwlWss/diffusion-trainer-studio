"""C3 instance-local physical SD3 Full-BF16 training observations.

A CPU caller can exercise these algorithms but is NEVER issued physical CUDA
qualification. The fixed production admission is in sd3_full_bf16_evidence_worker;
C4 save/load fingerprint binding has not been implemented.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
import random
import re
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



def audit_physical_optimizer_groups(
    *, session: Any, canonical_names_by_id: Mapping[int, str],
    routing_assignments: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Device-neutral real object identity audit; NEVER a CUDA qualification.

    Production invokes this after its fixed 263/526 BF16 CUDA scope gate.
    CPU tests cover the exact same group/Component/Route invariants.
    """
    optimizer = session.optimizer
    if not isinstance(optimizer, CompositeOptimizer):
        _fail("Physical group audit requires real DTS CompositeOptimizer")
    if len(session.runtime_spec.optimizers) != len(optimizer.entries):
        _fail("Mismatch in actual runtime Optimizer topology")
    assignments = routing_assignments
    by_id = canonical_names_by_id
    expected_names = set(by_id.values())
    runtime_names = set()
    children: dict[str, list[str]] = {}
    for runtime_spec, entry in zip(session.runtime_spec.optimizers, optimizer.entries):
        if runtime_spec.profile_name != entry.profile_name:
            _fail("Runtime Profile differs from physical Child ordering")
        if len(runtime_spec.groups) != len(entry.optimizer.param_groups):
            _fail("Physical Child has extra/missing RuntimeSpec Parameter Groups")
        spec_names = []
        group_names = []
        for group_spec, group in zip(runtime_spec.groups, entry.optimizer.param_groups):
            expected_ids = tuple(p.parameter_id for p in group_spec.parameters)
            actual_ids = tuple(id(p) for p in group["params"])
            if expected_ids != actual_ids:
                _fail("Physical Child Parameter Group/order differs from RuntimeSpec")
            if group.get("lr") != group_spec.learning_rate:
                _fail("Physical Child learning-rate group differs from reviewed RuntimeSpec")
            for p in group_spec.parameters:
                if p.parameter_id != id(p.parameter) or p.canonical_name not in assignments:
                    _fail("RuntimeSpec contains stale or unknown physical Parameter")
                route = assignments[p.canonical_name]
                if (p.parameter is not route.parameter or
                        p.component_id != route.component_id or
                        p.route_kind != route.route_kind or
                        p.parameter_class != route.parameter_class or
                        tuple(p.shape) != tuple(p.parameter.shape)):
                    _fail("RuntimeSpec and physical RoutingPlan disagree on Component/Route")
                spec_names.append(p.canonical_name)
            for param in group["params"]:
                if id(param) not in by_id:
                    _fail("Physical Child owns unknown or frozen Parameter")
                group_names.append(by_id[id(param)])
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
    return children



def validate_reference_ownership_rows(
    *, case_id: str, rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Reference-anchored 526-parameter geometry, route and Child mapping.

    Device-neutral so real CPU tests can reject a self-consistent *wrong*
    RuntimeSpec/RoutingPlan without pretending to establish CUDA ownership.
    """
    spec = host.build_sd35_medium_lora_target_spec()
    policy = host.build_reference_policy(case_id)
    expected = {module + suffix for module in spec
                for suffix in (".lora_down.weight", ".lora_up.weight")}
    if not isinstance(rows, Mapping) or set(rows) != expected:
        _fail("Actual parameter row set differs from exact 526 reference weights")
    component_counts: dict[str, int] = {}
    child_counts: dict[str, int] = {
        profile: 0 for profile in policy["optimizer_profiles"]
    }
    for module, target in spec.items():
        component = target["component_id"]
        route = policy["components"].get(component)
        if not isinstance(route, Mapping) or route.get("train") is not True:
            _fail("Reference Component must be explicitly trainable")
        fallback = case_id == "r2-muon-adamw-fallback-accum2" and (
            component == "mmdit.modulation_norm.adapter"
        )
        expected_kind = "fallback" if fallback else "primary"
        expected_profile = (
            route.get("fallback_optimizer_profile") if fallback
            else route.get("optimizer_profile")
        )
        if expected_profile not in child_counts:
            _fail("Reference optimizer Profile is absent")
        for suffix, dims in (
            (".lora_down.weight", [target["rank"], target["input_features"]]),
            (".lora_up.weight", [target["output_features"], target["rank"]]),
        ):
            name = module + suffix
            row = rows[name]
            if not isinstance(row, Mapping):
                _fail("Missing physical ownership row")
            if (row.get("canonical_name") != "network." + name
                    or row.get("component_id") != component
                    or row.get("route_kind") != expected_kind
                    or row.get("optimizer_profile") != expected_profile
                    or row.get("shape") != dims
                    or row.get("dtype") != "BF16"
                    or row.get("parameter_class") != "matrix_weight"
                    or row.get("requires_grad") is not True):
                _fail("Reference LoRA shape/rank/route/Component ownership mismatch: " + name)
            component_counts[component] = component_counts.get(component, 0) + 1
            child_counts[expected_profile] += 1
    expected_components = {
        "mmdit.attention.adapter": 242,
        "mmdit.mlp.adapter": 188,
        "mmdit.modulation_norm.adapter": 96,
    }
    if component_counts != expected_components:
        _fail("Physical Component reference tensor totals drifted")
    expected_children = (
        {"baseline": 526} if case_id == "r1-adamw-accum1"
        else {"muon_primary": 430, "adamw_fallback": 96}
    )
    if child_counts != expected_children:
        _fail("Physical optimizer Child reference totals drifted")
    return {
        "component_tensor_counts": component_counts,
        "child_tensor_counts": child_counts,
    }


def validate_reference_alpha_buffers(
    buffers: Mapping[str, torch.Tensor],
    target_spec: Mapping[str, Any],
    *, expected_device: torch.device,
) -> int:
    """Exact scalar BF16 alpha=4 reference, distinct from trainable weights."""
    expected = {module + ".alpha" for module in target_spec}
    if not isinstance(buffers, Mapping) or set(buffers) != expected:
        _fail("Reference Alpha Buffer names are incomplete/extra")
    for name, value in buffers.items():
        if (not isinstance(value, torch.Tensor) or value.is_meta
                or value.layout != torch.strided
                or value.dtype != torch.bfloat16 or value.device != expected_device
                or value.ndim != 0 or value.requires_grad
                or not bool(torch.isfinite(value).item())
                or float(value.item()) != 4.0):
            _fail("Reference Alpha must be a frozen BF16 scalar alpha=4: " + name)
    return len(expected)





def _audit_lora_target_execution(
    name: str, target: torch.nn.Linear,
    *, allowed_pre_hooks: Mapping[int, tuple[int, Callable]] | None = None,
) -> None:
    """Fail closed on outer target Hook registration, including late insertion.

    Legitimate runtime instrumentation belongs only to this Probe's exact,
    read-only pre-hook. The target can mutate its *output* through ordinary
    forward_hooks even though target.forward remains LoRAModule.forward.
    """
    if type(target) is not torch.nn.Linear:
        _fail("LoRA target execution no longer uses the reviewed Linear: " + name)
    if (getattr(target, "_forward_hooks", None)
            or getattr(target, "_forward_hooks_with_kwargs", None)
            or getattr(target, "_forward_hooks_always_called", None)
            or getattr(target, "_forward_pre_hooks_with_kwargs", None)):
        _fail("Unreviewed LoRA target Forward hook: " + name)
    current = getattr(target, "_forward_pre_hooks", None)
    if not isinstance(current, Mapping):
        _fail("LoRA target Forward pre-hook registry is malformed: " + name)
    allowed = (allowed_pre_hooks or {}).get(id(target))
    if allowed is None:
        if current:
            _fail("Unreviewed LoRA target Forward pre-hook: " + name)
    else:
        hook_id, callback = allowed
        if len(current) != 1 or current.get(hook_id) is not callback:
            _fail("LoRA target Forward pre-hook differs from Probe-owned hook: " + name)


def _audit_lora_projection_execution(
    name: str, role: str, projection: torch.nn.Module,
    *, allowed_pre_hooks: Mapping[int, tuple[int, Callable]] | None = None,
) -> None:
    """Validate genuine Linear execution, not just Parameters and dimensions.

    At Prepared there must be no per-Linear hooks. A live Probe may install
    exactly its own read-only pre-hook; all output-changing or unexpected
    input hooks remain forbidden. Hook identities are process-local and are
    never included in a portable evidence SHA.
    """
    if type(projection) is not torch.nn.Linear:
        _fail("LoRA projection changed its real Linear class: " + name + "/" + role)
    # PyTorch global Module Forward hooks run for ALL Linear modules and can
    # change inputs or returned activations without modifying this specific
    # projection's own hook registry. The fixed evidence reference has none.
    from torch.nn.modules import module as torch_module_runtime
    if (getattr(torch_module_runtime, "_global_forward_pre_hooks", None)
            or getattr(torch_module_runtime, "_global_forward_hooks", None)):
        _fail("Unreviewed global LoRA projection Forward hook: " + name + "/" + role)
    current = getattr(projection, "forward", None)
    if (getattr(current, "__self__", None) is not projection
            or getattr(current, "__func__", None) is not torch.nn.Linear.forward):
        _fail("LoRA projection Forward callable was replaced: " + name + "/" + role)
    if (getattr(projection, "_forward_hooks", None)
            or getattr(projection, "_forward_hooks_with_kwargs", None)
            or getattr(projection, "_forward_hooks_always_called", None)
            or getattr(projection, "_forward_pre_hooks_with_kwargs", None)):
        _fail("Unreviewed LoRA projection Forward hook: " + name + "/" + role)
    actual_pre_hooks = getattr(projection, "_forward_pre_hooks", None)
    if not isinstance(actual_pre_hooks, Mapping):
        _fail("LoRA projection Forward pre-hook registry is malformed: " + name + "/" + role)
    allowed = (allowed_pre_hooks or {}).get(id(projection))
    if allowed is None:
        if actual_pre_hooks:
            _fail("Unreviewed LoRA projection Forward pre-hook: " + name + "/" + role)
    else:
        hook_id, hook_callable = allowed
        if (len(actual_pre_hooks) != 1
                or actual_pre_hooks.get(hook_id) is not hook_callable):
            _fail("Unexpected LoRA projection Forward pre-hook: " + name + "/" + role)


def _lora_forward_module_payload(
    name: str, lora: Any,
    *, allowed_projection_pre_hooks: Mapping[int, tuple[int, Callable]] | None = None,
    allowed_target_pre_hooks: Mapping[int, tuple[int, Callable]] | None = None,
) -> tuple[dict[str, Any], torch.nn.Linear]:
    """Audit genuine LoRAModule effective Forward (not merely its state_dict).

    The real SD3 network patches the original Linear.forward with a bound
    LoRAModule.forward; alpha is a separate registered BF16 Buffer while
    scale/multiplier/dropout are regular mutable Python attributes.
    """
    from networks.lora_flux import LoRAModule

    if type(lora) is not LoRAModule or type(name) is not str or not name:
        _fail("Unreviewed actual LoRA Forward module type/name")
    if lora.lora_name != name or type(lora.lora_name) is not str:
        _fail("LoRA Forward canonical name drift: " + name)
    if type(lora.lora_dim) is not int or lora.lora_dim != 4:
        _fail("LoRA Forward rank differs from fixed dim=4: " + name)
    if type(lora.scale) is not float or lora.scale != 1.0:
        _fail("LoRA Forward scale differs from exact alpha/rank: " + name)
    if type(lora.multiplier) is not float or lora.multiplier != 1.0:
        _fail("LoRA Forward multiplier differs from fixed training reference: " + name)
    if lora.split_dims is not None:
        _fail("Unreviewed LoRA Forward split dimensions: " + name)
    if (lora.rank_dropout is not None or lora.module_dropout is not None
            or lora.ggpo_sigma is not None or lora.ggpo_beta is not None):
        _fail("Unreviewed LoRA Forward dropout/GGPO modifier: " + name)
    dropout = lora.dropout
    if dropout is not None and not (type(dropout) is float and dropout == 0.0):
        _fail("Unreviewed LoRA Forward neuron dropout: " + name)
    alpha = getattr(lora, "alpha", None)
    if (not isinstance(alpha, torch.Tensor)
            or lora._buffers.get("alpha") is not alpha
            or alpha.ndim != 0 or alpha.dtype != torch.bfloat16
            or alpha.requires_grad or alpha.layout != torch.strided
            or alpha.is_meta or not bool(torch.isfinite(alpha).item())
            or float(alpha.item()) != 4.0):
        _fail("LoRA Forward alpha Buffer differs from frozen BF16 scalar=4: " + name)
    for role in ("lora_down", "lora_up"):
        _audit_lora_projection_execution(
            name, role, getattr(lora, role, None),
            allowed_pre_hooks=allowed_projection_pre_hooks,
        )
    org_forward = getattr(lora, "org_forward", None)
    target = getattr(org_forward, "__self__", None)
    if (type(target) is not torch.nn.Linear
            or getattr(org_forward, "__func__", None) is not torch.nn.Linear.forward):
        _fail("LoRA Forward original Linear binding drift: " + name)
    live_forward = getattr(target, "forward", None)
    if (getattr(live_forward, "__self__", None) is not lora
            or getattr(live_forward, "__func__", None) is not LoRAModule.forward):
        _fail("LoRA Forward target Linear has been rebound: " + name)
    _audit_lora_target_execution(
        name, target, allowed_pre_hooks=allowed_target_pre_hooks,
    )
    if (lora.lora_down.in_features != target.in_features
            or lora.lora_down.out_features != 4
            or lora.lora_up.in_features != 4
            or lora.lora_up.out_features != target.out_features):
        _fail("LoRA Forward projection/target feature dimensions drift: " + name)
    if (alpha.device != lora.lora_down.weight.device
            or alpha.device != lora.lora_up.weight.device):
        _fail("LoRA Forward Alpha/Parameter device mismatch: " + name)
    return {
        "canonical_name": name,
        "module_type": "networks.lora_flux.LoRAModule",
        "lora_dim": lora.lora_dim, "scale": lora.scale,
        "multiplier": lora.multiplier,
        "dropout": dropout, "rank_dropout": lora.rank_dropout,
        "module_dropout": lora.module_dropout,
        "ggpo_sigma": lora.ggpo_sigma, "ggpo_beta": lora.ggpo_beta,
        "split_dims": lora.split_dims,
        "alpha": fingerprint_tensor(alpha),
        "target_type": "torch.nn.Linear",
        "target_features": (target.in_features, target.out_features),
    }, target


def _reference_sd3_mmdit_target_paths(target_spec: Mapping[str, Any]) -> dict[str, str]:
    """Independent 263-LoRA canonical-name -> genuine SD3 MMDiT module path.

    Must not derive this authority from a mutable LoRA target marker.
    """
    if (not isinstance(target_spec, Mapping)
            or target_spec != host.build_sd35_medium_lora_target_spec()):
        _fail("MMDiT target path spec differs from immutable SD3.5 Medium reference")
    suffixes = {
        "attn_qkv": "attn.qkv",
        "attn_proj": "attn.proj",
        "attn2_qkv": "attn2.qkv",
        "attn2_proj": "attn2.proj",
        "mlp_fc1": "mlp.fc1",
        "mlp_fc2": "mlp.fc2",
        "adaLN_modulation_1": "adaLN_modulation.1",
    }
    result = {}
    pattern = r"lora_unet_joint_blocks_([0-9]+)_(context_block|x_block)_(.+)"
    for name in target_spec:
        match = re.fullmatch(pattern, name)
        if match is None:
            _fail("Unreviewed LoRA canonical MMDiT target name: " + name)
        block, side, suffix = match.groups()
        if (block != str(int(block)) or int(block) not in range(24)
                or suffix not in suffixes):
            _fail("LoRA canonical MMDiT target path cannot be resolved: " + name)
        result[name] = f"joint_blocks.{block}.{side}.{suffixes[suffix]}"
    if len(result) != 263 or len(set(result.values())) != 263:
        _fail("LoRA canonical MMDiT target path inventory has duplicates/omissions")
    return result


def _audit_real_lora_target_path(
    name: str, lora: Any, target: torch.nn.Linear,
    *, mmdit_root: torch.nn.Module, expected_path: str,
) -> dict[str, str]:
    """Bind the observed patched Linear to its real MMDiT Module path."""
    if not isinstance(mmdit_root, torch.nn.Module) or (
        type(expected_path) is not str or not expected_path
    ):
        _fail("Missing actual MMDiT root or canonical LoRA target path: " + name)
    expected_type = "torch.nn.modules.linear.Linear"
    marker = getattr(lora, "_dts_parameter_policy_target_v1", None)
    if (type(marker) is not tuple or len(marker) != 3
            or any(type(value) is not str for value in marker)
            or marker != ("mmdit", expected_path, expected_type)):
        _fail("LoRA MMDiT target path/type metadata differs from frozen reference: " + name)
    try:
        live_target = mmdit_root.get_submodule(expected_path)
    except AttributeError:
        _fail("Real MMDiT target path disappeared: " + name + " -> " + expected_path)
    if type(live_target) is not torch.nn.Linear or live_target is not target:
        _fail("Real MMDiT target path points to a different Linear: " + name)
    return {"mmdit_target_path": expected_path, "mmdit_target_type": expected_type}


def inspect_reference_lora_forward_semantics(
    *, network: torch.nn.Module,
    expected_names: tuple[str, ...] | None = None,
    target_spec: Mapping[str, Any] | None = None,
    allowed_projection_pre_hooks: Mapping[int, tuple[int, Callable]] | None = None,
    allowed_target_pre_hooks: Mapping[int, tuple[int, Callable]] | None = None,
    lora_mmdit_root: torch.nn.Module | None = None,
    expected_lora_target_paths: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.nn.Linear]]:
    """Read-only real Forward binding audit, independent of trainable weights.

    Real CUDA additionally provides the immutable 263-module target specification;
    tiny CPU diagnostics may use a frozen initial name inventory, but never
    confer physical CUDA qualification.
    """
    from networks.lora_flux import LoRAModule

    if not isinstance(network, torch.nn.Module):
        _fail("Missing real LoRA Forward Network")
    root_multiplier = getattr(network, "multiplier", None)
    if type(root_multiplier) is not float or root_multiplier != 1.0:
        _fail("LoRA Forward root multiplier differs from fixed training reference")
    training_loras = getattr(network, "unet_loras", None)
    text_loras = getattr(network, "text_encoder_loras", None)
    if not isinstance(training_loras, (list, tuple)) or not training_loras:
        _fail("Missing actual LoRA Forward MMDiT modules")
    if not isinstance(text_loras, (list, tuple)) or len(text_loras) != 0:
        _fail("LoRA Forward reference does not train text encoder modules")
    if any(type(lora) is not LoRAModule for lora in training_loras):
        _fail("Unreviewed training LoRA Forward module class")
    names = [lora.lora_name for lora in training_loras]
    if (any(type(n) is not str or not n for n in names)
            or len(set(names)) != len(names)):
        _fail("LoRA Forward target names are invalid or duplicated")
    if expected_names is not None and set(names) != set(expected_names):
        _fail("LoRA Forward target inventory differs from Prepared reference")
    if target_spec is not None and (
        len(names) != 263 or set(names) != set(target_spec)
    ):
        _fail("LoRA Forward physical reference is not exactly 263 reviewed targets")
    if target_spec is not None:
        # The real CUDA path must use independent canonical target names, not
        # a caller-supplied mutable path map.
        if expected_lora_target_paths is not None:
            _fail("Real MMDiT target paths cannot be supplied by the current Network")
        target_paths = _reference_sd3_mmdit_target_paths(target_spec)
    else:
        target_paths = (
            dict(expected_lora_target_paths)
            if expected_lora_target_paths is not None else None
        )
    if target_paths is None:
        if lora_mmdit_root is not None:
            _fail("Real MMDiT root supplied without reviewed target paths")
    elif (lora_mmdit_root is None or set(target_paths) != set(names)
          or any(type(path) is not str or not path for path in target_paths.values())
          or len(set(target_paths.values())) != len(target_paths)):
        _fail("LoRA MMDiT target-path reference inventory is incomplete/duplicated")

    registered = {
        key: module
        for key, module in network.named_modules()
        if type(module) is LoRAModule
    }
    if len(registered) != len(names) or (
        {id(m) for m in registered.values()} != {id(m) for m in training_loras}
    ):
        _fail("LoRA Forward registered Module inventory differs from real training modules")
    registered_path = {id(module): key for key, module in registered.items()}
    modules = {lora.lora_name: lora for lora in training_loras}
    payload = {}
    targets = {}
    for name in sorted(modules):
        lora = modules[name]
        fields, target = _lora_forward_module_payload(
            name, lora, allowed_projection_pre_hooks=allowed_projection_pre_hooks,
            allowed_target_pre_hooks=allowed_target_pre_hooks,
        )
        if target_spec is not None:
            spec = target_spec[name]
            if registered_path[id(lora)] != name:
                _fail("LoRA Forward canonical registration path drift: " + name)
            if (target.in_features != spec["input_features"]
                    or target.out_features != spec["output_features"]):
                _fail("LoRA Forward target geometry differs from immutable reference: " + name)
        if target_paths is not None:
            fields.update(_audit_real_lora_target_path(
                name, lora, target, mmdit_root=lora_mmdit_root,
                expected_path=target_paths[name],
            ))
        fields["registered_path"] = registered_path[id(lora)]
        payload[name] = fields
        targets[name] = target
    return {"root_multiplier": root_multiplier, "modules": payload}, modules, targets



def _reference_lr_hex(lr: Any, *, label: str) -> str:
    if type(lr) is not float or not math.isfinite(lr) or lr < 0:
        # The production fixed Policy is strictly positive (0.0001).
        # Zero is allowed solely for a non-CUDA CPU diagnostic fixture that
        # proves F3 rejects a real state-only BF16 optimizer step.
        _fail("Physical learning rate must be a nonnegative finite Python float: " + label)
    return lr.hex()


def validate_reference_group_learning_rates(
    *, case_id: str, session: Any,
) -> dict[str, list[float]]:
    """Compare *three* independent LR sources to the frozen R1/R2 Policy.

    The caller must independently verify complete 526-parameter ownership via
    observe_prepared_ownership() before treating this as CUDA evidence. This
    device-neutral helper also permits small real-Optimizer CPU negatives.
    """
    policy = host.build_reference_policy(case_id)
    assignments_list = tuple(session.routing_plan.assignments)
    assignments = {a.canonical_name: a for a in assignments_list}
    if not assignments_list or len(assignments) != len(assignments_list):
        _fail("Reference LR audit has empty/duplicate routing assignments")
    runtime = tuple(session.runtime_spec.optimizers)
    entries = tuple(session.optimizer.entries)
    expected_profiles = set(policy["optimizer_profiles"])
    if (len(runtime) != len(entries)
            or {s.profile_name for s in runtime} != expected_profiles
            or {e.profile_name for e in entries} != expected_profiles):
        _fail("Physical reference learning-rate Profiles differ from frozen Policy")
    actual_names: set[str] = set()
    resolved: dict[str, list[float]] = {}
    for runtime_entry, physical_entry in zip(runtime, entries):
        name = runtime_entry.profile_name
        if name != physical_entry.profile_name:
            _fail("Physical learning-rate Child order differs from RuntimeSpec")
        if len(runtime_entry.groups) != len(physical_entry.optimizer.param_groups):
            _fail("Physical learning-rate Parameter Group cardinality differs")
        groups: list[float] = []
        for group_spec, physical_group in zip(
            runtime_entry.groups, physical_entry.optimizer.param_groups
        ):
            parameters = tuple(group_spec.parameters)
            actual_parameters = tuple(physical_group["params"])
            if not parameters or len(parameters) != len(actual_parameters):
                _fail("Reference LR group has wrong physical Parameter count")
            reference_values: set[str] = set()
            for p_spec, real_parameter in zip(parameters, actual_parameters):
                canonical = p_spec.canonical_name
                assignment = assignments.get(canonical)
                if (assignment is None or canonical in actual_names
                        or getattr(p_spec, "parameter", None) is not real_parameter):
                    _fail("Reference LR group has stale/duplicate physical Parameter")
                actual_names.add(canonical)
                component = policy["components"].get(assignment.component_id)
                if not isinstance(component, Mapping) or component.get("train") is not True:
                    _fail("Reference LR group targets an unknown/frozen Component")
                if assignment.route_kind == "primary":
                    profile = component["optimizer_profile"]
                    expected = component["learning_rate"]
                elif assignment.route_kind == "fallback":
                    profile = component.get("fallback_optimizer_profile")
                    expected = component.get(
                        "fallback_learning_rate", component["learning_rate"]
                    )
                else:
                    _fail("Unreviewed physical learning-rate Route")
                if assignment.optimizer_profile != profile or name != profile:
                    _fail("Reference LR group uses the wrong Component/Profile Route")
                expected_hex = _reference_lr_hex(expected, label=canonical + "/reference")
                if _reference_lr_hex(
                    assignment.learning_rate, label=canonical + "/routing"
                ) != expected_hex:
                    _fail("Routing assignment differs from independent reference learning rate")
                reference_values.add(expected_hex)
            if len(reference_values) != 1:
                _fail("Physical Group mixes incompatible reference learning rates")
            expected_hex = next(iter(reference_values))
            if (_reference_lr_hex(
                    group_spec.learning_rate, label=name + "/RuntimeSpec"
                ) != expected_hex
                    or _reference_lr_hex(
                        physical_group.get("lr"), label=name + "/param_groups"
                    ) != expected_hex):
                _fail("Actual physical Group/RuntimeSpec learning rate differs from reference")
            groups.append(group_spec.learning_rate)
        resolved[name] = groups
    if actual_names != set(assignments):
        _fail("Reference LR assignment set and physical Group ownership disagree")
    return resolved



def _reference_provider_semantics(case_id: str) -> dict[str, dict[str, Any]]:
    """Build frozen semantics from Policy and fresh pinned-provider objects."""
    from pytorch_optimizer import Muon

    policy = host.build_reference_policy(case_id)
    out = {}
    for profile, definition in policy["optimizer_profiles"].items():
        kind = definition["type"]
        sample = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.bfloat16))
        payload = [{"params": [sample], "lr": 0.0001}]
        if kind == "Muon":
            payload[0]["use_muon"] = True
            opt = Muon(payload, **definition["args"])
            fields = {"lr", "momentum", "nesterov", "weight_decay",
                      "weight_decouple", "ns_steps", "ns_coeffs",
                      "use_adjusted_lr", "use_muon"}
            instance = {"maximize": opt.maximize}
        elif kind == "AdamW":
            opt = torch.optim.AdamW(payload, **definition["args"])
            fields = {"lr", "betas", "eps", "weight_decay", "amsgrad",
                      "maximize", "foreach", "capturable", "differentiable",
                      "fused", "decoupled_weight_decay"}
            instance = {}
        else:
            _fail("Unreviewed F7 Provider: " + str(kind))
        if set(opt.param_groups[0]) - {"params", "step"} != fields:
            _fail("Pinned provider schema changed: " + profile)
        out[profile] = {
            "optimizer_type": kind, "fields": frozenset(fields),
            "group": {k: opt.param_groups[0][k] for k in sorted(fields)},
            "instance_fields": instance,
        }
    return out


def _actual_provider_semantics(entry: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    from pytorch_optimizer import Muon

    opt = entry.optimizer
    kind = expected["optimizer_type"]
    if (kind == "AdamW" and type(opt) is not torch.optim.AdamW) or (
        kind == "Muon" and type(opt) is not Muon
    ):
        _fail("Physical Optimizer provider type differs from fixed reference")
    instance = {k: getattr(opt, k, None) for k in expected["instance_fields"]}
    if fingerprint_state_tree(instance) != fingerprint_state_tree(expected["instance_fields"]):
        _fail("Optimizer instance semantic drift")
    groups = []
    for group in opt.param_groups:
        if set(group) - {"params", "step", "initial_lr"} != expected["fields"]:
            _fail("Optimizer Group semantic field set differs from reference")
        if "initial_lr" in group and _typed_tree(group["initial_lr"]) != _typed_tree(group["lr"]):
            _fail("Optimizer Group initial_lr differs from current rate")
        groups.append({k: copy.deepcopy(group[k]) for k in sorted(expected["fields"])})
    if not groups:
        _fail("No physical provider Group")
    return {"optimizer_type": kind, "groups": groups, "instance_fields": instance}


def validate_reference_optimizer_group_semantics(
    *, case_id: str, session: Any,
) -> dict[str, dict[str, Any]]:
    """Independently bind the actual provider Groups to frozen R1/R2 defaults."""
    refs = _reference_provider_semantics(case_id)
    entries = tuple(session.optimizer.entries)
    if len(entries) != len(refs) or {x.profile_name for x in entries} != set(refs):
        _fail("Reference Optimizer semantic Profile mismatch")
    result = {}
    for entry in entries:
        ref = refs[entry.profile_name]
        actual = _actual_provider_semantics(entry, ref)
        if any(fingerprint_state_tree(group) != fingerprint_state_tree(ref["group"])
               for group in actual["groups"]):
            _fail("Physical Optimizer Group semantic differs from reference: " + entry.profile_name)
        # Snapshot ONLY audited static provider values, not live CUDA Parameters.
        # Nested Muon ns_coeffs must not share a list with the actual Group.
        snapshot = copy.deepcopy(actual)
        result[entry.profile_name] = {
            **snapshot, "semantic_sha256": fingerprint_state_tree(snapshot),
        }
    return result


def observe_prepared_ownership(
    *, session: Any, accelerator: Any, network: torch.nn.Module,
    train_dataloader: Any, case_id: str,
    mmdit_root: torch.nn.Module | None = None,
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
    children = audit_physical_optimizer_groups(
        session=session, canonical_names_by_id=by_id,
        routing_assignments=assignments,
    )
    # F9-B: freeze the independently audited RuntimeSpec Group *order*
    # while still inside the Prepared evidence boundary. Use stable canonical
    # names in evidence; never serialize process-specific Python object IDs.
    reference_group_names = {
        runtime.profile_name: tuple(
            tuple(parameter.canonical_name for parameter in group.parameters)
            for group in runtime.groups
        )
        for runtime in session.runtime_spec.optimizers
    }
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
    alpha_count = validate_reference_alpha_buffers(
        buffers, target_spec, expected_device=expected_device,
    )
    forward_payload, _forward_modules, _forward_targets = inspect_reference_lora_forward_semantics(
        network=network, expected_names=tuple(sorted(target_spec)),
        target_spec=target_spec, lora_mmdit_root=mmdit_root,
    )
    forward_reference = {
        "module_names": tuple(sorted(target_spec)),
        "semantic_sha256": fingerprint_state_tree(forward_payload),
    }
    reference = validate_reference_ownership_rows(case_id=case_id, rows=rows)
    reference_lrs = validate_reference_group_learning_rates(
        case_id=case_id, session=session,
    )
    reference_semantics = validate_reference_optimizer_group_semantics(
        case_id=case_id, session=session,
    )
    if {profile: len(members) for profile, members in children.items()} != reference["child_tensor_counts"]:
        _fail("Physical Child Group ownership differs from frozen 430/96 or 526 reference")
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
        "alpha_buffer_count": alpha_count,
        "reference_lora_forward_semantics": forward_reference,
        "reference_component_tensor_counts": reference["component_tensor_counts"],
        "reference_child_tensor_counts": reference["child_tensor_counts"],
        "reference_group_learning_rates": reference_lrs,
        "reference_optimizer_group_semantics": reference_semantics,
        "reference_group_parameter_names": reference_group_names,
    }



def validate_muon_parameter_states(
    muon: Any, *, physical_step: int,
    canonical_names_by_id: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Audit pinned pytorch-optimizer Muon 3.10.0 per-Parameter state.

    Only observed live Tensor state, no guessed step from a serialized blob.
    Group.step advances even if a Parameter had grad=None, hence all required
    momentum_buffers are checked separately. No GPU-only behavior is faked.
    """
    from pytorch_optimizer import Muon
    if not isinstance(muon, Muon) or type(physical_step) is not int or physical_step < 1:
        _fail("Muon state audit requires real Muon and positive physical Step")
    names = {}
    seen_ids: set[int] = set()
    total = 0
    for group_index, group in enumerate(muon.param_groups):
        if (group.get("use_muon") is not True
                or type(group.get("step")) is not int
                or group["step"] != physical_step):
            _fail("Muon physical Group step/use_muon mismatch")
        for parameter_index, param in enumerate(group["params"]):
            param_id = id(param)
            if param_id in seen_ids:
                _fail("Duplicate Muon physical Parameter state")
            seen_ids.add(param_id)
            if canonical_names_by_id is not None:
                name = canonical_names_by_id.get(param_id)
                if not isinstance(name, str) or not name:
                    _fail("Muon physical State lacks canonical Parameter name")
            else:
                name = f"group{group_index}/param{parameter_index}"
            state = muon.state.get(param)
            if not isinstance(state, dict) or "momentum_buffer" not in state:
                _fail("Muon physical Parameter missing momentum_buffer: " + name)
            if "exp_avg" in state or "exp_avg_sq" in state:
                _fail("Muon physical Parameter incorrectly uses AdamW fallback state: " + name)
            buffer = state["momentum_buffer"]
            if (not isinstance(buffer, torch.Tensor) or buffer.is_meta
                    or buffer.layout != torch.strided
                    or buffer.dtype != param.dtype or buffer.device != param.device
                    or tuple(buffer.shape) != tuple(param.shape)
                    or buffer.dtype != torch.bfloat16):
                _fail("Muon momentum_buffer BF16 shape/device mismatch: " + name)
            if not bool(torch.isfinite(buffer).all().item()):
                _fail("Muon momentum_buffer contains non-finite values: " + name)
            names[name] = fingerprint_tensor(buffer)["sha256"]
            total += 1
    if total == 0:
        _fail("Muon physical Parameter state inventory is empty")
    return {
        "momentum_parameter_count": total,
        "momentum_buffer_sha256_by_parameter": dict(sorted(names.items())),
        "momentum_inventory_sha256": host._json_sha256(names),
        "physical_step": physical_step,
    }




def validate_adamw_parameter_states(
    adamw: Any, *, physical_step: int,
    canonical_names_by_id: Mapping[int, str],
) -> dict[str, Any]:
    """Audit actual pinned torch.optim.AdamW 2.7 default BF16 moment state.

    exp_avg/exp_avg_sq follow their Parameter's BF16 dtype/device/shape.
    Default noncapturable/nonfused AdamW 'step' is a CPU float32 scalar,
    even when trainable Parameters live on CUDA. No silent provider fallback.
    """
    if (type(adamw) is not torch.optim.AdamW
            or type(physical_step) is not int or physical_step < 1):
        _fail("AdamW state audit requires real torch.optim.AdamW and positive physical Step")
    if not isinstance(canonical_names_by_id, Mapping):
        _fail("AdamW state audit requires canonical physical Parameter names")
    seen: set[int] = set()
    moments: dict[str, Any] = {}
    for group in adamw.param_groups:
        if (group.get("amsgrad") is not False
                or group.get("capturable") is not False
                or group.get("fused") not in (None, False)):
            _fail("Unreviewed AdamW provider state mode (amsgrad/capturable/fused)")
        for param in group["params"]:
            identifier = id(param)
            name = canonical_names_by_id.get(identifier)
            if (identifier in seen or not isinstance(name, str) or not name
                    or param.dtype != torch.bfloat16 or param.is_meta):
                _fail("AdamW moment state has duplicate/unknown/non-BF16 Parameter")
            seen.add(identifier)
            state = adamw.state.get(param)
            if not isinstance(state, Mapping) or set(state) != {
                "step", "exp_avg", "exp_avg_sq",
            }:
                _fail("AdamW Parameter missing/unreviewed step or momentum buffers: " + name)
            step_tensor = state["step"]
            if (not isinstance(step_tensor, torch.Tensor) or step_tensor.is_meta
                    or step_tensor.ndim != 0
                    or step_tensor.dtype != torch.float32
                    or step_tensor.device.type != "cpu"
                    or not bool(torch.isfinite(step_tensor).item())
                    or float(step_tensor.item()) != physical_step):
                _fail("AdamW CPU scalar step does not match real physical Step: " + name)
            values: dict[str, str] = {}
            for field_name in ("exp_avg", "exp_avg_sq"):
                value = state[field_name]
                if (not isinstance(value, torch.Tensor) or value.is_meta
                        or value.layout != torch.strided
                        or value.dtype != param.dtype or value.device != param.device
                        or tuple(value.shape) != tuple(param.shape)):
                    _fail("AdamW momentum shape/dtype/device mismatch: " + name + "/" + field_name)
                if not bool(torch.isfinite(value).all().item()):
                    _fail("AdamW momentum contains non-finite values: " + name + "/" + field_name)
                values[field_name] = fingerprint_tensor(value)["sha256"]
            moments[name] = values
    if not moments or len(moments) != len(canonical_names_by_id):
        _fail("AdamW momentum inventory missing/extra physical Parameters")
    return {
        "moment_parameter_count": len(moments),
        "moments_by_parameter": dict(sorted(moments.items())),
        "moment_inventory_sha256": host._json_sha256(moments),
        "physical_step": physical_step,
    }



class PhysicalStepProbe:
    """Instance-only physical step recorder over REAL prepared torch objects.

    The CPU test path proves hook mechanics, never reference CUDA ownership.
    """
    def __init__(
        self, *, accelerator: Any, composite_optimizer: CompositeOptimizer,
        composite_scheduler: CompositeLRScheduler,
        parameter_names: Mapping[str, torch.nn.Parameter],
        component_by_parameter: Mapping[str, str],
        expected_group_lrs: Mapping[str, list[float]] | None = None,
        expected_group_semantics: Mapping[str, Any] | None = None,
        expected_group_parameter_order: Mapping[str, Any] | None = None,
        lora_network: torch.nn.Module | None = None,
        lora_mmdit_root: torch.nn.Module | None = None,
        expected_lora_target_paths: Mapping[str, str] | None = None,
        expected_lora_forward_semantics: Mapping[str, Any] | None = None,
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
        self.component_by_parameter = dict(component_by_parameter)
        expected_components = {
            "mmdit.attention.adapter",
            "mmdit.mlp.adapter",
            "mmdit.modulation_norm.adapter",
        }
        if (set(self.component_by_parameter) != set(self.params)
                or not self.component_by_parameter
                or not set(self.component_by_parameter.values()).issubset(expected_components)):
            _fail("Probe Component binding is missing or unreviewed")
        if len(self.params) == 526 and set(self.component_by_parameter.values()) != expected_components:
            _fail("Reference Probe requires all three trained Components")
        self.case_id, self.run_id, self.phase_id = case_id, run_id, phase_id
        self.source_commit, self.phase = source_commit, phase
        self.accumulation = spec["gradient_accumulation_steps"]
        self.start_step = start_step
        self.events: list[dict[str, Any]] = []
        self.sync_sequence: tuple[bool, ...] = ()
        self._installed = False
        self._patched: list[tuple[Any, str, bool, Any]] = []
        self._lora_forward_handles: list[Any] = []
        self._lora_projection_handles: list[Any] = []
        # Only Probe-installed read-only projection hooks are allowed while
        # observing actual training Forward. Never hold this in evidence SHA.
        self._lora_projection_allowed_pre_hooks: dict[int, tuple[int, Callable]] = {}
        self._lora_target_allowed_pre_hooks: dict[int, tuple[int, Callable]] = {}
        self._lora_projection_refs: dict[str, dict[str, torch.nn.Linear]] = {}
        self._lora_training_window = False
        self._lora_network = lora_network
        self._lora_mmdit_root = lora_mmdit_root
        self._lora_expected_target_paths: dict[str, str] | None = (
            dict(expected_lora_target_paths)
            if expected_lora_target_paths is not None else None
        )
        self._lora_live_target_paths: dict[str, str] | None = None
        self._lora_expected_sha256: str | None = None
        self._lora_module_digests: dict[str, str] = {}
        self._lora_registered_paths: dict[str, str] = {}
        self._lora_module_refs: dict[str, Any] = {}
        self._lora_target_refs: dict[str, torch.nn.Linear] = {}
        self._lora_target_spec: Mapping[str, Any] | None = None
        self._backwards = 0
        self._started = 0
        self._current_step: int | None = None
        self._current_micro: int | None = None
        self._current_sync: bool | None = None
        self._backward_seen = False
        self._pending_nosync_state: dict[str, str] | None = None
        self._pending_gradient_snapshot: str | None = None
        self._step_children: dict[int, dict[str, int]] = {}
        self._step_scheduler: dict[int, int] = {}
        self._sample_names: dict[int, set[str]] = {}
        self._pre_step_grad: dict[str, Any] | None = None
        self._last_muon_momentum_sha: dict[str, str] = {}
        self._last_adamw_moment_sha: dict[str, str] = {}
        # The accepted *whole logical Step* baseline is written only after
        # both physical Children and the actual Scheduler pass validation.
        self._last_completed_step: int | None = None
        self._last_completed_adapter_sha256: str | None = None
        required_types = tuple(spec["optimizer_types"])
        physical_types = tuple(entry.optimizer_type for entry in composite_optimizer.entries)
        if sorted(physical_types) != sorted(required_types):
            _fail("Probe Child optimizer types do not match fixed R1/R2")
        if len({entry.profile_name for entry in composite_optimizer.entries}) != len(required_types):
            _fail("Duplicate Profile name in physical optimizer")
        self.child_profiles = tuple(entry.profile_name for entry in composite_optimizer.entries)
        # Genuine CUDA reference ownership must supply an independently
        # checked Policy LR contract. Tiny diagnostic CPU Probes instead pin
        # their own immutable initial LR baseline (NOT CUDA qualification).
        if expected_group_lrs is None:
            if len(self.params) == 526 and accelerator.device.type == "cuda":
                _fail("Real CUDA Worker omitted independently audited reference learning rates")
            expected_group_lrs = {
                entry.profile_name: [
                    group.get("lr") for group in entry.optimizer.param_groups
                ] for entry in composite_optimizer.entries
            }
        if set(expected_group_lrs) != set(self.child_profiles):
            _fail("Reference LR Probe missing/extra physical Child Profiles")
        self.expected_group_lrs = {
            profile: tuple(
                float(lr) for lr in expected_group_lrs[profile]
                if _reference_lr_hex(lr, label=profile + "/prepared")
            )
            for profile in self.child_profiles
        }
        self._check_physical_group_lrs(phase="probe_init")
        refs = _reference_provider_semantics(case_id)
        real_reference_cuda = len(self.params) == 526 and accelerator.device.type == "cuda"
        diagnostic_semantics = expected_group_semantics is None
        if diagnostic_semantics:
            if real_reference_cuda:
                _fail("Real CUDA Worker omitted independently verified Group semantics")
            # Diagnostic CPU LR can exceed the reference; every other field
            # is still checked independently against the pinned provider.
            expected_group_semantics = {
                entry.profile_name: _actual_provider_semantics(entry, refs[entry.profile_name])
                for entry in composite_optimizer.entries
            }
        if not isinstance(expected_group_semantics, Mapping) or (
            set(expected_group_semantics) != set(self.child_profiles)
        ):
            _fail("Missing expected Group semantic Profiles")
        self._expected_group_semantics_sha256: dict[str, str] = {}
        for entry in composite_optimizer.entries:
            name = entry.profile_name
            ref = refs[name]
            supplied = expected_group_semantics[name]
            if not isinstance(supplied, Mapping) or (
                set(supplied) - {"semantic_sha256"}
                != {"optimizer_type", "groups", "instance_fields"}
            ):
                _fail("Malformed Prepared Optimizer Group semantic snapshot: " + name)
            frozen_view = {
                key: supplied[key]
                for key in ("optimizer_type", "groups", "instance_fields")
            }
            frozen_sha = fingerprint_state_tree(frozen_view)
            recorded_sha = supplied.get("semantic_sha256")
            if diagnostic_semantics:
                if recorded_sha is not None:
                    _fail("Unexpected diagnostic Group semantic signature")
            elif type(recorded_sha) is not str or recorded_sha != frozen_sha:
                _fail("Prepared Optimizer Group semantic digest mismatch: " + name)
            actual = _actual_provider_semantics(entry, ref)
            for group in actual["groups"]:
                # Genuine CUDA R1/R2 requires the exact independent Policy and
                # Provider defaults; diagnostic CPU permits a higher pinned LR.
                keys = ref["fields"] if real_reference_cuda else ref["fields"] - {"lr"}
                for key in keys:
                    if _typed_tree(group[key]) != _typed_tree(ref["group"][key]):
                        _fail("Physical Group differs from pinned Provider reference: " + name)
            if fingerprint_state_tree(actual) != frozen_sha:
                _fail("Physical Group changed since Prepared semantic audit: " + name)
            # Only SHA256 strings survive initialization. Neither nested
            # Group lists nor caller-owned Prepared dictionaries are retained.
            self._expected_group_semantics_sha256[name] = frozen_sha
        self._check_physical_group_semantics(phase="probe_init")
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
        self.profile_by_name = {}
        for name, param in self.params.items():
            profiles = [
                profile for profile, owned in self.child_params.items()
                if any(param is member for member in owned)
            ]
            if len(profiles) != 1:
                _fail("Parameter lacks exactly one physical Child Profile")
            self.profile_by_name[name] = profiles[0]
        # F9-B: every real CUDA Group must be bound to the independent
        # RuntimeSpec-ordered identity captured at Prepared, never to the
        # possibly mutated live Group read at Probe initialization.
        real_order_cuda = len(self.params) == 526 and accelerator.device.type == "cuda"
        if expected_group_parameter_order is None:
            if real_order_cuda:
                _fail("Real CUDA omitted independently audited RuntimeSpec Group order")
            # Tiny CPU-only diagnostics have no ParameterPolicy RuntimeSpec.
            # Their Probe-initial Group order is a diagnostic baseline ONLY.
            expected_group_parameter_order = {
                entry.profile_name: tuple(
                    tuple(id(parameter) for parameter in group["params"])
                    for group in entry.optimizer.param_groups
                )
                for entry in composite_optimizer.entries
            }
        if (not isinstance(expected_group_parameter_order, Mapping)
                or set(expected_group_parameter_order) != set(self.child_profiles)):
            _fail("Prepared RuntimeSpec Group order has missing/extra Profiles")
        frozen_group_order: dict[str, tuple[tuple[int, ...], ...]] = {}
        all_ordered_ids: list[int] = []
        for entry in composite_optimizer.entries:
            source_groups = expected_group_parameter_order[entry.profile_name]
            if (not isinstance(source_groups, (list, tuple))
                    or len(source_groups) != len(entry.optimizer.param_groups)):
                _fail("Prepared RuntimeSpec Group order has incorrect Group count")
            validated_groups: list[tuple[int, ...]] = []
            for source_group in source_groups:
                if not isinstance(source_group, (list, tuple)) or not source_group:
                    _fail("Prepared RuntimeSpec Group order has empty/malformed Group")
                if any(type(parameter_id) is not int or parameter_id not in named_ids
                       for parameter_id in source_group):
                    _fail("Prepared RuntimeSpec Group order contains unknown Parameter ID")
                group_ids = tuple(source_group)
                all_ordered_ids.extend(group_ids)
                validated_groups.append(group_ids)
            frozen_group_order[entry.profile_name] = tuple(validated_groups)
        if (len(all_ordered_ids) != len(set(all_ordered_ids))
                or set(all_ordered_ids) != set(named_ids)):
            _fail("Prepared RuntimeSpec Group order has duplicated/missing ownership")
        # Only independent integer tuples remain: no alias to optimizer's
        # mutable Group dictionaries, Lists or Parameter objects is retained.
        self._expected_group_parameter_order = frozen_group_order
        self._check_physical_group_order(phase="probe_init")
        # F9-A: the CPU-only legacy Probe remains compatible; real CUDA MUST
        # bind the exact independently audited 263-Module Prepared reference.
        real_forward_cuda = len(self.params) == 526 and accelerator.device.type == "cuda"
        if self._lora_network is None:
            if (real_forward_cuda or expected_lora_forward_semantics is not None
                    or self._lora_mmdit_root is not None
                    or self._lora_expected_target_paths is not None):
                _fail("Real CUDA LoRA Forward reference was not bound to its actual Network")
        else:
            if real_forward_cuda and expected_lora_forward_semantics is None:
                _fail("Real CUDA omitted independently audited LoRA Forward semantics")
            if (real_forward_cuda and (
                self._lora_mmdit_root is None
                or self._lora_expected_target_paths is not None
            )):
                _fail("Real CUDA must bind actual MMDiT root with fixed canonical paths")
            if expected_lora_forward_semantics is not None:
                if (not isinstance(expected_lora_forward_semantics, Mapping)
                        or set(expected_lora_forward_semantics) != {
                            "module_names", "semantic_sha256",
                        }):
                    _fail("Prepared LoRA Forward semantic reference is malformed")
                names = expected_lora_forward_semantics["module_names"]
                if (not isinstance(names, (list, tuple))
                        or len(names) != len(set(names))
                        or any(type(item) is not str or not item for item in names)
                        or type(expected_lora_forward_semantics["semantic_sha256"]) is not str):
                    _fail("Prepared LoRA Forward names or SHA256 are malformed")
                expected_names = tuple(names)
            else:
                expected_names = None
            self._lora_target_spec = (
                host.build_sd35_medium_lora_target_spec()
                if real_forward_cuda else None
            )
            self._lora_live_target_paths = (
                _reference_sd3_mmdit_target_paths(self._lora_target_spec)
                if self._lora_target_spec is not None
                else (dict(self._lora_expected_target_paths)
                      if self._lora_expected_target_paths is not None else None)
            )
            payload, modules, targets = inspect_reference_lora_forward_semantics(
                network=self._lora_network,
                expected_names=expected_names,
                target_spec=self._lora_target_spec,
                lora_mmdit_root=self._lora_mmdit_root,
                expected_lora_target_paths=self._lora_expected_target_paths,
            )
            sha = fingerprint_state_tree(payload)
            if (expected_lora_forward_semantics is not None
                    and sha != expected_lora_forward_semantics["semantic_sha256"]):
                _fail("Physical LoRA Forward differs from independent Prepared reference")
            self._lora_expected_sha256 = sha
            self._lora_module_digests = {
                name: fingerprint_state_tree(fields)
                for name, fields in payload["modules"].items()
            }
            self._lora_registered_paths = {
                name: fields["registered_path"]
                for name, fields in payload["modules"].items()
            }
            self._lora_module_refs = dict(modules)
            self._lora_target_refs = dict(targets)
            self._lora_projection_refs = {
                name: {
                    "lora_down": module.lora_down,
                    "lora_up": module.lora_up,
                }
                for name, module in modules.items()
            }
            self._check_lora_forward_semantics(phase="probe_init")

    def _check_physical_group_order(self, *, phase: str) -> None:
        """Check full ordered identity against immutable Prepared RuntimeSpec."""
        for entry in self.optimizer.entries:
            actual_groups = entry.optimizer.param_groups
            if any(not isinstance(group.get("params"), list) for group in actual_groups):
                _fail("Physical Parameter Group params container changed at " + phase)
            actual_order = tuple(
                tuple(id(parameter) for parameter in group["params"])
                for group in actual_groups
            )
            expected_order = self._expected_group_parameter_order.get(entry.profile_name)
            if actual_order != expected_order:
                _fail("Physical Parameter Group/order drift from Prepared RuntimeSpec at "
                      + phase + ": " + entry.profile_name)

    def _check_lora_forward_semantics(self, *, phase: str) -> None:
        if self._lora_network is None:
            return
        payload, modules, targets = inspect_reference_lora_forward_semantics(
            network=self._lora_network,
            expected_names=tuple(sorted(self._lora_module_refs)),
            target_spec=self._lora_target_spec,
            allowed_projection_pre_hooks=self._lora_projection_allowed_pre_hooks,
            allowed_target_pre_hooks=self._lora_target_allowed_pre_hooks,
            lora_mmdit_root=self._lora_mmdit_root,
            expected_lora_target_paths=self._lora_expected_target_paths,
        )
        for name in self._lora_module_refs:
            if (modules[name] is not self._lora_module_refs[name]
                    or targets[name] is not self._lora_target_refs[name]):
                _fail("LoRA Forward module/Linear identity drift at " + phase + ": " + name)
            for role, projection in self._lora_projection_refs[name].items():
                if getattr(modules[name], role) is not projection:
                    _fail("LoRA projection identity drift at " + phase + ": " + name + "/" + role)
        if fingerprint_state_tree(payload) != self._lora_expected_sha256:
            _fail("LoRA Forward semantic drift at " + phase)

    def _check_single_lora_forward(self, name: str, target: Any) -> None:
        if self._lora_network is None:
            return
        if target is not self._lora_target_refs[name]:
            _fail("LoRA Forward actual Linear target identity drift: " + name)
        if (type(self._lora_network.multiplier) is not float
                or self._lora_network.multiplier != 1.0):
            _fail("LoRA Forward root multiplier drift at actual physical Forward")
        fields, actual_target = _lora_forward_module_payload(
            name, self._lora_module_refs[name],
            allowed_projection_pre_hooks=self._lora_projection_allowed_pre_hooks,
            allowed_target_pre_hooks=self._lora_target_allowed_pre_hooks,
        )
        for role, projection in self._lora_projection_refs[name].items():
            if getattr(self._lora_module_refs[name], role) is not projection:
                _fail("LoRA projection module identity drift at real Forward: "
                      + name + "/" + role)
        if actual_target is not target:
            _fail("LoRA Forward original Linear target was replaced: " + name)
        if self._lora_live_target_paths is not None:
            fields.update(_audit_real_lora_target_path(
                name, self._lora_module_refs[name], target,
                mmdit_root=self._lora_mmdit_root,
                expected_path=self._lora_live_target_paths[name],
            ))
        # Immutable registration path is part of the Prepared semantic digest;
        # this single-Linear pre-hook checks real call-time scalars and Alpha.
        # Avoid scanning every one of the 263 registered modules per Linear
        # invocation; the constant-time named path is frozen at Probe init.
        registered_path = self._lora_registered_paths[name]
        try:
            registered_module = self._lora_network.get_submodule(registered_path)
        except AttributeError:
            _fail("LoRA Forward registered module path disappeared: " + name)
        if registered_module is not self._lora_module_refs[name]:
            _fail("LoRA Forward registered module identity was replaced: " + name)
        fields["registered_path"] = registered_path
        if fingerprint_state_tree(fields) != self._lora_module_digests[name]:
            _fail("LoRA Forward semantic drift at real Linear Forward: " + name)

    def _check_single_lora_projection(
        self, *, name: str, role: str, projection: torch.nn.Module,
    ) -> None:
        reviewed = self._lora_projection_refs[name][role]
        if (projection is not reviewed
                or getattr(self._lora_module_refs[name], role) is not reviewed):
            _fail("LoRA projection identity drift during actual Forward: " + name + "/" + role)
        _audit_lora_projection_execution(
            name, role, projection,
            allowed_pre_hooks=self._lora_projection_allowed_pre_hooks,
        )
        _audit_lora_target_execution(
            name, self._lora_target_refs[name],
            allowed_pre_hooks=self._lora_target_allowed_pre_hooks,
        )
        # The outer Linear pre-hook ran before later user hooks. Revalidate
        # provenance at the ACTUAL child projection invocation so metadata
        # or real-MMDiT path relinking cannot reach the true Backward.
        if self._lora_live_target_paths is not None:
            _audit_real_lora_target_path(
                name, self._lora_module_refs[name],
                self._lora_target_refs[name],
                mmdit_root=self._lora_mmdit_root,
                expected_path=self._lora_live_target_paths[name],
            )

    def _check_physical_group_lrs(self, *, phase: str) -> dict[str, list[float]]:
        actual: dict[str, list[float]] = {}
        for entry in self.optimizer.entries:
            rates = entry.optimizer.param_groups
            expected = self.expected_group_lrs.get(entry.profile_name, ())
            if len(rates) != len(expected):
                _fail("Physical learning-rate Group count drift at " + phase)
            normalized = []
            for i, (group, lr) in enumerate(zip(rates, expected)):
                current = group.get("lr")
                if _reference_lr_hex(current, label=entry.profile_name + "/" + phase) != lr.hex():
                    _fail("Physical Child learning rate drifted after preparation at " + phase)
                normalized.append(current)
            actual[entry.profile_name] = normalized
        return actual

    def _check_physical_group_semantics(self, *, phase: str) -> dict[str, str]:
        refs = _reference_provider_semantics(self.case_id)
        digests = {}
        for entry in self.optimizer.entries:
            name = entry.profile_name
            actual = _actual_provider_semantics(entry, refs[name])
            actual_sha = fingerprint_state_tree(actual)
            if actual_sha != self._expected_group_semantics_sha256[name]:
                _fail("Physical Optimizer Group semantic drift at " + phase + ": " + name)
            digests[name] = actual_sha
        return digests

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

    def _adapter_parameter_sha256(self) -> str:
        """One-Tensor-at-a-time hash of all canonical real BF16 LoRA weights."""
        return fingerprint_state_tree({
            name: parameter.detach()
            for name, parameter in sorted(self.params.items())
        })

    def _snapshot(self) -> dict[str, str]:
        return {
            "network": fingerprint_state_tree({name: p.detach() for name,p in sorted(self.params.items())}),
            "children": fingerprint_state_tree({
                entry.profile_name: entry.optimizer.state_dict()
                for entry in self.optimizer.entries
            }),
            "scheduler": fingerprint_state_tree(self.scheduler.state_dict()),
        }

    def _gradient_snapshot(self) -> str:
        """Physical gradient-state bytes, including None vs BF16 Tensor."""
        return fingerprint_state_tree({
            name: (None if param.grad is None else param.grad.detach())
            for name, param in sorted(self.params.items())
        })

    def _pre_backward_gradient_health(self, *, first_microbatch: bool) -> None:
        """Read-only baseline: stale gradients may not be called new Backward."""
        for name, param in sorted(self.params.items()):
            grad = param.grad
            if grad is None:
                continue
            if (grad.layout != torch.strided or grad.is_meta
                    or grad.dtype != torch.bfloat16
                    or grad.device != param.device
                    or tuple(grad.shape) != tuple(param.shape)):
                _fail("Unexpected non-BF16 or wrong-shape pre-Backward gradient: " + name)
            if not bool(torch.isfinite(grad).all().item()):
                _fail("Non-finite pre-Backward gradient: " + name)
            if first_microbatch and bool(torch.count_nonzero(grad).item()):
                _fail("First Microbatch has stale nonzero pre-existing BF16 gradient: " + name)

    def _gradient_value_snapshot(self) -> str:
        """Numeric BF16 state digest (signed zeros normalized, None distinct).

        CPU bytes are processed one Tensor at a time. No GPU clones, gradient
        mutations, FP32 conversion, nor retained full-model CPU snapshots.
        A None->all-zero transition cannot prove a contribution.
        """
        values: dict[str, Any] = {}
        for name, param in sorted(self.params.items()):
            grad = param.grad
            if grad is None:
                values[name] = None
                continue
            # _pre_backward_gradient_health and _gradients validate metadata/
            # finite values at the corresponding pre/post call boundaries.
            bits = grad.detach().contiguous().to(device="cpu").view(torch.int16)
            canonical_bits = bits.numpy().copy()
            canonical_bits[canonical_bits == -32768] = 0  # -0 BF16 == +0 BF16
            values[name] = {
                "shape": list(grad.shape), "dtype": str(grad.dtype),
                "numeric_sha256": hashlib.sha256(canonical_bits.tobytes()).hexdigest(),
            }
        return host._json_sha256(values)

    def _gradients(self, *, require_nonzero: bool) -> dict[str, Any]:
        """Audit every physical BF16 gradient once, grouped by Child AND Component."""
        child_stats: dict[str, dict[str, Any]] = {
            key: {"tensor_count": 0, "nonzero_tensor_count": 0,
                  "nonzero_element_count": 0, "hasher": hashlib.sha256()}
            for key in self.child_profiles
        }
        component_stats: dict[str, dict[str, Any]] = {
            key: {"tensor_count": 0, "nonzero_tensor_count": 0,
                  "nonzero_element_count": 0, "hasher": hashlib.sha256()}
            for key in sorted(set(self.component_by_parameter.values()))
        }
        for name, parameter in sorted(self.params.items()):
            gradient = parameter.grad
            if gradient is None or gradient.layout != torch.strided or gradient.is_meta:
                _fail("Missing or unsupported real Backward Gradient")
            if gradient.dtype != torch.bfloat16 or gradient.device != parameter.device:
                _fail("Non-BF16 or wrong-device physical gradient")
            if not bool(torch.isfinite(gradient).all().item()):
                _fail("Non-finite physical BF16 Gradient")
            desc = fingerprint_tensor(gradient)
            payload = host._canonical_json_bytes({
                "canonical_name": name,
                "dtype": desc["dtype"], "shape": desc["shape"],
                "sha256": desc["sha256"],
            }) + b"\n"
            n_nonzero = int(torch.count_nonzero(gradient).item())
            for stat in (
                child_stats[self.profile_by_name[name]],
                component_stats[self.component_by_parameter[name]],
            ):
                stat["tensor_count"] += 1
                stat["nonzero_tensor_count"] += int(n_nonzero > 0)
                stat["nonzero_element_count"] += n_nonzero
                stat["hasher"].update(payload)

        def finalize(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            result = {}
            for key, stat in stats.items():
                if require_nonzero and stat["nonzero_tensor_count"] == 0:
                    _fail("Physical Optimizer Child or Component has no nonzero BF16 Gradient: " + key)
                result[key] = {
                    "tensor_count": stat["tensor_count"],
                    "nonzero_tensor_count": stat["nonzero_tensor_count"],
                    "nonzero_element_count": stat["nonzero_element_count"],
                    "gradient_sha256": stat["hasher"].hexdigest(),
                }
            return result

        return {
            "child_gradients": finalize(child_stats),
            "component_gradients": finalize(component_stats),
            "gradient_phase": (
                "post_clip_pre_physical_step" if require_nonzero
                else "post_backward_pre_clip"
            ),
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
        self._check_physical_group_order(phase="pre_install")
        original_backward = self.accelerator.backward
        original_optimizer_step = self.optimizer.step
        original_scheduler_step = self.scheduler.step
        self._installed = True

        def backward_wrapper(*args, **kwargs):
            self._lora_training_window = False
            if self._current_step is None or self._backward_seen:
                _fail("Backward without unique active real Batch")
            self._check_physical_group_order(phase="pre_backward")
            first = self._current_micro == 1
            self._pre_backward_gradient_health(first_microbatch=first)
            before_gradient = self._gradient_snapshot()
            before_numeric = self._gradient_value_snapshot()
            if not first:
                if (self._pending_gradient_snapshot is None
                        or before_gradient != self._pending_gradient_snapshot):
                    _fail("Retained first-Microbatch BF16 gradient changed before actual Backward")
            elif self._pending_gradient_snapshot is not None:
                _fail("First Microbatch unexpectedly inherits retained gradients")
            # The actual production Accelerator.backward is called once, only
            # after all pre-Backward provenance assertions have passed.
            result = original_backward(*args, **kwargs)
            self._check_physical_group_order(phase="post_backward")
            summary = self._gradients(require_nonzero=False)
            if not any(
                info["nonzero_tensor_count"]
                for info in summary["component_gradients"].values()
            ):
                _fail("Microbatch produced no nonzero real BF16 gradient contribution")
            after_gradient = self._gradient_snapshot()
            after_numeric = self._gradient_value_snapshot()
            if (before_gradient == after_gradient
                    or before_numeric == after_numeric):
                _fail("Actual Backward produced no observable numeric BF16 gradient contribution")
            if first and self.accumulation > 1:
                self._pending_gradient_snapshot = after_gradient
            elif not first:
                self._pending_gradient_snapshot = None
            summary["pre_backward_gradient_sha256"] = before_gradient
            summary["post_backward_gradient_sha256"] = after_gradient
            summary["pre_backward_numeric_sha256"] = before_numeric
            summary["post_backward_numeric_sha256"] = after_numeric
            self._record("backward_completed", sample=self._sample,
                         sync=self._current_sync, details=summary)
            self._backward_seen = True
            self._backwards += 1
            return result

        def optimizer_wrapper(*args, **kwargs):
            self._check_physical_group_order(phase="pre_composite_step")
            self._check_lora_forward_semantics(phase="pre_composite_step")
            if not self._backward_seen or self._current_sync is not True:
                _fail("Physical CompositeOptimizer.step before synchronized Backward")
            step = self._current_step
            if sum(self._step_children[step].values()):
                _fail("Duplicate CompositeOptimizer physical Step")
            self._pre_step_grad = self._gradients(require_nonzero=True)
            result = original_optimizer_step(*args, **kwargs)
            self._check_physical_group_order(phase="post_composite_step")
            if set(k for k,v in self._step_children[step].items() if v) != set(self.child_profiles):
                _fail("CompositeOptimizer did not physically execute every Child once")
            return result

        def scheduler_wrapper(*args, **kwargs):
            self._check_physical_group_order(phase="pre_scheduler")
            self._check_lora_forward_semantics(phase="pre_scheduler")
            if not self._backward_seen or self._current_sync is not True:
                _fail("Scheduler advanced without synchronized Backward")
            step = self._current_step
            if self._step_scheduler[step] != 0:
                _fail("Duplicate physical Scheduler step")
            if set(k for k,v in self._step_children[step].items() if v) != set(self.child_profiles):
                _fail("Scheduler step preceded physical Child completion")
            self._check_physical_group_lrs(phase="pre_scheduler")
            self._check_physical_group_semantics(phase="pre_scheduler")
            before = fingerprint_state_tree(self.scheduler.state_dict())
            result = original_scheduler_step(*args, **kwargs)
            self._check_physical_group_order(phase="post_constant_scheduler")
            self._check_lora_forward_semantics(phase="post_constant_scheduler")
            self._check_physical_group_lrs(phase="post_constant_scheduler")
            self._check_physical_group_semantics(phase="post_constant_scheduler")
            after = fingerprint_state_tree(self.scheduler.state_dict())
            if after == before:
                _fail("Real CompositeLRScheduler step did not change state")
            # Capture the accepted full-Adapter fingerprint after both
            # physical Child steps and the constant Scheduler have succeeded.
            accepted_adapter_sha = self._adapter_parameter_sha256()
            self._step_scheduler[step] += 1
            self._record("physical_scheduler_stepped", details={
                "scheduler_before": before, "scheduler_after": after,
            })
            self._last_completed_step = step
            self._last_completed_adapter_sha256 = accepted_adapter_sha
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
                    child = next(e for e in self.optimizer.entries if e.profile_name == name)
                    self._check_physical_group_order(phase="pre_child_step")
                    self._check_lora_forward_semantics(phase="pre_child_step")
                    before_lrs = self._check_physical_group_lrs(phase="pre_child_step")
                    before_semantics = self._check_physical_group_semantics(phase="pre_child_step")
                    # One contiguous BF16 Tensor at a time: hashes are from
                    # original bytes, not cloned GPU parameters or FP32 casts.
                    members = {
                        canonical: param for canonical, param in sorted(self.params.items())
                        if self.profile_by_name[canonical] == name
                    }
                    before_devices = {
                        canonical: param.device for canonical, param in members.items()
                    }
                    before_shapes = {
                        canonical: tuple(param.shape) for canonical, param in members.items()
                    }
                    before_tensors = {
                        canonical: fingerprint_tensor(param)
                        for canonical, param in members.items()
                    }
                    before_params = host._json_sha256(before_tensors)
                    before_state = fingerprint_state_tree(child.optimizer.state_dict())
                    before_group_ids = tuple(
                        tuple(id(p) for p in group["params"])
                        for group in child.optimizer.param_groups
                    )
                    canonical_child_names = {
                        id(param): canonical for canonical, param in members.items()
                    }
                    muon_before = None
                    if child.optimizer_type == "Muon" and step > 1:
                        muon_before = validate_muon_parameter_states(
                            child.optimizer, physical_step=step - 1,
                            canonical_names_by_id=canonical_child_names,
                        )
                        previous_digest = self._last_muon_momentum_sha.get(name)
                        if self.phase == "fresh" and previous_digest is None:
                            _fail("Missing prior Fresh Muon momentum baseline")
                        # Resume begins in another process, without Fresh memory;
                        # only C4 may prove cross-process checkpoint continuity.
                        if (previous_digest is not None and
                                muon_before["momentum_inventory_sha256"] != previous_digest):
                            _fail("Muon momentum_buffer changed between physical Steps")
                    adamw_before = None
                    if child.optimizer_type == "AdamW" and step > 1:
                        adamw_before = validate_adamw_parameter_states(
                            child.optimizer, physical_step=step - 1,
                            canonical_names_by_id=canonical_child_names,
                        )
                        if self.phase == "fresh":
                            previous = self._last_adamw_moment_sha.get(name)
                            if previous is None:
                                _fail("Missing prior AdamW moment baseline")
                            if adamw_before["moment_inventory_sha256"] != previous:
                                _fail("AdamW moments changed between physical Steps")
                    result = original(*args, **kwargs)
                    self._check_physical_group_order(phase="post_child_step")
                    self._check_lora_forward_semantics(phase="post_child_step")
                    self._check_physical_group_lrs(phase="post_child_step")
                    self._check_physical_group_semantics(phase="post_child_step")
                    after_group_ids = tuple(
                        tuple(id(p) for p in group["params"])
                        for group in child.optimizer.param_groups
                    )
                    if after_group_ids != before_group_ids:
                        _fail("Physical Child Step changed Parameter ownership/order")
                    after_tensors = {}
                    changed_components = {
                        component: 0 for component in sorted(
                            {self.component_by_parameter[key] for key in members}
                        )
                    }
                    changed_parameters = 0
                    for canonical, param in members.items():
                        if (param.dtype != torch.bfloat16
                                or param.device != before_devices[canonical]
                                or tuple(param.shape) != before_shapes[canonical]):
                            _fail("Physical Child changed BF16 Parameter dtype/device/shape")
                        if not bool(torch.isfinite(param).all().item()):
                            _fail("Physical Child produced a non-finite BF16 weight")
                        result_fp = fingerprint_tensor(param)
                        after_tensors[canonical] = result_fp
                        if result_fp["sha256"] != before_tensors[canonical]["sha256"]:
                            changed_parameters += 1
                            changed_components[self.component_by_parameter[canonical]] += 1
                    after_params = host._json_sha256(after_tensors)
                    if not changed_parameters or before_params == after_params:
                        _fail("Physical Child advanced state but changed no BF16 LoRA weight")
                    after_state = fingerprint_state_tree(child.optimizer.state_dict())
                    if before_state == after_state:
                        _fail("Physical Child did not advance actual optimizer state")
                    muon_after = None
                    adamw_after = None
                    if child.optimizer_type == "Muon":
                        muon_after = validate_muon_parameter_states(
                            child.optimizer, physical_step=step,
                            canonical_names_by_id=canonical_child_names,
                        )
                        if (muon_before is not None and
                                muon_before["momentum_inventory_sha256"]
                                == muon_after["momentum_inventory_sha256"]):
                            _fail("Muon physical Step did not advance any BF16 momentum_buffer")
                        self._last_muon_momentum_sha[name] = muon_after["momentum_inventory_sha256"]
                    elif child.optimizer_type == "AdamW":
                        adamw_after = validate_adamw_parameter_states(
                            child.optimizer, physical_step=step,
                            canonical_names_by_id=canonical_child_names,
                        )
                        if (adamw_before is not None and
                                adamw_before["moment_inventory_sha256"] ==
                                adamw_after["moment_inventory_sha256"]):
                            # No independent AdamW momentum advancement despite
                            # a claimed real physical Step. Weight changes are
                            # verified independently above.
                            _fail("AdamW moments did not advance in real physical Step")
                        self._last_adamw_moment_sha[name] = adamw_after["moment_inventory_sha256"]
                    else:
                        _fail("Unreviewed physical Optimizer Child type")
                    self._step_children[step][name] += 1
                    self._record("physical_child_stepped", profile=name, details={
                        "optimizer_state_before": before_state,
                        "optimizer_state_after": after_state,
                        "parameter_before": before_params,
                        "parameter_after": after_params,
                        "changed_parameter_count": changed_parameters,
                        "changed_component_parameter_counts": changed_components,
                        "muon_momentum_state": muon_after,
                        "adamw_momentum_state": adamw_after,
                        "post_clip_gradient": self._pre_step_grad,
                        "physical_step": step,
                        "physical_group_learning_rates": before_lrs[name],
                        "physical_group_semantics_sha256": before_semantics[name],
                    })
                    return result
                return wrapped
            self._patch(entry.optimizer, "step", make_child_wrapper(original_child, profile))
        if self._lora_network is not None:
            try:
                for lora_name, target in sorted(self._lora_target_refs.items()):
                    def pre_forward(module, arguments, name=lora_name):
                        if self._lora_training_window:
                            self._check_single_lora_forward(name, module)
                    handle = target.register_forward_pre_hook(pre_forward)
                    self._lora_forward_handles.append(handle)
                    self._lora_target_allowed_pre_hooks[id(target)] = (
                        handle.id, pre_forward,
                    )
                # Checking the outer target alone cannot catch mutations made
                # after its pre-hook but before LoRAModule calls lora_down/up.
                # These real projection hooks are read-only; they never replace
                # inputs, outputs, gradients, or the provider Forward.
                for name in sorted(self._lora_projection_refs):
                    for role, projection in self._lora_projection_refs[name].items():
                        def projection_pre_forward(module, arguments, n=name, r=role):
                            if self._lora_training_window:
                                self._check_single_lora_projection(
                                    name=n, role=r, projection=module,
                                )
                        handle = projection.register_forward_pre_hook(projection_pre_forward)
                        self._lora_projection_handles.append(handle)
                        self._lora_projection_allowed_pre_hooks[id(projection)] = (
                            handle.id, projection_pre_forward,
                        )
            except Exception:
                self.close()
                raise

    def begin_batch(self, sample_marker: str) -> None:
        if not self._installed:
            _fail("Cannot begin a Batch before Probe installation")
        self._check_physical_group_order(phase="pre_training_forward")
        if self._pending_nosync_state is not None:
            if self._snapshot() != self._pending_nosync_state:
                _fail("premature/non-sync accumulation changed Adapter, Optimizer or Scheduler")
            if (self._pending_gradient_snapshot is None
                    or self._gradient_snapshot() != self._pending_gradient_snapshot):
                _fail("First Microbatch BF16 gradient was not retained through accumulation")
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
        # Exact static Forward semantics are enforced before each new physical
        # training Forward, including resumed and non-sync accumulation batches.
        self._check_lora_forward_semantics(phase="pre_training_forward")
        # A new process's first Step (Resume logical Step 2) is exempt:
        # cross-process Checkpoint continuity is exclusively a C4 obligation.
        # For an in-process Fresh successor, reject *before* new Backward.
        if current_micro == 1 and current_step > self.start_step + 1:
            if self._last_completed_step != current_step - 1:
                _fail("Missing prior completed Adapter physical Step")
            if self._last_completed_adapter_sha256 is None:
                _fail("Missing prior Adapter parameter baseline")
            if self._adapter_parameter_sha256() != self._last_completed_adapter_sha256:
                _fail("Adapter parameters changed between physical Steps")
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
        self._lora_training_window = self._lora_network is not None

    def _assert_step(self, step: int) -> None:
        counts = self._step_children.get(step, {})
        if set(counts) != set(self.child_profiles) or any(v != 1 for v in counts.values()):
            _fail("Physical Step missing/duplicate Muon or AdamW Child")
        if self._step_scheduler.get(step) != 1:
            _fail("Physical Step missing/duplicate real Scheduler advance")

    def assert_step_complete(self) -> None:
        self._check_physical_group_order(phase="completed_physical_step")
        self._check_lora_forward_semantics(phase="completed_physical_step")
        if (not self._installed or self._pending_nosync_state is not None
                or self._pending_gradient_snapshot is not None):
            _fail("Incomplete accumulation, retained gradient or detached physical Probe")
        if self._backwards != self._started or self._backwards % self.accumulation:
            _fail("Incomplete real Backward/Microbatch pair sequence")
        if not self._step_children:
            _fail("No physical optimization step was observed")
        for step in sorted(self._step_children):
            self._assert_step(step)

    def close(self) -> None:
        self._lora_training_window = False
        for handle in reversed(self._lora_projection_handles):
            handle.remove()
        self._lora_projection_handles.clear()
        self._lora_projection_allowed_pre_hooks.clear()
        for handle in reversed(self._lora_forward_handles):
            handle.remove()
        self._lora_forward_handles.clear()
        self._lora_target_allowed_pre_hooks.clear()
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
    "audit_physical_optimizer_groups",
    "validate_reference_ownership_rows", "validate_reference_alpha_buffers",
    "inspect_reference_lora_forward_semantics",
    "validate_reference_group_learning_rates",
    "validate_reference_optimizer_group_semantics",
    "validate_muon_parameter_states", "validate_adamw_parameter_states",
    "PhysicalStepProbe", "ObservedSd3NetworkTrainer",
]
