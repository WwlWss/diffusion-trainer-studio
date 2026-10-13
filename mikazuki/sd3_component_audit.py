"""Read-only, torch-free, fine-grained SD3/SD3.5 LoRA target audit.

The existing seven-component `sd3-lora` Parameter Policy is intentionally NOT
changed by this module. Eleven groups below are *diagnostic*, not a new policy
schema, routing table, optimizer owner, or permission to enable full-BF16.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from mikazuki.sd3_architecture_contract import SD3Architecture, SD3ArchitectureError


class SD3ComponentAuditError(ValueError):
    """Invalid, inconsistent, or unrecognized SD3 LoRA target metadata."""


SD3_DIAGNOSTIC_COMPONENTS = (
    "mmdit.x_attention.adapter",
    "mmdit.context_attention.adapter",
    "mmdit.x_extra_attention.adapter",
    "mmdit.x_mlp.adapter",
    "mmdit.context_mlp.adapter",
    "mmdit.x_modulation.adapter",
    "mmdit.context_modulation.adapter",
    "mmdit.input_output.adapter",
    "clip_l.adapter",
    "clip_g.adapter",
    "t5xxl.adapter",
)

_JOINT_PATH = re.compile(
    r"^joint_blocks\.(\d+)\.(x_block|context_block)\."
    r"(attn2|attn|mlp|adaLN_modulation)\.(.+)$"
)
_NONBLOCK_PATH = re.compile(
    r"^(?:context_embedder|x_embedder\.proj|"
    r"t_embedder\.mlp\.[02]|y_embedder\.mlp\.[02]|"
    r"final_layer\.(?:linear|adaLN_modulation\.1))$"
)
_CLIP_TARGET = re.compile(
    r"^text_model\.encoder\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))$"
)
_T5_TARGET = re.compile(
    r"^encoder\.block\.(\d+)\.layer\."
    r"(?:0\.SelfAttention\.(?:q|k|v|o)|1\.DenseReluDense\.(?:wi_0|wi_1|wo))$"
)
_TEXT_LIMITS = {"clip_l": 12, "clip_g": 32, "t5xxl": 24}
_TEXT_CANONICAL_ROOTS = {
    "clip_l": "clip_l", "text_encoder_1": "clip_l",
    "clip_g": "clip_g", "text_encoder_2": "clip_g",
    "t5xxl": "t5xxl", "text_encoder_3": "t5xxl",
}

_COMPONENT_BY_PATH = {
    ("x_block", "attn"): "mmdit.x_attention.adapter",
    ("context_block", "attn"): "mmdit.context_attention.adapter",
    ("x_block", "attn2"): "mmdit.x_extra_attention.adapter",
    ("x_block", "mlp"): "mmdit.x_mlp.adapter",
    ("context_block", "mlp"): "mmdit.context_mlp.adapter",
    ("x_block", "adaLN_modulation"): "mmdit.x_modulation.adapter",
    ("context_block", "adaLN_modulation"): "mmdit.context_modulation.adapter",
}
_TEXT_ROOTS = {
    "clip_l": "clip_l.adapter",
    "text_encoder_1": "clip_l.adapter",
    "clip_g": "clip_g.adapter",
    "text_encoder_2": "clip_g.adapter",
    "t5xxl": "t5xxl.adapter",
    "text_encoder_3": "t5xxl.adapter",
}


@dataclass(frozen=True)
class SD3AdapterTensor:
    target_root: str
    target_path: str
    target_type: str
    parameter_shape: tuple[int, ...]
    tensor_name: str | None = None


@dataclass(frozen=True)
class SD3TargetGroup:
    component_id: str
    block_index: int | None
    block_kind: str | None


def _clean_tensor(value: Any) -> SD3AdapterTensor:
    if isinstance(value, SD3AdapterTensor):
        item = value
    elif isinstance(value, Mapping):
        required = {"target_root", "target_path", "target_type", "parameter_shape"}
        allowed = required | {"tensor_name"}
        if not required.issubset(value) or set(value) - allowed:
            raise SD3ComponentAuditError(
                f"LoRA audit record keys mismatch: missing={sorted(required-set(value))}, "
                f"extra={sorted(set(value)-allowed)}"
            )
        item = SD3AdapterTensor(
            target_root=value["target_root"],
            target_path=value["target_path"],
            target_type=value["target_type"],
            parameter_shape=value["parameter_shape"],
            tensor_name=value.get("tensor_name"),
        )
    else:
        raise SD3ComponentAuditError(f"Unexpected LoRA record: {type(value).__name__}")
    if not isinstance(item.target_root, str) or not item.target_root.strip():
        raise SD3ComponentAuditError("LoRA target_root must be non-empty")
    if not isinstance(item.target_path, str) or not item.target_path.strip():
        raise SD3ComponentAuditError("LoRA target_path must be non-empty")
    if not isinstance(item.target_type, str) or not item.target_type.strip():
        raise SD3ComponentAuditError("LoRA target_type must be non-empty")
    if not isinstance(item.parameter_shape, (tuple, list)) or not item.parameter_shape:
        raise SD3ComponentAuditError("LoRA parameter_shape must be a nonempty list/tuple")
    if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in item.parameter_shape):
        raise SD3ComponentAuditError("LoRA parameter_shape has an invalid dimension")
    if item.tensor_name is not None and (
        not isinstance(item.tensor_name, str)
        or not item.tensor_name.strip()
        or item.tensor_name != item.tensor_name.strip()
    ):
        raise SD3ComponentAuditError("LoRA tensor_name must be a non-empty canonical string")
    return SD3AdapterTensor(
        target_root=item.target_root.strip().casefold(),
        target_path=item.target_path.strip("."),
        target_type=item.target_type.strip(),
        parameter_shape=tuple(item.parameter_shape),
        tensor_name=item.tensor_name,
    )


def verify_target_membership(
    target_root: str,
    target_path: str,
    target_type: str,
    *,
    module_inventory: Mapping[str, Mapping[str, str]],
) -> None:
    """Verify presence/type against *caller-provided* named_modules metadata.

    Inventory keys are canonical roots: mmdit, clip_l, clip_g, t5xxl.
    No model import or deserialization is performed in the audit layer.
    """
    if not isinstance(module_inventory, Mapping):
        raise SD3ComponentAuditError("module_inventory must be a root->module_path->type mapping")
    canonical_root = _TEXT_CANONICAL_ROOTS.get(target_root, "mmdit" if target_root in {"mmdit", "transformer"} else None)
    modules = module_inventory.get(canonical_root) if canonical_root is not None else None
    if not isinstance(modules, Mapping):
        raise SD3ComponentAuditError(f"No module inventory for target root {canonical_root!r}")
    actual_type = modules.get(target_path)
    if not isinstance(actual_type, str) or actual_type.rsplit(".", 1)[-1] != target_type.rsplit(".", 1)[-1]:
        raise SD3ComponentAuditError(
            f"LoRA target is absent or wrong class in live module inventory: {canonical_root}.{target_path}"
        )


def classify_sd3_adapter_target(
    architecture: SD3Architecture,
    target_root: str,
    target_path: str,
    target_type: str,
    *,
    module_inventory: Mapping[str, Mapping[str, str]] | None = None,
) -> SD3TargetGroup:
    """Classify ORIGINAL LoRA target metadata, never flattened adapter names."""
    if not isinstance(architecture, SD3Architecture):
        raise SD3ComponentAuditError("A validated SD3Architecture is required")
    if not all(isinstance(x, str) and x.strip() for x in (target_root, target_path, target_type)):
        raise SD3ComponentAuditError("LoRA target root/path/type must be non-empty strings")
    root = target_root.strip().casefold()
    path = target_path.strip(".")
    target_class = target_type.rsplit(".", 1)[-1]
    if target_class not in {"Linear", "Conv2d"}:
        raise SD3ComponentAuditError(f"Unsupported LoRA original target type: {target_type!r}")

    if root in _TEXT_ROOTS:
        if target_class != "Linear":
            raise SD3ComponentAuditError("SD3 CLIP/T5 LoRA must target a Linear module")
        canonical_root = _TEXT_CANONICAL_ROOTS[root]
        match = (_T5_TARGET if canonical_root == "t5xxl" else _CLIP_TARGET).fullmatch(path)
        if match is None or int(match.group(1)) >= _TEXT_LIMITS[canonical_root]:
            raise SD3ComponentAuditError(f"Unknown or out-of-range {canonical_root} LoRA target: {path!r}")
        if module_inventory is not None:
            verify_target_membership(root, path, target_type, module_inventory=module_inventory)
        return SD3TargetGroup(_TEXT_ROOTS[root], None, None)

    if root not in {"mmdit", "transformer"}:
        raise SD3ComponentAuditError(f"Unsupported SD3 target root: {root!r}")

    match = _JOINT_PATH.fullmatch(path)
    if match is None:
        if _NONBLOCK_PATH.fullmatch(path):
            if (path == "x_embedder.proj") != (target_class == "Conv2d"):
                raise SD3ComponentAuditError("SD3 x_embedder is Conv2d; other nonblock LoRA targets are Linear")
            if module_inventory is not None:
                verify_target_membership(root, path, target_type, module_inventory=module_inventory)
            return SD3TargetGroup("mmdit.input_output.adapter", None, None)
        raise SD3ComponentAuditError(f"Unrecognized MMDiT LoRA original target: {path!r}")

    block_index = int(match.group(1))
    stream = match.group(2)
    family = match.group(3)
    child = match.group(4)
    if block_index < 0 or block_index >= architecture.depth:
        raise SD3ComponentAuditError(f"MMDiT target block {block_index} outside depth {architecture.depth}")
    if target_class != "Linear":
        raise SD3ComponentAuditError("MMDiT joint-block adapter target must be Linear")
    if family in {"attn", "attn2"} and child not in {"qkv", "proj"}:
        raise SD3ComponentAuditError(f"Unexpected SD3 attention LoRA projection: {child!r}")
    if family == "mlp" and child not in {"fc1", "fc2"}:
        raise SD3ComponentAuditError(f"Unexpected SD3 MLP LoRA projection: {child!r}")
    if family == "adaLN_modulation" and child != "1":
        raise SD3ComponentAuditError(f"Unexpected SD3 modulation LoRA target: {child!r}")
    if family == "attn2" and (stream != "x_block" or block_index not in architecture.extra_attention_layers):
        raise SD3ComponentAuditError(f"Extra attention not present at {stream} block {block_index}")
    if block_index == architecture.depth - 1 and stream == "context_block":
        if family == "mlp" or (family == "attn" and child == "proj"):
            raise SD3ComponentAuditError("Last MMDiT Context block is pre-only")
    component = _COMPONENT_BY_PATH.get((stream, family))
    if component is None:
        raise SD3ComponentAuditError(f"Unrecognized MMDiT target path: {path!r}")
    if module_inventory is not None:
        verify_target_membership(root, path, target_type, module_inventory=module_inventory)
    return SD3TargetGroup(component, block_index, architecture.block_kind(block_index))


def audit_sd3_adapter_tensors(
    architecture: SD3Architecture,
    tensors: Iterable[SD3AdapterTensor | Mapping[str, Any]],
    *,
    module_inventory: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Count adapter tensor ownership by 11 diagnostic groups and actual block.

    Each row is one physical LoRA tensor, e.g. lora_down.weight or
    lora_up.weight. Different rows MAY share the same original LoRA target.
    This function never freezes/trains parameters or alters routing.
    """
    components: dict[str, dict[str, Any]] = {
        name: {"tensor_count": 0, "parameter_elements": 0, "target_module_count": 0,
               "target_examples": []}
        for name in SD3_DIAGNOSTIC_COMPONENTS
    }
    module_targets: dict[str, set[tuple[str, str]]] = {
        name: set() for name in SD3_DIAGNOSTIC_COMPONENTS
    }
    layers: dict[int, dict[str, Any]] = {
        i: {"block_index": i, "block_kind": architecture.block_kind(i),
            "tensor_count": 0, "parameter_elements": 0, "components": {}}
        for i in range(architecture.depth)
    }
    total = 0
    seen_names: set[str] = set()
    named = 0
    unnamed = 0
    for raw in tensors:
        item = _clean_tensor(raw)
        if item.tensor_name is None:
            unnamed += 1
        else:
            if item.tensor_name in seen_names:
                raise SD3ComponentAuditError(f"Duplicate LoRA tensor_name: {item.tensor_name!r}")
            seen_names.add(item.tensor_name)
            named += 1
        group = classify_sd3_adapter_target(
            architecture, item.target_root, item.target_path, item.target_type,
            module_inventory=module_inventory,
        )
        numel = math.prod(item.parameter_shape)
        component = components[group.component_id]
        component["tensor_count"] += 1
        component["parameter_elements"] += numel
        module_targets[group.component_id].add((item.target_root, item.target_path))
        if group.block_index is not None:
            row = layers[group.block_index]
            row["tensor_count"] += 1
            row["parameter_elements"] += numel
            row["components"][group.component_id] = row["components"].get(group.component_id, 0) + 1
        total += 1

    for name in SD3_DIAGNOSTIC_COMPONENTS:
        targets = sorted(module_targets[name])
        components[name]["target_module_count"] = len(targets)
        components[name]["target_examples"] = [f"{root}.{path}" for root, path in targets[:5]]
    return {
        "schema": "dts.sd3-component-audit",
        "version": 1,
        "model_architecture": architecture.identity(),
        "read_only": True,
        "changes_parameter_policy": False,
        "target_validation_status": "inventory_verified" if module_inventory is not None else "syntax_validated",
        "module_inventory_verified": module_inventory is not None,
        "parameter_identity_status": (
            "external_declared_unique" if named and not unnamed
            else "unverified_external"
        ),
        "physical_uniqueness_verified": False,
        "total_adapter_tensors": total,
        "components": components,
        "blocks": [layers[i] for i in range(architecture.depth)],
    }


def audit_sd3_parameter_descriptors(
    architecture: SD3Architecture,
    descriptors: Iterable[Any],
    *,
    module_inventory: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Audit Step-3 descriptors by adapter target without importing Torch.

    Descriptor aliases must agree on original targets. Step-3 should normally
    deduplicate physical parameters; repeat IDs are rejected rather than counted.
    """
    records = []
    seen_ids: set[int] = set()
    checked_descriptors = list(descriptors)
    for descriptor in checked_descriptors:
        parameter_id = getattr(descriptor, "parameter_id", None)
        if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
            raise SD3ComponentAuditError("Descriptor missing integer parameter_id")
        if parameter_id in seen_ids:
            raise SD3ComponentAuditError("Duplicate physical ParameterDescriptor ID")
        seen_ids.add(parameter_id)
        parameter = getattr(descriptor, "parameter", None)
        if parameter is not None and id(parameter) != parameter_id:
            raise SD3ComponentAuditError("Descriptor physical parameter_id mismatch")
        aliases = getattr(descriptor, "aliases", None)
        if not aliases:
            raise SD3ComponentAuditError("LoRA ParameterDescriptor has no original-target aliases")
        targets = set()
        for alias in aliases:
            target = getattr(alias, "adapter_target", None)
            if target is None:
                raise SD3ComponentAuditError("LoRA ParameterDescriptor lacks AdapterTargetMetadata")
            targets.add((target.root, target.module_path, target.module_type))
        if len(targets) != 1:
            raise SD3ComponentAuditError("LoRA ParameterDescriptor aliases disagree on original target")
        root, path, kind = next(iter(targets))
        records.append(SD3AdapterTensor(
            root, path, kind, tuple(descriptor.shape),
            tensor_name=f"descriptor-id:{parameter_id}",
        ))
    report = audit_sd3_adapter_tensors(architecture, records, module_inventory=module_inventory)
    report["parameter_identity_status"] = "descriptor_id_checked"
    report["physical_uniqueness_verified"] = bool(checked_descriptors) and all(
        getattr(descriptor, "parameter", None) is not None for descriptor in checked_descriptors
    )
    return report


__all__ = [
    "SD3AdapterTensor", "SD3ComponentAuditError", "SD3TargetGroup",
    "SD3_DIAGNOSTIC_COMPONENTS", "classify_sd3_adapter_target",
    "audit_sd3_adapter_tensors", "audit_sd3_parameter_descriptors",
    "verify_target_membership",
]
