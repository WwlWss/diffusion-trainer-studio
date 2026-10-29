"""Pure fail-closed boolean type boundary for Component trainer runtime.

This module does NOT normalize or mutate an Effective Config. It recognizes
the exact types that the stock trainer interprets unambiguously through
Python truthiness. Quoted TOML values such as "false" and "0" are strings,
and are therefore always rejected (bool("false") is True).

Standard mode never invokes this Component-only preflight.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Reviewed flags that are read as booleans by stock SD/SDXL/SD3/Flux or Anima
# training runtime, or by Component eligibility/qualification. This is NOT a
# registry of numeric flags such as blocks_to_swap or fused_optimizer_groups.
_COMPONENT_RUNTIME_BOOLEAN_FIELDS = frozenset({
    "full_bf16",
    "full_fp16",
    "fp8_base",
    "fp8_base_unet",
    "torch_compile",
    "compile",
    "deepspeed",
    "fused_backward_pass",
    "blockwise_fused_optimizers",
    "cpu_offload_checkpointing",
    "unsloth_offload_checkpointing",
    "gradient_checkpointing",
    "network_train_unet_only",
    "network_train_text_encoder_only",
    "cache_latents",
    "cache_latents_to_disk",
    "cache_text_encoder_outputs",
    "cache_text_encoder_outputs_to_disk",
    "enable_bucket",
    "persistent_data_loader_workers",
    "enable_scaled_pos_embed",
    "apply_lg_attn_mask",
    "apply_t5_attn_mask",
    "masked_loss",
    "lowram",
    "resume_from_huggingface",
})
_BACKEND_RUNTIME_BOOLEAN_FIELDS: dict[str, frozenset[str]] = {
    "sd-lora": frozenset({"v2"}),
    "sd-dreambooth": frozenset({"v2"}),
}


def strict_component_runtime_bool(value: object, *, field: str) -> bool:
    """Parse only runtime-equivalent boolean values, never string pseudo-bools.

    None represents an absent/off optional flag; bool, integer 0/1, and
    float 0.0/1.0 preserve existing recognized runtime semantics. String
    values, including empty, 'true' and 'false', are not safe: a trainer that
    reads these as normal Python booleans would treat all nonempty strings as
    True. No caller may use this helper to rewrite the supplied value.
    """
    if value is None:
        return False
    if type(value) is bool:
        return value
    if type(value) in (int, float):
        if value == 0:
            return False
        if value == 1:
            return True
    raise ValueError(
        f"Component runtime {field} must be a real boolean (true/false or 0/1), "
        f"not {value!r} ({type(value).__name__}). "
        "For TOML, use unquoted true/false."
    )


def component_runtime_boolean_type_blockers(
    effective_config: Mapping[str, Any],
    *,
    train_type: str,
) -> list[str]:
    """Return deterministic, pure type blockers before any training side effect."""
    if not isinstance(effective_config, Mapping):
        raise ValueError("Component runtime boolean preflight needs an Effective Config mapping.")
    normalized = str(train_type or "").strip().lower()
    fields = _COMPONENT_RUNTIME_BOOLEAN_FIELDS.union(
        _BACKEND_RUNTIME_BOOLEAN_FIELDS.get(normalized, frozenset())
    )
    blockers = []
    for field in sorted(fields):
        if field not in effective_config:
            continue
        try:
            strict_component_runtime_bool(effective_config[field], field=field)
        except ValueError as exc:
            blockers.append(str(exc))
    return blockers


def require_component_runtime_booleans(
    effective_config: Mapping[str, Any],
    *,
    train_type: str,
) -> None:
    """Early trainer fail-closed preflight, with zero model/optimizer work."""
    blockers = component_runtime_boolean_type_blockers(
        effective_config, train_type=train_type,
    )
    if blockers:
        raise ValueError(
            "Component runtime boolean preflight failed:\n- "
            + "\n- ".join(blockers)
        )


__all__ = [
    "strict_component_runtime_bool",
    "component_runtime_boolean_type_blockers",
    "require_component_runtime_booleans",
]
