"""Fail-closed SD3.5 Medium Component Full-BF16 *scope*, not qualification.

This host-side module is safe for Preview/Start and direct trainer preflight.
It loads only the safetensors header (no model values, Torch or CUDA) and
reuses the SD3 MMDiT architecture contract. The production sd3-lora Full-BF16
qualification remains pending until D1-SD3C2-B exact-head CUDA evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.sd3_architecture_contract import (
    validate_sd3_mmdit_checkpoint,
)


SD3_FULL_BF16_TRAINED_COMPONENTS = frozenset({
    "mmdit.attention.adapter",
    "mmdit.mlp.adapter",
    "mmdit.modulation_norm.adapter",
})
SD3_FULL_BF16_FROZEN_COMPONENTS = frozenset({
    "mmdit.other.adapter", "clip_l.adapter", "clip_g.adapter", "t5xxl.adapter",
})
_SD3_FULL_BF16_CASE_ACCUMULATION = {
    "adamw": 1,
    "muon_adamw_fallback": 2,
}
_EMBEDDED_SOURCE_KEYS = {
    "clip_l": "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight",
    "clip_g": "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight",
    "t5xxl": "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight",
}
_FALSE_VALUES = frozenset({"", "0", "false", "off", "no"})
_TRUE_VALUES = frozenset({"1", "true", "on", "yes"})
_UNQUALIFIED_SWITCHES = (
    "gradient_checkpointing",
    "fp8_base",
    "fp8_base_unet",
    "full_fp16",
    "deepspeed",
    "torch_compile",
    "compile",
    "fused_backward_pass",
    "blockwise_fused_optimizers",
    "cpu_offload_checkpointing",
    "unsloth_offload_checkpointing",
    "enable_scaled_pos_embed",
    "cache_text_encoder_outputs_to_disk",
    "cache_latents_to_disk",
    "lowram",
    "network_train_text_encoder_only",
)


def _is_true(value: object) -> bool:
    """Unparseable switch values are treated as active and therefore blocked."""
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_VALUES
    return True


def _is_explicit_true(value: object) -> bool:
    if value is True or value == 1:
        return True
    return isinstance(value, str) and value.strip().lower() in _TRUE_VALUES


def _is_exact_number(value: object, expected: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed == Decimal(expected)


def _is_unset(value: object) -> bool:
    return value is None or value == "" or value == () or value == []


def _positive_learning_rate(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed > 0


def _optimizer_case(policy: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Demand exactly the two reviewed canonical routing shapes (R1/R2)."""
    components = policy.get("components")
    profiles = policy.get("optimizer_profiles")
    if not isinstance(components, Mapping) or not isinstance(profiles, Mapping):
        return None, ["SD3 Full-BF16 requires canonical Component and Optimizer Profile mappings."]

    canonical_ids = set(get_model_component_profile("sd3-lora").components)
    if set(components) != canonical_ids:
        return None, ["SD3 Full-BF16 requires exactly the seven registered SD3 LoRA Components."]
    trainable = {
        name for name, route in components.items()
        if isinstance(route, Mapping) and route.get("train") is True
    }
    frozen = {
        name for name, route in components.items()
        if isinstance(route, Mapping) and route.get("train") is False
    }
    if trainable != SD3_FULL_BF16_TRAINED_COMPONENTS or frozen != SD3_FULL_BF16_FROZEN_COMPONENTS:
        return None, [
            "SD3 Full-BF16 covers exactly MMDiT Attention/MLP/Modulation LoRA; "
            "MMDiT Other and all CLIP/T5 adapters must be frozen."
        ]

    if not all(isinstance(v, Mapping) and isinstance(v.get("type"), str) for v in profiles.values()):
        return None, ["SD3 Full-BF16 Optimizer Profile types are invalid."]
    by_type = {v["type"]: name for name, v in profiles.items()}
    if len(profiles) == 1 and set(by_type) == {"AdamW"}:
        adamw_name = by_type["AdamW"]
        for component in SD3_FULL_BF16_TRAINED_COMPONENTS:
            route = components[component]
            if (
                route.get("optimizer_profile") != adamw_name
                or not _is_unset(route.get("fallback_optimizer_profile"))
                or not _is_unset(route.get("fallback_learning_rate"))
            ):
                return None, ["SD3 R1 must route every trained Component to sole AdamW, without fallback."]
        return "adamw", []

    if len(profiles) == 2 and set(by_type) == {"Muon", "AdamW"}:
        muon_name, adamw_name = by_type["Muon"], by_type["AdamW"]
        for component in SD3_FULL_BF16_TRAINED_COMPONENTS:
            route = components[component]
            if (
                route.get("optimizer_profile") != muon_name
                or route.get("fallback_optimizer_profile") != adamw_name
                or not _positive_learning_rate(route.get("fallback_learning_rate"))
            ):
                return None, [
                    "SD3 R2 requires Muon primary and an explicit AdamW fallback "
                    "Profile and LR on every trained MMDiT Component."
                ]
        return "muon_adamw_fallback", []

    return None, [
        "SD3 Full-BF16 permits only R1 sole AdamW or R2 Muon primary plus "
        "an independent explicit AdamW fallback Profile."
    ]


def _read_safetensors_shape_index(path: Path) -> dict[str, tuple[int, ...]]:
    """Only inspect metadata; no model tensor is materialized."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        return {
            name: tuple(file.get_slice(name).get_shape())
            for name in file.keys()
        }


def _model_scope_blockers(config: Mapping[str, Any]) -> list[str]:
    raw = config.get("pretrained_model_name_or_path")
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        return ["SD3 Full-BF16 requires an explicit local SD3.5 Medium safetensors model path."]

    model = Path(raw).expanduser()
    if not model.is_absolute() or model.suffix.lower() != ".safetensors" or not model.is_file():
        return [
            "SD3 Full-BF16 requires an existing absolute local .safetensors "
            "model checkpoint for SD3.5 Medium."
        ]
    try:
        shapes = _read_safetensors_shape_index(model)
        validated = validate_sd3_mmdit_checkpoint(
            shapes, require_loader_compatible_root=True,
        )
    except Exception as exc:
        # All header/read/topology failures must remain fail closed. The actual
        # production trainer will subsequently load/validate weight contents.
        return [f"SD3 Full-BF16 checkpoint header/architecture audit failed: {type(exc).__name__}: {exc}"]
    if validated.architecture.variant != "sd35-medium" or not validated.loader_prefix_compatible:
        return ["SD3 Full-BF16 qualifies only DTS-loader-compatible SD3.5 Medium MMDiT."]

    for name, embedded_key in _EMBEDDED_SOURCE_KEYS.items():
        if embedded_key in shapes:
            continue
        raw_source = config.get(name)
        if (
            not isinstance(raw_source, (str, Path))
            or not Path(raw_source).is_absolute()
            or not Path(raw_source).is_file()
        ):
            return [
                f"SD3 Full-BF16 needs an existing --{name} model file because "
                f"the base checkpoint does not embed {name}."
            ]
    if not any(key.startswith("first_stage_model.") for key in shapes):
        raw_vae = config.get("vae")
        if (
            not isinstance(raw_vae, (str, Path))
            or not Path(raw_vae).is_absolute()
            or not Path(raw_vae).is_file()
        ):
            return [
                "SD3 Full-BF16 needs an existing --vae model file because "
                "the base checkpoint does not embed VAE weights."
            ]
    return []


def sd3_full_bf16_scope_blockers(
    canonical_policy: Mapping[str, Any],
    effective_config: Mapping[str, Any],
) -> list[str]:
    """Validate the future C2-B candidate scope; does not grant qualification.

    The caller must first check sd3-lora's release qualification. This function
    is invoked by the shared Preview/Start/direct-trainer execution blocker path
    *only when* that record is qualified. In C2-A it is still pending, and no
    production Full-BF16 command is admitted.
    """
    if not isinstance(canonical_policy, Mapping) or not isinstance(effective_config, Mapping):
        return ["SD3 Full-BF16 needs canonical policy and effective-config mappings."]

    case, problems = _optimizer_case(canonical_policy)
    if problems:
        return problems

    checks = (
        ("network_module", "networks.lora_sd3"),
        ("optimizer_type", "AdamW"),
        ("lr_scheduler", "constant"),
        ("save_model_as", "safetensors"),
    )
    for field, expected in checks:
        if effective_config.get(field) != expected:
            problems.append(f"SD3 Full-BF16 requires {field}={expected!r}.")

    if not _is_explicit_true(effective_config.get("network_train_unet_only")):
        problems.append("SD3 Full-BF16 requires network_train_unet_only=true.")
    for flag in _UNQUALIFIED_SWITCHES:
        if _is_true(effective_config.get(flag)):
            problems.append(f"SD3 Full-BF16 does not qualify {flag}=true.")
    for flag in ("network_args", "network_weights", "base_weights", "optimizer_args"):
        if not _is_unset(effective_config.get(flag)):
            problems.append(f"SD3 Full-BF16 does not qualify nonempty {flag}.")
    for flag in ("blocks_to_swap", "fused_optimizer_groups", "pos_embed_random_crop_rate", "network_dropout"):
        value = effective_config.get(flag)
        if value is None or value is False:
            value = 0
        if not _is_exact_number(value, 0):
            problems.append(f"SD3 Full-BF16 requires {flag}=0.")

    for field, expected in (
        ("network_dim", 4),
        ("network_alpha", 4),
        ("train_batch_size", 1),
        ("gradient_accumulation_steps", _SD3_FULL_BF16_CASE_ACCUMULATION[case]),
        ("resolution", 512),
        ("max_data_loader_n_workers", 0),
    ):
        if not _is_exact_number(effective_config.get(field), expected):
            problems.append(f"SD3 Full-BF16 {case} requires {field}={expected}.")
    for flag in ("cache_latents", "cache_text_encoder_outputs"):
        if not _is_explicit_true(effective_config.get(flag)):
            problems.append(f"SD3 Full-BF16 {case} requires {flag}=true.")

    if problems:
        return problems
    return _model_scope_blockers(effective_config)


__all__ = [
    "SD3_FULL_BF16_TRAINED_COMPONENTS",
    "SD3_FULL_BF16_FROZEN_COMPONENTS",
    "sd3_full_bf16_scope_blockers",
]
