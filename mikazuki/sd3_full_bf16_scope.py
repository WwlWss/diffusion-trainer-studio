"""Fail-closed SD3.5 Medium Component Full-BF16 *scope*, not qualification.

This host-side module separates model-free Preview/Export config checks from
read-only Start/direct-trainer asset and safetensors-header checks, and reuses
the SD3 MMDiT architecture contract without loading weight values or CUDA. The production sd3-lora Full-BF16
qualification remains pending until D1-SD3C2-B exact-head CUDA evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy_runtime_bools import (
    component_runtime_boolean_type_blockers,
    strict_component_runtime_bool,
)
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
# Alternative Dataset Blueprint/custom loaders can override CLI batch/resolution.
# Keep C2-B CUDA admission on the reference single-DreamBooth dataset path.
_UNQUALIFIED_DATA_SOURCES = (
    "dataset_config", "dataset_class", "in_json", "reg_data_dir",
    "multi_caption_config", "conditioning_data_dir",
)
_UNQUALIFIED_SD3_MODEL_OVERRIDES = (
    "t5xxl_device", "t5xxl_dtype",
)
_OPTIONAL_ZERO_SCALARS = frozenset({
    # These stock argparse None values mean that the feature is disabled.
    # Other scalars (e.g. training_shift) must not inherit this equivalence.
    "blocks_to_swap", "fused_optimizer_groups", "network_dropout",
})
_REFERENCE_SD3_SCALARS = (
    ("blocks_to_swap", 0),
    ("fused_optimizer_groups", 0),
    # Actual argparse name is pos_emb_random_crop_rate (not pos_embed_...).
    ("pos_emb_random_crop_rate", 0),
    ("network_dropout", 0),
    ("clip_l_dropout_rate", 0),
    ("clip_g_dropout_rate", 0),
    ("t5_dropout_rate", 0),
    ("training_shift", 1),
    ("t5xxl_max_token_length", 256),
)
_REFERENCE_LEARNING_RATE = 0.0001
_REFERENCE_R1_ADAMW_ARGS = {"weight_decay": 0.01}
_REFERENCE_R2_MUON_ARGS: dict[str, Any] = {}
_REFERENCE_R2_FALLBACK_ADAMW_ARGS: dict[str, Any] = {}
_UNQUALIFIED_SWITCHES = (
    # Ordinary DreamBooth can use variable aspect-ratio buckets without
    # dataset_config. R1/R2 qualify fixed 512x512 geometry only.
    "enable_bucket",
    "gradient_checkpointing",
    "fp8_base",
    "fp8_base_unet",
    "full_fp16",
    # VAE float32 vs reference BF16 changes cached latents and device/dtype
    # execution. The R1/R2 physical evidence does not qualify this variant.
    "no_half_vae",
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
    "persistent_data_loader_workers",
    "apply_lg_attn_mask",
    "apply_t5_attn_mask",
    "masked_loss",
    "resume_from_huggingface",
)


def _is_exact_number(value: object, expected: int | float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation:
        return False
    # Decimal(float) exposes binary representation noise (e.g. 0.0001);
    # both canonical Policy floats and reference values use decimal text.
    return parsed.is_finite() and parsed == Decimal(str(expected))


def _is_reference_resolution(value: object) -> bool:
    """Accept exactly 512x512 in *production* GUI/CLI and Trainer forms.

    The raw SD3 UI/argparse uses a string ("512" or "512,512").
    train_util.prepare_dataset_args() later rewrites it to a 2-int tuple.
    Do not mutate either representation, or admit non-production list/int
    shapes that the stock string parser cannot consume.
    """
    if isinstance(value, str):
        tokens = value.split(",")
        if len(tokens) not in (1, 2) or any(not t.strip() for t in tokens):
            return False
        try:
            dimensions = tuple(int(t.strip()) for t in tokens)
        except ValueError:
            return False
        if len(dimensions) == 1:
            dimensions = (dimensions[0], dimensions[0])
    elif type(value) is tuple:
        if len(value) != 2 or any(type(item) is not int for item in value):
            return False
        dimensions = value
    else:
        return False
    return dimensions == (512, 512)


def _is_unset(value: object) -> bool:
    return value is None or value == "" or value == () or value == []


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
        if profiles[adamw_name].get("args") != _REFERENCE_R1_ADAMW_ARGS:
            return None, ["SD3 R1 AdamW Profile args must match the reviewed reference baseline."]
        for component in SD3_FULL_BF16_TRAINED_COMPONENTS:
            route = components[component]
            if (
                route.get("optimizer_profile") != adamw_name
                or not _is_unset(route.get("fallback_optimizer_profile"))
                or not _is_unset(route.get("fallback_learning_rate"))
                or not _is_exact_number(route.get("learning_rate"), _REFERENCE_LEARNING_RATE)
            ):
                return None, [
                    "SD3 R1 requires reference learning rate and sole AdamW routing "
                    "for every trained Component, without fallback."
                ]
        return "adamw", []

    if len(profiles) == 2 and set(by_type) == {"Muon", "AdamW"}:
        muon_name, adamw_name = by_type["Muon"], by_type["AdamW"]
        if (
            profiles[muon_name].get("args") != _REFERENCE_R2_MUON_ARGS
            or profiles[adamw_name].get("args") != _REFERENCE_R2_FALLBACK_ADAMW_ARGS
        ):
            return None, ["SD3 R2 Muon/AdamW Profile args must match the reviewed reference baseline."]
        for component in SD3_FULL_BF16_TRAINED_COMPONENTS:
            route = components[component]
            if (
                route.get("optimizer_profile") != muon_name
                or route.get("fallback_optimizer_profile") != adamw_name
                or not _is_exact_number(route.get("learning_rate"), _REFERENCE_LEARNING_RATE)
                or not _is_exact_number(route.get("fallback_learning_rate"), _REFERENCE_LEARNING_RATE)
            ):
                return None, [
                    "SD3 R2 requires reference learning rates, Muon primary and an "
                    "explicit AdamW fallback on every trained MMDiT Component."
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


def sd3_full_bf16_asset_blockers(config: Mapping[str, Any]) -> list[str]:
    """Read-only physical model/header check; never call from Preview/Export."""
    if not isinstance(config, Mapping):
        return ["SD3 Full-BF16 asset checks require an effective-config mapping."]
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


def sd3_full_bf16_config_blockers(
    canonical_policy: Mapping[str, Any],
    effective_config: Mapping[str, Any],
) -> list[str]:
    """Pure R1/R2 candidate scope validation (no filesystem or Torch imports)."""
    if not isinstance(canonical_policy, Mapping) or not isinstance(effective_config, Mapping):
        return ["SD3 Full-BF16 needs canonical policy and effective-config mappings."]

    # Type compatibility is an independent safety condition. A quoted TOML
    # "false" is True to the legacy Trainer, so no such string may be treated
    # as an off-switch in this candidate Full-BF16 scope.
    problems = component_runtime_boolean_type_blockers(
        effective_config, train_type="sd3-lora",
    )
    if problems:
        return problems

    case, problems = _optimizer_case(canonical_policy)
    if problems:
        return problems

    checks = (
        ("network_module", "networks.lora_sd3"),
        ("optimizer_type", "AdamW"),
        ("lr_scheduler", "constant"),
        ("save_model_as", "safetensors"),
        # SD3 sampling/conditioning modes are not qualified beyond defaults.
        ("weighting_scheme", "uniform"),
        ("loss_type", "l2"),
    )
    for field, expected in checks:
        value = effective_config.get(field, expected) if field in {"weighting_scheme", "loss_type"} else effective_config.get(field)
        if value != expected:
            problems.append(f"SD3 Full-BF16 requires {field}={expected!r}.")

    if not strict_component_runtime_bool(
        effective_config.get("network_train_unet_only"),
        field="network_train_unet_only",
    ):
        problems.append("SD3 Full-BF16 requires network_train_unet_only=true.")
    for flag in _UNQUALIFIED_SWITCHES:
        try:
            enabled = strict_component_runtime_bool(
                effective_config.get(flag), field=flag,
            )
        except ValueError as exc:
            # Newly scoped flags also fail closed even when they do not yet
            # belong to the shared early-preflight registry.
            problems.append(str(exc))
            continue
        if enabled:
            problems.append(f"SD3 Full-BF16 does not qualify {flag}=true.")
    for flag in (
        "network_args", "network_weights", "base_weights", "optimizer_args",
        *_UNQUALIFIED_DATA_SOURCES, *_UNQUALIFIED_SD3_MODEL_OVERRIDES,
        "min_timestep", "max_timestep",
    ):
        if not _is_unset(effective_config.get(flag)):
            problems.append(f"SD3 Full-BF16 does not qualify nonempty {flag}.")
    for flag, expected in _REFERENCE_SD3_SCALARS:
        value = effective_config.get(flag, expected)
        if value is None and flag in _OPTIONAL_ZERO_SCALARS:
            # A local comparison value only; never rewrite effective_config.
            value = 0
        if not _is_exact_number(value, expected):
            problems.append(f"SD3 Full-BF16 requires {flag}={expected}.")

    for field, expected in (
        ("network_dim", 4),
        ("network_alpha", 4),
        ("train_batch_size", 1),
        ("gradient_accumulation_steps", _SD3_FULL_BF16_CASE_ACCUMULATION[case]),
        ("max_data_loader_n_workers", 0),
    ):
        if not _is_exact_number(effective_config.get(field), expected):
            problems.append(f"SD3 Full-BF16 {case} requires {field}={expected}.")
    if not _is_reference_resolution(effective_config.get("resolution")):
        problems.append("SD3 Full-BF16 requires fixed resolution=512x512.")
    for flag in ("cache_latents", "cache_text_encoder_outputs"):
        if not strict_component_runtime_bool(effective_config.get(flag), field=flag):
            problems.append(f"SD3 Full-BF16 {case} requires {flag}=true.")

    return problems


def sd3_full_bf16_scope_blockers(
    canonical_policy: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    *,
    require_assets: bool = True,
) -> list[str]:
    """Combine pure scope and assets only for explicit Start/trainer preflight.

    The backend must first have an evidence-backed release qualification.
    This is a second, narrowly scoped gate; C2-A leaves sd3-lora pending.
    Unknown callers default to fail-closed asset checks.
    """
    if type(require_assets) is not bool:
        return ["SD3 Full-BF16 require_assets must be a boolean."]
    problems = sd3_full_bf16_config_blockers(canonical_policy, effective_config)
    if problems:
        return problems
    if require_assets:
        return sd3_full_bf16_asset_blockers(effective_config)
    return []


__all__ = [
    "SD3_FULL_BF16_TRAINED_COMPONENTS",
    "SD3_FULL_BF16_FROZEN_COMPONENTS",
    "sd3_full_bf16_config_blockers",
    "sd3_full_bf16_asset_blockers",
    "sd3_full_bf16_scope_blockers",
]
