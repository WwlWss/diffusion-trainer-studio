"""Host-only D1 stock-LoRA *reference-shape* coverage, not CUDA proof.

The D1 physical cases bind actual model SHA, input-policy identity, clean
exact HEAD and true Save -> new-process Resume. Those cannot be verified by
this I/O-free checker. A shape match NEVER certifies the user's current job
or supplies an authority token. Runtime eligibility is evaluated elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_D1_CASES = {
    ("sd-lora", "adamw"): "backend:sd-lora:full-bf16:adamw:v1",
    ("sd-lora", "muon_adamw_fallback"): (
        "backend:sd-lora:full-bf16:muon-adamw-fallback:v1"
    ),
    ("sdxl-lora", "adamw"): "backend:sdxl-lora:full-bf16:adamw:v1",
    ("sdxl-lora", "muon_adamw_fallback"): (
        "backend:sdxl-lora:full-bf16:muon-adamw-fallback:v1"
    ),
}
_D1_CORE_COMPONENTS = {
    "sd-lora": frozenset({
        "unet.attention.adapter", "unet.feed_forward.adapter",
        "unet.conv.adapter", "text_encoder.adapter",
    }),
    "sdxl-lora": frozenset({
        "unet.attention.adapter", "unet.feed_forward.adapter",
        "unet.conv.adapter", "text_encoder_1.adapter",
        "text_encoder_2.adapter",
    }),
}
_D1_LINEAR_COMPONENTS = {
    backend: members - {"unet.conv.adapter"}
    for backend, members in _D1_CORE_COMPONENTS.items()
}


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    """Only compares Host-visible reference fields; never asserts GPU proof."""

    covered: bool
    case_id: str | None
    reasons: tuple[str, ...]


def _reference_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _reference_resolution(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().replace(" ", "") == "512,512"
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return all(_reference_int(number, 512) for number in value)
    return False


def _network_args(config: Mapping[str, Any]) -> dict[str, str] | None:
    raw = config.get("network_args")
    if not isinstance(raw, (tuple, list)):
        return None
    out: dict[str, str] = {}
    for part in raw:
        if not isinstance(part, str) or "=" not in part:
            return None
        key, val = part.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key or key in out or not val:
            return None
        out[key] = val
    return out


def _profile_type(
    profiles: Mapping[str, Any],
    route: Mapping[str, Any],
    key: str,
) -> str:
    name = route.get(key)
    if not isinstance(name, str):
        return ""
    obj = profiles.get(name)
    if not isinstance(obj, Mapping):
        return ""
    value = obj.get("type")
    return value if isinstance(value, str) else ""


def _reference_routing(
    backend: str,
    trainable: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Any],
) -> str | None:
    if set(trainable) != _D1_CORE_COMPONENTS[backend]:
        return None

    def no_fallback(route: Mapping[str, Any]) -> bool:
        return route.get("fallback_optimizer_profile") in (None, "")

    if all(
        _profile_type(profiles, route, "optimizer_profile") == "AdamW"
        and no_fallback(route)
        and _reference_int_or_float(route.get("learning_rate"), 1e-4)
        for route in trainable.values()
    ):
        referenced = {
            route.get("optimizer_profile") for route in trainable.values()
        }
        if all(_empty_args(profiles.get(name)) for name in referenced):
            return "adamw"

    conv_route = trainable["unet.conv.adapter"]
    if (
        _profile_type(profiles, conv_route, "optimizer_profile") != "Muon"
        or _profile_type(profiles, conv_route, "fallback_optimizer_profile")
        != "AdamW"
        or not _reference_int_or_float(
            conv_route.get("fallback_learning_rate"), 1e-5
        )
    ):
        return None

    linear = _D1_LINEAR_COMPONENTS[backend]
    if not all(
        _profile_type(profiles, trainable[name], "optimizer_profile") == "Muon"
        and no_fallback(trainable[name])
        for name in linear
    ):
        return None

    if not all(
        _reference_int_or_float(route.get("learning_rate"), 1e-4)
        for route in trainable.values()
    ):
        return None
    referenced = {
        name
        for route in trainable.values()
        for name in (
            route.get("optimizer_profile"),
            route.get("fallback_optimizer_profile"),
        )
        if name
    }
    if not all(_empty_args(profiles.get(name)) for name in referenced):
        return None
    return "muon_adamw_fallback"


def _reference_int_or_float(value: object, expected: float) -> bool:
    return type(value) in (int, float) and float(value) == expected


def _empty_args(profile: object) -> bool:
    return (
        isinstance(profile, Mapping)
        and isinstance(profile.get("args"), Mapping)
        and not profile["args"]
    )


def match_full_bf16_evidence_coverage(
    *,
    train_type: str,
    canonical_policy: Mapping[str, Any],
    effective_config: Mapping[str, Any],
) -> EvidenceCoverage:
    """Compare visible shape to D1 stock SD1/SDXL reference cases.

    A successful match says *shape resembles D1*, not that D1's real model
    SHA, assets, exact HEAD, optimizer state or checkpoint lineage were
    reverified for this job. Consumers MUST NOT promote formally_qualified
    based on this shape match alone.
    """
    reasons: list[str] = []
    backend = str(train_type or "").strip().lower()
    if backend not in _D1_CORE_COMPONENTS:
        return EvidenceCoverage(False, None, ("backend.outside_d1_scope",))

    if not isinstance(canonical_policy, Mapping) or not isinstance(
        effective_config, Mapping
    ):
        return EvidenceCoverage(False, None, ("input.invalid_mapping",))

    components = canonical_policy.get("components")
    profiles = canonical_policy.get("optimizer_profiles")
    if not isinstance(components, Mapping) or not isinstance(profiles, Mapping):
        return EvidenceCoverage(False, None, ("policy.unavailable",))

    # D1 proof only covered true Full-BF16, single-process execution.
    # These are shape conditions, NOT extra producer runtime hard errors.
    if effective_config.get("full_bf16") is not True:
        reasons.append("execution.full_bf16_unconfirmed")
    if effective_config.get("mixed_precision") != "bf16":
        reasons.append("execution.mixed_precision_unconfirmed")
    if effective_config.get("num_processes", 1) != 1 or (
        type(effective_config.get("num_processes", 1)) is bool
    ):
        reasons.append("execution.process_count_unconfirmed")
    if effective_config.get("distributed_type") not in (None, "NO"):
        reasons.append("execution.distributed_mode_outside_scope")
    if effective_config.get("fused_optimizer_groups") not in (None, 0):
        reasons.append("execution.fused_optimizer_groups_outside_scope")

    active = {
        name: route
        for name, route in components.items()
        if isinstance(route, Mapping) and route.get("train") is True
    }
    if set(active) != _D1_CORE_COMPONENTS[backend]:
        reasons.append("components.outside_d1_scope")

    if effective_config.get("network_module") != "networks.lora":
        reasons.append("network.module_unconfirmed")
    if not _reference_int(effective_config.get("network_dim"), 4):
        reasons.append("network.rank_unconfirmed")
    if not _reference_int(effective_config.get("network_alpha"), 4):
        reasons.append("network.alpha_unconfirmed")
    if not _reference_resolution(effective_config.get("resolution")):
        reasons.append("dataset.resolution_unconfirmed")
    if not _reference_int(effective_config.get("train_batch_size"), 1):
        reasons.append("dataset.batch_unconfirmed")

    args = _network_args(effective_config)
    if args is None or set(args) != {"conv_dim", "conv_alpha"}:
        reasons.append("network.conv_geometry_unconfirmed")
    else:
        if args["conv_dim"] != "4" or args["conv_alpha"] != "4":
            reasons.append("network.conv_geometry_unconfirmed")

    if effective_config.get("save_model_as") != "safetensors":
        reasons.append("checkpoint.format_unconfirmed")
    for key in (
        "v2", "network_train_unet_only", "network_train_text_encoder_only",
        "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
        "gradient_checkpointing", "full_fp16", "fp8_base", "deepspeed",
        "torch_compile", "compile", "fused_backward_pass",
    ):
        if effective_config.get(key) not in (None, False, 0):
            reasons.append(f"feature.{key}_outside_scope")
    for key in ("network_weights", "base_weights", "resume_from_huggingface"):
        if effective_config.get(key) not in (None, False, 0, ""):
            reasons.append(f"inputs.{key}_unreviewed")

    mode = None
    if set(active) == _D1_CORE_COMPONENTS[backend]:
        mode = _reference_routing(backend, active, profiles)
    if mode is None:
        reasons.append("optimizer.reference_routing_mismatch")
    else:
        accumulation = 1 if mode == "adamw" else 2
        if not _reference_int(
            effective_config.get("gradient_accumulation_steps"),
            accumulation,
        ):
            reasons.append("optimizer.accumulation_outside_scope")

    if reasons:
        return EvidenceCoverage(False, None, tuple(sorted(set(reasons))))
    return EvidenceCoverage(True, _D1_CASES[(backend, mode)], ())


__all__ = ["EvidenceCoverage", "match_full_bf16_evidence_coverage"]
