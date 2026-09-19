"""Bootstrap helpers for migrating legacy Standard optimizer semantics.

This module deliberately avoids importing the FastAPI application layer. The
caller supplies the backend resolver, which keeps the helper testable in the
same lightweight host-only CI used by the rest of the semantic compiler.

Commit 4 now provides:
- one exact Standard effective-config snapshot helper;
- strict legacy learning-rate primitives;
- backend-specific Component/LR mapping;
- the fail-closed public Standard -> Component bootstrap.

It remains host-only and does not alter request/launch/runtime behavior.
"""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from mikazuki.model_component_profiles import (
    TrainingTargetProfile,
    get_model_component_profile,
    resolve_training_target_profile,
)
from mikazuki.parameter_policy import (
    PARAMETER_POLICY_GUI_KEYS,
    PARAMETER_POLICY_VERSION,
    bootstrap_legacy_optimizer_profile,
    validate_parameter_policy,
)
from mikazuki.parameter_policy_compat import (
    parameter_policy_compatibility_blockers,
)
from mikazuki.training_config import PreparedTrainingConfig, prepare_training_config
from mikazuki.utils import train_utils


LEGACY_BOOTSTRAP_PROFILE = "legacy_main"


def _prepare_standard_snapshot(
    config: Mapping[str, Any],
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> PreparedTrainingConfig:
    """Compile one immutable caller snapshot through the real Standard pipeline.

    Parameter Policy GUI/sidecar fields are host-owned migration inputs and are
    removed before compilation. All remaining Standard semantics, including
    ui_custom_params last-write-wins behavior, are delegated to the existing
    effective-config compiler.
    """

    if not isinstance(config, Mapping):
        raise ValueError("Parameter Policy bootstrap: config 必须是 mapping。")

    candidate = deepcopy(dict(config))
    for key in PARAMETER_POLICY_GUI_KEYS:
        candidate.pop(key, None)
    candidate.pop("parameter_policy_config", None)

    train_utils.fix_config_types(candidate)
    prepared = prepare_training_config(
        candidate,
        page_train_type=page_type,
        resolve_backend=resolve_backend,
        launch=False,
        toml_path=None,
    )

    # ui_custom_params is intentionally last-write-wins for legacy trainer
    # values, but bootstrap migration must never let it re-introduce
    # Parameter Policy host fields into the Standard snapshot.
    for key in PARAMETER_POLICY_GUI_KEYS:
        prepared.config.pop(key, None)
    prepared.config.pop("parameter_policy_config", None)
    return prepared


def _parse_legacy_lr(value: object, *, field: str) -> float:
    """Parse one already-effective legacy LR as a finite nonnegative float."""

    message = (
        f"Parameter Policy bootstrap: {field} 必须是有限非负数字；"
        "legacy LR=0 只允许迁移为 Train=false。"
    )
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(message)
    return result


def _resolve_legacy_lr(
    raw_value: object,
    *,
    field: str,
    fallback: float | None = None,
) -> float:
    """Resolve an optional legacy LR using only an explicitly verified fallback."""

    if raw_value in (None, ""):
        if fallback is None:
            raise ValueError(
                f"Parameter Policy bootstrap: 缺少必需学习率 {field}；"
                "不会猜测 trainer argparse 默认值。"
            )
        return _parse_legacy_lr(fallback, field=f"{field} fallback")
    return _parse_legacy_lr(raw_value, field=field)


def _component_route_from_legacy_lr(
    learning_rate: object,
    *,
    optimizer_profile: str = LEGACY_BOOTSTRAP_PROFILE,
) -> dict[str, Any]:
    """Translate historical zero-freeze semantics into explicit Train state."""

    lr = _parse_legacy_lr(learning_rate, field="component learning rate")
    if lr == 0:
        return {"train": False}
    return {
        "train": True,
        "optimizer_profile": optimizer_profile,
        "learning_rate": lr,
    }


def _expand_text_encoder_lrs(
    raw_value: object,
    *,
    count: int,
    base_lr: object,
) -> tuple[float, ...]:
    """Expand dev NetworkTrainer Text Encoder LR semantics deterministically.

    Semantics mirror the reviewed trainer behavior:
    - missing / empty -> base LR for every encoder;
    - scalar -> repeat for every encoder;
    - one-item sequence -> repeat the item;
    - shorter sequence -> pad with its final item;
    - longer sequence -> truncate to the requested encoder count.

    String values are scalar numeric values, not comma-separated lists.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            "Parameter Policy bootstrap: Text Encoder LR count 必须是正整数。"
        )

    base = _parse_legacy_lr(base_lr, field="learning_rate")

    if raw_value in (None, ""):
        values: list[object] = []
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        values = [raw_value]

    if not values:
        return tuple(base for _ in range(count))

    parsed = [
        _parse_legacy_lr(value, field=f"text_encoder_lr[{index}]")
        for index, value in enumerate(values)
    ]

    if len(parsed) == 1:
        return tuple(parsed[0] for _ in range(count))

    if len(parsed) < count:
        parsed.extend([parsed[-1]] * (count - len(parsed)))

    return tuple(parsed[:count])


def _target_route(
    target: TrainingTargetProfile,
    component_id: str,
    learning_rate: Callable[[], float],
) -> dict[str, Any]:
    """Build one Component row with higher-level target capability first."""

    if not target.is_available(component_id):
        return {"train": False}
    return _component_route_from_legacy_lr(learning_rate())


def _required_base_lr(config: Mapping[str, Any]) -> float:
    return _resolve_legacy_lr(
        config.get("learning_rate"),
        field="learning_rate",
    )


def _map_sd_lora_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
    *,
    sdxl: bool,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)

    def unet_lr() -> float:
        return _resolve_legacy_lr(
            config.get("unet_lr"),
            field="unet_lr",
            fallback=base,
        )

    def text_lr() -> float:
        return _resolve_legacy_lr(
            config.get("text_encoder_lr"),
            field="text_encoder_lr",
            fallback=base,
        )

    rows = {
        component_id: _target_route(target, component_id, unet_lr)
        for component_id in (
            "unet.attention.adapter",
            "unet.feed_forward.adapter",
            "unet.conv.adapter",
            "unet.other.adapter",
        )
    }

    text_components = (
        ("text_encoder_1.adapter", "text_encoder_2.adapter")
        if sdxl
        else ("text_encoder.adapter",)
    )
    for component_id in text_components:
        rows[component_id] = _target_route(target, component_id, text_lr)
    return rows


def _map_flux_family_lora_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
    *,
    train_type: str,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)

    def unet_lr() -> float:
        return _resolve_legacy_lr(
            config.get("unet_lr"),
            field="unet_lr",
            fallback=base,
        )

    if train_type == "sd3-lora":
        rows = {
            component_id: _target_route(target, component_id, unet_lr)
            for component_id in (
                "mmdit.attention.adapter",
                "mmdit.mlp.adapter",
                "mmdit.modulation_norm.adapter",
                "mmdit.other.adapter",
            )
        }
    else:
        rows = {
            component_id: _target_route(target, component_id, unet_lr)
            for component_id in (
                "transformer.double_stream.adapter",
                "transformer.single_stream.adapter",
                "transformer.input_conditioning.adapter",
            )
        }

    if train_type in {"flux-lora", "chroma-lora"}:
        text_lrs: tuple[float, ...] | None = None

        def text_lr(index: int) -> float:
            nonlocal text_lrs
            if text_lrs is None:
                text_lrs = _expand_text_encoder_lrs(
                    config.get("text_encoder_lr"),
                    count=2,
                    base_lr=base,
                )
            return text_lrs[index]

        if train_type == "flux-lora":
            rows["clip_l.adapter"] = _target_route(
                target,
                "clip_l.adapter",
                lambda: text_lr(0),
            )
        rows["t5xxl.adapter"] = _target_route(
            target,
            "t5xxl.adapter",
            lambda: text_lr(1),
        )
        return rows

    if train_type == "sd3-lora":
        text_lrs: tuple[float, ...] | None = None

        def text_lr(index: int) -> float:
            nonlocal text_lrs
            if text_lrs is None:
                text_lrs = _expand_text_encoder_lrs(
                    config.get("text_encoder_lr"),
                    count=3,
                    base_lr=base,
                )
            return text_lrs[index]

        rows.update(
            {
                "clip_l.adapter": _target_route(
                    target,
                    "clip_l.adapter",
                    lambda: text_lr(0),
                ),
                "clip_g.adapter": _target_route(
                    target,
                    "clip_g.adapter",
                    lambda: text_lr(1),
                ),
                "t5xxl.adapter": _target_route(
                    target,
                    "t5xxl.adapter",
                    lambda: text_lr(2),
                ),
            }
        )
        return rows

    raise AssertionError(f"Unhandled Flux-family LoRA backend {train_type!r}.")


def _map_anima_lora_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)

    def dit_lr() -> float:
        return _resolve_legacy_lr(
            config.get("unet_lr"),
            field="unet_lr",
            fallback=base,
        )

    def qwen_lr() -> float:
        return _resolve_legacy_lr(
            config.get("text_encoder_lr"),
            field="text_encoder_lr",
            fallback=base,
        )

    rows = {
        component_id: _target_route(target, component_id, dit_lr)
        for component_id in (
            "dit.self_attention.adapter",
            "dit.cross_attention.adapter",
            "dit.mlp.adapter",
            "dit.modulation.adapter",
            "dit.other.adapter",
            "llm_adapter.adapter",
        )
    }
    rows["qwen3.adapter"] = _target_route(
        target,
        "qwen3.adapter",
        qwen_lr,
    )
    return rows


def _map_sd_dreambooth_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)
    rows = {
        component_id: _target_route(
            target,
            component_id,
            lambda base=base: base,
        )
        for component_id in (
            "unet.transformer",
            "unet.conv_resnet",
            "unet.norm_bias_other",
            "unet.base_other",
        )
    }

    rows["text_encoder"] = _target_route(
        target,
        "text_encoder",
        lambda: _resolve_legacy_lr(
            config.get("learning_rate_te"),
            field="learning_rate_te",
            fallback=base,
        ),
    )
    return rows


def _map_sdxl_full_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)
    rows = {
        component_id: _target_route(
            target,
            component_id,
            lambda base=base: base,
        )
        for component_id in (
            "unet.transformer",
            "unet.conv_resnet",
            "unet.norm_bias_other",
            "unet.base_other",
        )
    }
    for component_id, field in (
        ("text_encoder_1", "learning_rate_te1"),
        ("text_encoder_2", "learning_rate_te2"),
    ):
        rows[component_id] = _target_route(
            target,
            component_id,
            lambda field=field: _resolve_legacy_lr(
                config.get(field),
                field=field,
                fallback=base,
            ),
        )
    return rows


def _map_flux_full_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)
    return {
        component_id: _target_route(
            target,
            component_id,
            lambda base=base: base,
        )
        for component_id in target.available_components
    }


def _map_anima_full_components(
    config: Mapping[str, Any],
    target: TrainingTargetProfile,
) -> dict[str, dict[str, Any]]:
    base = _required_base_lr(config)
    fields = {
        "dit.base_other": None,
        "dit.self_attention": "self_attn_lr",
        "dit.cross_attention": "cross_attn_lr",
        "dit.mlp": "mlp_lr",
        "dit.modulation": "mod_lr",
        "dit.llm_adapter": "llm_adapter_lr",
        "qwen3": "qwen3_lr",
    }

    rows: dict[str, dict[str, Any]] = {}
    for component_id, field in fields.items():
        if field is None:
            resolver = lambda base=base: base
        elif component_id == "qwen3":
            resolver = lambda field=field: _resolve_legacy_lr(
                config.get(field),
                field=field,
            )
        else:
            resolver = lambda field=field: _resolve_legacy_lr(
                config.get(field),
                field=field,
                fallback=base,
            )
        rows[component_id] = _target_route(
            target,
            component_id,
            resolver,
        )
    return rows


def _bootstrap_component_rows(
    effective_config: Mapping[str, Any],
    train_type: str,
) -> dict[str, dict[str, Any]]:
    """Map one Standard effective config into complete Component route rows.

    This is Commit 4C's pure mapping layer. It assumes compatibility checking is
    owned by Commit 4A / later public bootstrap orchestration and therefore does
    not inspect LoRA+, block-LR, fused optimizer, or custom-network blockers.
    """

    if not isinstance(effective_config, Mapping):
        raise ValueError(
            "Parameter Policy bootstrap: effective_config 必须是 mapping。"
        )

    profile = get_model_component_profile(train_type)
    target = resolve_training_target_profile(train_type, effective_config)

    if train_type == "sd-lora":
        rows = _map_sd_lora_components(
            effective_config,
            target,
            sdxl=False,
        )
    elif train_type == "sdxl-lora":
        rows = _map_sd_lora_components(
            effective_config,
            target,
            sdxl=True,
        )
    elif train_type in {"flux-lora", "chroma-lora", "sd3-lora"}:
        rows = _map_flux_family_lora_components(
            effective_config,
            target,
            train_type=train_type,
        )
    elif train_type == "anima-lora":
        rows = _map_anima_lora_components(
            effective_config,
            target,
        )
    elif train_type == "sd-dreambooth":
        rows = _map_sd_dreambooth_components(
            effective_config,
            target,
        )
    elif train_type == "sdxl-finetune":
        rows = _map_sdxl_full_components(
            effective_config,
            target,
        )
    elif train_type == "flux-finetune":
        rows = _map_flux_full_components(
            effective_config,
            target,
        )
    elif train_type == "anima-finetune":
        rows = _map_anima_full_components(
            effective_config,
            target,
        )
    else:
        raise ValueError(
            f"Parameter Policy bootstrap: 未处理 backend {train_type!r}。"
        )

    expected = set(profile.components)
    actual = set(rows)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("extra=" + ", ".join(extra))
        raise RuntimeError(
            f"Parameter Policy bootstrap mapper for {train_type!r} returned "
            "an incomplete Component schema: " + "; ".join(details)
        )

    # Stable key ordering makes future UI/sidecar bootstrap deterministic.
    return {component_id: rows[component_id] for component_id in sorted(rows)}



def _format_compatibility_failure(
    blockers: Sequence[str],
) -> str:
    return (
        "当前 Standard 配置不能无损迁移到 Component-wise Parameter Policy v1：\n"
        + "\n".join(f"- {message}" for message in blockers)
    )


def bootstrap_parameter_policy_from_standard(
    config: Mapping[str, Any],
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> dict[str, Any]:
    """Compile Standard once and return one strict canonical v1 policy.

    Migration is exact-or-fail:
    1. compile the ordinary Standard effective configuration;
    2. reject every known incompatible optimizer/LR/grouping semantic;
    3. derive exactly one legacy_main Optimizer Profile;
    4. map the backend's complete Model Component Profile;
    5. strict-validate the resulting sidecar-shaped policy.

    No fallback Optimizer Profile is invented.
    """

    prepared = _prepare_standard_snapshot(
        config,
        page_type,
        resolve_backend=resolve_backend,
    )

    blockers = parameter_policy_compatibility_blockers(
        prepared.config,
        prepared.train_type,
    )
    if blockers:
        raise ValueError(_format_compatibility_failure(blockers))

    optimizer_profiles = bootstrap_legacy_optimizer_profile(prepared.config)
    if set(optimizer_profiles) != {LEGACY_BOOTSTRAP_PROFILE}:
        raise RuntimeError(
            "Parameter Policy bootstrap invariant failed: expected exactly "
            f"{LEGACY_BOOTSTRAP_PROFILE!r}."
        )

    components = _bootstrap_component_rows(
        prepared.config,
        prepared.train_type,
    )

    candidate = {
        "version": PARAMETER_POLICY_VERSION,
        "optimizer_profiles": optimizer_profiles,
        "components": components,
    }
    canonical = validate_parameter_policy(candidate)

    # Commit 4 must never synthesize a fallback route. Keep this invariant
    # explicit so later runtime work cannot accidentally leak into migration.
    for component_id, route in canonical["components"].items():
        if "fallback_optimizer_profile" in route or "fallback_learning_rate" in route:
            raise RuntimeError(
                "Parameter Policy bootstrap invariant failed: migration invented "
                f"a fallback optimizer for Component {component_id!r}."
            )

    return canonical


def bootstrap_parameter_policy_optimizer_profile(
    config: Mapping[str, Any],
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> dict:
    """Compile the normal Standard pipeline, then derive one initial profile.

    The public behavior of the existing profile-only bootstrap is preserved.
    Full Component assignment is added by later Commit 4 stages.
    """

    prepared = _prepare_standard_snapshot(
        config,
        page_type,
        resolve_backend=resolve_backend,
    )
    return bootstrap_legacy_optimizer_profile(prepared.config)


__all__ = [
    "bootstrap_parameter_policy_from_standard",
    "bootstrap_parameter_policy_optimizer_profile",
]
