"""Fail-closed host compatibility checks for Parameter Policy v1.

Two layers intentionally live here:

* bootstrap semantics: whether an already-compiled Standard configuration can
  be represented losslessly by the Component Policy schema;
* runtime semantics: whether an existing Component policy is qualified for the
  requested trainer/runtime mode.

Neither layer inspects model parameters, imports torch, constructs optimizers
or schedulers, or mutates request/launch state. Non-Anima editor bootstrap uses
only representability blockers; Anima keeps its existing stricter bootstrap
gate. Runtime Preview/Start always uses the full v1 semantic blocker set.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from mikazuki.model_component_profiles import get_model_component_profile


_LORA_NETWORK_MODULES = {
    "sd-lora": "networks.lora",
    "sdxl-lora": "networks.lora",
    "flux-lora": "networks.lora_flux",
    "chroma-lora": "networks.lora_flux",
    "sd3-lora": "networks.lora_sd3",
    "anima-lora": "networks.lora_anima",
}

_LORAPLUS_KEYS = frozenset(
    {
        "loraplus_lr_ratio",
        "loraplus_unet_lr_ratio",
        "loraplus_text_encoder_lr_ratio",
    }
)
_SD_BLOCK_LR_KEYS = frozenset({"down_lr_weight", "mid_lr_weight", "up_lr_weight"})
_REGEX_LR_BACKENDS = frozenset({"flux-lora", "chroma-lora", "anima-lora"})
_CACHED_TE_PRELOAD_UNSAFE_LORA_BACKENDS = frozenset(
    {"sdxl-lora", "flux-lora", "chroma-lora", "sd3-lora", "anima-lora"}
)


def _as_bool(value: object, *, field: str) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是布尔值或 0/1，收到 {value!r}。"
        )
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(
        f"Parameter Policy compatibility: {field} 必须是布尔值或 0/1，收到 {value!r}。"
    )


def _positive_count(value: object, *, field: str) -> bool:
    if value in (None, "", 0, 0.0, "0", "0.0"):
        return False
    if isinstance(value, bool):
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是非负整数，收到 {value!r}。"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是非负整数，收到 {value!r}。"
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是非负整数，收到 {value!r}。"
        )
    if isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Parameter Policy compatibility: {field} 必须是非负整数，收到 {value!r}。"
            ) from exc
        if not numeric.is_integer():
            raise ValueError(
                f"Parameter Policy compatibility: {field} 必须是非负整数，收到 {value!r}。"
            )
    if parsed < 0:
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是非负整数，收到 {value!r}。"
        )
    return parsed > 0


def _positive_float(value: object, *, field: str) -> bool:
    if value in (None, "", 0, 0.0, "0", "0.0"):
        return False
    if isinstance(value, bool):
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是非负数值，收到 {value!r}。"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是非负数值，收到 {value!r}。"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是有限的非负数值，收到 {value!r}。"
        )
    return parsed > 0


def _nonempty_path(value: object, *, field: str) -> bool:
    if value in (None, ""):
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Parameter Policy compatibility: {field} 必须是字符串路径，收到 {value!r}。"
        )
    return bool(value.strip())


def _nonempty_paths(value: object, *, field: str) -> bool:
    if value in (None, "", []):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        found = False
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"Parameter Policy compatibility: {field} 必须是字符串路径或字符串路径列表，收到 {value!r}。"
                )
            found = found or bool(item.strip())
        return found
    raise ValueError(
        f"Parameter Policy compatibility: {field} 必须是字符串路径或字符串路径列表，收到 {value!r}。"
    )


def _parse_network_args(raw_args: object) -> dict[str, str]:
    """Parse normalized network_args by exact key, with later duplicates winning."""

    if raw_args in (None, "", []):
        return {}

    if isinstance(raw_args, str):
        items = [line.strip() for line in raw_args.splitlines() if line.strip()]
    elif isinstance(raw_args, (list, tuple)):
        items = []
        for item in raw_args:
            if not isinstance(item, str):
                raise ValueError(
                    "Parameter Policy compatibility: network_args 必须是字符串或字符串列表。"
                )
            stripped = item.strip()
            if stripped:
                items.append(stripped)
    else:
        raise ValueError(
            "Parameter Policy compatibility: network_args 必须是字符串或字符串列表。"
        )

    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Parameter Policy compatibility: network_args 项缺少 '=': {item!r}。"
            )
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip().casefold()
        if not key:
            raise ValueError(
                f"Parameter Policy compatibility: network_args key 不能为空: {item!r}。"
            )
        parsed[key] = raw_value.strip()
    return parsed


def _append_once(blockers: list[str], seen: set[str], message: str) -> None:
    if message in seen:
        return
    seen.add(message)
    blockers.append(message)


def parameter_policy_v1_semantic_blockers(
    effective_config: Mapping[str, Any],
    train_type: str,
) -> list[str]:
    """Return the full v1 runtime-qualification blocker set."""

    if not isinstance(effective_config, Mapping):
        raise ValueError(
            "Parameter Policy compatibility: effective_config 必须是 mapping。"
        )

    # Validate against the same registry used by the pure router.
    get_model_component_profile(train_type)

    network_args = _parse_network_args(effective_config.get("network_args"))
    blockers: list[str] = []
    seen: set[str] = set()

    if train_type in {"anima-lora", "anima-finetune"} and _as_bool(
        effective_config.get("compile"),
        field="compile",
    ):
        _append_once(
            blockers,
            seen,
            "Anima compile 会对 DiT block 应用独立 torch.compile 包装；"
            "Component-wise v1 尚未验证 adapter/parameter identity、CompositeOptimizer "
            "与 checkpoint save/resume 语义。",
        )

    if _as_bool(
        effective_config.get("torch_compile"),
        field="torch_compile",
    ):
        _append_once(
            blockers,
            seen,
            "torch_compile 会通过 Accelerate Dynamo 包装训练模型；Component-wise v1 "
            "尚未验证 compiled model 下的参数 identity、CompositeOptimizer 与 checkpoint "
            "save/resume 语义。",
        )

    for field, label in (
        ("full_fp16", "full FP16"),
        ("full_bf16", "full BF16"),
    ):
        if _as_bool(effective_config.get(field), field=field):
            _append_once(
                blockers,
                seen,
                f"{field} 会把训练模型/梯度切换为 {label} runtime；"
                "Component-wise v1 尚未完成真实 CUDA 下的 CompositeOptimizer、"
                "GradScaler/Accelerate prepare 与 checkpoint save/resume 验证。"
                "当前请使用普通 mixed_precision 模式。",
            )

    for field in ("fp8_base", "fp8_base_unet"):
        if _as_bool(effective_config.get(field), field=field):
            _append_once(
                blockers,
                seen,
                f"{field} 会改变基础模型/训练模块的 FP8 dtype 与设备路径；"
                "Component-wise v1 尚未完成真实 CUDA 下的 parameter identity、"
                "optimizer ownership 与 checkpoint save/resume 验证。"
                "当前请关闭 FP8 base 模式。",
            )

    if _as_bool(
        effective_config.get("fused_backward_pass"),
        field="fused_backward_pass",
    ):
        _append_once(
            blockers,
            seen,
            "fused_backward_pass 会改变 optimizer step/ownership 语义；"
            "Component-wise v1 尚未实现等价的 fused-backward runtime。",
        )

    if _positive_count(
        effective_config.get("fused_optimizer_groups"),
        field="fused_optimizer_groups",
    ):
        _append_once(
            blockers,
            seen,
            "fused_optimizer_groups 会创建并独立 step 多个 optimizer；"
            "Component-wise v1 尚未实现等价的多 optimizer runtime。",
        )

    if _as_bool(
        effective_config.get("blockwise_fused_optimizers"),
        field="blockwise_fused_optimizers",
    ):
        _append_once(
            blockers,
            seen,
            "blockwise_fused_optimizers 会按 block 创建/step optimizer；"
            "Component-wise v1 尚未实现等价的 blockwise fused runtime。",
        )

    if _as_bool(effective_config.get("deepspeed"), field="deepspeed"):
        _append_once(
            blockers,
            seen,
            "DeepSpeed 会接管 optimizer/distributed ownership；"
            "Component-wise v1 尚未完成 CompositeOptimizer + DeepSpeed 兼容验证。",
        )

    for field in ("cpu_offload_checkpointing", "unsloth_offload_checkpointing"):
        if _as_bool(effective_config.get(field), field=field):
            _append_once(
                blockers,
                seen,
                f"{field} 会改变参数驻留/反向图 ownership；"
                "Component-wise v1 尚未完成对应 device/offload runtime 验证。",
            )

    for field in ("blocks_to_swap", "double_blocks_to_swap", "single_blocks_to_swap"):
        if _positive_count(effective_config.get(field), field=field):
            _append_once(
                blockers,
                seen,
                f"{field} 会在训练期间动态迁移模型 block；"
                "Component-wise v1 尚未完成 optimizer parameter device 审计。",
            )

    if (
        train_type in _CACHED_TE_PRELOAD_UNSAFE_LORA_BACKENDS
        and _nonempty_path(
            effective_config.get("network_weights"),
            field="network_weights",
        )
        and (
            _as_bool(
                effective_config.get("cache_text_encoder_outputs"),
                field="cache_text_encoder_outputs",
            )
            or _as_bool(
                effective_config.get("cache_text_encoder_outputs_to_disk"),
                field="cache_text_encoder_outputs_to_disk",
            )
        )
    ):
        _append_once(
            blockers,
            seen,
            "Component-wise v1 不能同时使用预载 network_weights 与 Text Encoder "
            "输出缓存：缓存会在已有 Text Encoder adapter 权重应用前生成，导致 policy "
            "冻结的非零 adapter 不参与实际 conditioning。请关闭 "
            "cache_text_encoder_outputs/cache_text_encoder_outputs_to_disk，"
            "或不要预载 network_weights。",
        )

    if (
        train_type == "anima-lora"
        and _nonempty_paths(effective_config.get("base_weights"), field="base_weights")
        and (
            _as_bool(
                effective_config.get("cache_text_encoder_outputs"),
                field="cache_text_encoder_outputs",
            )
            or _as_bool(
                effective_config.get("cache_text_encoder_outputs_to_disk"),
                field="cache_text_encoder_outputs_to_disk",
            )
        )
    ):
        _append_once(
            blockers,
            seen,
            "Component-wise Anima LoRA 不能同时使用 base_weights 与 Text Encoder "
            "输出缓存：base_weights 会在缓存生成后 merge 到 Qwen3/DiT，包含 lora_te "
            "权重时会使实际 conditioning 与最终模型状态不一致。请关闭 "
            "cache_text_encoder_outputs/cache_text_encoder_outputs_to_disk，"
            "或不要使用 base_weights。",
        )

    if train_type in _LORA_NETWORK_MODULES and _positive_float(
        effective_config.get("scale_weight_norms"),
        field="scale_weight_norms",
    ):
        _append_once(
            blockers,
            seen,
            "scale_weight_norms 会直接修改 LoRA state_dict 中的 adapter 权重，"
            "包括 Parameter Policy 已冻结的组件；Component-wise v1 尚未实现"
            "仅对 policy-owned trainable adapter 应用 max-norm regularization。",
        )

    # Encoding/grouping semantics that Component Policy v1 cannot represent are
    # owned by the bootstrap helper below. Runtime preflight reuses that exact
    # source of truth instead of maintaining a second copy.
    for message in parameter_policy_bootstrap_semantic_blockers(
        effective_config,
        train_type,
        _network_args=network_args,
    ):
        _append_once(blockers, seen, message)

    return blockers


def parameter_policy_bootstrap_semantic_blockers(
    effective_config: Mapping[str, Any],
    train_type: str,
    *,
    _network_args: Mapping[str, str] | None = None,
) -> list[str]:
    """Return only Standard semantics that Component Policy v1 cannot encode.

    Runtime qualification modes such as FP8/full-precision training, DeepSpeed,
    compile, offload, fused optimizers, and block swapping are deliberately not
    bootstrap blockers.  They remain fail-closed in
    :func:`parameter_policy_v1_semantic_blockers` after the Component policy
    has been created, so the editor can surface an actionable Runtime blocker
    without silently rewriting the user's Standard configuration.
    """

    if not isinstance(effective_config, Mapping):
        raise ValueError(
            "Parameter Policy compatibility: effective_config 必须是 mapping。"
        )

    # Keep the same registry validation as runtime preflight.
    get_model_component_profile(train_type)

    network_args = (
        dict(_network_args)
        if _network_args is not None
        else _parse_network_args(effective_config.get("network_args"))
    )
    blockers: list[str] = []
    seen: set[str] = set()

    if train_type == "sdxl-finetune" and effective_config.get("block_lr") not in (
        None,
        "",
    ):
        _append_once(
            blockers,
            seen,
            "SDXL Full 的 block_lr 使用 23 组 U-Net block 学习率；"
            "当前 Component schema 不能无损表示该分组。",
        )

    if train_type == "sd-dreambooth":
        stop = effective_config.get("stop_text_encoder_training")
        if stop not in (None, ""):
            if isinstance(stop, bool):
                raise ValueError(
                    "Parameter Policy compatibility: "
                    "stop_text_encoder_training 必须是整数 step。"
                )
            try:
                stop_step = int(stop)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Parameter Policy compatibility: "
                    "stop_text_encoder_training 必须是整数 step。"
                ) from exc
            if isinstance(stop, float) and not stop.is_integer():
                raise ValueError(
                    "Parameter Policy compatibility: "
                    "stop_text_encoder_training 必须是整数 step。"
                )
            if isinstance(stop, str):
                try:
                    if not float(stop.strip()).is_integer():
                        raise ValueError
                except ValueError as exc:
                    raise ValueError(
                        "Parameter Policy compatibility: "
                        "stop_text_encoder_training 必须是整数 step。"
                    ) from exc
            if stop_step >= 0:
                _append_once(
                    blockers,
                    seen,
                    "SD DreamBooth 的 stop_text_encoder_training 会在训练中途冻结 "
                    "Text Encoder；静态 Component Train=true/false 不能无损表示该时序语义。",
                )

    if train_type in {"sd-lora", "sdxl-lora"} and (
        _SD_BLOCK_LR_KEYS.intersection(network_args)
    ):
        _append_once(
            blockers,
            seen,
            "SD/SDXL LoRA block LR weighting 会按 U-Net block 重写 adapter 学习率；"
            "当前 Component schema 不能无损表示该分组。",
        )

    if train_type in _LORA_NETWORK_MODULES and _LORAPLUS_KEYS.intersection(network_args):
        _append_once(
            blockers,
            seen,
            "LoRA+ 会将 lora_up 参数拆成独立学习率组；"
            "Component-wise v1 尚未拥有该参数级 LR grouping 语义。",
        )

    if train_type in _REGEX_LR_BACKENDS and "network_reg_lrs" in network_args:
        _append_once(
            blockers,
            seen,
            "network_reg_lrs 会按正则匹配创建独立 adapter 学习率组；"
            "Component-wise v1 不能无损表示该 regex-specific LR 语义。",
        )

    canonical_module = _LORA_NETWORK_MODULES.get(train_type)
    if canonical_module is not None:
        raw_module = effective_config.get("network_module")
        module = str(raw_module).strip() if raw_module not in (None, "") else ""
        if module and module != canonical_module:
            _append_once(
                blockers,
                seen,
                f"{train_type} 当前显式使用未审查的 network_module={module!r}；"
                f"Standard -> Component bootstrap 只验证过 {canonical_module!r} "
                "的 optimizer-group 语义。",
            )

    return blockers


def parameter_policy_compatibility_blockers(
    effective_config: Mapping[str, Any],
    train_type: str,
) -> list[str]:
    """Standard -> Component bootstrap gate.

    Non-Anima backends reject only semantics the v1 policy cannot represent.
    Runtime-only qualification modes are intentionally deferred to Preview /
    Start so switching modes never rewrites the user's trainer configuration.

    Anima keeps its existing stricter bootstrap behavior in this hardening
    branch; its already-qualified GUI/runtime flow is intentionally unchanged.
    """

    normalized_train_type = str(train_type or "").strip().lower()
    if normalized_train_type in {"anima-lora", "anima-finetune"}:
        return parameter_policy_v1_semantic_blockers(
            effective_config,
            normalized_train_type,
        )
    return parameter_policy_bootstrap_semantic_blockers(
        effective_config,
        normalized_train_type,
    )


__all__ = [
    "parameter_policy_bootstrap_semantic_blockers",
    "parameter_policy_compatibility_blockers",
    "parameter_policy_v1_semantic_blockers",
]
