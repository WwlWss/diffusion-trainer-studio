"""Fail-closed compatibility checks for Standard -> Component bootstrap.

This module answers one narrow question: can Parameter Policy v1 represent the
optimizer/LR/grouping semantics of an already-compiled Standard effective
configuration exactly?

It deliberately does not inspect model parameters, import torch, construct
optimizers/schedulers, decide optimizer runtime support, or alter request and
launch behavior. Optimizer capability/runtime support remains owned by
parameter_policy.py.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    except (TypeError, ValueError) as exc:
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


def parameter_policy_compatibility_blockers(
    effective_config: Mapping[str, Any],
    train_type: str,
) -> list[str]:
    """Return deterministic blockers for exact Standard -> Component migration."""

    if not isinstance(effective_config, Mapping):
        raise ValueError(
            "Parameter Policy compatibility: effective_config 必须是 mapping。"
        )

    # Validate against the same registry used by the pure router.
    get_model_component_profile(train_type)

    network_args = _parse_network_args(effective_config.get("network_args"))
    blockers: list[str] = []
    seen: set[str] = set()

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
            except (TypeError, ValueError) as exc:
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


__all__ = ["parameter_policy_compatibility_blockers"]
