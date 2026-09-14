"""Trainer semantic normalization owned by the Python backend.

This module is the single semantic layer between raw GUI state and trainer
arguments.  The legacy frontend must never run parseParams() first.
"""

from __future__ import annotations

from mikazuki.training_gui_args import (
    PRODIGY_TYPES,
    _arg_key,
    _as_bool,
    _is_empty,
    _items,
    apply_raw_gui_semantics,
)


DADAPT_PREFIX = "dadapt"


def apply_ui_custom_overrides(config: dict) -> None:
    overrides = config.pop("__ui_custom_overrides", None)
    if not overrides:
        return
    config.update(overrides)


def normalize_adaptive_optimizer_learning_rates(config: dict, warnings: list[str]) -> None:
    """Materialize the LR convention expected by D-Adaptation/Prodigy.

    The old frontend rewrote these values to 1.0. Doing it here prevents a raw
    semantic LR (for example Anima full's 1e-5 control) from conflicting with a
    second frontend-generated ``learning_rate=1`` value.
    """
    optimizer = str(config.get("optimizer_type") or "")
    lower = optimizer.lower()
    if not (lower.startswith(DADAPT_PREFIX) or lower in PRODIGY_TYPES):
        return

    changed = []
    for field in (
        "learning_rate",
        "unet_lr",
        "text_encoder_lr",
        "self_attn_lr",
        "cross_attn_lr",
        "mlp_lr",
        "mod_lr",
        "llm_adapter_lr",
    ):
        if field in config and not _is_empty(config[field]) and str(config[field]) not in {"0", "0.0", "1", "1.0"}:
            changed.append(field)
        if field in config and not _is_empty(config[field]) and str(config[field]) not in {"0", "0.0"}:
            config[field] = 1.0
    if "learning_rate" not in config:
        config["learning_rate"] = 1.0
    if changed:
        warnings.append(
            f"{optimizer} 按 trainer 约定使用 learning rate=1；已在后端统一规范化，不再由前端重写。"
        )


def normalize_sd_token_length(config: dict, warnings: list[str]) -> None:
    mode = config.pop("sd_max_token_length_mode", None)
    if mode not in (None, ""):
        mode = str(mode)
        if mode == "75":
            config.pop("max_token_length", None)
        elif mode in {"150", "225"}:
            config["max_token_length"] = int(mode)
        else:
            raise ValueError("SD/SDXL max token length 只能是 75 / 150 / 225。")
    elif config.get("max_token_length") == 255:
        config["max_token_length"] = 225
        warnings.append("旧 GUI 的 max_token_length=255 不被 trainer 接受；已迁移为 225。")


def normalize_common_dataloader(config: dict) -> None:
    workers = config.get("max_data_loader_n_workers")
    if workers in (None, ""):
        return
    try:
        workers_i = int(workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_data_loader_n_workers 必须是非负整数。") from exc
    if workers_i < 0:
        raise ValueError("max_data_loader_n_workers 不能小于 0。")
    config["max_data_loader_n_workers"] = workers_i
    if workers_i == 0:
        config["persistent_data_loader_workers"] = False


def normalize_dataset_source(config: dict) -> None:
    source = config.pop("dataset_source", None)
    has_config = bool(config.get("dataset_config"))
    if source not in (None, "") and str(source) not in {"folder", "config"}:
        raise ValueError("dataset_source 只能是 folder / config。")
    if str(source) == "config" and not has_config:
        raise ValueError("dataset_source=config 时必须填写 dataset_config。")
    if has_config:
        # dataset_config owns dataset/subset layout. Folder/metadata fields must
        # not silently leak into the trainer and change its source selection.
        for key in ("train_data_dir", "reg_data_dir", "in_json"):
            config.pop(key, None)
    else:
        config.pop("dataset_config", None)


def normalize_sd_lora_target(config: dict) -> None:
    target = config.pop("lora_target", None)
    if target in (None, ""):
        unet_only = _as_bool(config.get("network_train_unet_only"))
        te_only = _as_bool(config.get("network_train_text_encoder_only"))
        if unet_only and te_only:
            raise ValueError("LoRA 不能同时设置 network_train_unet_only 与 network_train_text_encoder_only。")
        return

    target = str(target)
    if target == "unet":
        config["network_train_unet_only"] = True
        config.pop("network_train_text_encoder_only", None)
    elif target == "text_encoder":
        config.pop("network_train_unet_only", None)
        config["network_train_text_encoder_only"] = True
    elif target == "unet_text_encoder":
        config.pop("network_train_unet_only", None)
        config.pop("network_train_text_encoder_only", None)
    else:
        raise ValueError("lora_target 只能是 unet / text_encoder / unet_text_encoder。")

    if target != "unet" and (
        _as_bool(config.get("cache_text_encoder_outputs"))
        or _as_bool(config.get("cache_text_encoder_outputs_to_disk"))
    ):
        raise ValueError("训练 Text Encoder LoRA 时不能启用 Text Encoder output cache。")


def normalize_flux_lora_target(config: dict, *, chroma: bool = False) -> None:
    target = config.pop("flux_lora_target", None)
    legacy_train_t5 = _as_bool(config.pop("train_t5xxl", False))
    if target in (None, ""):
        # Backward-compatible legacy state.
        target = "dit_t5xxl" if chroma and legacy_train_t5 else (
            "dit_clip_l_t5xxl" if legacy_train_t5 else None
        )
    if target is None:
        return

    target = str(target)
    if chroma:
        allowed = {"dit", "dit_t5xxl"}
        if target not in allowed:
            raise ValueError("Chroma LoRA target 只能是 dit / dit_t5xxl。")
        train_t5 = target == "dit_t5xxl"
        config["network_train_unet_only"] = not train_t5
    else:
        allowed = {"dit", "dit_clip_l", "dit_clip_l_t5xxl"}
        if target not in allowed:
            raise ValueError("Flux LoRA target 只能是 dit / dit_clip_l / dit_clip_l_t5xxl。")
        train_t5 = target == "dit_clip_l_t5xxl"
        config["network_train_unet_only"] = target == "dit"

    if train_t5 and (
        _as_bool(config.get("cache_text_encoder_outputs"))
        or _as_bool(config.get("cache_text_encoder_outputs_to_disk"))
    ):
        raise ValueError("训练 Flux/Chroma Text Encoder LoRA 时不能缓存 Text Encoder outputs。")

    args = [item for item in _items(config.get("network_args")) if _arg_key(item) != "train_t5xxl"]
    if train_t5:
        args.append("train_t5xxl=True")
    if args:
        config["network_args"] = args
    else:
        config.pop("network_args", None)


# Backward-compatible import surface for callers/tests.
from mikazuki.training_rehydrate import rehydrate_trainer_config
