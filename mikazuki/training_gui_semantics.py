"""Trainer semantic normalization owned by the Python backend.

This module is the single semantic layer between raw GUI state and trainer
arguments. The legacy frontend must never run parseParams()/checkParams() first.
"""

from __future__ import annotations

from mikazuki.training_gui_args import (
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
    """Materialize only the LR rewrite the legacy GUI actually performed.

    D-Adaptation rewrote all active learning rates to 1.0. Prodigy did not: the
    old UI merely warned when component LRs were not 1 and generated optimizer
    args. Keeping those behaviors distinct avoids silently changing existing
    Prodigy recipes while still moving the former frontend semantics to Python.
    """
    optimizer = str(config.get("optimizer_type") or "")
    lower = optimizer.lower()
    if not lower.startswith(DADAPT_PREFIX):
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
    elif "max_token_length" in config and config.get("max_token_length") not in (None, "", 150, 225):
        value = config.get("max_token_length")
        if value == 75:
            config.pop("max_token_length", None)
        else:
            raise ValueError("SD/SDXL max_token_length 只能是 75 / 150 / 225（75 在 TOML 中省略）。")


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
    if str(source) == "folder":
        config.pop("dataset_config", None)
        has_config = False
    if has_config:
        for key in ("train_data_dir", "reg_data_dir", "in_json"):
            config.pop(key, None)
    else:
        config.pop("dataset_config", None)


def _text_encoder_cache_enabled(config: dict) -> bool:
    return _as_bool(config.get("cache_text_encoder_outputs")) or _as_bool(config.get("cache_text_encoder_outputs_to_disk"))


def normalize_sd_lora_target(config: dict) -> None:
    target = config.pop("lora_target", None)
    if target in (None, ""):
        unet_only = _as_bool(config.get("network_train_unet_only"))
        te_only = _as_bool(config.get("network_train_text_encoder_only"))
        if unet_only and te_only:
            raise ValueError("LoRA 不能同时设置 network_train_unet_only 与 network_train_text_encoder_only。")
        # Both false/absent means joint U-Net + TE training in sd-scripts.
        if (te_only or not unet_only) and _text_encoder_cache_enabled(config):
            raise ValueError("训练 Text Encoder LoRA 时不能启用 Text Encoder output cache。")
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

    if target != "unet" and _text_encoder_cache_enabled(config):
        raise ValueError("训练 Text Encoder LoRA 时不能启用 Text Encoder output cache。")


def normalize_flux_lora_target(config: dict, *, chroma: bool = False) -> None:
    target = config.pop("flux_lora_target", None)
    legacy_train_t5 = _as_bool(config.pop("train_t5xxl", False))
    if target in (None, ""):
        target = "dit_t5xxl" if chroma and legacy_train_t5 else (
            "dit_clip_l_t5xxl" if legacy_train_t5 else None
        )
    if target is None:
        # Effective legacy/raw trainer state: false means at least CLIP-L is trainable.
        if not _as_bool(config.get("network_train_unet_only")) and _text_encoder_cache_enabled(config):
            raise ValueError("训练 Flux/Chroma Text Encoder LoRA 时不能缓存 Text Encoder outputs。")
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

    if train_t5 and _text_encoder_cache_enabled(config):
        raise ValueError("训练 Flux/Chroma Text Encoder LoRA 时不能缓存 Text Encoder outputs。")

    args = [item for item in _items(config.get("network_args")) if _arg_key(item) != "train_t5xxl"]
    if train_t5:
        args.append("train_t5xxl=True")
    if args:
        config["network_args"] = args
    else:
        config.pop("network_args", None)


def normalize_sd3_lora_target(config: dict) -> None:
    """Compile SD3 GUI target controls into the exact LoRA trainer contract.

    sd3_lora_target owns the mutually-exclusive MMDiT/Text Encoder target
    choice. train_t5xxl is materialized into network_args because
    networks.lora_sd3 reads it from network kwargs rather than argparse.
    Existing effective/rehydrated configs without semantic GUI fields are left
    intact apart from validating impossible legacy flag combinations.
    """

    target = config.pop("sd3_lora_target", None)
    semantic_t5_present = "train_t5xxl" in config
    semantic_train_t5 = (
        _as_bool(config.pop("train_t5xxl"))
        if semantic_t5_present
        else None
    )

    if target in (None, ""):
        unet_only = _as_bool(config.get("network_train_unet_only"))
        te_only = _as_bool(config.get("network_train_text_encoder_only"))
        if unet_only and te_only:
            raise ValueError(
                "SD3 LoRA 不能同时设置 network_train_unet_only 与 "
                "network_train_text_encoder_only。"
            )
    else:
        target = str(target)
        if target == "mmdit":
            config["network_train_unet_only"] = True
            config.pop("network_train_text_encoder_only", None)
        elif target == "text_encoder":
            config.pop("network_train_unet_only", None)
            config["network_train_text_encoder_only"] = True
        elif target == "mmdit_text_encoder":
            config.pop("network_train_unet_only", None)
            config.pop("network_train_text_encoder_only", None)
        else:
            raise ValueError(
                "sd3_lora_target 只能是 mmdit / text_encoder / "
                "mmdit_text_encoder。"
            )

    if semantic_t5_present:
        args = [
            item
            for item in _items(config.get("network_args"))
            if _arg_key(item) != "train_t5xxl"
        ]
        if semantic_train_t5:
            args.append("train_t5xxl=True")
        if args:
            config["network_args"] = args
        else:
            config.pop("network_args", None)

        # Standard mode follows the trainer's existing restriction. Component
        # mode defers to policy-aware host preflight because a T5 adapter may
        # exist for checkpoint/target purposes while remaining Train=false.
        if (
            semantic_train_t5
            and _text_encoder_cache_enabled(config)
            and config.get("parameter_policy_config") in (None, "")
        ):
            raise ValueError(
                "训练 SD3 T5XXL LoRA 时不能缓存 Text Encoder outputs。"
            )


def validate_legacy_common_conflicts(config: dict, effective_train_type: str) -> None:
    """Preserve checkParams() invariants after removing frontend validation."""
    if config.get("noise_offset") not in (None, "", 0, 0.0) and config.get("multires_noise_iterations") not in (None, "", 0, 0.0):
        raise ValueError("noise_offset 与 multires_noise_iterations 不能同时启用。")

    latent_cache = _as_bool(config.get("cache_latents")) or _as_bool(config.get("cache_latents_to_disk"))
    if latent_cache and (_as_bool(config.get("color_aug")) or _as_bool(config.get("random_crop"))):
        raise ValueError("latent cache 不能与 color_aug / random_crop 同时启用。")

    if _text_encoder_cache_enabled(config) and _as_bool(config.get("shuffle_caption")):
        raise ValueError("Text Encoder output cache 不能与 shuffle_caption 同时启用。")

    if str(config.get("network_module") or "") == "networks.oft" and effective_train_type != "sdxl-lora":
        raise ValueError("OFT 当前仅支持 SDXL LoRA。")


# Backward-compatible import surface for callers/tests.
from mikazuki.training_rehydrate import rehydrate_trainer_config
