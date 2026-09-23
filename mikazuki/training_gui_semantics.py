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
    return _as_bool(config.get("cache_text_encoder_outputs")) or _as_bool(
        config.get("cache_text_encoder_outputs_to_disk")
    )


def _component_policy_active(config: dict) -> bool:
    return bool(str(config.get("parameter_policy_config") or "").strip())


def _network_arg_exact_true(config: dict, key: str) -> bool | None:
    """Mirror the LoRA modules' exact network-kwargs wire semantics.

    train_network.py preserves the raw key/value around "=", then
    networks.lora_flux / networks.lora_sd3 enable train_t5xxl only when
    kwargs["train_t5xxl"] is exactly the string "True". Case variants or
    whitespace variants are different kwargs and must remain inert.
    """

    result: bool | None = None
    for item in _items(config.get("network_args")):
        if "=" not in item:
            continue
        raw_key, raw_value = item.split("=", 1)
        if raw_key != key:
            continue
        result = raw_value == "True"
    return result


def _set_network_arg_bool(config: dict, key: str, value: bool) -> None:
    # Remove only the exact kwarg consumed by the LoRA module. Malformed
    # aliases such as TRAIN_T5XXL or "train_t5xxl " are separate kwargs in
    # sd-scripts and must not be silently corrected by DTS.
    args = []
    for item in _items(config.get("network_args")):
        if "=" in item and item.split("=", 1)[0] == key:
            continue
        args.append(item)
    if value:
        args.append(f"{key}=True")
    if args:
        config["network_args"] = args
    else:
        config.pop("network_args", None)


def _validate_lora_only_flags(config: dict, *, label: str) -> tuple[bool, bool]:
    unet_only = _as_bool(config.get("network_train_unet_only"))
    te_only = _as_bool(config.get("network_train_text_encoder_only"))
    if unet_only and te_only:
        raise ValueError(
            f"{label} 不能同时设置 network_train_unet_only 与 "
            "network_train_text_encoder_only。"
        )
    return unet_only, te_only


def normalize_sd_lora_target(
    config: dict,
    *,
    defer_component_text_encoder_cache: bool = False,
) -> None:
    """Canonicalize SD/SDXL LoRA target semantics.

    Standard mode preserves the historical cache restriction. SDXL Component
    mode may defer that one decision because target availability and
    Component Train ownership are intentionally separate concepts.
    """

    target = config.pop("lora_target", None)
    if target not in (None, ""):
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
            raise ValueError(
                "lora_target 只能是 unet / text_encoder / unet_text_encoder。"
            )

    unet_only, te_only = _validate_lora_only_flags(config, label="LoRA")
    trains_text_encoder_target = te_only or not unet_only
    if (
        trains_text_encoder_target
        and _text_encoder_cache_enabled(config)
        and not defer_component_text_encoder_cache
    ):
        raise ValueError(
            "训练 Text Encoder LoRA 时不能启用 Text Encoder output cache。"
        )


def normalize_flux_lora_target(config: dict, *, chroma: bool = False) -> None:
    """Canonicalize Flux/Chroma target + T5 semantics after every override pass."""

    target = config.pop("flux_lora_target", None)
    legacy_t5_present = "train_t5xxl" in config
    legacy_train_t5 = (
        _as_bool(config.pop("train_t5xxl"))
        if legacy_t5_present
        else None
    )

    semantic_target = target not in (None, "")
    if not semantic_target and legacy_t5_present and legacy_train_t5:
        # Legacy/custom top-level train_t5xxl=true means the target must expose
        # T5 so the LoRA network can actually create those adapters.
        target = "dit_t5xxl" if chroma else "dit_clip_l_t5xxl"
        semantic_target = True

    if semantic_target:
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
                raise ValueError(
                    "Flux LoRA target 只能是 dit / dit_clip_l / "
                    "dit_clip_l_t5xxl。"
                )
            train_t5 = target == "dit_clip_l_t5xxl"
            config["network_train_unet_only"] = target == "dit"

        config.pop("network_train_text_encoder_only", None)
        _set_network_arg_bool(config, "train_t5xxl", train_t5)
    elif legacy_t5_present:
        # ui_custom_params is last-write-wins. In particular, an explicit
        # train_t5xxl=false must be able to clear the True emitted by the first
        # semantic pass. Chroma has no CLIP branch, so false canonicalizes to
        # transformer-only.
        _set_network_arg_bool(config, "train_t5xxl", bool(legacy_train_t5))
        if chroma and not legacy_train_t5:
            config["network_train_unet_only"] = True
            config.pop("network_train_text_encoder_only", None)

    unet_only, _te_only = _validate_lora_only_flags(
        config,
        label="Flux/Chroma LoRA",
    )
    train_t5 = _network_arg_exact_true(config, "train_t5xxl")
    train_t5 = bool(train_t5) if train_t5 is not None else False

    if unet_only and train_t5:
        raise ValueError(
            "Flux/Chroma LoRA target 不一致：network_train_unet_only=true "
            "不能与 network_args train_t5xxl=True 同时使用。"
        )

    # Preserve the network modules' final exact-"True" semantics while
    # collapsing duplicates to one canonical entry.
    _set_network_arg_bool(config, "train_t5xxl", train_t5)

    if (
        not _component_policy_active(config)
        and train_t5
        and _text_encoder_cache_enabled(config)
    ):
        # Flux/SD3 trainers support partial caching while CLIP adapters train;
        # only T5 training conflicts with the cached T5 output.
        raise ValueError(
            "训练 Flux/Chroma T5XXL LoRA 时不能缓存 Text Encoder outputs。"
        )

def normalize_sd3_lora_target(config: dict) -> None:
    """Canonicalize SD3 GUI target/T5 state into the real trainer contract."""

    target = config.pop("sd3_lora_target", None)
    semantic_t5_present = "train_t5xxl" in config
    semantic_train_t5 = (
        _as_bool(config.pop("train_t5xxl"))
        if semantic_t5_present
        else None
    )

    if target not in (None, ""):
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
        _set_network_arg_bool(
            config,
            "train_t5xxl",
            bool(semantic_train_t5),
        )

    unet_only, _te_only = _validate_lora_only_flags(config, label="SD3 LoRA")
    train_t5 = _network_arg_exact_true(config, "train_t5xxl")
    train_t5 = bool(train_t5) if train_t5 is not None else False

    if unet_only and train_t5:
        raise ValueError(
            "SD3 LoRA: Train T5XXL 要求 target 包含 Text Encoder；"
            "不能与 MMDiT-only target 同时使用。"
        )

    _set_network_arg_bool(config, "train_t5xxl", train_t5)

    if (
        train_t5
        and _text_encoder_cache_enabled(config)
        and not _component_policy_active(config)
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
