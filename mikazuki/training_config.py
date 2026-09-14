"""Single effective-config pipeline shared by preview and launch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from mikazuki.full_trainer_contract import (
    normalize_validate_flux_full,
    normalize_validate_sdxl_full,
)


PAGE_BACKEND_MAP = {
    "lora-basic": "sd-lora",
    "lora-master": "sd-lora",
    "sdxl-lora": "sdxl-lora",
    "dreambooth": "sd-dreambooth",
    "sdxl-full": "sdxl-finetune",
    "flux-lora": "flux-lora",
    "chroma-lora": "chroma-lora",
    "anima-lora": "anima-lora",
    "flux-finetune": "flux-finetune",
    "anima-finetune": "anima-finetune",
    "sd3-lora": "sd3-lora",
}


@dataclass
class PreparedTrainingConfig:
    train_type: str
    trainer_file: str
    config: dict
    gpu_ids: Optional[list] = None
    warnings: list[str] = field(default_factory=list)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _strip_network_training_keys(config: dict) -> None:
    for key in list(config.keys()):
        if key.startswith("network_") or key in {
            "scale_weight_norms",
            "enable_base_weight",
            "base_weights",
            "base_weights_multiplier",
            "unet_lr",
            "text_encoder_lr",
        }:
            config.pop(key, None)


def _resolve_requested_backend(config: dict, page_train_type: str | None, warnings: list[str]) -> str:
    embedded = config.pop("model_train_type", None)
    requested = PAGE_BACKEND_MAP.get(page_train_type or "", page_train_type)
    if requested:
        if embedded not in (None, "", requested):
            warnings.append(f"页面固定后端为 {requested}；已忽略 model_train_type={embedded!r}。")
        return str(requested)
    return str(embedded or "sd-lora")


def _normalize_memory_mode(config: dict) -> None:
    semantic = config.pop("memory_mode", None)
    if semantic not in (None, ""):
        mode = str(semantic).lower()
        if mode not in {"auto", "lowram", "highvram"}:
            raise ValueError("memory_mode 只能是 auto / lowram / highvram。")
        config["lowram"] = mode == "lowram"
        config["highvram"] = mode == "highvram"
    if _as_bool(config.get("lowram")) and _as_bool(config.get("highvram")):
        raise ValueError("lowram 与 highvram 不能同时启用。")


def normalize_flux_lora_text_encoder(config: dict) -> None:
    enabled = _as_bool(config.pop("train_t5xxl", False))
    network_args = [str(value) for value in (config.get("network_args") or [])]
    network_args = [value for value in network_args if not value.lower().startswith("train_t5xxl=")]
    if enabled:
        if _as_bool(config.get("network_train_unet_only")):
            raise ValueError("Flux/Chroma: train_t5xxl 不能与 network_train_unet_only 同时启用。")
        if _as_bool(config.get("cache_text_encoder_outputs")) or _as_bool(config.get("cache_text_encoder_outputs_to_disk")):
            raise ValueError("Flux/Chroma: 训练 T5XXL LoRA 时不能缓存 Text Encoder outputs。")
        network_args.append("train_t5xxl=True")
    if network_args:
        config["network_args"] = network_args
    else:
        config.pop("network_args", None)


def normalize_sd_dreambooth(config: dict, warnings: list[str]) -> None:
    mode = config.pop("sd_max_token_length_mode", None)
    if mode not in (None, ""):
        mode = str(mode)
        if mode == "75":
            config.pop("max_token_length", None)
        elif mode in {"150", "225"}:
            config["max_token_length"] = int(mode)
        else:
            raise ValueError("SD1/2 max token length 只能是 75 / 150 / 225。")
    elif config.get("max_token_length") == 255:
        config["max_token_length"] = 225
        warnings.append("旧 GUI 的 max_token_length=255 不被当前 trainer 接受；已迁移为 225。")
    if str(config.get("save_model_as") or "").lower() == "pt":
        raise ValueError("SD DreamBooth: 当前 trainer 不支持 save_model_as=pt。")


def prepare_non_anima_config(
    raw_config: dict,
    *,
    page_train_type: str | None,
    resolve_backend,
) -> PreparedTrainingConfig:
    """Prepare all non-Anima jobs; Anima is delegated to its existing contract."""
    config = deepcopy(raw_config)
    warnings: list[str] = []
    gpu_ids = config.pop("gpu_ids", None)
    requested = _resolve_requested_backend(config, page_train_type, warnings)
    effective_train_type, trainer_file = resolve_backend(config, requested)
    _normalize_memory_mode(config)

    if effective_train_type in {"flux-lora", "chroma-lora"}:
        normalize_flux_lora_text_encoder(config)
    elif effective_train_type == "sd-dreambooth":
        normalize_sd_dreambooth(config, warnings)
    elif effective_train_type == "sdxl-finetune":
        if config.get("max_train_steps") not in (None, "", 0, "0") and config.get("max_train_epochs") not in (None, "", 0, "0"):
            warnings.append("SDXL: 同时设置 step 与 epoch；按页面语义采用 max_train_steps。")
        _strip_network_training_keys(config)
        normalize_validate_sdxl_full(config)
    elif effective_train_type == "flux-finetune":
        if config.get("max_train_steps") not in (None, "", 0, "0") and config.get("max_train_epochs") not in (None, "", 0, "0"):
            warnings.append("Flux: 同时设置 step 与 epoch；按页面语义采用 max_train_steps。")
        _strip_network_training_keys(config)
        normalize_validate_flux_full(config)

    config.pop("model_train_type", None)
    config.pop("anima_training_mode", None)
    return PreparedTrainingConfig(effective_train_type, trainer_file, config, gpu_ids, warnings)
