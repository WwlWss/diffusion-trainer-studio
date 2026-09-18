"""Single effective-config pipeline shared by preview, export, import and launch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mikazuki.anima_effective_config import prepare_anima_config, validate_post_override_anima_config
from mikazuki.full_trainer_contract import normalize_validate_flux_full, normalize_validate_sdxl_full
from mikazuki.training_gui_args import normalize_numeric_fields
from mikazuki.training_gui_semantics import (
    apply_raw_gui_semantics,
    apply_ui_custom_overrides,
    normalize_adaptive_optimizer_learning_rates,
    normalize_common_dataloader,
    normalize_dataset_source,
    normalize_flux_lora_target,
    normalize_sd_lora_target,
    normalize_sd_token_length,
    validate_legacy_common_conflicts,
)

PAGE_BACKEND_MAP = {
    "lora-basic": "sd-lora", "lora-master": "sd-lora", "sd-lora": "sd-lora",
    "sdxl-lora": "sdxl-lora", "dreambooth": "sd-dreambooth", "sd-dreambooth": "sd-dreambooth",
    "sdxl-full": "sdxl-finetune", "sdxl-finetune": "sdxl-finetune", "flux-lora": "flux-lora",
    "chroma-lora": "chroma-lora", "anima-lora": "anima-lora", "flux-finetune": "flux-finetune",
    "anima-finetune": "anima-finetune", "sd3-lora": "sd3-lora",
}


@dataclass
class PreparedTrainingConfig:
    train_type: str
    trainer_file: str
    config: dict
    gpu_ids: Optional[list] = None
    warnings: list[str] = field(default_factory=list)
    sidecars: dict[str, str] = field(default_factory=dict)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _strip_network_training_keys(config: dict) -> None:
    for key in list(config.keys()):
        if key.startswith("network_") or key in {
            "scale_weight_norms", "enable_base_weight", "base_weights", "base_weights_multiplier",
            "unet_lr", "text_encoder_lr", "lora_target", "flux_lora_target",
        }:
            config.pop(key, None)


def _resolve_requested_backend(config: dict, page_train_type: str | None) -> str:
    embedded = config.pop("model_train_type", None)
    requested = PAGE_BACKEND_MAP.get(page_train_type or "", page_train_type)
    return str(requested or embedded or "sd-lora")


def _enforce_page_discriminators(config: dict, requested: str) -> None:
    if requested in {"flux-lora", "flux-finetune"}:
        config["model_type"] = "flux"
        config.pop("anima_training_mode", None)
    elif requested == "chroma-lora":
        config["model_type"] = "chroma"
        config.pop("anima_training_mode", None)
    elif requested in {"anima-lora", "anima-finetune"}:
        config["model_type"] = "anima"
        config["anima_training_mode"] = "lora" if requested == "anima-lora" else "finetune"
    else:
        config.pop("model_type", None)
        config.pop("anima_training_mode", None)


def _normalize_memory_mode(config: dict, effective_train_type: str) -> None:
    semantic = config.pop("memory_mode", None)
    if semantic not in (None, ""):
        mode = str(semantic).lower()
        allowed = {"auto", "highvram"} if effective_train_type == "anima-finetune" else {"auto", "lowram", "highvram"}
        if mode not in allowed:
            if effective_train_type == "anima-finetune" and mode == "lowram":
                raise ValueError("Anima Full 当前 trainer 尚未实现 Low RAM；请选择 Auto 或 High VRAM。")
            raise ValueError(f"memory_mode 只能是 {' / '.join(sorted(allowed))}。")
        config["lowram"] = mode == "lowram"
        config["highvram"] = mode == "highvram"
    if effective_train_type == "anima-finetune" and _as_bool(config.get("lowram")):
        raise ValueError("Anima Full 当前 trainer 尚未实现 lowram=true。")
    if _as_bool(config.get("lowram")) and _as_bool(config.get("highvram")):
        raise ValueError("lowram 与 highvram 不能同时启用。")


def _validate_final_effective_config(config: dict, effective_train_type: str) -> None:
    if _as_bool(config.get("lowram")) and _as_bool(config.get("highvram")):
        raise ValueError("ui_custom_params 产生了非法组合：lowram 与 highvram 不能同时为 true。")
    if effective_train_type == "anima-finetune" and _as_bool(config.get("lowram")):
        raise ValueError("ui_custom_params 不能为 Anima Full 开启尚未实现的 lowram。")
    workers = config.get("max_data_loader_n_workers")
    if workers not in (None, "") and int(workers) == 0 and _as_bool(config.get("persistent_data_loader_workers")):
        config["persistent_data_loader_workers"] = False


def _post_override_normalize(
    config: dict,
    effective_train_type: str,
    warnings: list[str],
) -> None:
    """Re-assert all trainer invariants after ui_custom_params has overwritten values."""
    # ui_custom_params is intentionally last-write-wins, but it must obey the
    # same type contract as ordinary GUI fields. sd-scripts seeds argparse from
    # TOML values directly, so quoted scientific-notation strings must be
    # coerced here rather than relying on argparse type=... conversions.
    normalize_numeric_fields(config)
    normalize_dataset_source(config)
    normalize_common_dataloader(config)
    _normalize_memory_mode(config, effective_train_type)

    if effective_train_type in {"sd-lora", "sdxl-lora"}:
        normalize_sd_token_length(config, warnings)
        normalize_sd_lora_target(config)
    elif effective_train_type == "sd-dreambooth":
        normalize_sd_token_length(config, warnings)
        if str(config.get("save_model_as") or "").lower() == "pt":
            raise ValueError("SD DreamBooth: 当前 trainer 不支持 save_model_as=pt。")
    elif effective_train_type in {"flux-lora", "chroma-lora"}:
        normalize_flux_lora_target(config, chroma=effective_train_type == "chroma-lora")
    elif effective_train_type == "sdxl-finetune":
        _strip_network_training_keys(config)
        normalize_validate_sdxl_full(config)
    elif effective_train_type == "flux-finetune":
        _strip_network_training_keys(config)
        normalize_validate_flux_full(config)
    elif effective_train_type in {"anima-lora", "anima-finetune"}:
        validate_post_override_anima_config(config, effective_train_type)

    normalize_adaptive_optimizer_learning_rates(config, warnings)
    validate_legacy_common_conflicts(config, effective_train_type)
    _validate_final_effective_config(config, effective_train_type)


def prepare_training_config(
    raw_config: dict, *, page_train_type: str | None, resolve_backend,
    launch: bool = False, toml_path: str | None = None,
) -> PreparedTrainingConfig:
    """Compile raw GUI state into the one effective config used everywhere."""
    config = apply_raw_gui_semantics(raw_config, page_train_type=page_train_type)
    warnings: list[str] = []
    gpu_ids = config.pop("gpu_ids", None)
    requested = _resolve_requested_backend(config, page_train_type)
    _enforce_page_discriminators(config, requested)
    effective_train_type, trainer_file = resolve_backend(config, requested)

    _normalize_memory_mode(config, effective_train_type)
    normalize_dataset_source(config)
    normalize_common_dataloader(config)

    if effective_train_type in {"sd-lora", "sdxl-lora"}:
        normalize_sd_token_length(config, warnings)
        normalize_sd_lora_target(config)
        normalize_adaptive_optimizer_learning_rates(config, warnings)
    elif effective_train_type == "sd-dreambooth":
        normalize_sd_token_length(config, warnings)
        normalize_adaptive_optimizer_learning_rates(config, warnings)
        if str(config.get("save_model_as") or "").lower() == "pt":
            raise ValueError("SD DreamBooth: 当前 trainer 不支持 save_model_as=pt。")
    elif effective_train_type in {"flux-lora", "chroma-lora"}:
        normalize_flux_lora_target(config, chroma=effective_train_type == "chroma-lora")
        normalize_adaptive_optimizer_learning_rates(config, warnings)
    elif effective_train_type == "sdxl-finetune":
        if config.get("max_train_steps") not in (None, "", 0, "0") and config.get("max_train_epochs") not in (None, "", 0, "0"):
            warnings.append("SDXL: 同时设置 step 与 epoch；按页面语义采用 max_train_steps。")
        _strip_network_training_keys(config)
        normalize_adaptive_optimizer_learning_rates(config, warnings)
        normalize_validate_sdxl_full(config)
    elif effective_train_type == "flux-finetune":
        if config.get("max_train_steps") not in (None, "", 0, "0") and config.get("max_train_epochs") not in (None, "", 0, "0"):
            warnings.append("Flux: 同时设置 step 与 epoch；按页面语义采用 max_train_steps。")
        _strip_network_training_keys(config)
        normalize_adaptive_optimizer_learning_rates(config, warnings)
        normalize_validate_flux_full(config)
    elif effective_train_type in {"anima-lora", "anima-finetune"}:
        prepare_anima_config(config, effective_train_type, trainer_file, launch=launch, toml_path=toml_path)
        normalize_adaptive_optimizer_learning_rates(config, warnings)
        config.pop("model_type", None)

    # Historical ui_custom_params behavior is intentionally last-write-wins for
    # trainer values, but it may not bypass page routing or trainer invariants.
    apply_ui_custom_overrides(config)
    _post_override_normalize(config, effective_train_type, warnings)

    config.pop("model_train_type", None)
    config.pop("anima_training_mode", None)
    config.pop("dataset_source", None)
    if effective_train_type in {"flux-lora", "flux-finetune"}:
        config["model_type"] = "flux"
    elif effective_train_type == "chroma-lora":
        config["model_type"] = "chroma"
    else:
        config.pop("model_type", None)
    # Two normalization passes may discover the same informational warning.
    warnings[:] = list(dict.fromkeys(warnings))
    return PreparedTrainingConfig(effective_train_type, trainer_file, config, gpu_ids, warnings)


def prepare_non_anima_config(raw_config: dict, *, page_train_type: str | None, resolve_backend) -> PreparedTrainingConfig:
    return prepare_training_config(raw_config, page_train_type=page_train_type, resolve_backend=resolve_backend, launch=False)
