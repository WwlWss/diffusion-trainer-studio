"""Anima sd-scripts UI parity normalization.

This module contains only Anima-specific semantic controls added on top of the
stable v2 training pipeline.  It deliberately avoids changing non-Anima pages.
Every GUI-only field is converted into a real sd-scripts argument before TOML
serialization; fields that would otherwise be ineffective are rejected or
removed rather than silently written to the config.
"""

from __future__ import annotations

from typing import Iterable


_LORA_ONLY_GUI_KEYS = {
    "anima_lora_checkpoint_mode",
    "anima_lora_compile_mode",
    "anima_lora_custom_optimizer_type",
    "anima_lora_custom_lr_scheduler_type",
    "anima_lora_train_llm_adapter",
    "anima_lora_include_patterns",
    "anima_lora_exclude_patterns",
    "anima_lora_rank_dropout",
    "anima_lora_module_dropout",
    "anima_lora_network_reg_dims",
    "anima_lora_network_reg_lrs",
    "anima_lora_loraplus_lr_ratio",
    "anima_lora_loraplus_unet_lr_ratio",
    "anima_lora_loraplus_text_encoder_lr_ratio",
    "anima_lora_network_verbose",
}

_FULL_ONLY_GUI_KEYS = {
    # Full finetune already owns its optimizer/scheduler/checkpoint semantic
    # controls in anima_finetune_config.py.  This set exists only so stale LoRA
    # parity fields cannot leak across pages through the legacy form state.
}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _items(value: object) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _arg_key(value: str) -> str:
    return value.split("=", 1)[0].strip().lower()


def _merge_network_args(existing: object, generated: Iterable[str]) -> list[str]:
    """Merge generated Anima LoRA args without clobbering explicit custom args.

    Existing network_args wins over generated values for the same key. This
    preserves the historical advanced escape hatch while making GUI controls
    first-class and deterministic.
    """
    result: list[str] = []
    index: dict[str, int] = {}
    for value in [*generated, *_items(existing)]:
        value = str(value).strip()
        if not value:
            continue
        key = _arg_key(value)
        if key in index:
            result[index[key]] = value
        else:
            index[key] = len(result)
            result.append(value)
    return result


def _normalize_lora_checkpoint_mode(config: dict) -> None:
    mode = config.pop("anima_lora_checkpoint_mode", None)
    if mode in (None, ""):
        return
    mode = str(mode).lower()
    mapping = {
        "off": (False, False, False),
        "standard": (True, False, False),
        "cpu": (True, True, False),
        "unsloth": (True, False, True),
    }
    if mode not in mapping:
        raise ValueError("Anima LoRA: checkpoint mode 只能是 off / standard / cpu / unsloth。")
    gradient, cpu, unsloth = mapping[mode]
    config["gradient_checkpointing"] = gradient
    config["cpu_offload_checkpointing"] = cpu
    config["unsloth_offload_checkpointing"] = unsloth


def _normalize_lora_compile_mode(config: dict) -> None:
    mode = config.pop("anima_lora_compile_mode", None)
    if mode in (None, ""):
        return
    mode = str(mode).lower()
    if mode not in {"off", "accelerate", "per_block"}:
        raise ValueError("Anima LoRA: compile mode 只能是 off / accelerate / per_block。")
    config["torch_compile"] = mode == "accelerate"
    config["compile"] = mode == "per_block"

    if mode != "accelerate":
        config.pop("dynamo_backend", None)
    if mode != "per_block":
        for key in (
            "compile_backend",
            "compile_mode",
            "compile_dynamic",
            "compile_fullgraph",
            "compile_cache_size_limit",
        ):
            config.pop(key, None)


def _normalize_custom_optimizer_scheduler(config: dict) -> None:
    optimizer = str(config.get("optimizer_type") or "").strip()
    custom_optimizer = config.pop("anima_lora_custom_optimizer_type", None)
    if optimizer.lower() == "custom":
        if custom_optimizer in (None, ""):
            raise ValueError("Anima LoRA: 选择 Custom optimizer 时必须填写完整 optimizer type/class。")
        config["optimizer_type"] = str(custom_optimizer).strip()
    else:
        config.pop("anima_lora_custom_optimizer_type", None)

    scheduler = str(config.get("lr_scheduler") or "").strip().lower()
    custom_scheduler = config.pop("anima_lora_custom_lr_scheduler_type", None)
    if scheduler == "custom":
        if custom_scheduler in (None, ""):
            raise ValueError("Anima LoRA: 选择 Custom scheduler 时必须填写 lr_scheduler_type。")
        config["lr_scheduler_type"] = str(custom_scheduler).strip()
        config["lr_scheduler"] = "constant"
    else:
        config.pop("anima_lora_custom_lr_scheduler_type", None)


def _normalize_lora_network_args(config: dict) -> None:
    generated: list[str] = []

    if _as_bool(config.pop("anima_lora_train_llm_adapter", False)):
        generated.append("train_llm_adapter=true")

    for gui_key, network_key in (
        ("anima_lora_rank_dropout", "rank_dropout"),
        ("anima_lora_module_dropout", "module_dropout"),
        ("anima_lora_network_reg_dims", "network_reg_dims"),
        ("anima_lora_network_reg_lrs", "network_reg_lrs"),
        ("anima_lora_loraplus_lr_ratio", "loraplus_lr_ratio"),
        ("anima_lora_loraplus_unet_lr_ratio", "loraplus_unet_lr_ratio"),
        ("anima_lora_loraplus_text_encoder_lr_ratio", "loraplus_text_encoder_lr_ratio"),
    ):
        value = config.pop(gui_key, None)
        if value not in (None, ""):
            generated.append(f"{network_key}={value}")

    for gui_key, network_key in (
        ("anima_lora_include_patterns", "include_patterns"),
        ("anima_lora_exclude_patterns", "exclude_patterns"),
    ):
        values = _items(config.pop(gui_key, None))
        if values:
            generated.append(f"{network_key}={values!r}")

    if _as_bool(config.pop("anima_lora_network_verbose", False)):
        generated.append("verbose=true")

    merged = _merge_network_args(config.get("network_args"), generated)
    if merged:
        config["network_args"] = merged
    else:
        config.pop("network_args", None)


def _validate_common_effective_controls(config: dict) -> None:
    weighting = str(config.get("weighting_scheme") or "uniform").lower()
    sampling = str(config.get("timestep_sampling") or "sigmoid").lower()
    if weighting in {"logit_normal", "mode"} and sampling != "sigma":
        raise ValueError(
            "Anima: weighting_scheme=logit_normal/mode 只有在 timestep_sampling=sigma 时"
            "才会真实改变 timestep 分布；请切换为 sigma，或选择其他 weighting scheme。"
        )

    if weighting == "logit_normal":
        config.setdefault("logit_mean", 0.0)
        config.setdefault("logit_std", 1.0)
        config.pop("mode_scale", None)
    elif weighting == "mode":
        config.setdefault("mode_scale", 1.29)
        config.pop("logit_mean", None)
        config.pop("logit_std", None)
    else:
        config.pop("logit_mean", None)
        config.pop("logit_std", None)
        config.pop("mode_scale", None)

    gamma = config.get("ip_noise_gamma")
    if gamma in (None, "", 0, 0.0):
        config.pop("ip_noise_gamma", None)
        config.pop("ip_noise_gamma_random_strength", None)
    else:
        if float(gamma) < 0:
            raise ValueError("Anima: ip_noise_gamma 不能为负数。")

    show_timesteps = str(config.get("show_timesteps") or "").lower()
    if show_timesteps in {"", "off", "none"}:
        config.pop("show_timesteps", None)
        config.pop("show_timesteps_resolution", None)
        config.pop("show_timesteps_offset", None)
    elif show_timesteps not in {"console", "image"}:
        raise ValueError("Anima: show_timesteps 只能是 off / console / image。")

    repo_id = config.get("huggingface_repo_id")
    if not repo_id:
        for key in (
            "huggingface_repo_type",
            "huggingface_path_in_repo",
            "huggingface_repo_visibility",
            "async_upload",
            "save_state_to_huggingface",
            "resume_from_huggingface",
        ):
            config.pop(key, None)
    elif _as_bool(config.get("resume_from_huggingface")) and not config.get("resume"):
        raise ValueError(
            "Anima: resume_from_huggingface=true 时必须同时填写 resume "
            "（例如 repo_id/path:revision:model）。"
        )


def _validate_lora_effective_controls(config: dict) -> None:
    blocks = int(config.get("blocks_to_swap") or 0)
    cpu = _as_bool(config.get("cpu_offload_checkpointing"))
    unsloth = _as_bool(config.get("unsloth_offload_checkpointing"))
    if cpu and unsloth:
        raise ValueError("Anima LoRA: CPU checkpoint offload 与 Unsloth offload 不能同时启用。")
    if blocks > 0 and (cpu or unsloth):
        raise ValueError("Anima LoRA: blocks_to_swap 不能与 checkpoint CPU/Unsloth offload 同时启用。")

    if _as_bool(config.get("torch_compile")) and _as_bool(config.get("compile")):
        raise ValueError("Anima LoRA: Accelerate torch_compile 与 per-block compile 不能同时启用。")
    if _as_bool(config.get("compile_fullgraph")) and _as_bool(config.get("split_attn")):
        raise ValueError("Anima LoRA: per-block compile_fullgraph 不能与 split_attn 同时启用。")

    if _as_bool(config.get("cache_text_encoder_outputs_to_disk")):
        config["cache_text_encoder_outputs"] = True
    if _as_bool(config.get("cache_latents_to_disk")):
        config["cache_latents"] = True


def normalize_anima_ui_parity(config: dict, mode: str) -> None:
    """Materialize new GUI controls into real sd-scripts arguments.

    Idempotent by design: preview/export/start and post-ui_custom validation all
    run the same function.
    """
    mode = str(mode).lower()
    if mode not in {"lora", "finetune"}:
        raise ValueError(f"Unsupported Anima mode: {mode}")

    _validate_common_effective_controls(config)

    if mode == "lora":
        _normalize_lora_checkpoint_mode(config)
        _normalize_lora_compile_mode(config)
        _normalize_custom_optimizer_scheduler(config)
        _normalize_lora_network_args(config)
        _validate_lora_effective_controls(config)
    else:
        for key in _LORA_ONLY_GUI_KEYS:
            config.pop(key, None)
        # The full trainer currently parses per-block compile/CUDA optimization
        # flags but does not execute them. Keep them out of exported TOML rather
        # than presenting no-op controls as supported functionality.
        for key in (
            "compile",
            "compile_backend",
            "compile_mode",
            "compile_dynamic",
            "compile_fullgraph",
            "compile_cache_size_limit",
            "cuda_allow_tf32",
            "cuda_cudnn_benchmark",
        ):
            config.pop(key, None)


__all__ = ["normalize_anima_ui_parity"]
