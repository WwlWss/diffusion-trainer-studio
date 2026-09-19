"""Inverse mapping from effective trainer TOML to semantic GUI state."""

from __future__ import annotations

from copy import deepcopy
import ast
import json
from pathlib import Path

from mikazuki.multi_caption_config import rehydrate_multi_caption_policy
from mikazuki.parameter_policy import rehydrate_parameter_policy, validate_parameter_policy
from mikazuki.training_gui_args import PRODIGY_TYPES, _arg_key, _as_bool, _items


def _extract_arg(args: list[str], key: str) -> tuple[object | None, list[str]]:
    key_lower = key.lower()
    found = None
    remaining: list[str] = []
    for item in args:
        if _arg_key(item) == key_lower:
            found = item.split("=", 1)[1] if "=" in item else ""
        else:
            remaining.append(item)
    return found, remaining


def _infer_anima_precision_mode(config: dict) -> str:
    mixed = str(config.pop("mixed_precision", "no") or "no").lower()
    full_fp16 = _as_bool(config.pop("full_fp16", False))
    full_bf16 = _as_bool(config.pop("full_bf16", False))
    if full_fp16:
        return "full_fp16"
    if full_bf16:
        return "full_bf16"
    if mixed == "fp16":
        return "mixed_fp16"
    if mixed == "bf16":
        return "mixed_bf16"
    return "fp32"


def _infer_cache_mode(config: dict, key: str, disk_key: str) -> str:
    disk = _as_bool(config.pop(disk_key, False))
    enabled = _as_bool(config.pop(key, False))
    return "disk" if disk else "memory" if enabled else "off"


def _parse_repr_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return [str(value)]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _infer_checkpoint_mode(config: dict) -> str:
    gradient = _as_bool(config.pop("gradient_checkpointing", False))
    cpu = _as_bool(config.pop("cpu_offload_checkpointing", False))
    unsloth = _as_bool(config.pop("unsloth_offload_checkpointing", False))
    if unsloth:
        return "unsloth"
    if cpu:
        return "cpu"
    return "standard" if gradient else "off"


def rehydrate_trainer_config(
    effective_config: dict,
    page_train_type: str,
    *,
    sidecars: dict[str, str] | None = None,
) -> dict:
    """Inverse-map an exported trainer TOML into current-page GUI state.

    The inverse is semantic rather than byte-for-byte historical state: values
    that were deliberately compiled away (for example a D-Adapt GUI LR before
    it became trainer LR=1) cannot be recovered, but re-preparing the returned
    state yields an equivalent trainer configuration.
    """
    config = deepcopy(effective_config)

    parameter_policy_path = config.pop("parameter_policy_config", None)
    if parameter_policy_path:
        path_key = str(parameter_policy_path)
        sidecar_content = (sidecars or {}).get(path_key)
        if sidecar_content is None:
            path = Path(path_key)
            if not path.is_file():
                raise ValueError(f"Parameter Policy sidecar 不存在，不能静默回退 Standard: {path}")
            try:
                sidecar_content = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Parameter Policy sidecar 读取失败: {path}: {exc}") from exc
        try:
            raw_policy = json.loads(sidecar_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Parameter Policy sidecar JSON 无效: {path_key}: {exc}") from exc
        policy = validate_parameter_policy(raw_policy)
        parameter_policy_gui = rehydrate_parameter_policy(policy)
    else:
        parameter_policy_gui = {"optimization_mode": "standard"}

    multi_caption_path = config.pop("multi_caption_config", None)
    if multi_caption_path:
        path_key = str(multi_caption_path)
        sidecar_content = (sidecars or {}).get(path_key)
        if sidecar_content is None:
            path = Path(path_key)
            if not path.is_file():
                raise ValueError(f"Multi-Caption sidecar 不存在，不能静默回退 Standard: {path}")
            try:
                sidecar_content = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Multi-Caption sidecar 读取失败: {path}: {exc}") from exc
        try:
            policy = json.loads(sidecar_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Multi-Caption sidecar JSON 无效: {path_key}: {exc}") from exc
        multi_caption_gui = rehydrate_multi_caption_policy(policy, page_train_type)
    else:
        multi_caption_gui = {"caption_mode": "standard"}

    if _as_bool(config.pop("lowram", False)):
        config["memory_mode"] = "lowram"
        config.pop("highvram", None)
    elif _as_bool(config.pop("highvram", False)):
        config["memory_mode"] = "highvram"
    else:
        config.pop("highvram", None)
        config["memory_mode"] = "auto"

    sd_pages = {"lora-basic", "lora-master", "sd-lora", "sdxl-lora", "dreambooth", "sd-dreambooth"}
    if page_train_type in sd_pages:
        token = config.pop("max_token_length", None)
        config["sd_max_token_length_mode"] = str(token) if token in {150, 225, "150", "225"} else "75"

    if page_train_type in {"lora-basic", "lora-master", "sd-lora", "sdxl-lora"}:
        unet_only = _as_bool(config.pop("network_train_unet_only", False))
        te_only = _as_bool(config.pop("network_train_text_encoder_only", False))
        config["lora_target"] = "unet" if unet_only else "text_encoder" if te_only else "unet_text_encoder"

    args = _items(config.pop("network_args", None))
    if page_train_type == "anima-lora":
        value, args = _extract_arg(args, "train_llm_adapter")
        if value is not None:
            config["anima_lora_train_llm_adapter"] = _as_bool(value)

        for arg_key, field in (
            ("rank_dropout", "anima_lora_rank_dropout"),
            ("module_dropout", "anima_lora_module_dropout"),
            ("network_reg_dims", "anima_lora_network_reg_dims"),
            ("network_reg_lrs", "anima_lora_network_reg_lrs"),
            ("loraplus_lr_ratio", "anima_lora_loraplus_lr_ratio"),
            ("loraplus_unet_lr_ratio", "anima_lora_loraplus_unet_lr_ratio"),
            ("loraplus_text_encoder_lr_ratio", "anima_lora_loraplus_text_encoder_lr_ratio"),
        ):
            value, args = _extract_arg(args, arg_key)
            if value is not None:
                config[field] = value

        for arg_key, field in (
            ("include_patterns", "anima_lora_include_patterns"),
            ("exclude_patterns", "anima_lora_exclude_patterns"),
        ):
            value, args = _extract_arg(args, arg_key)
            if value is not None:
                config[field] = "\n".join(_parse_repr_list(value))

        value, args = _extract_arg(args, "verbose")
        if value is not None:
            config["anima_lora_network_verbose"] = _as_bool(value)

    if page_train_type in {"flux-lora", "chroma-lora"}:
        train_t5_raw, args = _extract_arg(args, "train_t5xxl")
        train_t5 = _as_bool(train_t5_raw)
        unet_only = _as_bool(config.pop("network_train_unet_only", False))
        if page_train_type == "chroma-lora":
            config["flux_lora_target"] = "dit_t5xxl" if train_t5 else "dit"
        else:
            config["flux_lora_target"] = "dit" if unet_only else "dit_clip_l_t5xxl" if train_t5 else "dit_clip_l"

    if page_train_type == "anima-lora":
        unet_only = _as_bool(config.pop("network_train_unet_only", False))
        te_only = _as_bool(config.pop("network_train_text_encoder_only", False))
        config["anima_lora_target"] = "dit" if unet_only else "qwen3" if te_only else "dit_qwen3"
        if "text_encoder_lr" in config:
            config["anima_lora_text_encoder_lr"] = str(config.pop("text_encoder_lr"))
        if "learning_rate" in config:
            config["learning_rate"] = str(config["learning_rate"])
        for field in ("ip_noise_gamma", "logit_mean", "logit_std", "mode_scale"):
            if field in config and config[field] not in (None, ""):
                config[field] = str(config[field])

        # Restore semantic controls rather than forcing imported configs into
        # opaque raw booleans/custom args. Re-exporting the GUI state must
        # compile to the same effective sd-scripts configuration.
        config["anima_lora_checkpoint_mode"] = _infer_checkpoint_mode(config)

        accelerate_compile = _as_bool(config.pop("torch_compile", False))
        per_block_compile = _as_bool(config.pop("compile", False))
        if per_block_compile:
            config["anima_lora_compile_mode"] = "per_block"
            config.pop("dynamo_backend", None)
        elif accelerate_compile:
            config["anima_lora_compile_mode"] = "accelerate"
            for key in ("compile_backend", "compile_mode", "compile_dynamic", "compile_fullgraph", "compile_cache_size_limit"):
                config.pop(key, None)
        else:
            config["anima_lora_compile_mode"] = "off"
            config.pop("dynamo_backend", None)
            for key in ("compile_backend", "compile_mode", "compile_dynamic", "compile_fullgraph", "compile_cache_size_limit"):
                config.pop(key, None)

        optimizer = str(config.get("optimizer_type") or "")
        known_optimizers = {
            "AdamW", "AdamW8bit", "PagedAdamW8bit", "RAdamScheduleFree",
            "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit",
            "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdanIP",
            "DAdaptLion", "DAdaptSGD", "AdaFactor", "Prodigy",
            "prodigyplus.ProdigyPlusScheduleFree", "pytorch_optimizer.CAME",
        }
        if optimizer and optimizer not in known_optimizers:
            config["anima_lora_custom_optimizer_type"] = optimizer
            config["optimizer_type"] = "Custom"

        scheduler_type = config.pop("lr_scheduler_type", None)
        if scheduler_type:
            config["anima_lora_custom_lr_scheduler_type"] = str(scheduler_type)
            config["lr_scheduler"] = "custom"

    if page_train_type == "anima-finetune":
        if "learning_rate" in config:
            config["anima_finetune_learning_rate"] = str(config.pop("learning_rate"))
        for field in (
            "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr", "llm_adapter_lr",
            "qwen3_lr", "ip_noise_gamma", "logit_mean", "logit_std", "mode_scale",
        ):
            if field in config and config[field] not in (None, ""):
                config[field] = str(config[field])
        config["anima_precision_mode"] = _infer_anima_precision_mode(config)
        config["anima_latent_cache_mode"] = _infer_cache_mode(config, "cache_latents", "cache_latents_to_disk")
        config["anima_text_encoder_cache_mode"] = _infer_cache_mode(
            config, "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"
        )
        config["anima_checkpoint_mode"] = _infer_checkpoint_mode(config)

    module = str(config.get("network_module") or "")
    if module == "lycoris.kohya":
        reverse = {
            "algo": "lycoris_algo",
            "conv_dim": "conv_dim",
            "conv_alpha": "conv_alpha",
            "factor": "lokr_factor",
            "dropout": "dropout",
            "train_norm": "train_norm",
        }
        for arg_key, field in reverse.items():
            value, args = _extract_arg(args, arg_key)
            if value is not None:
                config[field] = value
    elif module == "networks.dylora":
        value, args = _extract_arg(args, "unit")
        if value is not None:
            config["dylora_unit"] = value

    block_found = False
    for field in ("down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold"):
        value, args = _extract_arg(args, field)
        if value is not None:
            config[field] = value
            block_found = True
    if block_found:
        config["enable_block_weights"] = True

    if args:
        config["network_args_custom"] = args
    else:
        config.pop("network_args", None)

    weights = config.get("base_weights")
    if isinstance(weights, list) and weights:
        config["enable_base_weight"] = True
        config["base_weights"] = "\n".join(str(item) for item in weights)
        multipliers = config.get("base_weights_multiplier")
        if isinstance(multipliers, list):
            config["base_weights_multiplier"] = "\n".join(str(item) for item in multipliers)

    opt_args = _items(config.pop("optimizer_args", None))
    optimizer = str(config.get("optimizer_type") or "").lower()
    if optimizer in PRODIGY_TYPES:
        value, opt_args = _extract_arg(opt_args, "d0")
        if value is not None:
            config["prodigy_d0"] = value
        value, opt_args = _extract_arg(opt_args, "d_coef")
        if value is not None:
            config["prodigy_d_coef"] = value
    if opt_args:
        config["optimizer_args_custom"] = opt_args

    config["dataset_source"] = "config" if config.get("dataset_config") else "folder"

    # Split-prompt pages can faithfully round-trip an exported generated sidecar
    # by treating the already-materialized file as their prompt_file. Basic's
    # historical inline sample_prompts control is kept as-is.
    if page_train_type != "lora-basic" and config.get("sample_prompts"):
        config["prompt_file"] = config.pop("sample_prompts")

    config.update(multi_caption_gui)
    config.update(parameter_policy_gui)
    return config
