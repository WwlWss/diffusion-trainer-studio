"""Side-effect controlled Anima normalization used by preview and launch."""

from __future__ import annotations

import json
import os
from typing import Optional

from mikazuki.anima_finetune_advanced import (
    normalize_anima_save_schedule,
    validate_anima_finetune_advanced_combinations,
    validate_effective_text_encoder_cache,
)
from mikazuki.anima_finetune_config import (
    normalize_anima_finetune_config,
    validate_anima_finetune_config,
)
from mikazuki.anima_qwen_config import normalize_qwen_training_config, trainer_supports_qwen_training


ANIMA_VARIANTS = {"base", "2.9b"}
ANIMA_OPTIONAL_FINETUNE_LRS = {"self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr", "llm_adapter_lr"}
ANIMA_LORA_ONLY_KEYS = {
    "network_module", "network_weights", "network_dim", "network_alpha", "network_dropout",
    "scale_weight_norms", "network_args", "network_args_custom", "network_train_unet_only",
    "network_train_text_encoder_only", "enable_base_weight", "base_weights", "base_weights_multiplier",
    "unet_lr", "text_encoder_lr",
}
ANIMA_FULL_ONLY_KEYS = {
    "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr", "llm_adapter_lr",
    "train_qwen3_text_encoder", "qwen3_lr", "qwen3_gradient_checkpointing", "qwen3_output_dir",
    "anima_finetune_learning_rate", "anima_precision_mode", "anima_latent_cache_mode",
    "anima_text_encoder_cache_mode", "anima_checkpoint_mode", "anima_custom_optimizer_type",
    "anima_custom_lr_scheduler_type", "cpu_offload_checkpointing", "fused_backward_pass", "deepspeed",
    "zero_stage", "offload_optimizer_device", "offload_optimizer_nvme_path", "offload_param_device",
    "offload_param_nvme_path", "zero3_init_flag", "zero3_save_16bit_model",
    "fp16_master_weights_and_gradients", "torch_compile", "dynamo_backend", "ddp_static_graph",
    "dataset_config", "in_json", "masked_loss", "conditioning_data_dir",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _pop_many(config: dict, keys) -> None:
    for key in keys:
        config.pop(key, None)


def _detect_variant(model_path: str) -> Optional[str]:
    if not model_path or not os.path.isfile(model_path) or not model_path.lower().endswith(".safetensors"):
        return None
    try:
        with open(model_path, "rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            metadata = json.loads(handle.read(length))
    except Exception:
        return None
    if any(key.endswith("blocks.39.mlp.layer1.weight") for key in metadata):
        return "2.9b"
    if any(key.endswith("blocks.27.mlp.layer1.weight") for key in metadata):
        return "base"
    return None


def _normalize_lora_target(config: dict) -> None:
    target = config.pop("anima_lora_target", None)
    if target in (None, ""):
        if _as_bool(config.get("network_train_text_encoder_only")):
            target = "qwen3"
        elif _as_bool(config.get("network_train_unet_only")):
            target = "dit"
        else:
            target = "dit_qwen3"
    target = str(target).lower()
    if target not in {"dit", "qwen3", "dit_qwen3"}:
        raise ValueError("Anima LoRA 训练目标只能是 dit / qwen3 / dit_qwen3。")

    te_lr = config.pop("anima_lora_text_encoder_lr", None)
    if te_lr not in (None, ""):
        try:
            config["text_encoder_lr"] = float(te_lr)
        except (TypeError, ValueError) as exc:
            raise ValueError("Anima LoRA: Qwen3 LoRA 学习率必须是有效数字。") from exc

    if target in {"qwen3", "dit_qwen3"} and (
        _as_bool(config.get("cache_text_encoder_outputs"))
        or _as_bool(config.get("cache_text_encoder_outputs_to_disk"))
    ):
        raise ValueError("Anima LoRA: 训练 Qwen3 LoRA 时必须关闭 Qwen3 输出缓存。")

    if target == "dit":
        config["network_train_unet_only"] = True
        config.pop("network_train_text_encoder_only", None)
        config.pop("text_encoder_lr", None)
    elif target == "qwen3":
        config["network_train_unet_only"] = False
        config["network_train_text_encoder_only"] = True
    else:
        config["network_train_unet_only"] = False
        config["network_train_text_encoder_only"] = False


def _materialize_flow_shift(config: dict, toml_path: str | None, launch: bool) -> None:
    flow_shift = config.pop("sample_flow_shift", None)
    if flow_shift in (None, ""):
        return
    try:
        value = float(flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError("Anima: 预览 sample_flow_shift 必须是有效数字。") from exc

    prompt_path = config.get("sample_prompts")
    if not prompt_path or not str(prompt_path).lower().endswith(".txt") or not os.path.isfile(str(prompt_path)):
        return
    with open(str(prompt_path), "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    rewritten = []
    changed = False
    for line in lines:
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]
        if stripped and not stripped.lstrip().startswith("#") and " --fs " not in stripped:
            stripped = f"{stripped} --fs {value:g}"
            changed = True
        rewritten.append(stripped + newline)
    if not changed:
        return

    derived = os.path.splitext(toml_path or "config/autosave/preview.toml")[0] + "-anima-prompts.txt"
    config["sample_prompts"] = derived
    if launch:
        with open(derived, "w", encoding="utf-8") as handle:
            handle.writelines(rewritten)


def prepare_anima_config(
    config: dict,
    train_type: str,
    trainer_file: str,
    *,
    launch: bool = False,
    toml_path: str | None = None,
) -> None:
    mode = "finetune" if train_type == "anima-finetune" else "lora"
    if config.get("max_train_steps") not in (None, "", 0, "0"):
        config.pop("max_train_epochs", None)
    if config.get("attn_mode") == "xformers":
        config["split_attn"] = True

    normalize_anima_finetune_config(config, mode)
    normalize_anima_save_schedule(config)
    if mode == "finetune":
        validate_anima_finetune_advanced_combinations(config)
    train_qwen3 = normalize_qwen_training_config(config, mode)
    validate_anima_finetune_config(config, mode)
    validate_effective_text_encoder_cache(config)

    variant = str(config.get("anima_model_variant", "base")).lower()
    if variant not in ANIMA_VARIANTS:
        raise ValueError(f"Unsupported Anima model variant: {variant}")
    detected = _detect_variant(str(config.get("pretrained_model_name_or_path") or ""))
    if detected is not None and detected != variant:
        raise ValueError(f"Anima 模型版本不匹配：界面选择 {variant}，checkpoint 检测为 {detected}。")
    max_blocks = 38 if variant == "2.9b" else 26
    if int(config.get("blocks_to_swap") or 0) > max_blocks:
        raise ValueError(f"Anima {variant} 最多允许 blocks_to_swap={max_blocks}。")
    config.pop("anima_model_variant", None)

    _materialize_flow_shift(config, toml_path, launch)

    if mode == "finetune":
        _pop_many(config, ANIMA_LORA_ONLY_KEYS)
        for key in ANIMA_OPTIONAL_FINETUNE_LRS:
            if config.get(key) in (None, ""):
                config.pop(key, None)
        # Capability is a launch-time requirement. Preview must remain useful
        # while the user is filling the form or before the local sd-scripts
        # patch has been applied.
        if launch and train_qwen3 and os.path.exists(trainer_file) and not trainer_supports_qwen_training(trainer_file):
            raise RuntimeError("当前 sd-scripts 尚未包含完整 Anima Qwen3 联合训练补丁。")
        if launch and train_qwen3 and config.get("qwen3_output_dir"):
            os.makedirs(str(config["qwen3_output_dir"]), exist_ok=True)
    else:
        _pop_many(config, ANIMA_FULL_ONLY_KEYS)
        _normalize_lora_target(config)
        config.setdefault("network_module", "networks.lora_anima")
