"""Validation helpers for optional Anima Qwen3 text-encoder finetuning.

This module intentionally owns only the new Qwen3-training fields. Existing
Anima routing and sd-scripts options stay where they are so disabling the new
feature preserves the current behaviour.
"""

from pathlib import Path


ANIMA_QWEN_TRAINING_KEYS = {
    "train_qwen3_text_encoder",
    "qwen3_lr",
    "qwen3_gradient_checkpointing",
    "qwen3_output_dir",
}

# Qwen3 needs a reliable independent parameter-group LR. AdaFactor is supported
# only in the proven fused-backward mode below; relative-step mode is disabled so
# both the DiT LR and qwen3_lr remain explicit.
SUPPORTED_QWEN_OPTIMIZERS = {
    "adamw",
    "adamw8bit",
    "pagedadamw8bit",
    "lion",
    "lion8bit",
    "pagedlion8bit",
    "sgdnesterov",
    "sgdnesterov8bit",
    "adafactor",
}


def strip_qwen_training_keys(config: dict) -> None:
    for key in ANIMA_QWEN_TRAINING_KEYS:
        config.pop(key, None)


def text_encoder_cache_enabled(config: dict) -> bool:
    """Match sd-scripts semantics: disk TE cache implies TE caching."""
    return bool(
        config.get("cache_text_encoder_outputs")
        or config.get("cache_text_encoder_outputs_to_disk")
    )


def _arg_key(item: object) -> str:
    return str(item).split("=", 1)[0].strip().lower()


def _ensure_fixed_adafactor_args(config: dict) -> None:
    """Keep explicit per-group LRs when Qwen3 joint uses fused AdaFactor."""
    raw = config.get("optimizer_args")
    if raw is None:
        args: list[str] = []
    elif isinstance(raw, (list, tuple)):
        args = [str(item) for item in raw]
    else:
        args = [str(raw)]

    required = {
        "relative_step": "relative_step=False",
        "scale_parameter": "scale_parameter=False",
        "warmup_init": "warmup_init=False",
    }
    existing = {_arg_key(item): index for index, item in enumerate(args)}
    for key, value in required.items():
        if key in existing:
            args[existing[key]] = value
        else:
            args.append(value)
    config["optimizer_args"] = args


def normalize_qwen_training_config(config: dict, anima_training_mode: str) -> bool:
    """Validate/normalize the optional Qwen3 joint-finetune fields.

    Returns True only when this job is an Anima full finetune that explicitly
    requests Qwen3 training. In every other case all new fields are removed so
    old LoRA/full-finetune configurations serialize exactly as before.
    """
    train_qwen3 = bool(config.get("train_qwen3_text_encoder"))

    if anima_training_mode != "finetune" or not train_qwen3:
        strip_qwen_training_keys(config)
        return False

    if text_encoder_cache_enabled(config):
        raise ValueError(
            "Anima: 训练 Qwen3 文本编码器时不能启用 cache_text_encoder_outputs "
            "或 cache_text_encoder_outputs_to_disk。请关闭两项 Qwen3 输出缓存；"
            "cache_latents 不受影响。"
        )

    try:
        dit_lr = float(config.get("learning_rate"))
    except (TypeError, ValueError):
        raise ValueError("Anima: Qwen3 联合训练要求有效的主 DiT learning_rate。")
    if dit_lr <= 0:
        raise ValueError("Anima: Qwen3 联合训练要求主 DiT learning_rate 大于 0；暂不支持仅训练 Qwen3。")

    try:
        qwen3_lr = float(config.get("qwen3_lr"))
    except (TypeError, ValueError):
        raise ValueError("Anima: 训练 Qwen3 时必须设置有效的 qwen3_lr（建议从 5e-7 起）。")
    if qwen3_lr <= 0:
        raise ValueError("Anima: 训练 Qwen3 时 qwen3_lr 必须大于 0。")

    optimizer = str(config.get("optimizer_type") or "AdamW").lower()
    if optimizer not in SUPPORTED_QWEN_OPTIMIZERS:
        raise ValueError(
            "Anima: 当前 Qwen3 联合训练仅支持具有可靠独立参数组学习率的优化器："
            "AdamW / AdamW8bit / PagedAdamW8bit / Lion / Lion8bit / "
            "PagedLion8bit / SGDNesterov / SGDNesterov8bit，或 fused AdaFactor。"
        )

    blocks_to_swap = int(config.get("blocks_to_swap") or 0)
    fused_backward = bool(config.get("fused_backward_pass"))
    if blocks_to_swap > 0:
        if optimizer != "adafactor" or not fused_backward:
            raise ValueError(
                "Anima: Qwen3 联合训练启用 blocks_to_swap 时必须使用 AdaFactor + fused_backward_pass。"
            )
    elif fused_backward and optimizer != "adafactor":
        raise ValueError("Anima: Qwen3 联合训练的 fused_backward_pass 当前仅支持 AdaFactor。")

    if optimizer == "adafactor":
        if not fused_backward:
            raise ValueError("Anima: Qwen3 联合训练的 AdaFactor 当前要求 fused_backward_pass=true。")
        _ensure_fixed_adafactor_args(config)

    if config.get("deepspeed"):
        raise ValueError("Anima: Qwen3 联合训练第一版暂不支持 DeepSpeed；普通 Anima DeepSpeed 不受影响。")

    if config.get("torch_compile"):
        raise ValueError("Anima: Qwen3 联合训练第一版暂不支持 torch_compile；请先关闭后进行联合训练。")

    # Empty output dir means: save Qwen sidecars beside the main checkpoint.
    if not config.get("qwen3_output_dir"):
        config.pop("qwen3_output_dir", None)
    else:
        output_dir = Path(str(config["qwen3_output_dir"]))
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"Anima: Qwen3 输出位置不是目录: {output_dir}")

    return True


def trainer_supports_qwen_training(trainer_path: str) -> bool:
    """Verify both the Anima trainer branch and its CLI argument definition.

    The CLI arguments live in ``library/anima_train_utils.py``, not directly in
    ``anima_train.py``. Requiring markers in both files avoids reporting support
    for a partially applied patch.
    """
    trainer = Path(trainer_path)
    args_file = trainer.parent / "library" / "anima_train_utils.py"
    try:
        trainer_text = trainer.read_text(encoding="utf-8", errors="ignore")
        args_text = args_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    return (
        "train_qwen3_text_encoder" in trainer_text
        and "--train_qwen3_text_encoder" in args_text
        and "--qwen3_lr" in args_text
    )
