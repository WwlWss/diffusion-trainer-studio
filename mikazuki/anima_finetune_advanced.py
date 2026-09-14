"""Pure validation helpers for advanced Anima full-finetune options.

Keep these helpers free of GUI/server/runtime imports so configuration semantics
can be regression-tested in the lightweight CI environment.
"""


def validate_effective_text_encoder_cache(config: dict) -> None:
    cache_enabled = bool(
        config.get("cache_text_encoder_outputs")
        or config.get("cache_text_encoder_outputs_to_disk")
    )
    if not cache_enabled:
        return
    if config.get("shuffle_caption"):
        raise ValueError("Anima: 缓存 Qwen3 输出时必须关闭 shuffle_caption")
    if float(config.get("caption_tag_dropout_rate") or 0) > 0:
        raise ValueError("Anima: 缓存 Qwen3 输出时不能启用 caption_tag_dropout_rate")
    if float(config.get("token_warmup_step") or 0) > 0:
        raise ValueError("Anima: 缓存 Qwen3 输出时不能启用 token_warmup_step")


def normalize_anima_save_schedule(config: dict) -> None:
    """Make mutually-overriding save cadence fields explicit before sd-scripts."""
    ratio = config.get("save_n_epoch_ratio")
    if ratio in (None, "", 0):
        return
    try:
        ratio_value = int(ratio)
    except (TypeError, ValueError) as e:
        raise ValueError("Anima: save_n_epoch_ratio 必须是正整数。") from e
    if ratio_value <= 0:
        raise ValueError("Anima: save_n_epoch_ratio 必须大于 0。")

    # sd-scripts recalculates save_every_n_epochs from this ratio. Remove the
    # competing GUI value rather than silently letting the trainer override it.
    config["save_n_epoch_ratio"] = ratio_value
    config.pop("save_every_n_epochs", None)


def validate_anima_finetune_advanced_combinations(config: dict) -> None:
    if config.get("dataset_config") and config.get("in_json"):
        raise ValueError("Anima: dataset_config 与 in_json 不能同时使用；请选择一种数据集来源。")
    if config.get("masked_loss") and not (config.get("conditioning_data_dir") or config.get("dataset_config")):
        raise ValueError("Anima: masked_loss 需要 conditioning_data_dir，或在 dataset_config 中提供 conditioning 数据。")

    deepspeed = bool(config.get("deepspeed"))
    fused_backward = bool(config.get("fused_backward_pass"))
    blocks_to_swap = int(config.get("blocks_to_swap") or 0)
    optimizer = str(config.get("optimizer_type") or "AdamW").strip().lower()
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps") or 1)

    if fused_backward and optimizer != "adafactor":
        raise ValueError("Anima: fused_backward_pass 当前仅支持 optimizer_type=Adafactor。")
    if fused_backward and gradient_accumulation_steps != 1:
        raise ValueError("Anima: fused_backward_pass 要求 gradient_accumulation_steps=1。")

    # Runtime smoke on the pinned Anima trainer confirmed that ordinary
    # end-of-backward optimizers are not safe with block swapping: the
    # ModelOffloader may have a parameter on CPU while its accumulated gradient
    # is still on CUDA when optimizer.step() runs. bitsandbytes then fails with
    # a CPU-parameter/CUDA-gradient device mismatch. The trainer already has an
    # Adafactor fused-backward path that steps each parameter from its gradient
    # hook before the block is offloaded, so require that path until Anima gains
    # Flux-style blockwise fused optimizers for other optimizer families.
    if blocks_to_swap > 0 and not fused_backward:
        raise ValueError(
            "Anima: blocks_to_swap 在全参微调中当前必须配合 fused_backward_pass=true 与 Adafactor；"
            "普通 optimizer.step() 会在 block swap 后遇到 CPU 参数 / CUDA 梯度设备不一致。"
        )
    if blocks_to_swap > 0 and optimizer != "adafactor":
        raise ValueError("Anima: blocks_to_swap 当前仅支持 Adafactor + fused_backward_pass。")

    if deepspeed and fused_backward:
        raise ValueError("Anima: DeepSpeed 与 fused_backward_pass 当前不能组合使用。")
    if deepspeed and blocks_to_swap > 0:
        raise ValueError("Anima: DeepSpeed 路径当前不支持 blocks_to_swap；请二选一。")
    if deepspeed and config.get("torch_compile"):
        raise ValueError("Anima: DeepSpeed + torch_compile 尚未完成 GPU 验证；当前请二选一。")
