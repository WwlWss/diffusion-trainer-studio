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
    if deepspeed and config.get("fused_backward_pass"):
        raise ValueError("Anima: DeepSpeed 与 fused_backward_pass 当前不能组合使用。")
    if deepspeed and int(config.get("blocks_to_swap") or 0) > 0:
        raise ValueError("Anima: DeepSpeed 路径当前不支持 blocks_to_swap；请二选一。")
    if deepspeed and config.get("torch_compile"):
        raise ValueError("Anima: DeepSpeed + torch_compile 尚未完成 GPU 验证；当前请二选一。")
