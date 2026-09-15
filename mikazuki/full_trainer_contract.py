"""Pure normalization/validation for non-Anima full trainers."""

from __future__ import annotations


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_number(value: object) -> bool:
    return value not in (None, "", 0, 0.0, "0", "0.0")


def normalize_common_full_config(config: dict) -> None:
    """Make trainer precedence/coercions explicit in the generated TOML."""
    if _positive_number(config.get("max_train_steps")):
        config.pop("max_train_epochs", None)

    if _as_bool(config.get("cache_latents_to_disk")):
        config["cache_latents"] = True
    if _as_bool(config.get("cache_text_encoder_outputs_to_disk")):
        config["cache_text_encoder_outputs"] = True
    if _as_bool(config.get("cpu_offload_checkpointing")):
        config["gradient_checkpointing"] = True

    ratio = config.get("save_n_epoch_ratio")
    if ratio not in (None, "", 0, "0"):
        try:
            ratio = int(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("save_n_epoch_ratio 必须是正整数。") from exc
        if ratio <= 0:
            raise ValueError("save_n_epoch_ratio 必须大于 0。")
        config["save_n_epoch_ratio"] = ratio
        config.pop("save_every_n_epochs", None)

    if config.get("sample_every_n_steps") not in (None, "", 0, "0"):
        config.pop("sample_every_n_epochs", None)


def _validate_precision(config: dict, label: str) -> None:
    mixed = str(config.get("mixed_precision") or "no").lower()
    full_fp16 = _as_bool(config.get("full_fp16"))
    full_bf16 = _as_bool(config.get("full_bf16"))
    if full_fp16 and full_bf16:
        raise ValueError(f"{label}: full_fp16 与 full_bf16 不能同时启用。")
    if full_fp16 and mixed != "fp16":
        raise ValueError(f"{label}: full_fp16 要求 mixed_precision=fp16。")
    if full_bf16 and mixed != "bf16":
        raise ValueError(f"{label}: full_bf16 要求 mixed_precision=bf16。")


def _validate_cacheability(config: dict, label: str) -> None:
    latent_cache = _as_bool(config.get("cache_latents")) or _as_bool(config.get("cache_latents_to_disk"))
    if latent_cache and (_as_bool(config.get("color_aug")) or _as_bool(config.get("random_crop"))):
        raise ValueError(f"{label}: latent cache 不能与 color_aug / random_crop 同时启用。")

    te_cache = _as_bool(config.get("cache_text_encoder_outputs")) or _as_bool(config.get("cache_text_encoder_outputs_to_disk"))
    if te_cache:
        incompatible = (
            _as_bool(config.get("shuffle_caption"))
            or float(config.get("caption_dropout_rate") or 0) > 0
            or float(config.get("caption_tag_dropout_rate") or 0) > 0
            or float(config.get("token_warmup_step") or 0) > 0
        )
        if incompatible:
            raise ValueError(f"{label}: text-encoder output cache 不能与 caption dropout / shuffle / token warmup / tag dropout 同时启用。")


def _validate_fused_backward(config: dict, label: str, *, other_fused: bool = False) -> None:
    if not _as_bool(config.get("fused_backward_pass")):
        return
    if str(config.get("optimizer_type") or "").lower() != "adafactor":
        raise ValueError(f"{label}: fused_backward_pass 要求 optimizer_type=AdaFactor。")
    if int(config.get("gradient_accumulation_steps") or 1) != 1:
        raise ValueError(f"{label}: fused_backward_pass 要求 gradient_accumulation_steps=1。")
    if other_fused:
        raise ValueError(f"{label}: fused_backward_pass 不能与另一种 fused optimizer 模式同时启用。")
    if _as_bool(config.get("deepspeed")):
        raise ValueError(f"{label}: DeepSpeed 与 fused_backward_pass 不能组合使用。")


def _validate_compile(config: dict, label: str, *, blocks_to_swap: int = 0) -> None:
    if not _as_bool(config.get("torch_compile")):
        return
    if not str(config.get("dynamo_backend") or "").strip():
        config["dynamo_backend"] = "inductor"
    if _as_bool(config.get("deepspeed")):
        raise ValueError(f"{label}: torch_compile + DeepSpeed 尚未完成当前仓库 GPU 验证，暂不允许组合。")
    if blocks_to_swap > 0:
        raise ValueError(f"{label}: torch_compile + blocks_to_swap 尚未完成当前仓库 GPU 验证，暂不允许组合。")


def normalize_validate_sdxl_full(config: dict) -> None:
    normalize_common_full_config(config)
    _validate_precision(config, "SDXL")
    _validate_cacheability(config, "SDXL")
    _validate_compile(config, "SDXL")

    te_cache = _as_bool(config.get("cache_text_encoder_outputs")) or _as_bool(config.get("cache_text_encoder_outputs_to_disk"))
    if _as_bool(config.get("train_text_encoder")) and te_cache:
        raise ValueError("SDXL: 训练文本编码器时不能缓存文本编码器输出。")

    block_lr = config.get("block_lr")
    if block_lr not in (None, ""):
        values = [part.strip() for part in str(block_lr).split(",")]
        if len(values) != 23:
            raise ValueError("SDXL: block_lr 必须恰好包含 23 个逗号分隔学习率。")
        try:
            [float(value) for value in values]
        except ValueError as exc:
            raise ValueError("SDXL: block_lr 的 23 项都必须是有效数字。") from exc

    if _as_bool(config.get("xformers")) and _as_bool(config.get("sdpa")):
        raise ValueError("SDXL: xformers 与 sdpa 只能选择一个。")
    if _as_bool(config.get("diffusers_xformers")) and (_as_bool(config.get("xformers")) or _as_bool(config.get("sdpa"))):
        raise ValueError("SDXL: diffusers_xformers 使用独立 attention 分支，不能与 U-Net xformers/sdpa 同时启用。")

    fused_groups = config.get("fused_optimizer_groups") not in (None, "", 0, "0")
    _validate_fused_backward(config, "SDXL", other_fused=fused_groups)
    if fused_groups:
        if int(config.get("gradient_accumulation_steps") or 1) != 1:
            raise ValueError("SDXL: fused_optimizer_groups 要求 gradient_accumulation_steps=1。")
        if _as_bool(config.get("deepspeed")):
            raise ValueError("SDXL: DeepSpeed 与 fused_optimizer_groups 不能组合使用。")


def normalize_validate_flux_full(config: dict) -> None:
    normalize_common_full_config(config)
    _validate_precision(config, "Flux")
    _validate_cacheability(config, "Flux")

    if _as_bool(config.get("xformers")) and _as_bool(config.get("sdpa")):
        raise ValueError("Flux: xformers 与 sdpa 只能选择一个。")

    blocks_to_swap = int(config.get("blocks_to_swap") or 0)
    _validate_compile(config, "Flux", blocks_to_swap=blocks_to_swap)
    if blocks_to_swap > 0 and _as_bool(config.get("cpu_offload_checkpointing")):
        raise ValueError("Flux: blocks_to_swap 不能与 cpu_offload_checkpointing 同时启用。")

    blockwise = _as_bool(config.get("blockwise_fused_optimizers"))
    _validate_fused_backward(config, "Flux", other_fused=blockwise)
    if blockwise:
        if int(config.get("gradient_accumulation_steps") or 1) != 1:
            raise ValueError("Flux: blockwise_fused_optimizers 要求 gradient_accumulation_steps=1。")
        if str(config.get("optimizer_type") or "").lower().endswith("schedulefree"):
            raise ValueError("Flux: blockwise_fused_optimizers 不支持 schedule-free optimizer。")
        if _as_bool(config.get("deepspeed")):
            raise ValueError("Flux: DeepSpeed 与 blockwise_fused_optimizers 不能组合使用。")
