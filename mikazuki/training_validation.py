"""Side-effect-free validation for prepared training configs."""

from __future__ import annotations

import os
from collections.abc import Callable

from mikazuki.training_data_contract import normalize_optional_dataset_paths, validate_dataset_source


def validate_prepared_config(
    prepared,
    check_paths: bool,
    *,
    is_file: Callable[[str], bool] | None = None,
    is_dir: Callable[[str], bool] | None = None,
    exists: Callable[[str], bool] | None = None,
    inspect_data_dir: Callable[[str], tuple[bool, str]] | None = None,
    validate_model: Callable[[str, str], tuple[bool, str]] | None = None,
    check_trainer: bool = True,
) -> None:
    """Validate semantic preview or launch-time runtime assets.

    Preview deliberately avoids local-path requirements. Start is strict, but
    its checks are read-only and ordered model/assets before dataset so a bad
    model can never trigger changes to a user's dataset directory.
    """
    config = prepared.config

    # Multi-Caption selects and processes text dynamically per exposure. V1
    # therefore cannot share a single cached Text Encoder embedding per image.
    # Keep latent caching allowed: image latents are independent of captions.
    if config.get("multi_caption_config"):
        if config.get("cache_text_encoder_outputs") or config.get("cache_text_encoder_outputs_to_disk"):
            raise ValueError(
                "Multi-Caption 当前不支持 Text Encoder Output Cache；"
                "请关闭 cache_text_encoder_outputs / cache_text_encoder_outputs_to_disk。"
                "Latent Cache 可以继续使用。"
            )
        if config.get("dataset_class"):
            raise ValueError("Multi-Caption v1 不支持自定义 dataset_class。")

    if not check_paths:
        normalize_optional_dataset_paths(config)
        for key in ("llm_adapter_path", "t5_tokenizer_path"):
            if not config.get(key):
                config.pop(key, None)
        return

    if is_file is None or is_dir is None or exists is None or inspect_data_dir is None or validate_model is None:
        from mikazuki.utils import train_utils

        is_file = is_file or os.path.isfile
        is_dir = is_dir or os.path.isdir
        exists = exists or os.path.exists
        inspect_data_dir = inspect_data_dir or train_utils.inspect_data_dir
        validate_model = validate_model or train_utils.validate_model

    if check_trainer and not exists(str(prepared.trainer_file)):
        raise ValueError(f"训练脚本不存在: {prepared.trainer_file}。请初始化/更新对应子模块。")

    # sd-scripts handles show_timesteps before loading model/text encoder/VAE
    # or constructing the dataset. Keep DTS equally side-effect-free and do not
    # make this diagnostic mode pretend to require runtime assets it never uses.
    if prepared.train_type in {"anima-lora", "anima-finetune"} and config.get("show_timesteps"):
        return

    model = config.get("pretrained_model_name_or_path")
    if not model:
        raise ValueError("必须指定 pretrained_model_name_or_path。")
    ok, message = validate_model(str(model), prepared.train_type)
    if not ok:
        raise ValueError(message)

    if prepared.train_type in {"anima-lora", "anima-finetune"}:
        required = (("qwen3", "Qwen3-0.6B"), ("vae", "Qwen-Image VAE"))
    elif prepared.train_type == "flux-lora":
        required = (("ae", "Flux AE"), ("t5xxl", "T5-XXL"), ("clip_l", "CLIP-L"))
    elif prepared.train_type == "chroma-lora":
        required = (("ae", "Chroma AE"), ("t5xxl", "T5-XXL"))
    else:
        required = ()

    for key, label in required:
        value = config.get(key)
        if not value:
            raise ValueError(f"{prepared.train_type} 训练需要指定 {label} 路径。")
        if not exists(str(value)):
            raise ValueError(f"{label} 路径不存在: {value}")

    for key in ("llm_adapter_path", "t5_tokenizer_path"):
        if not config.get(key):
            config.pop(key, None)

    prompt = config.get("sample_prompts")
    if isinstance(prompt, str) and prompt.strip():
        prompt_value = prompt.strip()
        looks_inline = any(flag in prompt_value for flag in ("--n", "--s", "--l", "--d"))
        looks_file = prompt_value.lower().endswith((".txt", ".toml", ".json"))
        pending_sidecar = prompt_value in getattr(prepared, "sidecars", {})
        if looks_file and not looks_inline and not pending_sidecar and not exists(prompt_value):
            raise ValueError(f"Prompt 文件不存在: {prompt_value}")

    validate_dataset_source(
        config,
        prepared.train_type,
        is_file=is_file,
        is_dir=is_dir,
        inspect_data_dir=inspect_data_dir,
    )


def validate_prepared_trainer(
    prepared,
    *,
    exists: Callable[[str], bool] | None = None,
) -> None:
    """Validate only the concrete trainer selected/materialized for launch."""

    exists = exists or os.path.exists
    if not exists(str(prepared.trainer_file)):
        raise ValueError(f"训练脚本不存在: {prepared.trainer_file}。请初始化/更新对应子模块。")
