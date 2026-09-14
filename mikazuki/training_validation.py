"""Pure validation for prepared training configs.

This module deliberately has no FastAPI/application imports so preview/launch
validation can be unit-tested without booting the web application.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from mikazuki.training_data_contract import normalize_optional_dataset_paths, validate_dataset_source


def validate_prepared_config(
    prepared,
    check_paths: bool,
    *,
    is_file: Callable[[str], bool] | None = None,
    validate_data_dir: Callable[[str], bool] | None = None,
    validate_model: Callable[[str, str], tuple[bool, str]] | None = None,
) -> None:
    """Validate semantic preview or launch-time runtime assets.

    Semantic contradictions are normalized/validated before this function is
    called. Preview must remain usable while paths are blank or files do not yet
    exist. Launch is strict and checks datasets, Anima side assets, trainer path,
    and the base model.
    """
    config = prepared.config

    if not check_paths:
        normalize_optional_dataset_paths(config)
        for key in ("llm_adapter_path", "t5_tokenizer_path"):
            if not config.get(key):
                config.pop(key, None)
        return

    if is_file is None or validate_data_dir is None or validate_model is None:
        from mikazuki.utils import train_utils

        is_file = is_file or os.path.isfile
        validate_data_dir = validate_data_dir or train_utils.validate_data_dir
        validate_model = validate_model or train_utils.validate_model

    validate_dataset_source(
        config,
        prepared.train_type,
        is_file=is_file,
        validate_data_dir=validate_data_dir,
    )

    if prepared.train_type in {"anima-lora", "anima-finetune"}:
        if not os.path.exists(prepared.trainer_file):
            raise ValueError("Anima 训练脚本不存在，请初始化 sd-scripts 子模块。")
        for key, label in (("qwen3", "Qwen3-0.6B"), ("vae", "Qwen-Image VAE")):
            value = config.get(key)
            if not value:
                raise ValueError(f"Anima 训练需要指定 {label} 路径。")
            if not os.path.exists(value):
                raise ValueError(f"{label} 路径不存在: {value}")
        for key in ("llm_adapter_path", "t5_tokenizer_path"):
            if not config.get(key):
                config.pop(key, None)

    model = config.get("pretrained_model_name_or_path")
    if not model:
        raise ValueError("必须指定 pretrained_model_name_or_path。")
    ok, message = validate_model(model, prepared.train_type)
    if not ok:
        raise ValueError(message)
