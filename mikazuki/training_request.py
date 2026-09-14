"""Request-level helpers for preview/run; no trainer launch code lives here."""

from __future__ import annotations

import json
import os

import mikazuki.app.api as legacy_api
from mikazuki.training_config import prepare_training_config
from mikazuki.training_data_contract import validate_dataset_source
from mikazuki.utils import train_utils


def decode_training_request(body: bytes) -> tuple[str | None, dict]:
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("训练配置必须是 JSON object。")
    if isinstance(payload.get("config"), dict):
        return payload.get("train_type"), dict(payload["config"])
    return None, dict(payload)


def prepare_prompt_fields(config: dict, stamp: str, launch: bool) -> list[str]:
    warnings: list[str] = []
    prompt_file = str(config.pop("prompt_file", "") or "").strip()
    if prompt_file:
        if launch and not os.path.exists(prompt_file):
            raise ValueError(f"Prompt 文件 {prompt_file} 不存在。")
        config["sample_prompts"] = prompt_file
        return warnings

    host_keys = {
        "positive_prompts", "negative_prompts", "sample_width", "sample_height",
        "sample_cfg", "sample_seed", "sample_steps", "randomly_choice_prompt",
    }
    if not any(key in config for key in host_keys):
        return warnings
    if not launch:
        for key in host_keys:
            config.pop(key, None)
        warnings.append("Prompt 临时文件只在启动训练时生成；preview 不产生文件副作用。")
        return warnings

    positive, prompt_arg = legacy_api.get_sample_prompts(config)
    if positive is not None and train_utils.is_promopt_like(prompt_arg):
        path = os.path.join(os.getcwd(), "config", "autosave", f"{stamp}-promopt.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(prompt_arg)
        config["sample_prompts"] = path
    return warnings


def prepare_request_config(config: dict, page_type: str | None, stamp: str, launch: bool, toml_path: str | None = None):
    train_utils.fix_config_types(config)
    prompt_warnings = prepare_prompt_fields(config, stamp, launch)
    prepared = prepare_training_config(
        config,
        page_train_type=page_type,
        resolve_backend=legacy_api.resolve_training_backend,
        launch=launch,
        toml_path=toml_path,
    )
    prepared.warnings.extend(prompt_warnings)
    return prepared


def validate_prepared_config(prepared, check_paths: bool) -> None:
    config = prepared.config
    validate_dataset_source(
        config,
        prepared.train_type,
        is_file=os.path.isfile if check_paths else lambda _path: True,
        validate_data_dir=train_utils.validate_data_dir if check_paths else lambda _path: True,
    )
    if prepared.train_type in {"anima-lora", "anima-finetune"}:
        if check_paths and not os.path.exists(prepared.trainer_file):
            raise ValueError("Anima 训练脚本不存在，请初始化 sd-scripts 子模块。")
        for key, label in (("qwen3", "Qwen3-0.6B"), ("vae", "Qwen-Image VAE")):
            value = config.get(key)
            if not value:
                raise ValueError(f"Anima 训练需要指定 {label} 路径。")
            if check_paths and not os.path.exists(value):
                raise ValueError(f"{label} 路径不存在: {value}")
        for key in ("llm_adapter_path", "t5_tokenizer_path"):
            if not config.get(key):
                config.pop(key, None)
    if check_paths:
        model = config.get("pretrained_model_name_or_path")
        if not model:
            raise ValueError("必须指定 pretrained_model_name_or_path。")
        ok, message = train_utils.validate_model(model, prepared.train_type)
        if not ok:
            raise ValueError(message)
