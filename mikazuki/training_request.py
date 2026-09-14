"""Request-level helpers for preview/run; no trainer launch code lives here."""

from __future__ import annotations

import json
import os

import mikazuki.app.api as legacy_api
from mikazuki.training_config import prepare_training_config
from mikazuki.training_validation import validate_prepared_config
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


__all__ = [
    "decode_training_request",
    "prepare_prompt_fields",
    "prepare_request_config",
    "validate_prepared_config",
]
