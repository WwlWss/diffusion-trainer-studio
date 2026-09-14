"""Raw Schemastery GUI argument compilation owned by the Python backend.

This module contains the former parseParams() conversions for network, optimizer,
base-weight, custom TOML, path and Basic-page compatibility controls.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import re

try:  # Python 3.11+
    import tomllib as _toml_reader
except ModuleNotFoundError:  # pragma: no cover - compatibility with older app Python
    import toml as _toml_reader


PRODIGY_TYPES = {"prodigy", "prodigyplus.prodigyplusschedulefree"}
_BASIC_LORA_DEFAULTS = {
    "save_model_as": "safetensors",
    "enable_bucket": True,
    "min_bucket_reso": 256,
    "max_bucket_reso": 1024,
    "learning_rate": "1e-4",
    "lr_scheduler_num_cycles": 1,
    "network_module": "networks.lora",
    "logging_dir": "./logs",
    "caption_extension": ".txt",
    "seed": 1337,
    "prior_loss_weight": 1,
    "clip_skip": 2,
    "persistent_data_loader_workers": False,
}
_PATH_FIELDS = {
    "pretrained_model_name_or_path", "train_data_dir", "reg_data_dir", "output_dir",
    "network_weights", "dataset_config", "in_json", "vae", "ae", "t5xxl", "clip_l",
    "qwen3", "prompt_file", "llm_adapter_path", "t5_tokenizer_path",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == []


def _items(value: object) -> list[str]:
    if _is_empty(value):
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _arg_key(value: str) -> str:
    return value.split("=", 1)[0].strip().lower()


def _merge_args(existing: object, generated: Iterable[str] = (), custom: object = None) -> list[str]:
    result: list[str] = []
    key_to_index: dict[str, int] = {}
    for value in [*_items(existing), *[str(x) for x in generated], *_items(custom)]:
        value = value.strip()
        if not value:
            continue
        key = _arg_key(value)
        if key in key_to_index:
            result[key_to_index[key]] = value
        else:
            key_to_index[key] = len(result)
            result.append(value)
    return result


def _normalize_paths_and_gpu(config: dict) -> None:
    for field in _PATH_FIELDS:
        value = config.get(field)
        if isinstance(value, str) and value:
            config[field] = value.replace("\\", "/")
    gpu_ids = config.get("gpu_ids")
    if isinstance(gpu_ids, list):
        normalized = []
        for value in gpu_ids:
            match = re.search(r"GPU\s+(\d+):", str(value))
            normalized.append(match.group(1) if match else str(value))
        config["gpu_ids"] = normalized


def _parse_ui_custom_params(config: dict) -> None:
    payload = config.pop("ui_custom_params", None)
    if _is_empty(payload):
        return
    if not isinstance(payload, str):
        raise ValueError("ui_custom_params 必须是 TOML 文本。")
    try:
        parsed = _toml_reader.loads(payload)
    except Exception as exc:
        raise ValueError(f"ui_custom_params TOML 解析失败: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("ui_custom_params 必须解析为 TOML 顶层键值。")
    config["__ui_custom_overrides"] = parsed


def _normalize_network_args(config: dict) -> None:
    custom = config.pop("network_args_custom", None)
    generated: list[str] = []
    module = str(config.get("network_module") or "")
    if module == "lycoris.kohya":
        mapping = {
            "lycoris_algo": "algo", "conv_dim": "conv_dim", "conv_alpha": "conv_alpha",
            "lokr_factor": "factor", "dropout": "dropout", "train_norm": "train_norm",
        }
        for field, argument in mapping.items():
            if field not in config:
                continue
            value = config.pop(field)
            if _is_empty(value):
                continue
            if isinstance(value, bool):
                value = str(value).lower()
            generated.append(f"{argument}={value}")
    else:
        for field in ("lycoris_algo", "conv_dim", "conv_alpha", "lokr_factor", "dropout", "train_norm"):
            config.pop(field, None)

    if module == "networks.dylora":
        value = config.pop("dylora_unit", None)
        if not _is_empty(value):
            generated.append(f"unit={value}")
    else:
        config.pop("dylora_unit", None)

    if _as_bool(config.pop("enable_block_weights", False)):
        for field in ("down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold"):
            value = config.pop(field, None)
            if not _is_empty(value):
                generated.append(f"{field}={value}")
    else:
        for field in ("down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold"):
            config.pop(field, None)

    merged = _merge_args(config.get("network_args"), generated, custom)
    if merged:
        config["network_args"] = merged
    else:
        config.pop("network_args", None)


def _normalize_base_weights(config: dict) -> None:
    enabled = _as_bool(config.pop("enable_base_weight", False))
    if not enabled:
        config.pop("base_weights", None)
        config.pop("base_weights_multiplier", None)
        return
    weights = _items(config.get("base_weights"))
    multipliers_raw = _items(config.get("base_weights_multiplier"))
    if not weights:
        config.pop("base_weights", None)
        config.pop("base_weights_multiplier", None)
        return
    config["base_weights"] = weights
    if multipliers_raw:
        try:
            multipliers = [float(value) for value in multipliers_raw]
        except ValueError as exc:
            raise ValueError("base_weights_multiplier 每一行都必须是数字。") from exc
        if len(multipliers) not in {1, len(weights)}:
            raise ValueError("base_weights_multiplier 必须只有 1 个值，或与 base_weights 数量一致。")
        config["base_weights_multiplier"] = multipliers
    else:
        config.pop("base_weights_multiplier", None)


def _normalize_optimizer_args(config: dict) -> None:
    custom = config.pop("optimizer_args_custom", None)
    generated: list[str] = []
    optimizer = str(config.get("optimizer_type") or "")
    lower = optimizer.lower()
    if lower.startswith("dadapt"):
        if lower in {"dadaptation", "dadaptadam"}:
            generated.extend(("decouple=True", "weight_decay=0.01"))
    if lower in PRODIGY_TYPES:
        generated.extend(("decouple=True", "weight_decay=0.01", "use_bias_correction=True"))
        d0 = config.pop("prodigy_d0", None)
        d_coef = config.pop("prodigy_d_coef", None)
        if not _is_empty(d_coef):
            generated.append(f"d_coef={d_coef}")
        if not _is_empty(d0):
            generated.append(f"d0={d0}")
        if config.get("lr_warmup_steps"):
            generated.append("safeguard_warmup=True")
    else:
        config.pop("prodigy_d0", None)
        config.pop("prodigy_d_coef", None)
    merged = _merge_args(config.get("optimizer_args"), generated, custom)
    if merged:
        config["optimizer_args"] = merged
    else:
        config.pop("optimizer_args", None)


def apply_raw_gui_semantics(raw_config: dict, *, page_train_type: str | None = None) -> dict:
    """Compile raw Schemastery state into a backend-owned intermediate config."""
    config = deepcopy(raw_config)
    if page_train_type == "lora-basic":
        config = {**_BASIC_LORA_DEFAULTS, **config}
    _normalize_paths_and_gpu(config)
    _parse_ui_custom_params(config)
    _normalize_network_args(config)
    _normalize_base_weights(config)
    _normalize_optimizer_args(config)
    return config
