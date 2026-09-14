"""Dataset source contract shared by every training backend."""

from __future__ import annotations

from collections.abc import Callable


DATASET_CONFIG_TRAIN_TYPES = {
    "sd-lora",
    "sdxl-lora",
    "sd-dreambooth",
    "sdxl-finetune",
    "flux-lora",
    "chroma-lora",
    "flux-finetune",
    "anima-lora",
    "anima-finetune",
    "sd3-lora",
}

OPTIONAL_DATASET_PATH_KEYS = (
    "dataset_config",
    "in_json",
    "conditioning_data_dir",
)


def normalize_optional_dataset_paths(config: dict) -> None:
    for key in OPTIONAL_DATASET_PATH_KEYS:
        if not config.get(key):
            config.pop(key, None)


def validate_dataset_source(
    config: dict,
    effective_train_type: str,
    *,
    is_file: Callable[[str], bool],
    is_dir: Callable[[str], bool],
    inspect_data_dir: Callable[[str], tuple[bool, str]],
) -> None:
    """Validate the selected dataset source without modifying user files."""
    normalize_optional_dataset_paths(config)

    dataset_config = config.get("dataset_config")
    if dataset_config:
        if effective_train_type not in DATASET_CONFIG_TRAIN_TYPES:
            raise ValueError("当前训练后端不支持 dataset_config。")
        if not is_file(str(dataset_config)):
            raise ValueError(f"dataset_config 文件不存在: {dataset_config}")
        for key in ("train_data_dir", "reg_data_dir", "in_json"):
            config.pop(key, None)
        return

    train_data_dir = config.get("train_data_dir")
    if not train_data_dir:
        raise ValueError("训练数据集路径不能为空；或请选择 dataset_config 数据源。")

    in_json = config.get("in_json")
    if in_json:
        if not is_file(str(in_json)):
            raise ValueError(f"metadata JSON 文件不存在: {in_json}")
        if not is_dir(str(train_data_dir)):
            raise ValueError(f"metadata 训练图片目录不存在: {train_data_dir}")
        return

    valid, message = inspect_data_dir(str(train_data_dir))
    if not valid:
        raise ValueError(message)
