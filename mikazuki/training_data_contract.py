"""Pure dataset-source normalization for training backends."""

from __future__ import annotations

from collections.abc import Callable


DATASET_CONFIG_TRAIN_TYPES = {
    "sdxl-finetune",
    "flux-finetune",
    "anima-finetune",
}

OPTIONAL_DATASET_PATH_KEYS = (
    "dataset_config",
    "in_json",
    "conditioning_data_dir",
)


def normalize_optional_dataset_paths(config: dict) -> None:
    """Remove empty optional paths so argparse receives None, not an empty path."""
    for key in OPTIONAL_DATASET_PATH_KEYS:
        if not config.get(key):
            config.pop(key, None)


def validate_dataset_source(
    config: dict,
    effective_train_type: str,
    *,
    is_file: Callable[[str], bool],
    validate_data_dir: Callable[[str], bool],
) -> None:
    """Validate and normalize the effective dataset source before launch.

    SDXL, Flux and Anima full trainers all give dataset_config precedence over
    train_data_dir and in_json. In directory/metadata mode, train_data_dir is
    still required because in_json only changes subset metadata construction.
    """
    normalize_optional_dataset_paths(config)

    dataset_config = config.get("dataset_config")
    if dataset_config:
        if effective_train_type not in DATASET_CONFIG_TRAIN_TYPES:
            raise ValueError("当前页面不支持 dataset_config。")
        if not is_file(dataset_config):
            raise ValueError(f"dataset_config 文件不存在: {dataset_config}")
        # These are explicitly ignored by all three full trainers when a dataset
        # config is present. Remove them so the final TOML reflects reality and
        # callers do not scan stale/default image directories unnecessarily.
        config.pop("train_data_dir", None)
        config.pop("in_json", None)
        return

    train_data_dir = config.get("train_data_dir")
    if not train_data_dir or not validate_data_dir(train_data_dir):
        raise ValueError("训练数据集路径不存在或没有图片，请检查目录。")

    in_json = config.get("in_json")
    if in_json and not is_file(in_json):
        raise ValueError(f"metadata JSON 文件不存在: {in_json}")
