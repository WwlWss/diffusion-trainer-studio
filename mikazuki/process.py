
import asyncio
import json
import os
import sys
from typing import Optional

import toml

from mikazuki.anima_qwen_config import (
    normalize_qwen_training_config,
    text_encoder_cache_enabled,
    trainer_supports_qwen_training,
)
from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm


ANIMA_LORA_ONLY_KEYS = {
    "network_module",
    "network_weights",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "scale_weight_norms",
    "network_args",
    "network_args_custom",
    "network_train_unet_only",
    "network_train_text_encoder_only",
    "enable_base_weight",
    "base_weights",
    "base_weights_multiplier",
    "unet_lr",
    "text_encoder_lr",
}

ANIMA_OPTIONAL_FINETUNE_LRS = {
    "self_attn_lr",
    "cross_attn_lr",
    "mlp_lr",
    "mod_lr",
    "llm_adapter_lr",
}

ANIMA_VARIANTS = {"base", "2.9b"}


def _detect_anima_variant(model_path: str) -> Optional[str]:
    """Detect standard 28-block Anima vs expanded 40-block Anima 2.9B.

    Safetensors keeps its tensor names in a JSON header, so this check does not
    load model tensors into RAM or VRAM. Prefixes such as ``net.`` and
    ``model.diffusion_model.`` are intentionally ignored by using ``endswith``.
    """
    if not model_path or not os.path.isfile(model_path) or not model_path.lower().endswith(".safetensors"):
        return None

    try:
        with open(model_path, "rb") as f:
            metadata_length = int.from_bytes(f.read(8), "little")
            metadata = json.loads(f.read(metadata_length))
    except Exception as e:
        log.warning(f"Unable to inspect Anima checkpoint block count: {e}")
        return None

    keys = metadata.keys()
    if any(key.endswith("blocks.39.mlp.layer1.weight") for key in keys):
        return "2.9b"
    if any(key.endswith("blocks.27.mlp.layer1.weight") for key in keys):
        return "base"
    return None


def _validate_effective_text_encoder_cache(config: dict) -> None:
    """Mirror sd-scripts semantics before launching the subprocess.

    sd-scripts treats cache_text_encoder_outputs_to_disk as implicitly enabling
    the text-encoder cache. Validate the effective state here so disk-only cache
    configurations cannot bypass the GUI-side compatibility checks.
    """
    if not text_encoder_cache_enabled(config):
        return
    if config.get("shuffle_caption"):
        raise ValueError("Anima: 缓存 Qwen3 输出时必须关闭 shuffle_caption")
    if float(config.get("caption_tag_dropout_rate") or 0) > 0:
        raise ValueError("Anima: 缓存 Qwen3 输出时不能启用 caption_tag_dropout_rate")


def _resolve_anima_trainer(toml_path: str, trainer_file: str) -> str:
    """Prepare Anima jobs and switch between LoRA and full finetune.

    The packaged frontend uses the existing Flux expert route, so the GUI sends
    lightweight routing fields for the Anima training mode and model variant.
    Those fields are removed before sd-scripts sees the config.
    """
    normalized_trainer = trainer_file.replace("\\", "/")
    is_anima_lora = normalized_trainer.endswith("/sd-scripts/anima_train_network.py")
    is_anima_finetune = normalized_trainer.endswith("/sd-scripts/anima_train.py")
    if not (is_anima_lora or is_anima_finetune):
        return trainer_file

    try:
        config = toml.load(toml_path)
    except Exception as e:
        log.warning(f"Unable to inspect Anima training config, using selected trainer unchanged: {e}")
        return trainer_file

    default_mode = "finetune" if is_anima_finetune else "lora"
    mode = str(config.pop("anima_training_mode", default_mode)).lower()
    if mode not in {"lora", "finetune"}:
        log.warning(f"Unknown Anima training mode '{mode}', falling back to {default_mode}")
        mode = default_mode

    # New Qwen3 fields are strictly opt-in. When disabled (or when LoRA is
    # selected) they are removed completely before sd-scripts sees the config.
    train_qwen3 = normalize_qwen_training_config(config, mode)
    _validate_effective_text_encoder_cache(config)

    variant = str(config.pop("anima_model_variant", "base")).lower()
    if variant not in ANIMA_VARIANTS:
        raise ValueError(f"Unsupported Anima model variant: {variant}")

    detected_variant = _detect_anima_variant(config.get("pretrained_model_name_or_path", ""))
    if detected_variant is not None and detected_variant != variant:
        selected_name = "Anima 2.9B (40 blocks)" if variant == "2.9b" else "Anima (28 blocks)"
        detected_name = "Anima 2.9B (40 blocks)" if detected_variant == "2.9b" else "Anima (28 blocks)"
        raise ValueError(f"选择的是 {selected_name}，但检测到底模为 {detected_name}。请切换 Anima 模型版本选项或选择正确的底模。")

    blocks_to_swap = int(config.get("blocks_to_swap") or 0)
    max_blocks_to_swap = 38 if variant == "2.9b" else 26
    if blocks_to_swap > max_blocks_to_swap:
        raise ValueError(
            f"{('Anima 2.9B' if variant == '2.9b' else 'Anima')} 最多允许 blocks_to_swap={max_blocks_to_swap}，"
            f"当前值为 {blocks_to_swap}。"
        )

    log.info(
        "Anima model variant: %s",
        "2.9B / 40 blocks" if variant == "2.9b" else "standard / 28 blocks",
    )

    if mode == "finetune":
        for key in ANIMA_LORA_ONLY_KEYS:
            config.pop(key, None)

        # Empty optional component-LR fields must be omitted, otherwise argparse's
        # float conversion sees an empty string instead of the intended None value.
        for key in ANIMA_OPTIONAL_FINETUNE_LRS:
            if config.get(key) in (None, ""):
                config.pop(key, None)

        trainer_file = "./sd-scripts/anima_train.py"
        if not os.path.exists(trainer_file):
            raise FileNotFoundError(
                "Anima full finetune script is missing. Run `git submodule update --init --recursive`."
            )
        if train_qwen3 and not trainer_supports_qwen_training(trainer_file):
            raise RuntimeError(
                "当前 sd-scripts 子模块尚未包含 Anima Qwen3 联合训练补丁。"
                "普通 Anima LoRA / 全参微调不受影响；请更新到支持 "
                "--train_qwen3_text_encoder 的 sd-scripts 版本。"
            )
        if train_qwen3 and config.get("qwen3_output_dir"):
            try:
                os.makedirs(config["qwen3_output_dir"], exist_ok=True)
            except OSError as e:
                raise ValueError(f"Anima: 无法创建 Qwen3 输出目录 {config['qwen3_output_dir']}: {e}") from e
        log.info("Anima full finetune selected; using sd-scripts/anima_train.py")
    else:
        # Component LRs belong to full finetune only. Remove stale values when a
        # finetune preset was loaded and the user switched the form back to LoRA.
        for key in ANIMA_OPTIONAL_FINETUNE_LRS | {"cpu_offload_checkpointing"}:
            config.pop(key, None)
        trainer_file = "./sd-scripts/anima_train_network.py"
        log.info("Anima LoRA selected; using sd-scripts/anima_train_network.py")

    # GUI-only routing keys must never reach sd-scripts.
    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return trainer_file


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")

    try:
        trainer_file = _resolve_anima_trainer(toml_path, trainer_file)
    except Exception as e:
        log.error(f"Failed to prepare Anima training / Anima 训练准备失败: {e}")
        return APIResponse(status="error", message=str(e))

    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
            if sys.platform == "win32":
                customize_env["USE_LIBUV"] = "0"
                args[3:3] = ["--rdzv_backend", "c10d"]

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    def _run():
        try:
            task.execute()
            result = task.communicate()
            if result.returncode != 0:
                log.error(f"Training failed / 训练失败")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")