
import asyncio
import json
import os
import sys
from typing import Optional

import toml

from mikazuki.anima_finetune_advanced import (
    normalize_anima_save_schedule,
    validate_anima_finetune_advanced_combinations,
    validate_effective_text_encoder_cache,
)
from mikazuki.anima_finetune_config import (
    normalize_anima_finetune_config,
    validate_anima_finetune_config,
)
from mikazuki.anima_qwen_config import (
    normalize_qwen_training_config,
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

ANIMA_FINETUNE_ONLY_KEYS = {
    "fused_backward_pass",
    "deepspeed",
    "zero_stage",
    "offload_optimizer_device",
    "offload_optimizer_nvme_path",
    "offload_param_device",
    "offload_param_nvme_path",
    "zero3_init_flag",
    "zero3_save_16bit_model",
    "fp16_master_weights_and_gradients",
    "torch_compile",
    "dynamo_backend",
    "ddp_static_graph",
    "dataset_config",
    "in_json",
    "masked_loss",
    "conditioning_data_dir",
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


def _prepare_anima_preview_flow_shift(config: dict, toml_path: str) -> None:
    """Apply the GUI preview flow shift to text prompt files safely.

    Anima's sampler reads ``--fs`` from each prompt entry; there is no global
    trainer argument for it. The GUI therefore sends ``sample_flow_shift`` as
    a host-only field. We create an autosave-side copy of a text prompt file
    and append ``--fs`` only to lines that do not already provide one, leaving
    user-owned prompt files untouched. TOML/JSON prompt files keep their own
    per-prompt flow_shift values and simply ignore the GUI-only default.
    """
    flow_shift = config.pop("sample_flow_shift", None)
    if flow_shift in (None, ""):
        return

    try:
        flow_shift_value = float(flow_shift)
    except (TypeError, ValueError) as e:
        raise ValueError("Anima: 预览 sample_flow_shift 必须是有效数字。") from e

    prompt_path = config.get("sample_prompts")
    if not prompt_path:
        return

    prompt_path = str(prompt_path)
    if not prompt_path.lower().endswith(".txt"):
        # Structured prompt files already support their own flow_shift key.
        return
    if not os.path.isfile(prompt_path):
        return

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        raise ValueError(f"Anima: 无法读取预览 Prompt 文件 {prompt_path}: {e}") from e

    changed = False
    rewritten = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]
        if stripped and not stripped.lstrip().startswith("#") and " --fs " not in stripped:
            stripped = f"{stripped} --fs {flow_shift_value:g}"
            changed = True
        rewritten.append(stripped + newline)

    if not changed:
        return

    derived_path = os.path.splitext(toml_path)[0] + "-anima-prompts.txt"
    try:
        with open(derived_path, "w", encoding="utf-8") as f:
            f.writelines(rewritten)
    except OSError as e:
        raise ValueError(f"Anima: 无法写入预览 Prompt 临时文件 {derived_path}: {e}") from e

    config["sample_prompts"] = derived_path


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

    # Convert the GUI's mutually-exclusive semantic controls first. Qwen3
    # validation must see the resulting effective cache/LR state rather than
    # the pre-normalized form values.
    normalize_anima_finetune_config(config, mode)
    normalize_anima_save_schedule(config)

    if mode == "finetune":
        validate_anima_finetune_advanced_combinations(config)

    # New Qwen3 fields are strictly opt-in. When disabled (or when LoRA is
    # selected) they are removed completely before sd-scripts sees the config.
    train_qwen3 = normalize_qwen_training_config(config, mode)
    validate_anima_finetune_config(config, mode)
    validate_effective_text_encoder_cache(config)
    _prepare_anima_preview_flow_shift(config, toml_path)

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
        # Full-finetune settings must not leak into anima_train_network.py when
        # the user loads a finetune preset and then switches back to LoRA.
        for key in ANIMA_OPTIONAL_FINETUNE_LRS | ANIMA_FINETUNE_ONLY_KEYS | {"cpu_offload_checkpointing"}:
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