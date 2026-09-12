import asyncio
import hashlib
import json
import os
import re
import random

from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import toml
from fastapi import APIRouter, BackgroundTasks, Request
from starlette.requests import Request

import mikazuki.process as process
from mikazuki import launch_utils
from mikazuki.anima_qwen_config import (
    normalize_qwen_training_config,
    text_encoder_cache_enabled,
    trainer_supports_qwen_training,
)
from mikazuki.app.config import app_config
from mikazuki.app.models import (APIResponse, APIResponseFail,
                                 APIResponseSuccess, TaggerInterrogateRequest)
from mikazuki.log import log
from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)
from mikazuki.tasks import tm
from mikazuki.utils import train_utils
from mikazuki.utils.devices import printable_devices
from mikazuki.utils.tk_window import (open_directory_selector,
                                      open_file_selector)

router = APIRouter()

avaliable_scripts = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py",
    "networks/merge_lora.py",
    "tools/merge_models.py",
]

avaliable_schemas = []
avaliable_presets = []

trainer_mapping = {
    "sd-lora": "./scripts/stable/train_network.py",
    "sdxl-lora": "./scripts/stable/sdxl_train_network.py",

    "sd-dreambooth": "./scripts/stable/train_db.py",
    "sdxl-finetune": "./scripts/stable/sdxl_train.py",

    "sd3-lora": "./scripts/dev/sd3_train_network.py",
    "flux-lora": "./scripts/dev/flux_train_network.py",
    "flux-finetune": "./scripts/dev/flux_train.py",
    "anima-lora": "./sd-scripts/anima_train_network.py",
    "anima-finetune": "./sd-scripts/anima_train.py",
}

ANIMA_ONLY_KEYS = {
    "qwen3",
    "vae",
    "llm_adapter_path",
    "t5_tokenizer_path",
    "qwen3_max_token_length",
    "t5_max_token_length",
    "attn_mode",
    "split_attn",
    "vae_chunk_size",
    "vae_disable_cache",
    "qwen_image_vae_2d",
    "blocks_to_swap",
    "unsloth_offload_checkpointing",
    "compile",
    "compile_backend",
    "compile_mode",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_cache_size_limit",
    "cuda_allow_tf32",
    "cuda_cudnn_benchmark",
}

ANIMA_FINETUNE_ONLY_KEYS = {
    "self_attn_lr",
    "cross_attn_lr",
    "mlp_lr",
    "mod_lr",
    "llm_adapter_lr",
    "cpu_offload_checkpointing",
}

# Old SD/SDXL fields that are either meaningless or actively misleading for
# Anima's DiT / Rectified Flow path. Keep them out even when an older preset or
# ui_custom_params injects them.
ANIMA_LEGACY_KEYS = {
    "unet_lr",
    "text_encoder_lr",
    "clip_skip",
    "noise_offset",
    "multires_noise_iterations",
    "multires_noise_discount",
    "min_snr_gamma",
    "weighted_captions",
}

FLUX_ONLY_KEYS = {
    "ae",
    "clip_l",
    "t5xxl",
    "t5xxl_max_token_length",
    "train_t5xxl",
    "apply_t5_attn_mask",
    "model_prediction_type",
    "guidance_scale",
    "fp8_base",
    "fp8_base_unet",
}


def _strip_keys(config: dict, keys) -> None:
    for key in keys:
        config.pop(key, None)


def _strip_network_training_keys(config: dict) -> None:
    """Remove LoRA/network-only values before launching a full finetune."""
    for key in list(config.keys()):
        if key.startswith("network_") or key in {
            "scale_weight_norms",
            "enable_base_weight",
            "base_weights",
            "base_weights_multiplier",
        }:
            config.pop(key, None)


def _normalize_anima_common_config(config: dict) -> None:
    """Normalize shared GUI values to the semantics expected by sd-scripts Anima."""
    _strip_keys(config, FLUX_ONLY_KEYS | ANIMA_LEGACY_KEYS)

    # Step-based training is much more useful for heavily repeated datasets.
    # Avoid passing both limits because max_train_epochs can recalculate the
    # effective step count in sd-scripts.
    if config.get("max_train_steps") not in (None, "", 0):
        config.pop("max_train_epochs", None)

    # Anima's xformers implementation requires split attention.
    if config.get("attn_mode") == "xformers":
        config["split_attn"] = True

    blocks_to_swap = int(config.get("blocks_to_swap") or 0)
    unsloth_offload = bool(config.get("unsloth_offload_checkpointing"))
    cpu_offload = bool(config.get("cpu_offload_checkpointing"))

    if blocks_to_swap > 0 and unsloth_offload:
        raise ValueError("Anima: blocks_to_swap 不能与 unsloth_offload_checkpointing 同时启用")
    if blocks_to_swap > 0 and cpu_offload:
        raise ValueError("Anima: blocks_to_swap 不能与 cpu_offload_checkpointing 同时启用")
    if unsloth_offload and cpu_offload:
        raise ValueError("Anima: unsloth_offload_checkpointing 不能与 cpu_offload_checkpointing 同时启用")

    # sd-scripts treats *_to_disk as enabling the text-encoder cache even when
    # cache_text_encoder_outputs itself is false. Validate the effective state,
    # not only the visible checkbox.
    if text_encoder_cache_enabled(config):
        if config.get("shuffle_caption"):
            raise ValueError("Anima: 缓存 Qwen3 输出时必须关闭 shuffle_caption")
        if float(config.get("caption_tag_dropout_rate") or 0) > 0:
            raise ValueError("Anima: 缓存 Qwen3 输出时不能启用 caption_tag_dropout_rate")


def resolve_training_backend(config: dict, model_train_type: str) -> Tuple[str, str]:
    """Resolve GUI model/mode selection to a concrete trainer.

    The shared Flux/Chroma/Anima page is still loaded through the historical
    `flux-lora` frontend route. Do not trust that route value for Anima: the
    actual backend is selected from `model_type` + `anima_training_mode`.
    """
    model_type = config.get("model_type")
    anima_training_mode = config.pop("anima_training_mode", None)

    if model_type == "anima":
        if anima_training_mode is None:
            anima_training_mode = "finetune" if model_train_type == "anima-finetune" else "lora"
        if anima_training_mode not in {"lora", "finetune"}:
            raise ValueError(f"Unsupported Anima training mode: {anima_training_mode}")

        config.pop("model_type", None)
        _normalize_anima_common_config(config)
        normalize_qwen_training_config(config, anima_training_mode)

        if anima_training_mode == "finetune":
            _strip_network_training_keys(config)
            for key in ANIMA_FINETUNE_ONLY_KEYS:
                if config.get(key) in (None, ""):
                    config.pop(key, None)
            effective_train_type = "anima-finetune"
        else:
            _strip_keys(config, ANIMA_FINETUNE_ONLY_KEYS)
            # The GUI intentionally trains only the DiT LoRA. Qwen3 remains
            # frozen; this is also required when text encoder outputs are cached.
            config["network_train_unet_only"] = True
            config.pop("network_train_text_encoder_only", None)
            effective_train_type = "anima-lora"

        return effective_train_type, trainer_mapping[effective_train_type]

    _strip_keys(config, ANIMA_ONLY_KEYS | ANIMA_FINETUNE_ONLY_KEYS)
    normalize_qwen_training_config(config, "disabled")
    return model_train_type, trainer_mapping[model_train_type]


async def load_schemas():
    avaliable_schemas.clear()

    schema_dir = os.path.join(os.getcwd(), "mikazuki", "schema")
    schemas = os.listdir(schema_dir)

    def lambda_hash(x):
        return hashlib.md5(x.encode()).hexdigest()

    for schema_name in schemas:
        with open(os.path.join(schema_dir, schema_name), encoding="utf-8") as f:
            content = f.read()
            avaliable_schemas.append({
                "name": schema_name.rstrip(".ts"),
                "schema": content,
                "hash": lambda_hash(content)
            })


async def load_presets():
    avaliable_presets.clear()

    preset_dir = os.path.join(os.getcwd(), "config", "presets")
    presets = os.listdir(preset_dir)

    for preset_name in presets:
        with open(os.path.join(preset_dir, preset_name), encoding="utf-8") as f:
            content = f.read()
            avaliable_presets.append(toml.loads(content))


def get_sample_prompts(config: dict) -> Tuple[Optional[str], str]:
    # backward compatibility
    if "sample_prompts" in config and "positive_prompts" not in config:
        return None, config["sample_prompts"]

    train_data_dir = config["train_data_dir"]
    sub_dir = [dir for dir in glob(os.path.join(train_data_dir, '*')) if os.path.isdir(dir)]

    positive_prompts = config.pop('positive_prompts', None)
    negative_prompts = config.pop('negative_prompts', '')
    sample_width = config.pop('sample_width', 512)
    sample_height = config.pop('sample_height', 512)
    sample_cfg = config.pop('sample_cfg', 7)
    sample_seed = config.pop('sample_seed', 2333)
    sample_steps = config.pop('sample_steps', 24)
    randomly_choice_prompt = config.pop('randomly_choice_prompt', False)

    if randomly_choice_prompt:
        if len(sub_dir) != 1:
            raise ValueError('训练数据集下有多个子文件夹，无法启用随机选取 Prompt 功能')

        txt_files = glob(os.path.join(sub_dir[0], '*.txt'))
        if not txt_files:
            raise ValueError('训练数据集路径没有 txt 文件')
        try:
            sample_prompt_file = random.choice(txt_files)
            with open(sample_prompt_file, 'r', encoding='utf-8') as f:
                positive_prompts = f.read()
        except IOError:
            log.error(f"读取 {sample_prompt_file} 文件失败")

    return positive_prompts, f'{positive_prompts} --n {negative_prompts}  --w {sample_width} --h {sample_height} --l {sample_cfg}  --s {sample_steps}  --d {sample_seed}'


@router.post("/run")
async def create_toml_file(request: Request):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}.toml")
    json_data = await request.body()

    config: dict = json.loads(json_data.decode("utf-8"))
    train_utils.fix_config_types(config)

    gpu_ids = config.pop("gpu_ids", None)

    suggest_cpu_threads = 8 if len(train_utils.get_total_images(config["train_data_dir"])) > 200 else 2
    model_train_type = config.pop("model_train_type", "sd-lora")
    try:
        effective_train_type, trainer_file = resolve_training_backend(config, model_train_type)
    except (KeyError, ValueError) as e:
        return APIResponseFail(message=f"训练类型配置无效: {e}")

    if effective_train_type != "sdxl-finetune":
        if not train_utils.validate_data_dir(config["train_data_dir"]):
            return APIResponseFail(message="训练数据集路径不存在或没有图片，请检查目录。")

    if effective_train_type in {"anima-lora", "anima-finetune"}:
        if not os.path.exists(trainer_file):
            return APIResponseFail(
                message="Anima 训练脚本不存在。请运行 `git submodule update --init --recursive` 后重启 GUI。"
            )
        for key, label in [("qwen3", "Qwen3-0.6B"), ("vae", "Qwen-Image VAE")]:
            value = config.get(key)
            if not value:
                return APIResponseFail(message=f"Anima 训练需要指定 {label} 路径。")
            if not os.path.exists(value):
                return APIResponseFail(message=f"{label} 路径不存在: {value}")

        # Optional Anima paths must be omitted instead of serialized as empty strings.
        # sd-scripts treats a non-None empty path as a real directory and fails to load it.
        for key in ("llm_adapter_path", "t5_tokenizer_path"):
            if not config.get(key):
                config.pop(key, None)

        if effective_train_type == "anima-lora":
            config.setdefault("network_module", "networks.lora_anima")

        if config.get("train_qwen3_text_encoder") and not trainer_supports_qwen_training(trainer_file):
            return APIResponseFail(
                message=(
                    "当前 sd-scripts 子模块尚未包含 Anima Qwen3 联合训练补丁。"
                    "普通 Anima LoRA / 全参微调不受影响；请先更新到支持 "
                    "--train_qwen3_text_encoder 的 sd-scripts 版本。"
                )
            )

    validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"], effective_train_type)
    if not validated:
        return APIResponseFail(message=message)

    if "prompt_file" in config and config["prompt_file"].strip() != "":
        prompt_file = config["prompt_file"].strip()
        if not os.path.exists(prompt_file):
            return APIResponseFail(message=f"Prompt 文件 {prompt_file} 不存在，请检查路径。")
        config["sample_prompts"] = prompt_file
    else:
        try:
            positive_prompt, sample_prompts_arg = get_sample_prompts(config=config)

            if positive_prompt is not None and train_utils.is_promopt_like(sample_prompts_arg):
                sample_prompts_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}-promopt.txt")
                with open(sample_prompts_file, "w", encoding="utf-8") as f:
                    f.write(sample_prompts_arg)
                config["sample_prompts"] = sample_prompts_file
                log.info(f"Wrote prompts to file {sample_prompts_file}")

        except ValueError as e:
            log.error(f"Error while processing prompts: {e}")
            return APIResponseFail(message=str(e))

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))

    result = process.run_train(toml_file, trainer_file, gpu_ids, suggest_cpu_threads)

    return result


@router.post("/run_script")
async def run_script(request: Request, background_tasks: BackgroundTasks):
    paras = await request.body()
    j = json.loads(paras.decode("utf-8"))
    script_name = j["script_name"]
    if script_name not in avaliable_scripts:
        return APIResponseFail(message="Script not found")
    del j["script_name"]
    result = []
    for k, v in j.items():
        result.append(f"--{k}")
        if not isinstance(v, bool):
            value = str(v)
            if " " in value:
                value = f'"{v}"'
            result.append(value)
    script_args = " ".join(result)
    script_path = Path(os.getcwd()) / "scripts" / script_name
    cmd = f"{launch_utils.python_bin} {script_path} {script_args}"
    background_tasks.add_task(launch_utils.run, cmd)
    return APIResponseSuccess()


@router.post("/interrogate")
async def run_interrogate(req: TaggerInterrogateRequest, background_tasks: BackgroundTasks):
    interrogator = available_interrogators.get(req.interrogator_model, available_interrogators["wd14-convnextv2-v2"])
    background_tasks.add_task(
        on_interrogate,
        image=None,
        batch_input_glob=req.path,
        batch_input_recursive=req.batch_input_recursive,
        batch_output_dir="",
        batch_output_filename_format="[name].[output_extension]",
        batch_output_action_on_conflict=req.batch_output_action_on_conflict,
        batch_remove_duplicated_tag=True,
        batch_output_save_json=False,
        interrogator=interrogator,
        threshold=req.threshold,
        character_threshold=req.character_threshold,
        add_rating_tag=req.add_rating_tag,
        add_model_tag=req.add_model_tag,
        additional_tags=req.additional_tags,
        exclude_tags=req.exclude_tags,
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=req.replace_underscore,
        replace_underscore_excludes=req.replace_underscore_excludes,
        escape_tag=req.escape_tag,
        unload_model_after_running=True
    )
    return APIResponseSuccess()


@router.get("/pick_file")
async def pick_file(picker_type: str):
    if picker_type == "folder":
        coro = asyncio.to_thread(open_directory_selector, "")
    elif picker_type == "model-file":
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt;*.pth"), ("all files", "*.*")]
        coro = asyncio.to_thread(open_file_selector, "", "Select file", file_types)

    result = await coro
    if result == "":
        return APIResponseFail(message="用户取消选择")

    return APIResponseSuccess(data={
        "path": result
    })


@router.get("/get_files")
async def get_files(pick_type) -> APIResponse:
    pick_preset = {
        "model-file": {
            "type": "file",
            "path": "./sd-models",
            "filter": "(.safetensors|.ckpt|.pt|.pth)"
        },
        "model-saved-file": {
            "type": "file",
            "path": "./output",
            "filter": "(.safetensors|.ckpt|.pt)"
        },
        "train-dir": {
            "type": "folder",
            "path": "./train",
            "filter": None
        },
    }

    folder_blacklist = [".ipynb_checkpoints", ".DS_Store"]

    def list_path_or_files(preset_info):
        path = Path(preset_info["path"])
        file_type = preset_info["type"]
        regex_filter = preset_info["filter"]
        result_list = []

        if file_type == "file":
            if regex_filter:
                pattern = re.compile(regex_filter)
                files = [f for f in path.glob("**/*") if f.is_file() and pattern.search(f.name)]
            else:
                files = [f for f in path.glob("**/*") if f.is_file()]
            for file in files:
                result_list.append({
                    "path": str(file.resolve().absolute()).replace("\\", "/"),
                    "name": file.name,
                    "size": f"{round(file.stat().st_size / (1024**3),2)} GB"
                })
        elif file_type == "folder":
            folders = [f for f in path.iterdir() if f.is_dir()]
            for folder in folders:
                if folder.name in folder_blacklist:
                    continue
                result_list.append({
                    "path": str(folder.resolve().absolute()).replace("\\", "/"),
                    "name": folder.name,
                    "size": 0
                })

        return result_list

    if pick_type not in pick_preset:
        return APIResponseFail(message="Invalid request")

    dirs = list_path_or_files(pick_preset[pick_type])
    return APIResponseSuccess(data={
        "files": dirs
    })


@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    return APIResponseSuccess(data={
        "tasks": tm.dump()
    })


@router.get("/tasks/terminate/{task_id}", response_model_exclude_none=True)
async def terminate_task(task_id: str):
    tm.terminate_task(task_id)
    return APIResponseSuccess()


@router.get("/graphic_cards")
async def list_avaliable_cards() -> APIResponse:
    if not printable_devices:
        return APIResponse(status="pending")

    return APIResponseSuccess(data={
        "cards": printable_devices
    })
