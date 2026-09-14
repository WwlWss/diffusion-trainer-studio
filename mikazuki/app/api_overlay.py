"""Narrow API overlay for backend-specific training pages.

Keep the existing API router intact and replace only POST /run so the new page
contracts can validate trainer-specific dataset sources without rewriting the
legacy API module wholesale.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import toml
from fastapi import Request

import mikazuki.app.api as legacy_api
import mikazuki.process as process
from mikazuki.app.models import APIResponseFail
from mikazuki.full_trainer_contract import normalize_validate_flux_full, normalize_validate_sdxl_full
from mikazuki.log import log
from mikazuki.training_data_contract import validate_dataset_source
from mikazuki.training_schema_factory import fixed_flux_family_schema, fixed_sd_schema
from mikazuki.utils import train_utils


# load_schemas() resolves these names from the legacy module at call time. Point
# them at the pure, independently testable implementations used by the new page
# contracts instead of maintaining two runtime copies.
legacy_api._fixed_sd_schema = fixed_sd_schema
legacy_api._fixed_flux_family_schema = fixed_flux_family_schema

router = legacy_api.router

router.routes[:] = [
    route
    for route in router.routes
    if not (getattr(route, "path", None) == "/run" and "POST" in getattr(route, "methods", set()))
]


@router.post("/run")
async def create_toml_file(request: Request):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), "config", "autosave", f"{timestamp}.toml")
    config: dict = json.loads((await request.body()).decode("utf-8"))
    train_utils.fix_config_types(config)

    gpu_ids = config.pop("gpu_ids", None)
    model_train_type = config.pop("model_train_type", "sd-lora")
    try:
        effective_train_type, trainer_file = legacy_api.resolve_training_backend(config, model_train_type)
        if effective_train_type == "sdxl-finetune":
            normalize_validate_sdxl_full(config)
        elif effective_train_type == "flux-finetune":
            normalize_validate_flux_full(config)
    except (KeyError, TypeError, ValueError) as e:
        return APIResponseFail(message=f"训练类型配置无效: {e}")

    try:
        validate_dataset_source(
            config,
            effective_train_type,
            is_file=os.path.isfile,
            validate_data_dir=train_utils.validate_data_dir,
        )
    except ValueError as e:
        return APIResponseFail(message=str(e))

    train_data_dir = config.get("train_data_dir")
    suggest_cpu_threads = 8 if train_data_dir and len(train_utils.get_total_images(train_data_dir)) > 200 else 2

    if effective_train_type in {"anima-lora", "anima-finetune"}:
        if not os.path.exists(trainer_file):
            return APIResponseFail(message="Anima 训练脚本不存在。请运行 `git submodule update --init --recursive` 后重启 GUI。")
        for key, label in (("qwen3", "Qwen3-0.6B"), ("vae", "Qwen-Image VAE")):
            value = config.get(key)
            if not value:
                return APIResponseFail(message=f"Anima 训练需要指定 {label} 路径。")
            if not os.path.exists(value):
                return APIResponseFail(message=f"{label} 路径不存在: {value}")

        for key in ("llm_adapter_path", "t5_tokenizer_path"):
            if not config.get(key):
                config.pop(key, None)
        if effective_train_type == "anima-lora":
            config.setdefault("network_module", "networks.lora_anima")

    validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"], effective_train_type)
    if not validated:
        return APIResponseFail(message=message)

    if config.get("prompt_file", "").strip():
        prompt_file = config.pop("prompt_file").strip()
        if not os.path.exists(prompt_file):
            return APIResponseFail(message=f"Prompt 文件 {prompt_file} 不存在，请检查路径。")
        config["sample_prompts"] = prompt_file
    else:
        config.pop("prompt_file", None)
        try:
            positive_prompt, sample_prompts_arg = legacy_api.get_sample_prompts(config)
            if positive_prompt is not None and train_utils.is_promopt_like(sample_prompts_arg):
                sample_prompts_file = os.path.join(os.getcwd(), "config", "autosave", f"{timestamp}-promopt.txt")
                with open(sample_prompts_file, "w", encoding="utf-8") as f:
                    f.write(sample_prompts_arg)
                config["sample_prompts"] = sample_prompts_file
                log.info(f"Wrote prompts to file {sample_prompts_file}")
        except ValueError as e:
            log.error(f"Error while processing prompts: {e}")
            return APIResponseFail(message=str(e))

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))
    return process.run_train(toml_file, trainer_file, gpu_ids, suggest_cpu_threads)
