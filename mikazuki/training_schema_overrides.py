"""Source-level schema fixes applied before runtime specialization."""

from __future__ import annotations

import re

from mikazuki.training_schema_factory import (
    fixed_flux_family_schema as _fixed_flux_family_schema,
    fixed_sd_schema as _fixed_sd_schema,
)

_MEMORY_MODE = (
    'memory_mode: Schema.union(["auto", "lowram", "highvram"]).default("auto")'
    '.description("模型加载模式：Auto=默认；Low RAM=优先降低主机内存；High VRAM=减少 CPU/GPU 搬运")'
)

_DATASET_SOURCE = '''
    Schema.intersect([
        Schema.object({
            dataset_source: Schema.union(["folder", "config"]).default("folder").description("数据集来源：普通目录或 sd-scripts dataset_config")
        }).description("数据集来源"),
        Schema.union([
            Schema.object({
                dataset_source: Schema.const("config").required(),
                dataset_config: Schema.string().role('filepicker', { type: "file" }).description("dataset TOML/JSON；启用后后端忽略 train_data_dir/reg_data_dir/in_json")
            }),
            Schema.object({})
        ])
    ]),
'''


def _replace_dreambooth_logging(content: str) -> str:
    pattern = re.compile(
        r'\n    Schema\.intersect\(\[\n'
        r'        Schema\.object\(\{\n'
        r'            log_with: Schema\.union\(\["tensorboard", "wandb"\]\).*?'
        r'\n    \]\),\n',
        re.DOTALL,
    )
    replaced, count = pattern.subn("\n    SHARED_SCHEMAS.LOG_SETTINGS,\n", content, count=1)
    if count != 1:
        raise RuntimeError("DreamBooth logging schema anchor changed")
    return replaced


def _inject_dataset_source(source: str) -> str:
    if "dataset_source:" in source:
        return source
    anchor = "    // 数据集设置\n"
    if anchor in source:
        return source.replace(anchor, anchor + _DATASET_SOURCE, 1)
    anchor = "    Schema.object(\n        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {"
    if anchor in source:
        return source.replace(anchor, _DATASET_SOURCE + "\n" + anchor, 1)
    return _DATASET_SOURCE + "\n" + source


def _replace_sd_caption_token_control(source: str) -> str:
    return source.replace(
        'Schema.object(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS).description("caption（Tag）选项")',
        'Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {'
        'sd_max_token_length_mode: Schema.union(["75", "150", "225"]).default("75").description("CLIP 最大 token 长度；75 使用 trainer 默认，150/225 为扩展长度")'
        '}, ["max_token_length"])).description("caption（Tag）选项")',
        1,
    )


def override_raw_schema(name: str, content: str) -> str:
    if name == "shared":
        return content.replace(
            'wandb_api_key: Schema.string().required().description("wandb 的 api 密钥")',
            'wandb_api_key: Schema.string().description("可选：WandB API key；留空使用现有 wandb login / 环境变量")',
        )
    if name == "sdxl-full":
        content = content.replace('        v_parameterization: Schema.boolean().default(false).description("v-parameterization training"),\n', "")
        content = content.replace('        scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("缩放 v-pred loss"),\n', "")
        content = content.replace(
            '        highvram: Schema.boolean().default(false).description("高显存加载路径"),',
            '        memory_mode: Schema.union(["auto", "lowram", "highvram"]).default("auto").description("模型加载模式：Auto / Low RAM / High VRAM"),\n'
            '        torch_compile: Schema.boolean().default(false).description("启用 PyTorch/Accelerate torch.compile；实验性组合默认关闭"),\n'
            '        dynamo_backend: Schema.string().default("inductor").description("Accelerate dynamo backend；trainer 默认 inductor"),',
            1,
        )
    elif name == "flux-finetune":
        content = content.replace(
            '        highvram: Schema.boolean().default(false).description("高显存优化路径"),',
            '        memory_mode: Schema.union(["auto", "lowram", "highvram"]).default("auto").description("模型加载模式：Auto / Low RAM / High VRAM"),\n'
            '        torch_compile: Schema.boolean().default(false).description("启用 PyTorch/Accelerate torch.compile；与 block swap/DeepSpeed 组合尚未验证"),\n'
            '        dynamo_backend: Schema.string().default("inductor").description("Accelerate dynamo backend；trainer 默认 inductor"),',
            1,
        )
    return content


def fixed_sd_schema(template: str, train_type: str) -> str:
    source = template
    if train_type in {"sd-lora", "sdxl-lora"}:
        source = _inject_dataset_source(source)
        source = _replace_sd_caption_token_control(source)
        source = source.replace(
            '        network_train_unet_only: Schema.boolean().default(false).description("仅训练 U-Net 训练SDXL Lora时推荐开启"),\n'
            '        network_train_text_encoder_only: Schema.boolean().default(false).description("仅训练文本编码器"),',
            '        lora_target: Schema.union(["unet", "text_encoder", "unet_text_encoder"]).default("unet_text_encoder").description("LoRA 训练目标；互斥选择，避免两个 raw boolean 形成非法组合"),',
            1,
        )
        source = source.replace(
            'Schema.object(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH).description("速度优化选项")',
            f'Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {{{_MEMORY_MODE}}}, ["lowram"])).description("速度优化选项")',
            1,
        )
    if train_type == "sdxl-lora":
        source = source.replace(
            'Schema.object(SHARED_SCHEMAS.RAW.DATASET_SETTINGS).description("数据集设置")',
            'Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {'
            'resolution: Schema.string().default("1024,1024").description("SDXL 默认训练分辨率"),'
            'max_bucket_reso: Schema.number().default(2048).description("SDXL bucket 最大分辨率"),'
            'bucket_reso_steps: Schema.number().default(32).description("SDXL bucket step；trainer 要求 32 的倍数")'
            '})).description("数据集设置")',
            1,
        )
        source = source.replace(
            "    SHARED_SCHEMAS.OTHER,",
            "    Schema.object({seed: Schema.number().default(1337).description(\"随机种子\"), ui_custom_params: Schema.string().role('textarea').description(\"危险：自定义 TOML 参数会覆盖界面参数\")}).description(\"其他设置\"),",
            1,
        )
    elif train_type == "sd-dreambooth":
        source = _inject_dataset_source(source)
        source = source.replace(
            'save_model_as: Schema.union(["safetensors", "pt", "ckpt"]).default("safetensors").description("模型保存格式"),',
            'save_model_as: Schema.union(["safetensors", "ckpt", "diffusers", "diffusers_safetensors"]).default("safetensors").description("模型保存格式"),',
            1,
        )
        source = source.replace(
            'max_token_length: Schema.number().default(255).description("最大 token 长度"),',
            'sd_max_token_length_mode: Schema.union(["75", "150", "225"]).default("75").description("CLIP 最大 token 长度；75 使用 trainer 默认，150/225 为扩展长度"),',
            1,
        )
        source = source.replace(
            'lowram: Schema.boolean().default(false).description("低内存模式 该模式下会将 U-net、文本编码器、VAE 直接加载到显存中"),',
            'memory_mode: Schema.union(["auto", "lowram", "highvram"]).default("auto").description("模型加载模式：Auto / Low RAM / High VRAM"),',
            1,
        )
        source = _replace_dreambooth_logging(source)
    return _fixed_sd_schema(source, train_type)


def fixed_flux_family_schema(template: str, model_type: str, train_type: str, anima_mode: str | None = None) -> str:
    source = template
    if train_type.endswith("lora"):
        source = _inject_dataset_source(source)
    if model_type in {"flux", "chroma"}:
        source = source.replace(
            'timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift"]).default("sigmoid")',
            'timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid")',
            1,
        )
        target_schema = (
            'flux_lora_target: Schema.union(["dit", "dit_t5xxl"]).default("dit").description("Chroma LoRA 训练目标；Chroma 使用 dummy CLIP-L")'
            if model_type == "chroma" else
            'flux_lora_target: Schema.union(["dit", "dit_clip_l", "dit_clip_l_t5xxl"]).default("dit").description("Flux LoRA 训练目标；默认仅 DiT，不再隐式训练 CLIP-L")'
        )
        source = source.replace('train_t5xxl: Schema.boolean().default(false).description("训练 T5XXL（不推荐）")', target_schema, 1)
        source = source.replace(
            'UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {',
            'UpdateSchema(UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {'
            'memory_mode: Schema.union(["auto", "lowram", "highvram"]).default("auto").description("模型加载模式：Auto / Low RAM / High VRAM")'
            '}, ["lowram"]), {',
            1,
        )
    if model_type == "anima" and anima_mode == "lora":
        source = source.replace(
            'network_train_unet_only: Schema.boolean().default(true).disabled().description("固定为仅训练 Anima DiT LoRA；字段名沿用 sd-scripts 历史 U-Net 命名")',
            'anima_lora_target: Schema.union(["dit", "qwen3", "dit_qwen3"]).default("dit").description("LoRA 训练目标。dit 是绝大多数角色/风格 LoRA 的默认；只有要改变文本编码器行为时才选 qwen3 或 dit_qwen3，后两者更吃显存且更易影响提示词泛化。"),\n'
            '            anima_lora_text_encoder_lr: Schema.string().default("5e-6").description("Qwen3 LoRA 学习率，仅 qwen3/dit_qwen3 使用。建议显著低于 DiT LoRA LR，可从约 1e-6~5e-6 起做短跑，过高容易破坏文本表示。")',
            1,
        )
        source = source.replace(
            'lowram: Schema.boolean().default(false).description("低主机内存模式")',
            'memory_mode: Schema.union(["auto", "lowram", "highvram"]).default("auto").description("模型加载模式。Auto 为稳妥默认；Low RAM 减少主机内存占用但可能增加搬运；High VRAM 减少 CPU/GPU 来回搬运但需要更多显存。")',
            1,
        )
    elif model_type == "anima" and anima_mode == "finetune":
        source = source.replace(
            'highvram: Schema.boolean().default(false).description("启用 sd-scripts High VRAM 模式，减少缓存阶段频繁清理 CUDA cache；仅在显存余量充足时建议开启")',
            'memory_mode: Schema.union(["auto", "highvram"]).default("auto").description("模型加载模式。Auto 为推荐默认；High VRAM 在显存明显有余量时减少缓存阶段的 CUDA 清理/搬运，可能更快但会提高峰值显存。Anima Full 当前没有 Low RAM 路径。")',
            1,
        )
    return _fixed_flux_family_schema(source, model_type, train_type, anima_mode)
