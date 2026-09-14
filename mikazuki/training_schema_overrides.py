"""Small source-level fixes applied before runtime schema specialization."""

from __future__ import annotations

from mikazuki.training_schema_factory import (
    fixed_flux_family_schema as _fixed_flux_family_schema,
    fixed_sd_schema as _fixed_sd_schema,
)


def override_raw_schema(name: str, content: str) -> str:
    """Correct dedicated schema files before they are published to the GUI."""
    if name == "sdxl-full":
        # The pinned frontend deletes these as SD1/2-only fields. Keeping them
        # visible creates controls that can never reach sdxl_train.py.
        content = content.replace(
            '        v_parameterization: Schema.boolean().default(false).description("v-parameterization training"),\n',
            "",
        )
        content = content.replace(
            '        scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("缩放 v-pred loss"),\n',
            "",
        )
    return content


def fixed_sd_schema(template: str, train_type: str) -> str:
    source = template
    if train_type == "sdxl-lora":
        # Do not reuse SHARED_SCHEMAS.OTHER here because it contains clip_skip,
        # which is a SD1/2 option and is silently deleted by the old frontend.
        source = source.replace(
            "    SHARED_SCHEMAS.OTHER,",
            "    Schema.object({seed: Schema.number().default(1337).description(\"随机种子\"), ui_custom_params: Schema.string().role('textarea').description(\"危险：自定义 TOML 参数会覆盖界面参数\")}).description(\"其他设置\"),",
            1,
        )
    elif train_type == "sd-dreambooth":
        source = source.replace(
            'save_model_as: Schema.union(["safetensors", "pt", "ckpt"]).default("safetensors").description("模型保存格式"),',
            'save_model_as: Schema.union(["safetensors", "ckpt", "diffusers", "diffusers_safetensors"]).default("safetensors").description("模型保存格式"),',
            1,
        )
        source = source.replace(
            'max_token_length: Schema.number().default(255).description("最大 token 长度"),',
            'sd_max_token_length_mode: Schema.union(["75", "150", "225"]).default("225").description("CLIP 最大 token 长度；75 使用 trainer 默认，150/225 为扩展长度"),',
            1,
        )
    return _fixed_sd_schema(source, train_type)


def fixed_flux_family_schema(template: str, model_type: str, train_type: str, anima_mode: str | None = None) -> str:
    source = template
    if model_type in {"flux", "chroma"}:
        source = source.replace(
            'timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift"]).default("sigmoid")',
            'timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid")',
            1,
        )
    if model_type == "anima" and anima_mode == "lora":
        source = source.replace(
            'network_train_unet_only: Schema.boolean().default(true).disabled().description("固定为仅训练 Anima DiT LoRA；字段名沿用 sd-scripts 历史 U-Net 命名"),',
            'anima_lora_target: Schema.union(["dit", "qwen3", "dit_qwen3"]).default("dit").description("LoRA 训练目标：仅 DiT、仅 Qwen3、或 DiT + Qwen3"),\n'
            '            anima_lora_text_encoder_lr: Schema.string().default("5e-6").description("Qwen3 LoRA 学习率；仅 Qwen3/联合目标使用"),',
            1,
        )
    return _fixed_flux_family_schema(source, model_type, train_type, anima_mode)
