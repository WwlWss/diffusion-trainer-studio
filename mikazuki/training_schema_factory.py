"""Pure schema specialization helpers for backend-specific training pages.

The shipped frontend still consumes schema text dynamically. These helpers take
legacy switchable schema templates and freeze them to exactly one backend so
navigation pages cannot switch Python trainers internally.
"""

from __future__ import annotations

import re


def replace_once(content: str, old: str, new: str, label: str) -> str:
    if old not in content:
        raise RuntimeError(f"Unable to build training schema: missing {label} anchor")
    return content.replace(old, new, 1)


def fixed_sd_schema(template: str, train_type: str) -> str:
    """Freeze the historical SD/SDXL selector to one concrete trainer."""
    if train_type in {"sd-lora", "sdxl-lora"}:
        old = 'model_train_type: Schema.union(["sd-lora", "sdxl-lora"]).default("sd-lora").description("训练种类"),'
    elif train_type in {"sd-dreambooth", "sdxl-finetune"}:
        old = 'model_train_type: Schema.union(["sd-dreambooth", "sdxl-finetune"]).default("sd-dreambooth").description("训练种类"),'
    else:
        raise ValueError(f"Unsupported SD schema backend: {train_type}")

    new = (
        f'model_train_type: Schema.string().default("{train_type}").disabled()'
        f'.description("固定训练后端：{train_type}"),'
    )
    return replace_once(template, old, new, f"{train_type} model_train_type")


def fixed_flux_family_schema(
    template: str,
    model_type: str,
    train_type: str,
    anima_mode: str | None = None,
) -> str:
    """Freeze the shared Flux/Chroma/Anima template to one concrete page."""
    if model_type not in {"flux", "chroma", "anima"}:
        raise ValueError(f"Unsupported Flux-family model type: {model_type}")

    old_model = 'model_type: Schema.union(["flux", "chroma", "anima"]).default("flux").description("模型架构：FLUX / Chroma / Anima"),'
    new_model = (
        f'model_type: Schema.string().default("{model_type}").disabled()'
        f'.description("固定模型 backend：{model_type}"),'
    )
    content = replace_once(template, old_model, new_model, f"{train_type} model_type")

    if model_type == "anima":
        if anima_mode not in {"lora", "finetune"}:
            raise ValueError("Anima schema requires anima_mode=lora or finetune")
        old_mode = 'anima_training_mode: Schema.union(["lora", "finetune"]).description("Anima 训练方式：请选择 LoRA 或全参微调；显式选择可确保旧版 GUI 正确切换条件设置"),'
        new_mode = (
            f'anima_training_mode: Schema.string().default("{anima_mode}").disabled()'
            f'.description("固定 Anima 训练后端：{anima_mode}"),\n'
            f'            model_train_type: Schema.string().default("{train_type}").disabled()'
            f'.description("固定训练后端：{train_type}"),'
        )
        content = replace_once(content, old_mode, new_mode, f"{train_type} Anima mode")
    else:
        old_type = 'model_train_type: Schema.string().default("flux-lora").disabled().description("实际训练种类"),'
        new_type = (
            f'model_train_type: Schema.string().default("{train_type}").disabled()'
            f'.description("固定训练后端：{train_type}"),'
        )
        content = replace_once(content, old_type, new_type, f"{train_type} inner train type")

    if model_type == "chroma":
        content, removed = re.subn(
            r'^\s*clip_l: Schema\.string\(\)\.role\([^\n]+\n',
            "",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if removed != 1:
            raise RuntimeError("Unable to build Chroma schema: CLIP-L anchor not found")
        content = replace_once(
            content,
            'apply_t5_attn_mask: Schema.boolean().default(true).description("对 T5-XXL 编码器和 FLUX double block 应用注意力掩码"),',
            'apply_t5_attn_mask: Schema.boolean().default(true).disabled().description("Chroma 固定启用 T5 attention mask"),',
            "Chroma T5 attention mask",
        )

    return content
