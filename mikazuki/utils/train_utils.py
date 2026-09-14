from enum import Enum
import glob
import json
import os
import re
import sys
from typing import Dict

from mikazuki.log import log

python_bin = sys.executable


class ModelType(Enum):
    UNKNOWN = -1
    SD15 = 1
    SD2 = 2
    SDXL = 3
    SD3 = 4
    FLUX = 5
    LUMINA = 6
    ANIMA = 7
    CHROMA = 8
    LoRA = 10


MODEL_SIGNATURE = [
    {"type": ModelType.ANIMA, "signature": ["blocks.0.self_attn.q_proj.weight", "llm_adapter.out_proj.weight"]},
    {"type": ModelType.CHROMA, "signature": ["distilled_guidance_layer.in_proj.weight", "distilled_guidance_layer"]},
    {"type": ModelType.LUMINA, "signature": ["cap_embedder.0.weight", "context_refiner.0.attention.k_norm.weight"]},
    {"type": ModelType.FLUX, "signature": [
        "double_blocks.0.img_mlp.0.weight",
        "guidance_in.in_layer.weight",
        "model.diffusion_model.double_blocks",
        "double_blocks.0.img_attn.norm.query_norm.scale",
    ]},
    {"type": ModelType.SD3, "signature": [
        "model.diffusion_model.x_embedder.proj.weight",
        "model.diffusion_model.joint_blocks.0.context_block.attn.proj.weight",
    ]},
    {"type": ModelType.SDXL, "signature": ["conditioner.embedders.1.model.transformer.resblocks"]},
    {"type": ModelType.SD15, "signature": ["model.diffusion_model", "cond_stage_model.transformer.text_model"]},
    {"type": ModelType.LoRA, "signature": [
        "lora_te_text_model_encoder",
        "lora_unet_up_blocks",
        "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.alpha",
        "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
        "lora_unet",
        "lora_te",
        "lora_A.weight",
    ]},
]

ACCEPTED_MODEL_TYPES = {
    "sd-lora": {ModelType.SD15, ModelType.SD2},
    "sd-dreambooth": {ModelType.SD15, ModelType.SD2},
    "sdxl-lora": {ModelType.SDXL},
    "sdxl-finetune": {ModelType.SDXL},
    "sd3-lora": {ModelType.SD3},
    "flux-lora": {ModelType.FLUX},
    "flux-finetune": {ModelType.FLUX},
    "chroma-lora": {ModelType.CHROMA},
    "anima-lora": {ModelType.ANIMA},
    "anima-finetune": {ModelType.ANIMA},
}


def is_promopt_like(s):
    if not isinstance(s, str):
        return False
    return any(p in s for p in ["--n", "--s", "--l", "--d"])


def match_model_type_legacy(sig_content: bytes):
    if b"distilled_guidance_layer" in sig_content:
        return ModelType.CHROMA
    if b"model.diffusion_model.double_blocks" in sig_content or b"double_blocks.0.img_attn.norm.query_norm.scale" in sig_content:
        return ModelType.FLUX
    if b"model.diffusion_model.x_embedder.proj.weight" in sig_content:
        return ModelType.SD3
    if b"conditioner.embedders.1.model.transformer.resblocks" in sig_content:
        return ModelType.SDXL
    if b"model.diffusion_model" in sig_content or b"cond_stage_model.transformer.text_model" in sig_content:
        return ModelType.SD15
    if b"lora_unet" in sig_content or b"lora_te" in sig_content:
        return ModelType.LoRA
    return ModelType.UNKNOWN


def read_safetensors_metadata(path) -> Dict:
    if not os.path.exists(path):
        log.error(f"Can't find safetensors metadata file {path}")
        return None
    with open(path, "rb") as f:
        meta_length = int.from_bytes(f.read(8), "little")
        meta = f.read(meta_length)
        return json.loads(meta)


def guess_model_type(path):
    lower = path.lower()
    if lower.endswith(".safetensors"):
        metadata = read_safetensors_metadata(path)
        if not metadata:
            return ModelType.UNKNOWN
        model_keys = "\n".join(metadata.keys())
        for candidate in MODEL_SIGNATURE:
            if any(signature in model_keys for signature in candidate["signature"]):
                return candidate["type"]
        return ModelType.UNKNOWN
    if lower.endswith(".pt") or lower.endswith(".ckpt") or lower.endswith(".pth"):
        with open(path, "rb") as f:
            return match_model_type_legacy(f.read(1024 * 1000))
    return ModelType.UNKNOWN


def _model_label(model_type: ModelType) -> str:
    return {
        ModelType.SD15: "SD1.x/SD2", ModelType.SD2: "SD2", ModelType.SDXL: "SDXL",
        ModelType.SD3: "SD3", ModelType.FLUX: "FLUX", ModelType.CHROMA: "Chroma",
        ModelType.LUMINA: "Lumina", ModelType.ANIMA: "Anima", ModelType.LoRA: "LoRA adapter",
        ModelType.UNKNOWN: "unknown",
    }.get(model_type, model_type.name)


def validate_model(model_name: str, training_type: str = "sd-lora"):
    if os.path.exists(model_name):
        if os.path.isdir(model_name):
            files = os.listdir(model_name)
            if "model_index.json" not in files:
                log.warning("Model directory family cannot be identified reliably; deferring to trainer loader")
            return True, "ok"
        try:
            model_type = guess_model_type(model_name)
        except Exception as exc:
            return False, f"无法读取本地模型文件以验证类型: {exc}"
        if model_type == ModelType.UNKNOWN:
            return False, f"无法从本地 checkpoint 识别模型类型: {model_name}"
        accepted = ACCEPTED_MODEL_TYPES.get(training_type)
        if accepted is None:
            if model_type == ModelType.LoRA:
                return False, "选择的文件是 LoRA/adapter，不是可作为底模的 checkpoint。"
            return True, "ok"
        if model_type not in accepted:
            expected = "/".join(sorted(_model_label(value) for value in accepted))
            return False, f"模型类型不匹配：{training_type} 页面要求 {expected}，但本地 checkpoint 检测为 {_model_label(model_type)}。"
        return True, "ok"
    if (
        model_name.count("/") == 1
        and model_name[0] not in [".", "/"]
        and model_name.split(".")[-1].lower() not in ["pt", "pth", "ckpt", "safetensors"]
    ):
        log.warning("HuggingFace repo family cannot be checked locally; deferring to trainer loader")
        return True, "ok"
    return False, "model not found"


def inspect_data_dir(path: str) -> tuple[bool, str]:
    """Read-only inspection of the classic repeat-folder dataset layout."""
    if not os.path.isdir(path):
        return False, f"训练数据集目录不存在: {path}"
    try:
        dir_content = os.listdir(path)
    except OSError as exc:
        return False, f"无法读取训练数据集目录 {path}: {exc}"
    if not dir_content:
        return False, f"训练数据集目录为空: {path}"
    subdirs = [name for name in dir_content if os.path.isdir(os.path.join(path, name))]
    legal = [name for name in subdirs if re.fullmatch(r"\d+_.+", name)]
    if legal:
        for name in legal:
            if get_total_images(os.path.join(path, name), recursive=True):
                return True, f"Found {len(legal)} legal dataset folder(s)"
        return False, "找到了 repeat_名称 子目录，但其中没有支持的 jpg/jpeg/png 图片。"
    if get_total_images(path, recursive=False):
        return False, (
            "训练目录根部有图片，但没有 '数字_名称' 数据子目录。Start 只做只读校验，"
            "不会再自动创建目录或移动文件；请显式整理数据集后再启动。"
        )
    return False, "训练目录中没有合法的 '数字_名称' 数据子目录，也没有可训练图片。"


def validate_data_dir(path):
    """Compatibility boolean wrapper. This function is intentionally pure."""
    valid, message = inspect_data_dir(path)
    if valid:
        log.info(message)
    else:
        log.error(message)
    return valid


def suggest_num_repeat(img_count):
    if img_count <= 10:
        return 7
    if img_count <= 50:
        return 5
    if img_count <= 100:
        return 3
    return 1


def check_training_params(data):
    potential_path = ["train_data_dir", "reg_data_dir", "output_dir"]
    file_paths = ["sample_prompts"]
    for path in potential_path:
        if path in data and not os.path.exists(data[path]):
            return False
    for path in file_paths:
        if path in data and not os.path.exists(data[path]):
            return False
    return True


def get_total_images(path, recursive=True):
    if recursive:
        image_files = glob.glob(path + "/**/*.jpg", recursive=True)
        image_files += glob.glob(path + "/**/*.jpeg", recursive=True)
        image_files += glob.glob(path + "/**/*.png", recursive=True)
    else:
        image_files = glob.glob(path + "/*.jpg")
        image_files += glob.glob(path + "/*.jpeg")
        image_files += glob.glob(path + "/*.png")
    return image_files


def fix_config_types(config: dict):
    keep_float_params = ["guidance_scale", "sigmoid_scale", "discrete_flow_shift"]
    for key in keep_float_params:
        if key in config:
            config[key] = float(config[key])
