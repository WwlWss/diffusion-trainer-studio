"""Normalize and validate Anima full-finetune GUI semantics.

The legacy GUI exposes several low-level sd-scripts booleans independently.
That makes it easy to create contradictory configurations (for example disk
cache without cache, or full_fp16 together with bf16 mixed precision). New
Anima full-finetune UI fields describe the user's intent as mutually-exclusive
modes; this module converts them to the raw sd-scripts arguments immediately
before launch.

Legacy presets remain supported: if a semantic field is absent, the existing
raw sd-scripts fields are normalized to their effective behaviour and then
validated. The functions are intentionally idempotent so the API and process
boundaries may both call them in the future.
"""

from __future__ import annotations

from typing import Mapping


ANIMA_FINETUNE_GUI_KEYS = {
    "anima_finetune_learning_rate",
    "anima_precision_mode",
    "anima_latent_cache_mode",
    "anima_text_encoder_cache_mode",
    "anima_checkpoint_mode",
    "anima_custom_optimizer_type",
    "anima_custom_lr_scheduler_type",
}

ANIMA_COMMON_GUI_KEYS = {
    "anima_preview_cadence",
    "anima_preview_interval",
}

PRECISION_MODES: Mapping[str, dict[str, object]] = {
    "fp32": {
        "mixed_precision": "no",
        "full_fp16": False,
        "full_bf16": False,
    },
    "mixed_fp16": {
        "mixed_precision": "fp16",
        "full_fp16": False,
        "full_bf16": False,
    },
    "full_fp16": {
        "mixed_precision": "fp16",
        "full_fp16": True,
        "full_bf16": False,
    },
    "mixed_bf16": {
        "mixed_precision": "bf16",
        "full_fp16": False,
        "full_bf16": False,
    },
    "full_bf16": {
        "mixed_precision": "bf16",
        "full_fp16": False,
        "full_bf16": True,
    },
}

CACHE_MODES: Mapping[str, tuple[bool, bool]] = {
    "off": (False, False),
    "memory": (True, False),
    "disk": (True, True),
}

CHECKPOINT_MODES: Mapping[str, tuple[bool, bool, bool]] = {
    "off": (False, False, False),
    "standard": (True, False, False),
    "cpu": (True, True, False),
    "unsloth": (True, False, True),
}

# Loss weighting itself has special Anima formulas only for sigma_sqrt/cosmap,
# but logit_normal/mode are still meaningful when timestep_sampling=sigma:
# flux_train_utils uses them to change the actual timestep sampling density.
# anima_ui_parity validates that those modes are never exposed as a no-op.
SUPPORTED_ANIMA_WEIGHTING_SCHEMES = {
    "uniform",
    "none",
    "sigma_sqrt",
    "cosmap",
    "logit_normal",
    "mode",
}


SUPPORTED_DEEPSPEED_OFFLOAD_DEVICES = {None, "", "cpu", "nvme"}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _values_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return _as_bool(actual) == expected
    return actual == expected


def _apply_semantic_mapping(
    config: dict,
    semantic_key: str,
    mapping: Mapping[str, dict[str, object]],
    label: str,
) -> bool:
    """Apply one semantic mode and reject conflicting explicit raw arguments."""
    raw_mode = config.pop(semantic_key, None)
    if raw_mode in (None, ""):
        return False

    mode = str(raw_mode).lower()
    if mode not in mapping:
        choices = ", ".join(mapping.keys())
        raise ValueError(f"Anima: 未知的{label}模式 '{raw_mode}'；可选值为 {choices}。")

    expected = mapping[mode]
    conflicts = []
    for raw_key, raw_value in expected.items():
        if raw_key in config and config[raw_key] not in (None, ""):
            if not _values_equal(config[raw_key], raw_value):
                conflicts.append(f"{raw_key}={config[raw_key]!r}")

    if conflicts:
        raise ValueError(
            f"Anima: {label}模式 {mode!r} 与显式底层参数冲突："
            + ", ".join(conflicts)
            + "。请删除 ui_custom_params / 旧 preset 中冲突的底层参数。"
        )

    config.update(expected)
    return True


def _normalize_learning_rate(config: dict) -> None:
    semantic_lr = config.pop("anima_finetune_learning_rate", None)
    if semantic_lr in (None, ""):
        return

    try:
        semantic_value = float(semantic_lr)
    except (TypeError, ValueError) as e:
        raise ValueError("Anima: 全参微调学习率必须是有效数字。") from e

    if "learning_rate" in config and config["learning_rate"] not in (None, ""):
        try:
            raw_value = float(config["learning_rate"])
        except (TypeError, ValueError) as e:
            raise ValueError("Anima: learning_rate 必须是有效数字。") from e
        if raw_value != semantic_value:
            raise ValueError(
                "Anima: anima_finetune_learning_rate 与显式 learning_rate 冲突；"
                "请删除 ui_custom_params / 旧 preset 中重复的 learning_rate。"
            )

    config["learning_rate"] = semantic_value


def _normalize_cache_mode(config: dict, semantic_key: str, raw_key: str, disk_key: str, label: str) -> bool:
    raw_mode = config.pop(semantic_key, None)
    if raw_mode in (None, ""):
        # Match sd-scripts effective behaviour for legacy configs: disk cache
        # implicitly enables the corresponding in-memory/cache pipeline.
        if _as_bool(config.get(disk_key)):
            config[raw_key] = True
        return False

    mode = str(raw_mode).lower()
    if mode not in CACHE_MODES:
        raise ValueError(f"Anima: 未知的{label}缓存模式 '{raw_mode}'；可选值为 off / memory / disk。")

    expected_enabled, expected_disk = CACHE_MODES[mode]
    conflicts = []
    if raw_key in config and config[raw_key] not in (None, ""):
        if _as_bool(config[raw_key]) != expected_enabled:
            conflicts.append(f"{raw_key}={config[raw_key]!r}")
    if disk_key in config and config[disk_key] not in (None, ""):
        if _as_bool(config[disk_key]) != expected_disk:
            conflicts.append(f"{disk_key}={config[disk_key]!r}")
    if conflicts:
        raise ValueError(
            f"Anima: {label}缓存模式 {mode!r} 与显式底层参数冲突："
            + ", ".join(conflicts)
            + "。请删除 ui_custom_params / 旧 preset 中冲突的缓存参数。"
        )

    config[raw_key] = expected_enabled
    config[disk_key] = expected_disk
    return True


def _normalize_checkpoint_mode(config: dict) -> bool:
    raw_mode = config.pop("anima_checkpoint_mode", None)
    if raw_mode in (None, ""):
        # Preserve sd-scripts' legacy coercion while turning impossible two-
        # offload combinations into an early, clear error.
        cpu = _as_bool(config.get("cpu_offload_checkpointing"))
        unsloth = _as_bool(config.get("unsloth_offload_checkpointing"))
        if cpu and unsloth:
            raise ValueError(
                "Anima: cpu_offload_checkpointing 与 unsloth_offload_checkpointing 不能同时启用。"
            )
        if cpu or unsloth:
            config["gradient_checkpointing"] = True
        return False

    mode = str(raw_mode).lower()
    if mode not in CHECKPOINT_MODES:
        raise ValueError(
            f"Anima: 未知的 Gradient Checkpointing 模式 '{raw_mode}'；"
            "可选值为 off / standard / cpu / unsloth。"
        )

    gradient, cpu, unsloth = CHECKPOINT_MODES[mode]
    expected = {
        "gradient_checkpointing": gradient,
        "cpu_offload_checkpointing": cpu,
        "unsloth_offload_checkpointing": unsloth,
    }
    conflicts = []
    for key, value in expected.items():
        if key in config and config[key] not in (None, "") and _as_bool(config[key]) != value:
            conflicts.append(f"{key}={config[key]!r}")
    if conflicts:
        raise ValueError(
            f"Anima: Gradient Checkpointing 模式 {mode!r} 与显式底层参数冲突："
            + ", ".join(conflicts)
            + "。请删除 ui_custom_params / 旧 preset 中冲突的 checkpoint 参数。"
        )

    config.update(expected)
    return True


def _normalize_optimizer(config: dict) -> None:
    custom_optimizer = config.pop("anima_custom_optimizer_type", None)
    optimizer = str(config.get("optimizer_type") or "").strip()
    if optimizer.lower() != "custom":
        return
    if custom_optimizer in (None, ""):
        raise ValueError("Anima: 选择 Custom optimizer 时必须填写完整 optimizer class / type。")
    config["optimizer_type"] = str(custom_optimizer).strip()


def _normalize_scheduler(config: dict) -> None:
    custom_scheduler = config.pop("anima_custom_lr_scheduler_type", None)
    scheduler = str(config.get("lr_scheduler") or "").strip().lower()
    if scheduler != "custom":
        return
    if custom_scheduler in (None, ""):
        raise ValueError("Anima: 选择 Custom scheduler 时必须填写 lr_scheduler_type。")
    config["lr_scheduler_type"] = str(custom_scheduler).strip()
    # get_scheduler_fix checks lr_scheduler_type first. Keep the ordinary name
    # valid as well so downstream logging/config inspection never sees an
    # invented scheduler enum value.
    config["lr_scheduler"] = "constant"


def _normalize_preview_cadence(config: dict) -> None:
    cadence = config.pop("anima_preview_cadence", None)
    interval = config.pop("anima_preview_interval", None)
    if cadence in (None, "") and interval in (None, ""):
        return
    if cadence in (None, "") or interval in (None, ""):
        raise ValueError("Anima: 预览频率必须同时指定 cadence 和 interval。")

    cadence = str(cadence).lower()
    try:
        interval = int(interval)
    except (TypeError, ValueError) as e:
        raise ValueError("Anima: 预览 interval 必须是正整数。") from e
    if interval <= 0:
        raise ValueError("Anima: 预览 interval 必须大于 0。")

    if cadence == "epoch":
        config["sample_every_n_epochs"] = interval
        config.pop("sample_every_n_steps", None)
    elif cadence == "step":
        config["sample_every_n_steps"] = interval
        config.pop("sample_every_n_epochs", None)
    else:
        raise ValueError("Anima: 预览 cadence 只能是 epoch 或 step。")


def _normalize_deepspeed_effective_state(config: dict) -> None:
    if not _as_bool(config.get("deepspeed")):
        # Empty optional offload values should not become argparse strings.
        for key in (
            "offload_optimizer_device",
            "offload_optimizer_nvme_path",
            "offload_param_device",
            "offload_param_nvme_path",
        ):
            if config.get(key) in (None, ""):
                config.pop(key, None)
        return

    # sd-scripts itself forces this to one in prepare_deepspeed_args(). Make the
    # generated TOML reflect the effective value instead of hiding a coercion.
    config["max_data_loader_n_workers"] = 1
    for key in (
        "offload_optimizer_device",
        "offload_optimizer_nvme_path",
        "offload_param_device",
        "offload_param_nvme_path",
    ):
        if config.get(key) in (None, ""):
            config.pop(key, None)


def normalize_anima_finetune_config(config: dict, anima_training_mode: str) -> None:
    """Convert semantic Anima GUI fields to raw sd-scripts args.

    Preview cadence is common to Anima LoRA/full finetune. Full-finetune-only
    semantic controls are discarded for LoRA so switching modes cannot leak
    stale values into anima_train_network.py.
    """
    _normalize_preview_cadence(config)

    if anima_training_mode != "finetune":
        for key in ANIMA_FINETUNE_GUI_KEYS:
            config.pop(key, None)
        return

    _normalize_learning_rate(config)
    _normalize_optimizer(config)
    _normalize_scheduler(config)

    _apply_semantic_mapping(
        config,
        "anima_precision_mode",
        PRECISION_MODES,
        "训练精度",
    )
    _normalize_cache_mode(
        config,
        "anima_latent_cache_mode",
        "cache_latents",
        "cache_latents_to_disk",
        "Latent",
    )
    _normalize_cache_mode(
        config,
        "anima_text_encoder_cache_mode",
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk",
        "Qwen3 输出",
    )
    _normalize_checkpoint_mode(config)
    _normalize_deepspeed_effective_state(config)


def validate_anima_finetune_config(config: dict, anima_training_mode: str) -> None:
    """Fail fast for ineffective or contradictory Anima full-finetune options."""
    if anima_training_mode != "finetune":
        return

    try:
        learning_rate = float(config.get("learning_rate"))
    except (TypeError, ValueError) as e:
        raise ValueError("Anima: 全参微调要求有效的 learning_rate。") from e
    if learning_rate <= 0:
        raise ValueError(
            "Anima: 全参微调要求 learning_rate > 0。learning_rate=0 会冻结整个 DiT，"
            "分组件学习率不能重新启用被冻结的 DiT。"
        )

    mixed_precision = str(config.get("mixed_precision") or "no").lower()
    full_fp16 = _as_bool(config.get("full_fp16"))
    full_bf16 = _as_bool(config.get("full_bf16"))
    if full_fp16 and full_bf16:
        raise ValueError("Anima: full_fp16 与 full_bf16 不能同时启用。")
    if full_fp16 and mixed_precision != "fp16":
        raise ValueError("Anima: full_fp16 要求 mixed_precision='fp16'。")
    if full_bf16 and mixed_precision != "bf16":
        raise ValueError("Anima: full_bf16 要求 mixed_precision='bf16'。")

    latent_cache = _as_bool(config.get("cache_latents")) or _as_bool(config.get("cache_latents_to_disk"))
    if latent_cache and _as_bool(config.get("color_aug")):
        raise ValueError("Anima: cache_latents 与 color_aug 不兼容；请关闭其中一项。")
    if latent_cache and _as_bool(config.get("random_crop")):
        raise ValueError("Anima: cache_latents 与 random_crop 不兼容；请关闭其中一项。")

    cpu_offload = _as_bool(config.get("cpu_offload_checkpointing"))
    unsloth_offload = _as_bool(config.get("unsloth_offload_checkpointing"))
    if cpu_offload and unsloth_offload:
        raise ValueError(
            "Anima: cpu_offload_checkpointing 与 unsloth_offload_checkpointing 不能同时启用。"
        )

    if _as_bool(config.get("fused_backward_pass")):
        optimizer = str(config.get("optimizer_type") or "AdamW").lower()
        if optimizer != "adafactor":
            raise ValueError("Anima: fused_backward_pass 当前仅支持 optimizer_type=AdaFactor。")
        if int(config.get("gradient_accumulation_steps") or 1) != 1:
            raise ValueError("Anima: fused_backward_pass 要求 gradient_accumulation_steps=1。")

    if _as_bool(config.get("deepspeed")):
        try:
            zero_stage = int(config.get("zero_stage", 2))
        except (TypeError, ValueError) as e:
            raise ValueError("Anima: DeepSpeed zero_stage 必须是 0 / 1 / 2 / 3。") from e
        if zero_stage not in {0, 1, 2, 3}:
            raise ValueError("Anima: DeepSpeed zero_stage 必须是 0 / 1 / 2 / 3。")

        optimizer_offload = config.get("offload_optimizer_device")
        param_offload = config.get("offload_param_device")
        if optimizer_offload not in SUPPORTED_DEEPSPEED_OFFLOAD_DEVICES:
            raise ValueError("Anima: DeepSpeed optimizer offload 只能是 cpu / nvme / 空。")
        if param_offload not in SUPPORTED_DEEPSPEED_OFFLOAD_DEVICES:
            raise ValueError("Anima: DeepSpeed parameter offload 只能是 cpu / nvme / 空。")
        if optimizer_offload and zero_stage not in {2, 3}:
            raise ValueError("Anima: DeepSpeed optimizer offload 仅适用于 ZeRO stage 2/3。")
        if param_offload and zero_stage != 3:
            raise ValueError("Anima: DeepSpeed parameter offload 仅适用于 ZeRO stage 3。")
        if config.get("offload_optimizer_nvme_path") and optimizer_offload != "nvme":
            raise ValueError("Anima: offload_optimizer_nvme_path 要求 optimizer offload=nvme。")
        if config.get("offload_param_nvme_path") and param_offload != "nvme":
            raise ValueError("Anima: offload_param_nvme_path 要求 parameter offload=nvme。")
        if (_as_bool(config.get("zero3_init_flag")) or _as_bool(config.get("zero3_save_16bit_model"))) and zero_stage != 3:
            raise ValueError("Anima: zero3_init_flag / zero3_save_16bit_model 仅适用于 ZeRO stage 3。")
        if _as_bool(config.get("fp16_master_weights_and_gradients")):
            if zero_stage != 2 or optimizer_offload != "cpu" or mixed_precision != "fp16":
                raise ValueError(
                    "Anima: fp16_master_weights_and_gradients 要求 ZeRO-2 + optimizer CPU offload + FP16 训练。"
                )

    if _as_bool(config.get("torch_compile")) and int(config.get("blocks_to_swap") or 0) > 0:
        raise ValueError("Anima: torch_compile 与 blocks_to_swap 的组合尚未完成 GPU 验证；当前请二选一。")

    # Current anima_train.py always writes safetensors. Accept and remove the
    # legacy shared-GUI default, but reject choices that would falsely imply a
    # different output format.
    save_model_as = config.pop("save_model_as", None)
    if save_model_as not in (None, "", "safetensors"):
        raise ValueError(
            f"Anima: 当前全参 trainer 只保存 safetensors，save_model_as={save_model_as!r} 不受支持。"
        )

    # These options currently parse successfully but do not affect the full
    # trainer. Reject true/non-empty values instead of silently contaminating
    # experiments with no-op controls.
    unsupported_truthy = {
        "no_half_vae": "no_half_vae",
        "lowram": "lowram",
        "compile": "compile / per-block torch.compile",
        "cuda_allow_tf32": "cuda_allow_tf32",
        "cuda_cudnn_benchmark": "cuda_cudnn_benchmark",
    }
    for key, label in unsupported_truthy.items():
        if _as_bool(config.get(key)):
            raise ValueError(f"Anima: 当前全参 trainer 尚未实现 {label}；请关闭该选项。")
        config.pop(key, None)

    if config.get("llm_adapter_path"):
        raise ValueError(
            "Anima: 当前全参 trainer 尚未实际加载独立 llm_adapter_path；"
            "为避免静默 no-op，请先留空。"
        )
    config.pop("llm_adapter_path", None)

    # The Anima sampler is fixed to its rectified-flow Euler implementation.
    # sample_sampler belongs to the shared SD preview UI and is ineffective here.
    config.pop("sample_sampler", None)

    weighting_scheme = str(config.get("weighting_scheme") or "uniform").lower()
    if weighting_scheme not in SUPPORTED_ANIMA_WEIGHTING_SCHEMES:
        allowed = ", ".join(sorted(SUPPORTED_ANIMA_WEIGHTING_SCHEMES))
        raise ValueError(
            f"Anima: weighting_scheme={weighting_scheme!r} 不受当前 trainer 支持；请选择 {allowed}。"
        )

    # Raw checkpoint-offload modes imply gradient checkpointing in sd-scripts;
    # keep that effective state explicit in the final TOML.
    if cpu_offload or unsloth_offload:
        config["gradient_checkpointing"] = True
