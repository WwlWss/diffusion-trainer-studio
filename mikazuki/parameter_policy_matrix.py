"""Release matrix for Parameter Policy Component Start.

This module is host-side and deliberately torch-free. It records the model
families whose trainer integration is complete enough for baseline Component
Start and the execution modes that still require dedicated qualification or
new semantics.

"Baseline" means one CUDA process using the ordinary trainer optimizer/forward
lifecycle. Feature-specific blockers remain authoritative in
parameter_policy_compat.py.
"""

from __future__ import annotations

from typing import Final


PARAMETER_POLICY_BACKEND_MATRIX_VERSION: Final = 1

PARAMETER_POLICY_BACKEND_MATRIX: Final[dict[str, dict[str, str]]] = {
    "sd-lora": {
        "trainer_family": "stable-network",
        "trainer": "./scripts/stable/train_network.py",
    },
    "sdxl-lora": {
        "trainer_family": "stable-network",
        "trainer": "./scripts/stable/sdxl_train_network.py",
    },
    "sd-dreambooth": {
        "trainer_family": "stable-full",
        "trainer": "./scripts/stable/train_db.py",
    },
    "sdxl-finetune": {
        "trainer_family": "stable-full",
        "trainer": "./scripts/stable/sdxl_train.py",
    },
    "sd3-lora": {
        "trainer_family": "dev-network",
        "trainer": "./scripts/dev/sd3_train_network.py",
    },
    "flux-lora": {
        "trainer_family": "dev-network",
        "trainer": "./scripts/dev/flux_train_network.py",
    },
    "chroma-lora": {
        "trainer_family": "dev-network",
        "trainer": "./scripts/dev/flux_train_network.py",
    },
    "flux-finetune": {
        "trainer_family": "dev-full",
        "trainer": "./scripts/dev/flux_train.py",
    },
    "anima-lora": {
        "trainer_family": "anima-staged-network",
        "trainer": "./sd-scripts/anima_train_network.py",
    },
    "anima-finetune": {
        "trainer_family": "anima-staged-full",
        "trainer": "./sd-scripts/anima_train.py",
    },
}

PARAMETER_POLICY_RUNTIME_TRAIN_TYPES: Final[frozenset[str]] = frozenset(
    PARAMETER_POLICY_BACKEND_MATRIX
)

# These modes change optimizer ownership, parameter residency, model wrapping,
# or distributed state. They remain fail-closed until a dedicated Component
# runtime contract exists and is qualified.
PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "torch_compile",
        "compile",
        "deepspeed",
        "fused_backward_pass",
        "fused_optimizer_groups",
        "blockwise_fused_optimizers",
        "cpu_offload_checkpointing",
        "unsloth_offload_checkpointing",
        "blocks_to_swap",
        "double_blocks_to_swap",
        "single_blocks_to_swap",
        "full_fp16",
        "full_bf16",
        "fp8_base",
        "fp8_base_unet",
    }
)

# These are not merely untested execution modes. Parameter Policy v1 cannot
# represent their semantics exactly, so GPU smoke alone must never unblock them.
PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "lora_plus",
        "regex_lr",
        "sd_lora_block_lr",
        "sdxl_full_block_lr",
        "scale_weight_norms",
        "dreambooth_dynamic_text_encoder_stop",
        "preloaded_adapter_text_encoder_cache",
        "anima_qwen_only",
        "unreviewed_network_module",
    }
)


def parameter_policy_gpu_selection_blockers(gpu_ids: object) -> list[str]:
    """Fail closed on explicit multi-GPU Component launch until DDP is qualified."""

    if gpu_ids in (None, "", []):
        return []
    if not isinstance(gpu_ids, (list, tuple)):
        raise ValueError(
            "Parameter Policy GPU selection must be a list/tuple when supplied."
        )
    normalized = [str(value).strip() for value in gpu_ids if str(value).strip()]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Parameter Policy GPU selection contains duplicate device ids: "
            + ", ".join(normalized)
        )
    if len(normalized) > 1:
        return [
            "Component-wise v1 当前只开放单 GPU Start；显式多 GPU/DDP "
            "尚未完成 CompositeOptimizer、checkpoint 与跨进程 ownership 验证。"
        ]
    return []


__all__ = [
    "PARAMETER_POLICY_BACKEND_MATRIX",
    "PARAMETER_POLICY_BACKEND_MATRIX_VERSION",
    "PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS",
    "PARAMETER_POLICY_RUNTIME_TRAIN_TYPES",
    "PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES",
    "parameter_policy_gpu_selection_blockers",
]
