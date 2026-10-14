"""Checkpoint-only forward-semantics guard for SD3 extra-attention models.

This is intentionally separate from Component manifest v2, optimizer ownership,
full-BF16 qualification, and weight format. It validates *algorithm* identity
before Accelerate loads saved optimizer/model state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SD3_FORWARD_SEMANTICS_FILE = "dts_sd3_forward_semantics.json"
SD3_FORWARD_SEMANTICS_SCHEMA = "dts.sd3.forward-semantics"
SD3_FORWARD_SEMANTICS_VERSION = 1
SD3_EXTRA_ATTENTION_ALGORITHM = "extra-attention-head-dim-v1"


class SD3ForwardSemanticsError(ValueError):
    """The checkpoint cannot resume under the currently loaded SD3 forward."""


def _positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SD3ForwardSemanticsError(
            f"Invalid SD3 forward {description}: {value!r}"
        )
    return value


def forward_semantics_for_mmdit(mmdit: Any) -> dict[str, Any] | None:
    """Compute the runtime forward contract from actual constructed block objects.

    None means this architecture has no extra-attention algorithm touched by B1.
    Do not infer release identity from a checkpoint file name or a shape label.
    """
    blocks = getattr(mmdit, "joint_blocks", None)
    if blocks is None:
        raise SD3ForwardSemanticsError("MMDiT has no joint_blocks")
    try:
        depth = len(blocks)
    except TypeError as exc:
        raise SD3ForwardSemanticsError("MMDiT joint_blocks are not sized") from exc
    if depth == 0:
        raise SD3ForwardSemanticsError("MMDiT has no joint blocks")

    extra_layers = [
        i for i, block in enumerate(blocks)
        if bool(getattr(block.x_block, "x_block_self_attn", False))
    ]
    if not extra_layers:
        return None

    hidden = _positive_int(getattr(mmdit, "hidden_size", None), "hidden_size")
    heads = _positive_int(getattr(mmdit, "num_heads", None), "num_heads")
    if hidden % heads != 0:
        raise SD3ForwardSemanticsError("MMDiT hidden size is not divisible by num_heads")
    head_dim = hidden // heads
    for index in extra_layers:
        attn2 = getattr(blocks[index].x_block, "attn2", None)
        if attn2 is None:
            raise SD3ForwardSemanticsError(
                f"SD3 extra-attention block {index} has no attn2"
            )
        if _positive_int(getattr(attn2, "head_dim", None), "extra head_dim") != head_dim:
            raise SD3ForwardSemanticsError(
                f"SD3 extra-attention block {index} has incompatible head_dim"
            )
        if _positive_int(getattr(attn2, "num_heads", None), "extra num_heads") != heads:
            raise SD3ForwardSemanticsError(
                f"SD3 extra-attention block {index} has incompatible num_heads"
            )
    return {
        "schema": SD3_FORWARD_SEMANTICS_SCHEMA,
        "version": SD3_FORWARD_SEMANTICS_VERSION,
        "attention_contract": SD3_EXTRA_ATTENTION_ALGORITHM,
        "model_type_hint": str(getattr(mmdit, "model_type", "")),
        "depth": depth,
        "hidden_size": hidden,
        "num_heads": heads,
        "head_dim": head_dim,
        "extra_attention_layers": extra_layers,
    }


def validate_sd3_forward_checkpoint(directory: str | os.PathLike[str], contract: dict[str, Any]) -> None:
    """Called as an Accelerate load-pre-hook, before optimizer state restore."""
    path = Path(directory) / SD3_FORWARD_SEMANTICS_FILE
    if not path.is_file():
        raise SD3ForwardSemanticsError(
            "SD3 extra-attention checkpoint lacks a forward-semantics sidecar: "
            f"{path}. Old forward semantics cannot safely resume after B1; "
            "load LoRA weights as a new stage instead of --resume."
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SD3ForwardSemanticsError(
            f"Invalid SD3 forward-semantics sidecar {path}: {exc}"
        ) from exc
    if actual != contract:
        raise SD3ForwardSemanticsError(
            "SD3 checkpoint forward-semantics mismatch: expected "
            f"{contract!r}, got {actual!r}. Start a new stage instead of --resume."
        )


def register_sd3_forward_semantics_hooks(accelerator: Any, mmdit: Any) -> bool:
    """Register scoped Accelerate hooks; return False for unaffected architectures."""
    contract = forward_semantics_for_mmdit(mmdit)
    if contract is None:
        return False
    save_hook = getattr(accelerator, "register_save_state_pre_hook", None)
    load_hook = getattr(accelerator, "register_load_state_pre_hook", None)
    if not callable(save_hook) or not callable(load_hook):
        raise SD3ForwardSemanticsError("Accelerator save/load pre-hooks are unavailable")

    def write_semantics(models, weights, output_dir):
        del models, weights
        if not getattr(accelerator, "is_main_process", True):
            return
        path = Path(output_dir) / SD3_FORWARD_SEMANTICS_FILE
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)

    def check_semantics(models, input_dir):
        del models
        validate_sd3_forward_checkpoint(input_dir, contract)

    save_hook(write_semantics)
    load_hook(check_semantics)
    return True


__all__ = [
    "SD3_FORWARD_SEMANTICS_FILE", "SD3ForwardSemanticsError",
    "forward_semantics_for_mmdit", "validate_sd3_forward_checkpoint",
    "register_sd3_forward_semantics_hooks",
]
