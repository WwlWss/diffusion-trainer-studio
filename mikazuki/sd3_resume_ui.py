"""SD3 LoRA WebUI duration and read-only checkpoint preflight.

The SD3 trainer remains authoritative for checkpoint load and forward identity.
This layer makes GUI Preview/Export/Start agree on an explicit total-step target
and rejects predictable resume mistakes before allocating model memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

_PROGRESS_SCHEMA = "dts.sd3.resume-progress"
_PROGRESS_VERSION = 1


@dataclass(frozen=True)
class SD3ResumePreview:
    completed_steps: int
    current_epoch: int
    dataloader_batches: int
    gradient_accumulation_steps: int
    steps_per_epoch: int


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"SD3: {label} 必须是正整数。")
    if isinstance(value, str) and value.strip().isdecimal():
        value = int(value.strip())
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"SD3: {label} 必须是正整数。")
    return value


def _read_object(folder: Path, filename: str) -> dict[str, Any]:
    path = folder / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"SD3 Resume: 无法读取有效的 {filename}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"SD3 Resume: {filename} 必须是 JSON object。")
    return payload


def inspect_sd3_resume_checkpoint(raw_dir: str) -> SD3ResumePreview:
    """Read the PR #60 sidecar without touching Accelerate or creating files."""
    folder = Path(raw_dir).expanduser()
    if not folder.is_dir():
        raise ValueError(f"SD3 Resume: Checkpoint 目录不存在: {folder}")
    progress = _read_object(folder, "dts_sd3_resume_progress.json")
    state = _read_object(folder, "train_state.json")
    step = _positive_int(state.get("current_step"), "train_state.current_step")
    epoch = _positive_int(state.get("current_epoch"), "train_state.current_epoch")
    batches = _positive_int(progress.get("dataloader_batches"), "dataloader_batches")
    accumulation = _positive_int(
        progress.get("gradient_accumulation_steps"), "gradient_accumulation_steps"
    )
    processes = _positive_int(progress.get("num_processes"), "num_processes")
    expected = {
        "schema": _PROGRESS_SCHEMA,
        "version": _PROGRESS_VERSION,
        "completed_optimizer_steps": step,
        "current_epoch": epoch,
        "dataloader_batches": batches,
        "gradient_accumulation_steps": accumulation,
        "num_processes": processes,
    }
    # Mirror the trainer's exact saved-progress / train-state comparison.
    if progress != expected:
        raise ValueError("SD3 Resume: progress sidecar 与 train_state 不匹配，或 Schema/Version 不兼容。")
    if processes != 1:
        raise ValueError("SD3 Resume: 当前只支持单进程 Checkpoint。")
    steps_per_epoch = (batches + accumulation - 1) // accumulation
    if step % steps_per_epoch:
        raise ValueError("SD3 Resume: 尚不支持从 Epoch 中途保存的状态恢复。")
    expected_epoch = (step - 1) // steps_per_epoch + 1
    if epoch != expected_epoch:
        raise ValueError("SD3 Resume: 保存的 Epoch 与完成的优化步数不一致。")
    # The full forward-semantics contract depends on the *loaded* MMDiT,
    # and is deliberately still validated by the trainer's pre-load hook.
    return SD3ResumePreview(step, epoch, batches, accumulation, steps_per_epoch)


def normalize_sd3_training_duration(config: dict, warnings: list[str]) -> None:
    """Compile the SD3-only GUI target into existing sd-scripts CLI fields.

    This MUST run after ui_custom_params. Explicit GUI targets cannot coexist
    with legacy trainer duration overrides, even if those overrides are equal.
    Legacy (no GUI target mode) preserves the original epoch-first behavior.
    """
    mode = config.pop("sd3_training_target_mode", None)
    gui_epochs = config.pop("sd3_training_target_epochs", None)
    gui_steps = config.pop("sd3_training_target_steps", None)
    raw_epochs = config.pop("max_train_epochs", None)
    raw_steps = config.pop("max_train_steps", None)
    has_epochs = raw_epochs not in (None, "")
    has_steps = raw_steps not in (None, "")
    resume = str(config.get("resume") or "").strip()
    if not resume:
        config.pop("resume", None)
    else:
        # The exact path inspected by Preview must also reach the trainer.
        config["resume"] = resume
        conflicting = [
            name for name in ("initial_step", "initial_epoch")
            if config.get(name) is not None
        ]
        if config.get("skip_until_initial_step"):
            conflicting.append("skip_until_initial_step")
        if conflicting:
            raise ValueError(
                "SD3 Resume: 不能同时设置旧式训练游标参数: "
                + ", ".join(sorted(conflicting))
            )
    if mode not in (None, "", "epochs", "steps"):
        raise ValueError("SD3: 训练目标只能选择 epochs 或 steps。")

    if mode in ("epochs", "steps"):
        if has_epochs or has_steps:
            raise ValueError(
                "SD3: 训练目标已由 UI 模式指定，不能再通过旧字段 "
                "max_train_epochs / max_train_steps 覆盖；请使用对应的训练目标控件。"
            )
        if mode == "epochs":
            epochs = _positive_int(gui_epochs, "目标 Epoch")
            steps = None
        else:
            steps = _positive_int(gui_steps, "目标 Optimizer Steps")
            epochs = None
    elif has_epochs:
        # Legacy sd-scripts honors max_train_epochs over max_train_steps.
        epochs = _positive_int(raw_epochs, "max_train_epochs")
        steps = None
        if has_steps:
            _positive_int(raw_steps, "max_train_steps")
            if not resume:
                config["max_train_steps"] = int(raw_steps)
            warnings.append("SD3 旧配置同时指定 Epoch/Steps；沿用原训练器的 Epoch 优先语义。")
    elif has_steps:
        steps = _positive_int(raw_steps, "max_train_steps")
        epochs = None
    else:
        if resume:
            raise ValueError("SD3 Resume: 必须明确指定目标总 Optimizer Steps 或目标 Epoch。")
        # Historical headless / API callers without a GUI target keep the
        # trainer's original default step count.
        return

    if not resume:
        if epochs is not None:
            config["max_train_epochs"] = epochs
        else:
            config["max_train_steps"] = steps
        return

    saved = inspect_sd3_resume_checkpoint(resume)
    current_accumulation = _positive_int(
        config.get("gradient_accumulation_steps", 1),
        "gradient_accumulation_steps",
    )
    if current_accumulation != saved.gradient_accumulation_steps:
        raise ValueError(
            "SD3 Resume: 当前 gradient_accumulation_steps 与 Checkpoint 不一致。"
        )
    total = epochs * saved.steps_per_epoch if epochs is not None else steps
    if total <= saved.completed_steps:
        raise ValueError(
            f"SD3 Resume: 目标总步数 {total} 必须大于 Checkpoint 已完成步数 "
            f"{saved.completed_steps}。"
        )
    config["max_train_steps"] = total
    if epochs is not None:
        warnings.append(
            f"SD3 Resume: 目标 {epochs} Epoch × {saved.steps_per_epoch} "
            f"optimizer steps/Epoch = {total} 总步数；"
            f"Checkpoint 已完成 {saved.completed_steps} 步。"
        )
    warnings.append(
        "SD3 Resume: Checkpoint 未记录原训练的目标总步数；"
        "请确认本次总步数与原训练计划一致。调度器状态由 Accelerate 恢复，"
        "此预检不能证明原数据顺序或模型前向语义兼容。"
    )


def validate_sd3_resume_gpu_selection(config: dict, gpu_ids: Any) -> None:
    if not config.get("resume") or gpu_ids is None:
        return
    if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) > 1:
        raise ValueError("SD3 Resume: 当前只支持单 GPU / 单进程恢复。")


__all__ = [
    "SD3ResumePreview", "inspect_sd3_resume_checkpoint",
    "normalize_sd3_training_duration", "validate_sd3_resume_gpu_selection",
]
