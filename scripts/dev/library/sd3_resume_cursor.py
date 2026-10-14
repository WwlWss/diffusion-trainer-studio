"""Fail-closed, SD3 LoRA-only checkpoint cursor and progress sidecar.

Only single-process, *epoch-boundary* resumes are qualified by this first
implementation. The ordinary Dev NetworkTrainer cursor remains unchanged for
Flux/Chroma; unsupported SD3 mid-epoch or dataset order cases fail early.

Accelerate owns optimizer/scheduler/RNG restoration. This module owns logical
optimizer-step/epoch units and saves a matching data-shape progress contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

SD3_RESUME_PROGRESS_FILE = "dts_sd3_resume_progress.json"
SD3_RESUME_PROGRESS_SCHEMA = "dts.sd3.resume-progress"
SD3_RESUME_PROGRESS_VERSION = 1


class SD3ResumeCursorError(ValueError):
    """State cannot establish a safe SD3 LoRA resume position."""


@dataclass(frozen=True)
class SD3ResumeCursor:
    completed_steps: int
    epoch_to_start: int
    batches_to_skip: int
    remaining_steps: int


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SD3ResumeCursorError(f"SD3 resume requires positive integer {name}; got {value!r}")
    return value


def plan_sd3_lora_resume_cursor(
    *, saved_step: Any, saved_epoch: Any, max_train_steps: Any,
    dataloader_batches: Any, gradient_accumulation_steps: Any,
    num_processes: Any = 1, initial_step: Any = None,
    initial_epoch: Any = None, max_train_epochs: Any = None,
    skip_until_initial_step: bool = False,
) -> SD3ResumeCursor:
    """Plan one absolute checkpoint continuation; steps are optimizer updates.

    A step is counted only after a completed accumulation group; trailing
    partial accumulation groups also complete the epoch. Epoch-local resumes
    are deliberately disallowed until sampler/data-order identity is proven.
    """
    conflicting = sorted(
        name for name, value in (
            ("initial_step", initial_step), ("initial_epoch", initial_epoch),
            ("max_train_epochs", max_train_epochs),
            ("skip_until_initial_step", skip_until_initial_step or None),
        ) if value is not None
    )
    if conflicting:
        raise SD3ResumeCursorError(
            "SD3 checkpoint progress cannot mix explicit lifecycle overrides: "
            f"{conflicting!r}"
        )
    processes = _positive(num_processes, "num_processes")
    if processes != 1:
        raise SD3ResumeCursorError("SD3 LoRA checkpoint cursor only supports one process")
    step = _positive(saved_step, "saved_step")
    epoch = _positive(saved_epoch, "saved_epoch")
    total = _positive(max_train_steps, "max_train_steps")
    batches = _positive(dataloader_batches, "dataloader_batches")
    accum = _positive(gradient_accumulation_steps, "gradient_accumulation_steps")
    if step >= total:
        raise SD3ResumeCursorError(
            f"SD3 checkpoint has no remaining optimizer updates: {step} >= {total}"
        )
    steps_per_epoch = (batches + accum - 1) // accum
    expected_epoch = (step - 1) // steps_per_epoch + 1
    if epoch != expected_epoch:
        raise SD3ResumeCursorError(
            f"SD3 checkpoint epoch/step mismatch: saved={epoch}, "
            f"expected={expected_epoch}, completed_steps={step}"
        )
    epoch_to_start, steps_in_epoch = divmod(step, steps_per_epoch)
    batches_to_skip = min(steps_in_epoch * accum, batches)
    if batches_to_skip != 0:
        raise SD3ResumeCursorError(
            "SD3 mid-epoch resume is not qualified: sampler order and RNG "
            "continuity are not proven. Save an epoch-boundary state."
        )
    return SD3ResumeCursor(
        completed_steps=step,
        epoch_to_start=epoch_to_start,
        batches_to_skip=0,
        remaining_steps=total - step,
    )


def _progress_contract(
    *, saved_step: Any, saved_epoch: Any, dataloader_batches: Any,
    gradient_accumulation_steps: Any, num_processes: Any,
) -> dict[str, Any]:
    step = _positive(saved_step, "saved_step")
    epoch = _positive(saved_epoch, "saved_epoch")
    batches = _positive(dataloader_batches, "dataloader_batches")
    accum = _positive(gradient_accumulation_steps, "gradient_accumulation_steps")
    processes = _positive(num_processes, "num_processes")
    return {
        "schema": SD3_RESUME_PROGRESS_SCHEMA,
        "version": SD3_RESUME_PROGRESS_VERSION,
        "completed_optimizer_steps": step,
        "current_epoch": epoch,
        "dataloader_batches": batches,
        "gradient_accumulation_steps": accum,
        "num_processes": processes,
    }


def validate_sd3_resume_progress(
    directory: str | os.PathLike[str], *,
    dataloader_batches: int, gradient_accumulation_steps: int,
    num_processes: int,
) -> dict[str, Any]:
    """Check saved geometry and train_state before any optimizer state is loaded."""
    folder = Path(directory)
    path = folder / SD3_RESUME_PROGRESS_FILE
    if not path.is_file():
        raise SD3ResumeCursorError(
            "SD3 resume requires a new-format progress sidecar: "
            f"{path}. An older state cannot be resumed without a verified cursor."
        )
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        train_state = json.loads((folder / "train_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SD3ResumeCursorError(f"Invalid SD3 resume state: {exc}") from exc
    if not isinstance(saved, dict) or not isinstance(train_state, dict):
        raise SD3ResumeCursorError("SD3 resume progress and train_state must be JSON objects")
    expected = _progress_contract(
        saved_step=train_state.get("current_step"),
        saved_epoch=train_state.get("current_epoch"),
        dataloader_batches=dataloader_batches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_processes=num_processes,
    )
    if saved != expected:
        raise SD3ResumeCursorError(
            "SD3 resume progress conflicts with saved train_state or "
            "current DataLoader/accumulation/process geometry."
        )
    return saved


def register_sd3_resume_progress_hooks(
    accelerator: Any, *, args: Any, train_dataloader: Any,
    current_epoch: Any, current_step: Any,
) -> None:
    """Register per-SD3 save and load checks; never runs for other trainers."""
    save = getattr(accelerator, "register_save_state_pre_hook", None)
    load = getattr(accelerator, "register_load_state_pre_hook", None)
    if not callable(save) or not callable(load):
        raise SD3ResumeCursorError("Accelerator state pre-hooks are unavailable")

    def save_hook(models, weights, output_dir):
        del models, weights
        if not getattr(accelerator, "is_main_process", True):
            return
        payload = _progress_contract(
            saved_step=current_step.value + 1,
            saved_epoch=current_epoch.value,
            dataloader_batches=len(train_dataloader),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_processes=accelerator.num_processes,
        )
        path = Path(output_dir) / SD3_RESUME_PROGRESS_FILE
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)

    def load_hook(models, input_dir):
        del models
        validate_sd3_resume_progress(
            input_dir,
            dataloader_batches=len(train_dataloader),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_processes=accelerator.num_processes,
        )

    save(save_hook)
    load(load_hook)


__all__ = [
    "SD3_RESUME_PROGRESS_FILE", "SD3ResumeCursor", "SD3ResumeCursorError",
    "plan_sd3_lora_resume_cursor", "validate_sd3_resume_progress",
    "register_sd3_resume_progress_hooks",
]
