"""Pure, fail-closed resume cursor for qualified SD1/SDXL stock-LoRA full-BF16.

Accelerate restores model, optimizer, scheduler, and RNG state, but the legacy
network trainer separately owns the global optimizer step and epoch-local data
position.  Keep those two counters separate when resuming qualified stock LoRA.
No torch, CUDA, or trainer imports are allowed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass


class SdLoraResumeCursorError(ValueError):
    """The checkpoint cannot establish one unambiguous D1 resume position."""


@dataclass(frozen=True)
class SdLoraResumeCursor:
    completed_steps: int
    epoch_to_start: int
    batches_to_skip: int
    remaining_steps: int

    def skipped_batches_for_epoch(self, epoch: int) -> int:
        """Skip processed batches only in the first resumed epoch."""
        return self.batches_to_skip if epoch == self.epoch_to_start else 0


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SdLoraResumeCursorError(
            f"D1 stock LoRA resume requires {field} to be a positive integer; "
            f"got {value!r}."
        )
    return value


def plan_stock_lora_full_bf16_resume(
    *,
    saved_step: object,
    saved_epoch: object,
    max_train_steps: object,
    dataloader_batches: object,
    gradient_accumulation_steps: object,
    initial_step: object = None,
    initial_epoch: object = None,
    max_train_epochs: object = None,
    skip_until_initial_step: bool = False,
) -> SdLoraResumeCursor:
    """Return the logical step and epoch-local batch offset after a D1 resume.

    Each saved step is a *completed* optimizer update, not a microbatch.
    Accelerate's non-stateful DataLoader resumes at an epoch boundary and must
    skip the already-processed microbatches in its first resumed epoch.
    Last, incomplete accumulation groups still count as one completed step.
    """
    overrides = {
        "initial_step": initial_step,
        "initial_epoch": initial_epoch,
        "max_train_epochs": max_train_epochs,
    }
    active = [name for name, value in overrides.items() if value is not None]
    if skip_until_initial_step:
        active.append("skip_until_initial_step")
    if active:
        raise SdLoraResumeCursorError(
            "D1 stock LoRA full-BF16 resume cannot combine checkpoint progress "
            f"with explicit lifecycle override(s): {sorted(active)!r}."
        )

    step = _positive_integer(saved_step, "checkpoint current_step")
    checkpoint_epoch = _positive_integer(saved_epoch, "checkpoint current_epoch")
    target = _positive_integer(max_train_steps, "max_train_steps")
    batches = _positive_integer(dataloader_batches, "dataloader batch count")
    accumulation = _positive_integer(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    if step >= target:
        raise SdLoraResumeCursorError(
            "D1 stock LoRA resume checkpoint has no remaining optimizer steps: "
            f"completed={step}, max_train_steps={target}."
        )

    steps_per_epoch = (batches + accumulation - 1) // accumulation
    expected_checkpoint_epoch = (step - 1) // steps_per_epoch + 1
    if checkpoint_epoch != expected_checkpoint_epoch:
        raise SdLoraResumeCursorError(
            "D1 stock LoRA checkpoint epoch/optimizer-step mismatch: "
            f"current_epoch={checkpoint_epoch}, current_step={step}, "
            f"expected_epoch={expected_checkpoint_epoch}, "
            f"steps_per_epoch={steps_per_epoch}."
        )

    epoch_to_start, steps_in_epoch = divmod(step, steps_per_epoch)
    # A partial final accumulation group completes the epoch; do not attempt
    # to skip more batches than the epoch contains.
    batches_to_skip = min(steps_in_epoch * accumulation, batches)
    return SdLoraResumeCursor(
        completed_steps=step,
        epoch_to_start=epoch_to_start,
        batches_to_skip=batches_to_skip,
        remaining_steps=target - step,
    )


def plan_sd_lora_full_bf16_resume(**kwargs: object) -> SdLoraResumeCursor:
    """Compatibility entrypoint for the previously qualified SD1 lifecycle."""
    return plan_stock_lora_full_bf16_resume(**kwargs)


__all__ = [
    "SdLoraResumeCursor",
    "SdLoraResumeCursorError",
    "plan_sd_lora_full_bf16_resume",
    "plan_stock_lora_full_bf16_resume",
]
