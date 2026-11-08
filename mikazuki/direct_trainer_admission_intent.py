"""Ephemeral, strict Direct Trainer CLI acknowledgement intent.

This module is deliberately dependency-light: it does not import torch,
Accelerate, sd-scripts, or the model-aware Parameter Policy Trainer.  The
intent is per process invocation and must never be serialized into Trainer
configuration, task metadata, sidecars, or Checkpoint state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ACK_ARGUMENT = "experimental_full_bf16_ack"
ACK_OPTION = "--experimental_full_bf16_ack"


@dataclass(frozen=True, slots=True)
class DirectTrainerAdmissionIntent:
    """One invocation's explicit CLI acknowledgement; never a permission."""

    experimental_full_bf16_ack: bool = False

    def __post_init__(self) -> None:
        if type(self.experimental_full_bf16_ack) is not bool:
            raise ValueError(
                "Direct Trainer acknowledgement must be an explicit boolean."
            )


def add_direct_trainer_admission_argument(parser: Any) -> None:
    """Register the one-shot Direct Trainer CLI acknowledgement switch."""
    parser.add_argument(
        ACK_OPTION,
        dest=ACK_ARGUMENT,
        action="store_true",
        default=False,
        help=(
            "Acknowledge this single Direct Trainer Full-BF16 candidate "
            "(does not override admission or runtime blockers)."
        ),
    )


def extract_direct_trainer_admission_intent(
    args: Any,
) -> DirectTrainerAdmissionIntent:
    """Consume the raw argparse flag and return a strict immutable intent."""
    if args is None:
        raise TypeError("Direct Trainer CLI arguments are required.")
    value = getattr(args, ACK_ARGUMENT, False)
    if type(value) is not bool:
        raise ValueError(
            "Direct Trainer acknowledgement must be supplied as a CLI boolean."
        )
    try:
        delattr(args, ACK_ARGUMENT)
    except AttributeError:
        pass
    return DirectTrainerAdmissionIntent(experimental_full_bf16_ack=value)


def reject_direct_trainer_ack_toml(values: Mapping[str, Any]) -> None:
    """Reject a reserved control-plane key after TOML section flattening."""
    if not isinstance(values, Mapping):
        raise TypeError("Flattened TOML configuration must be a mapping.")
    if ACK_ARGUMENT in values:
        raise ValueError(
            f"{ACK_ARGUMENT} is a per-invocation CLI control and cannot be "
            "supplied by TOML configuration."
        )


__all__ = [
    "ACK_ARGUMENT",
    "ACK_OPTION",
    "DirectTrainerAdmissionIntent",
    "add_direct_trainer_admission_argument",
    "extract_direct_trainer_admission_intent",
    "reject_direct_trainer_ack_toml",
]
