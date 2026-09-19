"""Bootstrap helpers for migrating legacy Standard optimizer semantics.

This module deliberately avoids importing the FastAPI application layer. The
caller supplies the backend resolver, which keeps the helper testable in the
same lightweight host-only CI used by the rest of the semantic compiler.

Commit 4B provides only:
- one exact Standard effective-config snapshot helper;
- strict legacy learning-rate primitives.

Backend-specific Component mapping is added in Commit 4C.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from mikazuki.parameter_policy import (
    PARAMETER_POLICY_GUI_KEYS,
    bootstrap_legacy_optimizer_profile,
)
from mikazuki.training_config import PreparedTrainingConfig, prepare_training_config
from mikazuki.utils import train_utils


LEGACY_BOOTSTRAP_PROFILE = "legacy_main"


def _prepare_standard_snapshot(
    config: Mapping[str, Any],
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> PreparedTrainingConfig:
    """Compile one immutable caller snapshot through the real Standard pipeline.

    Parameter Policy GUI/sidecar fields are host-owned migration inputs and are
    removed before compilation. All remaining Standard semantics, including
    ui_custom_params last-write-wins behavior, are delegated to the existing
    effective-config compiler.
    """

    if not isinstance(config, Mapping):
        raise ValueError("Parameter Policy bootstrap: config 必须是 mapping。")

    candidate = dict(config)
    for key in PARAMETER_POLICY_GUI_KEYS:
        candidate.pop(key, None)
    candidate.pop("parameter_policy_config", None)

    train_utils.fix_config_types(candidate)
    return prepare_training_config(
        candidate,
        page_train_type=page_type,
        resolve_backend=resolve_backend,
        launch=False,
        toml_path=None,
    )


def _parse_legacy_lr(value: object, *, field: str) -> float:
    """Parse one already-effective legacy LR as a finite nonnegative float."""

    message = (
        f"Parameter Policy bootstrap: {field} 必须是有限非负数字；"
        "legacy LR=0 只允许迁移为 Train=false。"
    )
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(message)
    return result


def _resolve_legacy_lr(
    raw_value: object,
    *,
    field: str,
    fallback: float | None = None,
) -> float:
    """Resolve an optional legacy LR using only an explicitly verified fallback."""

    if raw_value in (None, ""):
        if fallback is None:
            raise ValueError(
                f"Parameter Policy bootstrap: 缺少必需学习率 {field}；"
                "不会猜测 trainer argparse 默认值。"
            )
        return _parse_legacy_lr(fallback, field=f"{field} fallback")
    return _parse_legacy_lr(raw_value, field=field)


def _component_route_from_legacy_lr(
    learning_rate: object,
    *,
    optimizer_profile: str = LEGACY_BOOTSTRAP_PROFILE,
) -> dict[str, Any]:
    """Translate historical zero-freeze semantics into explicit Train state."""

    lr = _parse_legacy_lr(learning_rate, field="component learning rate")
    if lr == 0:
        return {"train": False}
    return {
        "train": True,
        "optimizer_profile": optimizer_profile,
        "learning_rate": lr,
    }


def _expand_text_encoder_lrs(
    raw_value: object,
    *,
    count: int,
    base_lr: object,
) -> tuple[float, ...]:
    """Expand dev NetworkTrainer Text Encoder LR semantics deterministically.

    Semantics mirror the reviewed trainer behavior:
    - missing / empty -> base LR for every encoder;
    - scalar -> repeat for every encoder;
    - one-item sequence -> repeat the item;
    - shorter sequence -> pad with its final item;
    - longer sequence -> truncate to the requested encoder count.

    String values are scalar numeric values, not comma-separated lists.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            "Parameter Policy bootstrap: Text Encoder LR count 必须是正整数。"
        )

    base = _parse_legacy_lr(base_lr, field="learning_rate")

    if raw_value in (None, ""):
        values: list[object] = []
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        values = [raw_value]

    if not values:
        return tuple(base for _ in range(count))

    parsed = [
        _parse_legacy_lr(value, field=f"text_encoder_lr[{index}]")
        for index, value in enumerate(values)
    ]

    if len(parsed) == 1:
        return tuple(parsed[0] for _ in range(count))

    if len(parsed) < count:
        parsed.extend([parsed[-1]] * (count - len(parsed)))

    return tuple(parsed[:count])


def bootstrap_parameter_policy_optimizer_profile(
    config: Mapping[str, Any],
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> dict:
    """Compile the normal Standard pipeline, then derive one initial profile.

    The public behavior of the existing profile-only bootstrap is preserved.
    Full Component assignment is added by later Commit 4 stages.
    """

    prepared = _prepare_standard_snapshot(
        config,
        page_type,
        resolve_backend=resolve_backend,
    )
    return bootstrap_legacy_optimizer_profile(prepared.config)


__all__ = [
    "bootstrap_parameter_policy_optimizer_profile",
]
