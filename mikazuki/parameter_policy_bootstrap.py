"""Bootstrap helpers for migrating legacy Standard optimizer semantics.

This module deliberately avoids importing the FastAPI application layer.  The
caller supplies the backend resolver, which keeps the helper testable in the
same lightweight host-only CI used by the rest of the semantic compiler.
"""

from __future__ import annotations

from typing import Callable

from mikazuki.parameter_policy import (
    PARAMETER_POLICY_GUI_KEYS,
    bootstrap_legacy_optimizer_profile,
)
from mikazuki.training_config import prepare_training_config
from mikazuki.utils import train_utils


def bootstrap_parameter_policy_optimizer_profile(
    config: dict,
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> dict:
    """Compile the normal Standard pipeline, then derive one initial profile.

    Step 2 intentionally stops at optimizer-profile bootstrap. Model Component
    assignment is Step 3.
    """

    candidate = dict(config)
    for key in PARAMETER_POLICY_GUI_KEYS:
        candidate.pop(key, None)
    candidate.pop("parameter_policy_config", None)
    train_utils.fix_config_types(candidate)

    prepared = prepare_training_config(
        candidate,
        page_train_type=page_type,
        resolve_backend=resolve_backend,
        launch=False,
        toml_path=None,
    )
    return bootstrap_legacy_optimizer_profile(prepared.config)


__all__ = ["bootstrap_parameter_policy_optimizer_profile"]
