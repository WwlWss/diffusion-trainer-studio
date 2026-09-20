"""Lazy bridge from sd-scripts trainers into DTS Parameter Policy runtime.

Standard mode never imports this module from trainer code. When Component-wise
integration opts in, the bridge locates the DTS repository root without assuming
that it is already present on sys.path.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _find_dts_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "mikazuki" / "parameter_policy_trainer.py").is_file():
            return parent
    raise ImportError(
        "Unable to locate DTS root containing mikazuki/parameter_policy_trainer.py."
    )


def load_parameter_policy_trainer():
    root = _find_dts_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module("mikazuki.parameter_policy_trainer")


def create_parameter_policy_session(*args, **kwargs):
    return load_parameter_policy_trainer().create_parameter_policy_session(*args, **kwargs)


def load_parameter_policy_file(*args, **kwargs):
    return load_parameter_policy_trainer().load_parameter_policy_file(*args, **kwargs)


def make_legacy_scheduler_factory(*args, **kwargs):
    return load_parameter_policy_trainer().make_legacy_scheduler_factory(*args, **kwargs)


__all__ = [
    "create_parameter_policy_session",
    "load_parameter_policy_file",
    "load_parameter_policy_trainer",
    "make_legacy_scheduler_factory",
]
