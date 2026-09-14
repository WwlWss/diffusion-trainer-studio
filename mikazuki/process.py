"""Compatibility entrypoint for already-prepared training TOMLs.

Historically this module performed a second round of Anima/config normalization.
That created a second semantic engine beside the web API. All preparation now
lives in ``training_config`` / ``training_request``; this module only delegates
an already-final TOML to the common launcher.
"""

from __future__ import annotations

from typing import Optional

from mikazuki.training_launcher import run_prepared_train


def run_train(
    toml_path: str,
    trainer_file: str = "./scripts/stable/train_network.py",
    gpu_ids: Optional[list] = None,
    cpu_threads: Optional[int] = 2,
):
    return run_prepared_train(
        toml_path,
        trainer_file,
        gpu_ids,
        cpu_threads,
    )
