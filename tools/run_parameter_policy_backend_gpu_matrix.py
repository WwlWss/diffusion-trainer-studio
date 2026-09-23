#!/usr/bin/env python3
"""Execute the Step 6F real-backend CUDA qualification matrix.

The manifest supplies machine-local model/dataset paths through explicit trainer
commands. Each of the ten released backends must run a fresh Component training
case, produce the Step 6E checkpoint manifest v2, then successfully resume from
that checkpoint. Trainer lifecycle assertions provide the ownership/device
checks; this harness makes the backend coverage and evidence auditable.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.parameter_policy_trainer import (
    PARAMETER_POLICY_CHECKPOINT_MANIFEST,
    PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION,
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Backend GPU manifest must be schema_version=1.")
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Backend GPU manifest cases must be a list.")

    by_backend: dict[str, dict] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every backend GPU case must be an object.")
        train_type = str(case.get("train_type") or "").strip()
        if train_type in by_backend:
            raise ValueError(f"Duplicate backend GPU case: {train_type!r}.")
        for key in ("fresh_command", "resume_command", "checkpoint_dir"):
            if key not in case:
                raise ValueError(f"Backend {train_type!r} is missing {key}.")
        for key in ("fresh_command", "resume_command"):
            command = case[key]
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(item, str) and item for item in command)
            ):
                raise ValueError(
                    f"Backend {train_type!r} {key} must be a non-empty argv list."
                )
        if not isinstance(case["checkpoint_dir"], str) or not case["checkpoint_dir"].strip():
            raise ValueError(f"Backend {train_type!r} checkpoint_dir must be a path string.")
        environment = case.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError(
                f"Backend {train_type!r} environment must be a string mapping."
            )
        by_backend[train_type] = case

    expected = set(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES)
    actual = set(by_backend)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Backend GPU matrix must cover the exact Step 6F release set; "
            f"missing={missing!r}, extra={extra!r}."
        )
    return [by_backend[name] for name in sorted(by_backend)]


def _run_command(command: list[str], *, cwd: str | None, environment: dict[str, str]):
    env = dict(os.environ)
    env.update(environment)
    completed = subprocess.run(
        command,
        cwd=cwd or None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.returncode, completed.stdout


def _validate_checkpoint(case: dict) -> dict:
    manifest_path = (
        Path(case["checkpoint_dir"]) / PARAMETER_POLICY_CHECKPOINT_MANIFEST
    )
    if not manifest_path.is_file():
        raise AssertionError(
            f"missing checkpoint manifest after fresh run: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION:
        raise AssertionError(
            f"checkpoint manifest version={payload.get('version')!r}, "
            f"expected={PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION}"
        )
    if payload.get("train_type") != case["train_type"]:
        raise AssertionError(
            f"checkpoint train_type={payload.get('train_type')!r}, "
            f"expected={case['train_type']!r}"
        )
    for key in (
        "policy_hash",
        "runtime_topology_fingerprint",
        "trainable_components",
        "frozen_components",
        "trainable_parameter_tensors",
        "trainable_parameter_elements",
    ):
        if key not in payload:
            raise AssertionError(f"checkpoint manifest missing {key!r}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="parameter-policy-backend-gpu-matrix.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Backend GPU matrix requires a real CUDA runtime.")

    cases = _load_manifest(Path(args.manifest))
    evidence = {
        "schema_version": 1,
        "commit": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": {
            name: _version(name)
            for name in (
                "accelerate",
                "bitsandbytes",
                "lion-pytorch",
                "schedulefree",
                "pytorch-optimizer",
            )
        },
        "cases": [],
    }

    failed = False
    for case in cases:
        row = {"train_type": case["train_type"]}
        started = time.time()
        try:
            fresh_code, fresh_log = _run_command(
                case["fresh_command"],
                cwd=case.get("cwd"),
                environment=case.get("environment", {}),
            )
            row["fresh_exit_code"] = fresh_code
            row["fresh_log_tail"] = fresh_log[-12000:]
            if fresh_code != 0:
                raise AssertionError(f"fresh trainer exited with {fresh_code}")

            checkpoint = _validate_checkpoint(case)
            row["checkpoint"] = {
                "version": checkpoint["version"],
                "train_type": checkpoint["train_type"],
                "policy_hash": checkpoint["policy_hash"],
                "runtime_topology_fingerprint": checkpoint[
                    "runtime_topology_fingerprint"
                ],
                "trainable_components": checkpoint["trainable_components"],
                "frozen_components": checkpoint["frozen_components"],
                "trainable_parameter_tensors": checkpoint[
                    "trainable_parameter_tensors"
                ],
                "trainable_parameter_elements": checkpoint[
                    "trainable_parameter_elements"
                ],
            }

            resume_code, resume_log = _run_command(
                case["resume_command"],
                cwd=case.get("cwd"),
                environment=case.get("environment", {}),
            )
            row["resume_exit_code"] = resume_code
            row["resume_log_tail"] = resume_log[-12000:]
            if resume_code != 0:
                raise AssertionError(f"resume trainer exited with {resume_code}")

            # Re-read after resume so a destructive/mismatched resume cannot
            # silently replace the identity evidence.
            resumed_checkpoint = _validate_checkpoint(case)
            if resumed_checkpoint != checkpoint:
                raise AssertionError(
                    "checkpoint manifest identity changed across resume"
                )
            row["status"] = "pass"
        except Exception as exc:
            failed = True
            row["status"] = "fail"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["duration_seconds"] = round(time.time() - started, 4)
        evidence["cases"].append(row)
        print(f"{row['status'].upper():4} {row['train_type']}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
