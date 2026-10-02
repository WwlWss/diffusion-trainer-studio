#!/usr/bin/env python3
"""Exact-head CUDA evidence harness for Parameter Policy execution qualification.

Phase C0/C1 established the strict exact-head CUDA/BF16 evidence infrastructure.
Phase C2 extends that protocol with AdamW full-BF16 optimizer lifecycle evidence.
Phase C3 adds a separate pure-Muon lifecycle bundle that proves the pinned Muon
provider uses only explicit Muon groups/state, without its internal AdamW fallback.
Both train/save flows are followed by resume/second-step in a fresh subprocess.

The coordinator launches every evidence phase in a fresh Python subprocess.
Optimizer evidence remains non-promoting until the final qualification gate,
and the tiny Flux scaffold is never backend-qualification evidence.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STABLE_SCRIPTS_ROOT = REPO_ROOT / "scripts" / "stable"


class ExecutionGpuMatrixBootstrapError(RuntimeError):
    pass


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ExecutionGpuMatrixBootstrapError(
            f"git {' '.join(args)} failed: {exc.output.strip()}"
        ) from exc


def _git_sha() -> str:
    sha = _git("rev-parse", "HEAD")
    if not sha:
        raise ExecutionGpuMatrixBootstrapError(
            "git rev-parse HEAD returned an empty SHA."
        )
    return sha


def _expected_commit_from_argv(argv: list[str]) -> str:
    for index, value in enumerate(argv):
        if value == "--expected-commit":
            if index + 1 >= len(argv):
                break
            expected = argv[index + 1].strip()
            if expected:
                return expected
            break
        if value.startswith("--expected-commit="):
            expected = value.split("=", 1)[1].strip()
            if expected:
                return expected
            break
    raise ExecutionGpuMatrixBootstrapError("--expected-commit is required.")


def _assert_exact_clean_head(expected_commit: str) -> str:
    expected = str(expected_commit or "").strip()
    if not expected:
        raise ExecutionGpuMatrixBootstrapError("--expected-commit is required.")

    actual = _git_sha()
    if actual != expected:
        raise ExecutionGpuMatrixBootstrapError(
            "Execution GPU qualification must run on the requested exact head: "
            f"expected={expected}, actual={actual}."
        )

    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:12])
        raise ExecutionGpuMatrixBootstrapError(
            "Execution GPU qualification requires a completely clean workspace, "
            "including untracked files. Remove or move evidence outputs outside "
            f"the repository before running. git status:\n{preview}"
        )
    return actual


def _output_from_argv(argv: list[str], *, default: str) -> str:
    output = default
    for index, value in enumerate(argv):
        if value == "--output":
            if index + 1 >= len(argv):
                raise ExecutionGpuMatrixBootstrapError(
                    "--output requires a non-empty value."
                )
            candidate = argv[index + 1].strip()
            if not candidate:
                raise ExecutionGpuMatrixBootstrapError(
                    "--output requires a non-empty value."
                )
            output = candidate
        elif value.startswith("--output="):
            candidate = value.split("=", 1)[1].strip()
            if not candidate:
                raise ExecutionGpuMatrixBootstrapError(
                    "--output requires a non-empty value."
                )
            output = candidate
    return output


def _worker_mode_from_argv(argv: list[str]) -> bool:
    worker_flags = ("--worker-case", "--worker-phase", "--worker-result", "--worker-dir")
    return any(
        value == flag or value.startswith(flag + "=")
        for value in argv
        for flag in worker_flags
    )


def _single_worker_path_from_argv(
    argv: list[str],
    *,
    option: str,
) -> str:
    candidate: str | None = None
    prefix = option + "="
    for index, value in enumerate(argv):
        parsed: str | None = None
        if value == option:
            if index + 1 >= len(argv):
                raise ExecutionGpuMatrixBootstrapError(
                    f"{option} requires a non-empty value."
                )
            parsed = argv[index + 1].strip()
        elif value.startswith(prefix):
            parsed = value.split("=", 1)[1].strip()
        else:
            continue

        if candidate is not None:
            raise ExecutionGpuMatrixBootstrapError(
                f"Worker mode requires {option} exactly once before runtime imports."
            )
        if not parsed:
            raise ExecutionGpuMatrixBootstrapError(
                f"{option} requires a non-empty value."
            )
        candidate = parsed

    if candidate is None:
        raise ExecutionGpuMatrixBootstrapError(
            f"Worker mode requires {option} before runtime imports."
        )
    return candidate


def _worker_dir_from_argv(argv: list[str]) -> str:
    return _single_worker_path_from_argv(argv, option="--worker-dir")


def _assert_worker_dir_outside_repo(worker_dir: str) -> Path:
    resolved = Path(worker_dir).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=False))
    except ValueError:
        return resolved
    raise ExecutionGpuMatrixBootstrapError(
        "Execution GPU qualification worker requires --worker-dir outside "
        f"the repository; received {resolved}."
    )


def _worker_result_from_argv(argv: list[str]) -> str:
    return _single_worker_path_from_argv(argv, option="--worker-result")


def _assert_worker_result_in_dir(worker_result: str, worker_dir: Path) -> Path:
    resolved = Path(worker_result).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(worker_dir)
    except ValueError as exc:
        raise ExecutionGpuMatrixBootstrapError(
            "Execution GPU qualification worker requires --worker-result "
            f"inside --worker-dir; received {resolved}."
        ) from exc
    return resolved


def _assert_output_outside_repo(output: str) -> Path:
    resolved = Path(output).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=False))
    except ValueError:
        return resolved
    raise ExecutionGpuMatrixBootstrapError(
        "Execution GPU qualification requires --output outside the repository; "
        f"received {resolved}."
    )


_BOOTSTRAP_EXPECTED_COMMIT = _expected_commit_from_argv(sys.argv[1:])
_BOOTSTRAP_COMMIT = _assert_exact_clean_head(_BOOTSTRAP_EXPECTED_COMMIT)
_BOOTSTRAP_WORKER_MODE = _worker_mode_from_argv(sys.argv[1:])
_BOOTSTRAP_WORKER_DIR = (
    _assert_worker_dir_outside_repo(_worker_dir_from_argv(sys.argv[1:]))
    if _BOOTSTRAP_WORKER_MODE
    else None
)
_BOOTSTRAP_WORKER_RESULT = (
    _assert_worker_result_in_dir(
        _worker_result_from_argv(sys.argv[1:]),
        _BOOTSTRAP_WORKER_DIR,
    )
    if _BOOTSTRAP_WORKER_MODE and _BOOTSTRAP_WORKER_DIR is not None
    else None
)
_BOOTSTRAP_OUTPUT = (
    None
    if _BOOTSTRAP_WORKER_MODE
    else _assert_output_outside_repo(
        _output_from_argv(
            sys.argv[1:],
            default="parameter-policy-execution-gpu-matrix.json",
        )
    )
)


import torch
from accelerate import Accelerator


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mikazuki import parameter_policy_execution as execution
from mikazuki.parameter_policy_trainer import (
    create_parameter_policy_session,
    make_legacy_scheduler_factory,
)
from tools.parameter_policy_execution_gpu_runtime import (
    gradient_evidence,
    model_parameter_evidence,
    optimizer_state_evidence,
    scheduler_state_evidence,
)
from tools.parameter_policy_execution_gpu_support import (
    ADAMW_FULL_BF16_CASE_IDS,
    ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
    EXECUTION_GPU_CASE_PHASES,
    ExecutionGpuMatrixError,
    MUON_FULL_BF16_ARGUMENT_FAMILY,
    MUON_FULL_BF16_CASE_IDS,
    MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
    summarize_adamw_full_bf16_bundle,
    summarize_muon_full_bf16_bundle,
    temporary_execution_qualification,
)


EVIDENCE_SCHEMA = "dts.parameter-policy.execution-gpu-matrix"
EVIDENCE_VERSION = 2
_CASES = tuple(EXECUTION_GPU_CASE_PHASES)
_ADAMW_CASES = ADAMW_FULL_BF16_CASE_IDS
_MUON_CASES = MUON_FULL_BF16_CASE_IDS
_PINNED_MUON_PROVIDER_VERSION = "3.10.0"
_STDIO_TAIL_LIMIT = 16000


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    versions = sorted(
        {
            line.strip()
            for line in output.splitlines()
            if line.strip()
        }
    )
    return ",".join(versions) if versions else None


def _cuda_environment() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ExecutionGpuMatrixError(
            "Parameter Policy execution GPU qualification requires a real CUDA device."
        )
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(is_bf16_supported) or not bool(is_bf16_supported()):
        raise ExecutionGpuMatrixError(
            "Parameter Policy execution GPU qualification requires CUDA BF16 support."
        )

    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
            }
        )

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": _nvidia_driver_version(),
        "bf16_supported": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {
            name: _package_version(name)
            for name in (
                "accelerate",
                "bitsandbytes",
                "lion-pytorch",
                "schedulefree",
                "pytorch-optimizer",
            )
        },
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
    }


def _qualification_snapshot() -> dict[str, dict[str, dict[str, Any]]]:
    def serialize(table):
        return {
            name: {
                "status": row.status,
                "reason": row.reason,
                "evidence_case_id": row.evidence_case_id,
            }
            for name, row in sorted(table.items())
        }

    return {
        "backends": serialize(execution.FULL_BF16_BACKEND_QUALIFICATIONS),
        "optimizers": serialize(execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS),
    }


class _TinyFlux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Linear(8, 8)
        with torch.no_grad():
            layer.weight.fill_(0.125)
            layer.bias.fill_(0.25)
        self.double_blocks = torch.nn.ModuleList([layer])

    def forward(self, value):
        return self.double_blocks[0](value)


def _initialize_muon_linear(
    layer: torch.nn.Linear,
    *,
    diagonal_scale: float,
    fill_scale: float,
) -> None:
    with torch.no_grad():
        value = torch.eye(
            layer.out_features,
            layer.in_features,
            dtype=layer.weight.dtype,
        )
        value.mul_(diagonal_scale)
        value.add_(torch.ones_like(value) * fill_scale)
        layer.weight.copy_(value)


class _TinyMuonFlux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        expand = torch.nn.Linear(8, 12, bias=False)
        contract = torch.nn.Linear(12, 8, bias=False)
        _initialize_muon_linear(
            expand,
            diagonal_scale=0.125,
            fill_scale=0.015625,
        )
        _initialize_muon_linear(
            contract,
            diagonal_scale=0.09375,
            fill_scale=0.0078125,
        )
        self.double_blocks = torch.nn.ModuleList([expand, contract])

    def forward(self, value):
        value = self.double_blocks[0](value)
        return self.double_blocks[1](value)


def _policy() -> dict[str, Any]:
    return {
        "version": 1,
        "optimizer_profiles": {
            "main": {"type": "AdamW", "args": {}},
        },
        "components": {
            "transformer.double_stream": {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 1e-2,
            },
        },
    }


def _muon_policy() -> dict[str, Any]:
    return {
        "version": 1,
        "optimizer_profiles": {
            "main": {
                "type": "Muon",
                "args": {
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                    "weight_decouple": True,
                    "nesterov": True,
                    "ns_steps": 5,
                    "ns_coeffs": "original",
                    "use_adjusted_lr": False,
                },
            },
        },
        "components": {
            "transformer.double_stream": {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 2e-2,
            },
        },
    }


def _scheduler_args(
    policy_path: Path,
    *,
    optimizer_type: str = "AdamW",
) -> SimpleNamespace:
    return SimpleNamespace(
        parameter_policy_config=str(policy_path),
        optimizer_type=optimizer_type,
        full_bf16=True,
        mixed_precision="bf16",
        lr_scheduler="constant",
        lr_scheduler_type="",
        lr_scheduler_args=None,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        max_train_steps=8,
    )


def _load_production_get_scheduler_fix():
    expected_module = (STABLE_SCRIPTS_ROOT / "library" / "train_util.py").resolve(
        strict=False
    )
    if not expected_module.is_file():
        raise ExecutionGpuMatrixError(
            "Production scheduler provider is unavailable at "
            f"{expected_module}."
        )

    stable_root = str(STABLE_SCRIPTS_ROOT.resolve(strict=False))
    if stable_root not in sys.path:
        sys.path.insert(0, stable_root)

    try:
        train_util = importlib.import_module("library.train_util")
    except Exception as exc:
        raise ExecutionGpuMatrixError(
            "Could not import production scheduler provider "
            "library.train_util.get_scheduler_fix."
        ) from exc

    module_file = getattr(train_util, "__file__", None)
    if module_file is None or Path(module_file).resolve(strict=False) != expected_module:
        raise ExecutionGpuMatrixError(
            "Production scheduler provider resolved from an unexpected module: "
            f"{module_file!r}; expected {expected_module}."
        )

    get_scheduler_fix = getattr(train_util, "get_scheduler_fix", None)
    if not callable(get_scheduler_fix):
        raise ExecutionGpuMatrixError(
            "Production scheduler provider does not expose callable "
            "library.train_util.get_scheduler_fix."
        )
    return get_scheduler_fix


def _scheduler_factory(args: object):
    return make_legacy_scheduler_factory(
        args=args,
        get_scheduler_fix=_load_production_get_scheduler_fix(),
        num_processes=1,
    )


def _assert_cuda_accelerator(accelerator: Accelerator) -> str:
    if accelerator.num_processes != 1:
        raise AssertionError(
            "Phase C shared execution qualification requires one process; "
            f"got {accelerator.num_processes}."
        )
    distributed_type = str(
        getattr(
            accelerator.distributed_type,
            "value",
            accelerator.distributed_type,
        )
    ).strip().upper()
    if distributed_type != "NO":
        raise AssertionError(
            "Phase C shared execution qualification requires "
            "Accelerate distributed_type=NO; "
            f"got {distributed_type!r}."
        )
    if torch.device(accelerator.device).type != "cuda":
        raise AssertionError(
            "Phase C shared execution qualification requires CUDA; "
            f"got accelerator.device={accelerator.device}."
        )
    return distributed_type


def _adamw_accumulation_steps(case_id: str) -> int:
    if case_id == "optimizer:adamw:full-bf16:accum1:v1":
        return 1
    if case_id == "optimizer:adamw:full-bf16:accum2:v1":
        return 2
    raise ExecutionGpuMatrixError(f"Unknown AdamW lifecycle case {case_id!r}.")


def _muon_accumulation_steps(case_id: str) -> int:
    if case_id == MUON_FULL_BF16_CASE_IDS[0]:
        return 1
    if case_id == MUON_FULL_BF16_CASE_IDS[1]:
        return 2
    raise ExecutionGpuMatrixError(f"Unknown Muon lifecycle case {case_id!r}.")


def _assert_pinned_muon_provider() -> dict[str, str]:
    version = _package_version("pytorch-optimizer")
    if version != _PINNED_MUON_PROVIDER_VERSION:
        raise ExecutionGpuMatrixError(
            "Muon full-BF16 qualification requires pinned pytorch-optimizer "
            f"{_PINNED_MUON_PROVIDER_VERSION}; found {version!r}."
        )
    try:
        import pytorch_optimizer
    except ImportError as exc:
        raise ExecutionGpuMatrixError(
            "Muon full-BF16 qualification could not import pytorch_optimizer."
        ) from exc

    muon_class = getattr(pytorch_optimizer, "Muon", None)
    if muon_class is None:
        raise ExecutionGpuMatrixError(
            "Pinned pytorch_optimizer does not expose Muon."
        )
    provider_class = (
        f"{muon_class.__module__}."
        f"{getattr(muon_class, '__qualname__', muon_class.__name__)}"
    )
    return {
        "package": "pytorch-optimizer",
        "version": version,
        "class": provider_class,
    }


def _muon_child(session):
    entries = tuple(session.optimizer.entries)
    if len(entries) != 1:
        raise AssertionError(
            f"Pure Muon evidence requires exactly one child optimizer; got {len(entries)}."
        )
    entry = entries[0]
    if entry.profile_name != "main" or entry.optimizer_type != "Muon":
        raise AssertionError(
            "Pure Muon evidence resolved unexpected optimizer entry "
            f"{entry.profile_name!r}/{entry.optimizer_type!r}."
        )
    return entry.optimizer


def _muon_routing_evidence(session) -> dict[str, Any]:
    rows = []
    for assignment in session.routing_plan.assignments:
        rows.append(
            {
                "canonical_name": assignment.canonical_name,
                "component_id": assignment.component_id,
                "route_kind": assignment.route_kind,
                "optimizer_profile": assignment.optimizer_profile,
                "parameter_class": assignment.parameter_class,
                "shape": [int(dim) for dim in assignment.parameter.shape],
            }
        )
    trainable = [
        row for row in rows if row["route_kind"] in {"primary", "fallback"}
    ]
    if not trainable:
        raise AssertionError("Pure Muon evidence produced no trainable routing assignments.")
    if any(row["route_kind"] != "primary" for row in trainable):
        raise AssertionError(
            "Pure Muon evidence unexpectedly used fallback routing."
        )
    if any(row["optimizer_profile"] != "main" for row in trainable):
        raise AssertionError(
            "Pure Muon evidence routed a trainable parameter outside the Muon profile."
        )
    if any(row["parameter_class"] != "matrix_weight" for row in trainable):
        raise AssertionError(
            "Pure Muon evidence routed a non-matrix trainable parameter."
        )

    shapes = [tuple(row["shape"]) for row in trainable]
    if len(trainable) != 2 or shapes != [(12, 8), (8, 12)]:
        raise AssertionError(
            "Pure Muon evidence must cover exactly the tall/wide reference shapes "
            f"[(12, 8), (8, 12)]; got {shapes!r}."
        )
    if not any(rows_count > cols_count for rows_count, cols_count in shapes):
        raise AssertionError("Pure Muon evidence does not cover transpose=True.")
    if not any(rows_count < cols_count for rows_count, cols_count in shapes):
        raise AssertionError("Pure Muon evidence does not cover transpose=False.")

    return {
        "assignments": rows,
        "trainable_count": len(trainable),
        "primary_count": sum(row["route_kind"] == "primary" for row in trainable),
        "fallback_count": sum(row["route_kind"] == "fallback" for row in trainable),
        "matrix_shapes": [list(shape) for shape in shapes],
        "covers_transpose_true": any(
            rows_count > cols_count for rows_count, cols_count in shapes
        ),
        "covers_transpose_false": any(
            rows_count < cols_count for rows_count, cols_count in shapes
        ),
    }


def _muon_group_step_counters(session) -> list[int]:
    child = _muon_child(session)
    counters: list[int] = []
    for group in child.param_groups:
        raw = group.get("step", 0)
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                raise AssertionError("Muon group step tensor must be scalar.")
            raw = raw.detach().cpu().item()
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise AssertionError(f"Muon group step is not numeric: {raw!r}.")
        number = int(raw)
        if float(raw) != float(number) or number < 0:
            raise AssertionError(f"Muon group step is invalid: {raw!r}.")
        counters.append(number)
    return counters


def _muon_path_evidence(session) -> dict[str, Any]:
    child = _muon_child(session)
    groups = []
    for index, group in enumerate(child.param_groups):
        use_muon = group.get("use_muon")
        if use_muon is not True:
            raise AssertionError(
                f"Muon group {index} is not explicitly use_muon=True."
            )
        groups.append(
            {
                "index": index,
                "use_muon": True,
                "step": _muon_group_step_counters(session)[index],
                "parameter_count": len(group["params"]),
            }
        )

    state_keys: list[list[str]] = []
    for parameter_state in child.state.values():
        keys = sorted(str(key) for key in parameter_state)
        if "exp_avg" in keys or "exp_avg_sq" in keys:
            raise AssertionError(
                "Pure Muon evidence entered the provider's internal AdamW state path."
            )
        if parameter_state and "momentum_buffer" not in keys:
            raise AssertionError(
                f"Muon parameter state lacks momentum_buffer: {keys!r}."
            )
        state_keys.append(keys)

    return {
        "child_class": (
            f"{child.__class__.__module__}."
            f"{child.__class__.__qualname__}"
        ),
        "groups": groups,
        "state_keys": state_keys,
        "state_parameter_count": len(state_keys),
        "internal_adamw_state_present": any(
            "exp_avg" in keys or "exp_avg_sq" in keys
            for keys in state_keys
        ),
    }


def _muon_child_state_is_empty(session) -> bool:
    child = _muon_child(session)
    return len(child.state) == 0 and all(
        int(group.get("step", 0)) == 0
        for group in child.param_groups
    )


def _assert_muon_gradient_evidence(gradients: dict[str, Any]) -> None:
    if not gradients:
        raise AssertionError("Muon step produced no gradient evidence.")
    for name, evidence in gradients.items():
        if evidence is None:
            raise AssertionError(f"Muon trainable parameter {name} has no gradient.")
        if evidence.get("finite") is not True:
            raise AssertionError(f"Muon trainable parameter {name} has non-finite gradient.")
        if evidence.get("nonzero") is not True:
            raise AssertionError(f"Muon trainable parameter {name} has zero gradient.")


def _muon_qualification_contract(
    session,
    provider: dict[str, str],
) -> dict[str, Any]:
    optimizers = tuple(session.runtime_spec.optimizers)
    if len(optimizers) != 1:
        raise AssertionError(
            "Muon qualification contract requires exactly one runtime optimizer spec."
        )
    spec = optimizers[0]
    if spec.profile_name != "main" or spec.optimizer_type != "Muon":
        raise AssertionError(
            "Muon qualification contract resolved unexpected runtime optimizer "
            f"{spec.profile_name!r}/{spec.optimizer_type!r}."
        )

    reference_arguments = spec.optimizer_arguments
    expected_family = tuple(sorted(MUON_FULL_BF16_ARGUMENT_FAMILY))
    actual_family = tuple(sorted(reference_arguments))
    if actual_family != expected_family:
        raise AssertionError(
            "Muon reference arguments must explicitly cover the full C3 argument family: "
            f"expected={expected_family!r}, actual={actual_family!r}."
        )

    return {
        "schema": "dts.parameter-policy.muon-full-bf16-qualification-family",
        "version": 1,
        "provider": dict(provider),
        "routing_family": "pure_muon",
        "state_family": "momentum_buffer",
        "argument_family": list(expected_family),
        "reference_arguments": reference_arguments,
    }


def _trainable_dtypes(session) -> list[str]:
    return sorted({str(parameter.dtype) for parameter in session.trainable_parameters})


def _adamw_child_state_is_empty(session) -> bool:
    payload = session.optimizer.state_dict()
    child = payload["children"]["main"]["state_dict"]
    return child["state"] == {}


def _adamw_step_counters(session) -> list[int]:
    payload = session.optimizer.state_dict()
    child_state = payload["children"]["main"]["state_dict"]["state"]
    counters: list[int] = []
    for parameter_state in child_state.values():
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            continue
        raw = parameter_state["step"]
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                raise AssertionError("AdamW step counter tensor must be scalar.")
            raw = raw.detach().cpu().item()
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise AssertionError(f"AdamW step counter is not numeric: {raw!r}.")
        number = int(raw)
        if float(raw) != float(number) or number < 0:
            raise AssertionError(f"AdamW step counter is invalid: {raw!r}.")
        counters.append(number)
    return sorted(counters)


def _scheduler_step_count(raw_scheduler: object) -> int:
    state = raw_scheduler.state_dict()
    value = state.get("step_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssertionError(f"Composite scheduler step_count is invalid: {value!r}.")
    return value


def _assert_tensor_evidence_finite(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        if "finite" in value and value.get("finite") is not True:
            raise AssertionError(f"{label} contains non-finite tensor evidence.")
        tensor = value.get("tensor")
        if isinstance(tensor, dict) and tensor.get("finite") is not True:
            raise AssertionError(f"{label} contains non-finite tensor evidence.")
        for item in value.values():
            _assert_tensor_evidence_finite(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_tensor_evidence_finite(item, label=label)


def _assert_gradient_evidence(gradients: dict[str, Any]) -> None:
    if not gradients:
        raise AssertionError("AdamW step produced no gradient evidence.")
    for name, evidence in gradients.items():
        if evidence is None:
            raise AssertionError(f"AdamW trainable parameter {name} has no gradient.")
        if evidence.get("finite") is not True:
            raise AssertionError(f"AdamW trainable parameter {name} has non-finite gradient.")
        if evidence.get("nonzero") is not True:
            raise AssertionError(f"AdamW trainable parameter {name} has zero gradient.")


def _run_logical_step(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: object,
    scheduler: object,
    raw_scheduler: object,
    session: object,
    accumulation_steps: int,
    logical_step: int,
) -> dict[str, Any]:
    before_model = model_parameter_evidence(accelerator.unwrap_model(model))
    before_optimizer = optimizer_state_evidence(session.optimizer)
    before_scheduler = scheduler_state_evidence(raw_scheduler)
    before_counters = _adamw_step_counters(session)
    before_scheduler_step = _scheduler_step_count(raw_scheduler)
    _assert_tensor_evidence_finite(before_model, label="AdamW model state")
    _assert_tensor_evidence_finite(before_optimizer, label="AdamW optimizer state")
    _assert_tensor_evidence_finite(before_scheduler, label="AdamW scheduler state")
    microsteps: list[dict[str, Any]] = []

    for microstep in range(accumulation_steps):
        model_before = model_parameter_evidence(accelerator.unwrap_model(model))
        optimizer_before = optimizer_state_evidence(session.optimizer)
        scheduler_before = scheduler_state_evidence(raw_scheduler)
        with accelerator.accumulate(model):
            value = torch.full(
                (2, 8),
                0.25 + 0.125 * (logical_step + microstep),
                device=accelerator.device,
                dtype=torch.bfloat16,
            )
            output = model(value)
            loss = output.float().square().mean()
            accelerator.backward(loss)
            sync_gradients = bool(accelerator.sync_gradients)
            gradients = gradient_evidence(list(session.trainable_parameters))
            _assert_gradient_evidence(gradients)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model_after = model_parameter_evidence(accelerator.unwrap_model(model))
        optimizer_after = optimizer_state_evidence(session.optimizer)
        scheduler_after = scheduler_state_evidence(raw_scheduler)
        counters_after = _adamw_step_counters(session)
        scheduler_step_after = _scheduler_step_count(raw_scheduler)
        _assert_tensor_evidence_finite(model_after, label="AdamW model state")
        _assert_tensor_evidence_finite(optimizer_after, label="AdamW optimizer state")
        _assert_tensor_evidence_finite(scheduler_after, label="AdamW scheduler state")
        microsteps.append(
            {
                "microstep": microstep + 1,
                "sync_gradients": sync_gradients,
                "loss": float(loss.detach().cpu()),
                "gradients": gradients,
                "model_changed": model_after != model_before,
                "optimizer_state_changed": optimizer_after["sha256"]
                != optimizer_before["sha256"],
                "scheduler_state_changed": scheduler_after["sha256"]
                != scheduler_before["sha256"],
                "adamw_step_counters": counters_after,
                "scheduler_step_count": scheduler_step_after,
            }
        )

        if accumulation_steps > 1 and microstep < accumulation_steps - 1:
            if sync_gradients:
                raise AssertionError("Intermediate accumulation microstep unexpectedly synced.")
            if model_after != model_before:
                raise AssertionError("Intermediate accumulation microstep changed parameters.")
            if optimizer_after["sha256"] != optimizer_before["sha256"]:
                raise AssertionError("Intermediate accumulation microstep changed optimizer state.")
            if scheduler_after["sha256"] != scheduler_before["sha256"]:
                raise AssertionError("Intermediate accumulation microstep advanced scheduler.")
            if counters_after != before_counters:
                raise AssertionError("Intermediate accumulation microstep advanced AdamW step counter.")
            if scheduler_step_after != before_scheduler_step:
                raise AssertionError("Intermediate accumulation microstep advanced scheduler step count.")

    after_model = model_parameter_evidence(accelerator.unwrap_model(model))
    after_optimizer = optimizer_state_evidence(session.optimizer)
    after_scheduler = scheduler_state_evidence(raw_scheduler)
    after_counters = _adamw_step_counters(session)
    after_scheduler_step = _scheduler_step_count(raw_scheduler)
    _assert_tensor_evidence_finite(after_model, label="AdamW model state")
    _assert_tensor_evidence_finite(after_optimizer, label="AdamW optimizer state")
    _assert_tensor_evidence_finite(after_scheduler, label="AdamW scheduler state")
    if before_model == after_model:
        raise AssertionError("AdamW logical step did not update model parameters.")
    if before_optimizer["sha256"] == after_optimizer["sha256"]:
        raise AssertionError("AdamW logical step did not update optimizer state.")
    if before_scheduler["sha256"] == after_scheduler["sha256"]:
        raise AssertionError("AdamW logical step did not advance scheduler.")
    if not microsteps[-1]["sync_gradients"]:
        raise AssertionError("Final accumulation microstep did not synchronize gradients.")
    if logical_step == 1:
        if before_counters:
            raise AssertionError("Fresh AdamW state unexpectedly has step counters.")
    elif before_counters and any(value != logical_step - 1 for value in before_counters):
        raise AssertionError(
            f"AdamW resumed step counters are not at logical step {logical_step - 1}: "
            f"{before_counters!r}."
        )
    if not after_counters or any(value != logical_step for value in after_counters):
        raise AssertionError(
            f"AdamW step counters did not reach logical step {logical_step}: "
            f"{after_counters!r}."
        )
    if after_scheduler_step != before_scheduler_step + 1:
        raise AssertionError(
            "Composite scheduler did not advance exactly one step for one logical step."
        )

    return {
        "logical_step": logical_step,
        "accumulation_steps": accumulation_steps,
        "before": {
            "model": before_model,
            "optimizer": before_optimizer,
            "scheduler": before_scheduler,
            "adamw_step_counters": before_counters,
            "scheduler_step_count": before_scheduler_step,
        },
        "after": {
            "model": after_model,
            "optimizer": after_optimizer,
            "scheduler": after_scheduler,
            "adamw_step_counters": after_counters,
            "scheduler_step_count": after_scheduler_step,
        },
        "microsteps": microsteps,
    }


def _run_muon_logical_step(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: object,
    scheduler: object,
    raw_scheduler: object,
    session: object,
    accumulation_steps: int,
    logical_step: int,
) -> dict[str, Any]:
    before_model = model_parameter_evidence(accelerator.unwrap_model(model))
    before_optimizer = optimizer_state_evidence(session.optimizer)
    before_scheduler = scheduler_state_evidence(raw_scheduler)
    before_counters = _muon_group_step_counters(session)
    before_scheduler_step = _scheduler_step_count(raw_scheduler)
    before_path = _muon_path_evidence(session)
    _assert_tensor_evidence_finite(before_model, label="Muon model state")
    _assert_tensor_evidence_finite(before_optimizer, label="Muon optimizer state")
    _assert_tensor_evidence_finite(before_scheduler, label="Muon scheduler state")
    microsteps: list[dict[str, Any]] = []

    for microstep in range(accumulation_steps):
        model_before = model_parameter_evidence(accelerator.unwrap_model(model))
        optimizer_before = optimizer_state_evidence(session.optimizer)
        scheduler_before = scheduler_state_evidence(raw_scheduler)
        counters_before = _muon_group_step_counters(session)
        with accelerator.accumulate(model):
            scale = 0.5 + 0.125 * (logical_step + microstep)
            value = torch.eye(
                8,
                device=accelerator.device,
                dtype=torch.bfloat16,
            )
            value.mul_(scale)
            output = model(value)
            loss = output.float().square().mean()
            accelerator.backward(loss)
            sync_gradients = bool(accelerator.sync_gradients)
            gradients = gradient_evidence(list(session.trainable_parameters))
            _assert_muon_gradient_evidence(gradients)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model_after = model_parameter_evidence(accelerator.unwrap_model(model))
        optimizer_after = optimizer_state_evidence(session.optimizer)
        scheduler_after = scheduler_state_evidence(raw_scheduler)
        counters_after = _muon_group_step_counters(session)
        scheduler_step_after = _scheduler_step_count(raw_scheduler)
        path_after = _muon_path_evidence(session)
        _assert_tensor_evidence_finite(model_after, label="Muon model state")
        _assert_tensor_evidence_finite(optimizer_after, label="Muon optimizer state")
        _assert_tensor_evidence_finite(scheduler_after, label="Muon scheduler state")
        microsteps.append(
            {
                "microstep": microstep + 1,
                "sync_gradients": sync_gradients,
                "loss": float(loss.detach().cpu()),
                "gradients": gradients,
                "model_changed": model_after != model_before,
                "optimizer_state_changed": optimizer_after["sha256"]
                != optimizer_before["sha256"],
                "scheduler_state_changed": scheduler_after["sha256"]
                != scheduler_before["sha256"],
                "muon_group_step_counters": counters_after,
                "scheduler_step_count": scheduler_step_after,
                "muon_path": path_after,
            }
        )

        if accumulation_steps > 1 and microstep < accumulation_steps - 1:
            if sync_gradients:
                raise AssertionError("Intermediate Muon accumulation microstep unexpectedly synced.")
            if model_after != model_before:
                raise AssertionError("Intermediate Muon accumulation microstep changed parameters.")
            if optimizer_after["sha256"] != optimizer_before["sha256"]:
                raise AssertionError("Intermediate Muon accumulation microstep changed optimizer state.")
            if scheduler_after["sha256"] != scheduler_before["sha256"]:
                raise AssertionError("Intermediate Muon accumulation microstep advanced scheduler.")
            if counters_after != counters_before:
                raise AssertionError("Intermediate Muon accumulation microstep advanced group step.")
            if scheduler_step_after != before_scheduler_step:
                raise AssertionError("Intermediate Muon accumulation microstep advanced scheduler step count.")
            if any(keys for keys in path_after["state_keys"]):
                raise AssertionError("Intermediate Muon accumulation microstep created optimizer state.")

    after_model = model_parameter_evidence(accelerator.unwrap_model(model))
    after_optimizer = optimizer_state_evidence(session.optimizer)
    after_scheduler = scheduler_state_evidence(raw_scheduler)
    after_counters = _muon_group_step_counters(session)
    after_scheduler_step = _scheduler_step_count(raw_scheduler)
    after_path = _muon_path_evidence(session)
    _assert_tensor_evidence_finite(after_model, label="Muon model state")
    _assert_tensor_evidence_finite(after_optimizer, label="Muon optimizer state")
    _assert_tensor_evidence_finite(after_scheduler, label="Muon scheduler state")

    if before_model == after_model:
        raise AssertionError("Muon logical step did not update model parameters.")
    if before_optimizer["sha256"] == after_optimizer["sha256"]:
        raise AssertionError("Muon logical step did not update optimizer state.")
    if before_scheduler["sha256"] == after_scheduler["sha256"]:
        raise AssertionError("Muon logical step did not advance scheduler.")
    if not microsteps[-1]["sync_gradients"]:
        raise AssertionError("Final Muon accumulation microstep did not synchronize gradients.")
    if logical_step == 1:
        if any(value != 0 for value in before_counters):
            raise AssertionError(
                f"Fresh Muon group steps are not zero: {before_counters!r}."
            )
    elif any(value != logical_step - 1 for value in before_counters):
        raise AssertionError(
            f"Muon resumed group steps are not at logical step {logical_step - 1}: "
            f"{before_counters!r}."
        )
    if not after_counters or any(value != logical_step for value in after_counters):
        raise AssertionError(
            f"Muon group steps did not reach logical step {logical_step}: "
            f"{after_counters!r}."
        )
    if after_scheduler_step != before_scheduler_step + 1:
        raise AssertionError(
            "Composite scheduler did not advance exactly one step for one Muon logical step."
        )
    if after_path["internal_adamw_state_present"]:
        raise AssertionError("Muon logical step entered internal AdamW state path.")
    expected_state_count = sum(
        group["parameter_count"] for group in after_path["groups"]
    )
    if after_path["state_parameter_count"] != expected_state_count:
        raise AssertionError(
            "Muon logical step did not create state for every routed parameter: "
            f"expected={expected_state_count}, "
            f"actual={after_path['state_parameter_count']}."
        )
    if not after_path["state_keys"] or any(
        "momentum_buffer" not in keys for keys in after_path["state_keys"]
    ):
        raise AssertionError("Muon logical step did not create momentum_buffer state.")

    return {
        "logical_step": logical_step,
        "accumulation_steps": accumulation_steps,
        "before": {
            "model": before_model,
            "optimizer": before_optimizer,
            "scheduler": before_scheduler,
            "muon_group_step_counters": before_counters,
            "scheduler_step_count": before_scheduler_step,
            "muon_path": before_path,
        },
        "after": {
            "model": after_model,
            "optimizer": after_optimizer,
            "scheduler": after_scheduler,
            "muon_group_step_counters": after_counters,
            "scheduler_step_count": after_scheduler_step,
            "muon_path": after_path,
        },
        "microsteps": microsteps,
    }


def _build_muon_runtime(case_dir: Path, *, accumulation_steps: int):
    provider = _assert_pinned_muon_provider()
    policy_path = case_dir / "muon-policy.json"
    policy_path.write_text(
        json.dumps(_muon_policy(), sort_keys=True),
        encoding="utf-8",
    )
    args = _scheduler_args(policy_path, optimizer_type="Muon")
    model = _TinyMuonFlux()
    session = create_parameter_policy_session(
        args=args,
        train_type="flux-finetune",
        roots={"transformer": model},
    )
    routing = _muon_routing_evidence(session)
    if routing["primary_count"] != 2 or routing["fallback_count"] != 0:
        raise AssertionError(
            "Pure Muon runtime must contain exactly two primary assignments and no fallback."
        )
    qualification_contract = _muon_qualification_contract(session, provider)
    raw_scheduler = session.build_scheduler(_scheduler_factory(args))
    if session.execution_contract is None:
        raise AssertionError("Muon full-BF16 session has no execution contract.")
    path = _muon_path_evidence(session)
    if path["internal_adamw_state_present"]:
        raise AssertionError("Fresh Muon runtime already exposes internal AdamW state.")
    if path["child_class"] != provider["class"]:
        raise AssertionError(
            "Muon runtime resolved a different provider class: "
            f"{path['child_class']!r} != {provider['class']!r}."
        )

    model.to(torch.bfloat16)
    if _trainable_dtypes(session) != ["torch.bfloat16"]:
        raise AssertionError("Muon trainable parameters are not true BF16 before prepare.")

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=accumulation_steps,
    )
    distributed_type = _assert_cuda_accelerator(accelerator)
    model, optimizer, scheduler = accelerator.prepare(
        model,
        session.optimizer,
        raw_scheduler,
    )
    session.finalize_after_prepare(
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if _trainable_dtypes(session) != ["torch.bfloat16"]:
        raise AssertionError("Muon trainable parameters are not true BF16 after prepare.")
    return (
        args,
        model,
        optimizer,
        scheduler,
        raw_scheduler,
        session,
        accelerator,
        distributed_type,
        provider,
        routing,
        qualification_contract,
    )


def _muon_handoff_path(case_dir: Path) -> Path:
    return case_dir / "muon-handoff.json"


def _muon_checkpoint_dir(case_dir: Path) -> Path:
    return case_dir / "muon-checkpoint"


def _muon_train_save(case_id: str, case_dir: Path) -> dict[str, Any]:
    accumulation_steps = _muon_accumulation_steps(case_id)
    with temporary_execution_qualification(
        backend="flux-finetune",
        optimizers=("Muon",),
        evidence_case_id=MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
    ) as lease:
        (
            _args,
            model,
            optimizer,
            scheduler,
            raw_scheduler,
            session,
            accelerator,
            distributed_type,
            provider,
            routing,
            qualification_contract,
        ) = _build_muon_runtime(case_dir, accumulation_steps=accumulation_steps)
        try:
            if not _muon_child_state_is_empty(session):
                raise AssertionError("Fresh Muon optimizer state must start empty.")
            step = _run_muon_logical_step(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                raw_scheduler=raw_scheduler,
                session=session,
                accumulation_steps=accumulation_steps,
                logical_step=1,
            )
            checkpoint_dir = _muon_checkpoint_dir(case_dir)
            accelerator.save_state(checkpoint_dir)
            saved = {
                "schema": "dts.parameter-policy.muon-full-bf16-handoff",
                "version": 1,
                "case_id": case_id,
                "evidence_bundle_id": MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
                "commit": _BOOTSTRAP_COMMIT,
                "accumulation_steps": accumulation_steps,
                "provider": provider,
                "routing": routing,
                "qualification_contract": qualification_contract,
                "execution_identity": session.execution_contract.execution_identity(),
                "execution_signature": session.execution_contract.execution_signature(),
                "model": model_parameter_evidence(accelerator.unwrap_model(model)),
                "optimizer": optimizer_state_evidence(session.optimizer),
                "scheduler": scheduler_state_evidence(raw_scheduler),
                "muon_path": _muon_path_evidence(session),
            }
            _write_json(_muon_handoff_path(case_dir), saved)
            return {
                "scope": "optimizer",
                "qualification_target": {
                    "feature": "full_bf16",
                    "kind": "optimizer",
                    "name": "Muon",
                },
                "evidence_bundle_id": MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
                "promotion_eligible": False,
                "optimizer_qualification_eligible": False,
                "qualification_evidence_component": True,
                "backend_scaffold": "flux-finetune",
                "backend_qualification_eligible": False,
                "qualification_lease": lease,
                "provider": provider,
                "routing": routing,
                "qualification_contract": qualification_contract,
                "accelerator": {
                    "device": str(accelerator.device),
                    "num_processes": int(accelerator.num_processes),
                    "distributed_type": distributed_type,
                },
                "trainable_dtypes": _trainable_dtypes(session),
                "step": step,
                "handoff": saved,
            }
        finally:
            accelerator.end_training()


def _muon_resume_second_step(case_id: str, case_dir: Path) -> dict[str, Any]:
    handoff_path = _muon_handoff_path(case_dir)
    if not handoff_path.is_file():
        raise ExecutionGpuMatrixError("Muon resume phase is missing train/save handoff.")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    accumulation_steps = _muon_accumulation_steps(case_id)
    expected = {
        "schema": "dts.parameter-policy.muon-full-bf16-handoff",
        "version": 1,
        "case_id": case_id,
        "evidence_bundle_id": MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
        "commit": _BOOTSTRAP_COMMIT,
        "accumulation_steps": accumulation_steps,
    }
    for key, value in expected.items():
        if handoff.get(key) != value:
            raise ExecutionGpuMatrixError(
                f"Muon resume handoff mismatch for {key}: "
                f"expected={value!r}, actual={handoff.get(key)!r}."
            )

    with temporary_execution_qualification(
        backend="flux-finetune",
        optimizers=("Muon",),
        evidence_case_id=MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
    ) as lease:
        (
            _args,
            model,
            optimizer,
            scheduler,
            raw_scheduler,
            session,
            accelerator,
            distributed_type,
            provider,
            routing,
            qualification_contract,
        ) = _build_muon_runtime(case_dir, accumulation_steps=accumulation_steps)
        try:
            if provider != handoff.get("provider"):
                raise AssertionError("Muon provider identity changed across fresh resume.")
            if routing != handoff.get("routing"):
                raise AssertionError("Muon routing identity changed across fresh resume.")
            if qualification_contract != handoff.get("qualification_contract"):
                raise AssertionError(
                    "Muon qualification-family contract changed across fresh resume."
                )
            if not _muon_child_state_is_empty(session):
                raise AssertionError("Fresh resume Muon optimizer state must start empty.")

            accelerator.load_state(_muon_checkpoint_dir(case_dir))
            session.assert_runtime_contract(
                phase="post_resume",
                accelerator=accelerator,
                optimizer=optimizer,
            )
            restored_model = model_parameter_evidence(accelerator.unwrap_model(model))
            restored_optimizer = optimizer_state_evidence(session.optimizer)
            restored_scheduler = scheduler_state_evidence(raw_scheduler)
            restored_path = _muon_path_evidence(session)

            if restored_model != handoff.get("model"):
                raise AssertionError("Muon resumed model evidence does not match saved state.")
            if restored_optimizer["sha256"] != handoff["optimizer"]["sha256"]:
                raise AssertionError("Muon resumed optimizer state does not match saved state.")
            if restored_scheduler["sha256"] != handoff["scheduler"]["sha256"]:
                raise AssertionError("Muon resumed scheduler state does not match saved state.")
            if restored_path != handoff.get("muon_path"):
                raise AssertionError("Muon resumed path/state evidence changed.")
            if session.execution_contract.execution_identity() != handoff.get("execution_identity"):
                raise AssertionError("Muon resumed execution identity changed.")
            if session.execution_contract.execution_signature() != handoff.get("execution_signature"):
                raise AssertionError("Muon resumed execution signature changed.")
            if any(value != 1 for value in _muon_group_step_counters(session)):
                raise AssertionError(
                    "Muon resumed group steps did not restore logical step 1."
                )

            step = _run_muon_logical_step(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                raw_scheduler=raw_scheduler,
                session=session,
                accumulation_steps=accumulation_steps,
                logical_step=2,
            )
            return {
                "scope": "optimizer",
                "qualification_target": {
                    "feature": "full_bf16",
                    "kind": "optimizer",
                    "name": "Muon",
                },
                "evidence_bundle_id": MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
                "promotion_eligible": False,
                "optimizer_qualification_eligible": False,
                "qualification_evidence_component": True,
                "backend_scaffold": "flux-finetune",
                "backend_qualification_eligible": False,
                "qualification_lease": lease,
                "provider": provider,
                "routing": routing,
                "qualification_contract": qualification_contract,
                "accelerator": {
                    "device": str(accelerator.device),
                    "num_processes": int(accelerator.num_processes),
                    "distributed_type": distributed_type,
                },
                "restored": {
                    "model": restored_model,
                    "optimizer": restored_optimizer,
                    "scheduler": restored_scheduler,
                    "muon_path": restored_path,
                },
                "step": step,
            }
        finally:
            accelerator.end_training()


def _build_adamw_runtime(case_dir: Path, *, accumulation_steps: int):
    policy_path = case_dir / "policy.json"
    policy_path.write_text(json.dumps(_policy(), sort_keys=True), encoding="utf-8")
    args = _scheduler_args(policy_path)
    model = _TinyFlux()
    session = create_parameter_policy_session(
        args=args,
        train_type="flux-finetune",
        roots={"transformer": model},
    )
    raw_scheduler = session.build_scheduler(_scheduler_factory(args))
    if session.execution_contract is None:
        raise AssertionError("AdamW full-BF16 session has no execution contract.")
    model.to(torch.bfloat16)
    if _trainable_dtypes(session) != ["torch.bfloat16"]:
        raise AssertionError("AdamW trainable parameters are not true BF16 before prepare.")

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=accumulation_steps,
    )
    distributed_type = _assert_cuda_accelerator(accelerator)
    model, optimizer, scheduler = accelerator.prepare(
        model,
        session.optimizer,
        raw_scheduler,
    )
    session.finalize_after_prepare(
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if _trainable_dtypes(session) != ["torch.bfloat16"]:
        raise AssertionError("AdamW trainable parameters are not true BF16 after prepare.")
    return (
        args,
        model,
        optimizer,
        scheduler,
        raw_scheduler,
        session,
        accelerator,
        distributed_type,
    )


def _adamw_handoff_path(case_dir: Path) -> Path:
    return case_dir / "adamw-handoff.json"


def _adamw_checkpoint_dir(case_dir: Path) -> Path:
    return case_dir / "checkpoint"


def _adamw_train_save(case_id: str, case_dir: Path) -> dict[str, Any]:
    accumulation_steps = _adamw_accumulation_steps(case_id)
    with temporary_execution_qualification(
        backend="flux-finetune",
        optimizers=("AdamW",),
        evidence_case_id=ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
    ) as lease:
        (
            _args,
            model,
            optimizer,
            scheduler,
            raw_scheduler,
            session,
            accelerator,
            distributed_type,
        ) = _build_adamw_runtime(case_dir, accumulation_steps=accumulation_steps)
        try:
            if not _adamw_child_state_is_empty(session):
                raise AssertionError("Fresh AdamW optimizer state must start empty.")
            step = _run_logical_step(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                raw_scheduler=raw_scheduler,
                session=session,
                accumulation_steps=accumulation_steps,
                logical_step=1,
            )
            checkpoint_dir = _adamw_checkpoint_dir(case_dir)
            accelerator.save_state(checkpoint_dir)
            saved = {
                "schema": "dts.parameter-policy.adamw-full-bf16-handoff",
                "version": 1,
                "case_id": case_id,
                "evidence_bundle_id": ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
                "commit": _BOOTSTRAP_COMMIT,
                "accumulation_steps": accumulation_steps,
                "execution_identity": session.execution_contract.execution_identity(),
                "execution_signature": session.execution_contract.execution_signature(),
                "model": model_parameter_evidence(accelerator.unwrap_model(model)),
                "optimizer": optimizer_state_evidence(session.optimizer),
                "scheduler": scheduler_state_evidence(raw_scheduler),
            }
            _write_json(_adamw_handoff_path(case_dir), saved)
            return {
                "scope": "optimizer",
                "qualification_target": {
                    "feature": "full_bf16",
                    "kind": "optimizer",
                    "name": "AdamW",
                },
                "evidence_bundle_id": ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
                "promotion_eligible": False,
                "optimizer_qualification_eligible": False,
                "qualification_evidence_component": True,
                "backend_scaffold": "flux-finetune",
                "backend_qualification_eligible": False,
                "qualification_lease": lease,
                "accelerator": {
                    "device": str(accelerator.device),
                    "num_processes": int(accelerator.num_processes),
                    "distributed_type": distributed_type,
                },
                "trainable_dtypes": _trainable_dtypes(session),
                "step": step,
                "handoff": saved,
            }
        finally:
            accelerator.end_training()


def _adamw_resume_second_step(case_id: str, case_dir: Path) -> dict[str, Any]:
    handoff_path = _adamw_handoff_path(case_dir)
    if not handoff_path.is_file():
        raise ExecutionGpuMatrixError("AdamW resume phase is missing train/save handoff.")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    accumulation_steps = _adamw_accumulation_steps(case_id)
    expected = {
        "schema": "dts.parameter-policy.adamw-full-bf16-handoff",
        "version": 1,
        "case_id": case_id,
        "evidence_bundle_id": ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
        "commit": _BOOTSTRAP_COMMIT,
        "accumulation_steps": accumulation_steps,
    }
    for key, value in expected.items():
        if handoff.get(key) != value:
            raise ExecutionGpuMatrixError(
                f"AdamW resume handoff mismatch for {key}: "
                f"expected={value!r}, actual={handoff.get(key)!r}."
            )

    with temporary_execution_qualification(
        backend="flux-finetune",
        optimizers=("AdamW",),
        evidence_case_id=ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
    ) as lease:
        (
            _args,
            model,
            optimizer,
            scheduler,
            raw_scheduler,
            session,
            accelerator,
            distributed_type,
        ) = _build_adamw_runtime(case_dir, accumulation_steps=accumulation_steps)
        try:
            if not _adamw_child_state_is_empty(session):
                raise AssertionError("Fresh resume AdamW optimizer state must start empty.")
            accelerator.load_state(_adamw_checkpoint_dir(case_dir))
            session.assert_runtime_contract(
                phase="post_resume",
                accelerator=accelerator,
                optimizer=optimizer,
            )
            restored_model = model_parameter_evidence(accelerator.unwrap_model(model))
            restored_optimizer = optimizer_state_evidence(session.optimizer)
            restored_scheduler = scheduler_state_evidence(raw_scheduler)
            if restored_model != handoff.get("model"):
                raise AssertionError("AdamW resumed model evidence does not match saved state.")
            if restored_optimizer["sha256"] != handoff["optimizer"]["sha256"]:
                raise AssertionError("AdamW resumed optimizer state does not match saved state.")
            if restored_scheduler["sha256"] != handoff["scheduler"]["sha256"]:
                raise AssertionError("AdamW resumed scheduler state does not match saved state.")
            if session.execution_contract.execution_identity() != handoff.get("execution_identity"):
                raise AssertionError("AdamW resumed execution identity changed.")
            if session.execution_contract.execution_signature() != handoff.get("execution_signature"):
                raise AssertionError("AdamW resumed execution signature changed.")

            step = _run_logical_step(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                raw_scheduler=raw_scheduler,
                session=session,
                accumulation_steps=accumulation_steps,
                logical_step=2,
            )
            return {
                "scope": "optimizer",
                "qualification_target": {
                    "feature": "full_bf16",
                    "kind": "optimizer",
                    "name": "AdamW",
                },
                "evidence_bundle_id": ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
                "promotion_eligible": False,
                "optimizer_qualification_eligible": False,
                "qualification_evidence_component": True,
                "backend_scaffold": "flux-finetune",
                "backend_qualification_eligible": False,
                "qualification_lease": lease,
                "accelerator": {
                    "device": str(accelerator.device),
                    "num_processes": int(accelerator.num_processes),
                    "distributed_type": distributed_type,
                },
                "restored": {
                    "model": restored_model,
                    "optimizer": restored_optimizer,
                    "scheduler": restored_scheduler,
                },
                "step": step,
            }
        finally:
            accelerator.end_training()


def _case_cuda_bf16_capability() -> dict[str, Any]:
    environment = _cuda_environment()
    return {
        "scope": "infrastructure",
        "promotion_eligible": False,
        "backend_qualification_eligible": False,
        "environment": environment,
    }


def _case_full_bf16_session() -> dict[str, Any]:
    environment = _cuda_environment()
    if environment["gpu_count"] < 1:
        raise AssertionError("CUDA environment probe returned no GPUs.")

    with tempfile.TemporaryDirectory() as temp_dir:
        policy_path = Path(temp_dir) / "policy.json"
        policy_path.write_text(
            json.dumps(_policy(), sort_keys=True),
            encoding="utf-8",
        )
        args = SimpleNamespace(
            parameter_policy_config=str(policy_path),
            full_bf16=True,
            mixed_precision="bf16",
        )
        model = _TinyFlux()

        with temporary_execution_qualification(
            backend="flux-finetune",
            optimizers=("AdamW",),
            evidence_case_id="phase-c:infra-lease:v1",
        ) as lease:
            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
            )
            if session.execution_contract is None:
                raise AssertionError("full-BF16 execution contract was not created.")
            if session.execution_contract.features != ("full_bf16",):
                raise AssertionError(
                    f"unexpected execution features {session.execution_contract.features!r}"
                )

            model.to(torch.bfloat16)
            trainable_dtypes_before_prepare = sorted(
                {str(parameter.dtype) for parameter in session.trainable_parameters}
            )
            if trainable_dtypes_before_prepare != ["torch.bfloat16"]:
                raise AssertionError(
                    "trainable parameters were not true BF16 before prepare: "
                    f"{trainable_dtypes_before_prepare!r}"
                )

            accelerator = Accelerator(mixed_precision="bf16")
            try:
                distributed_type = _assert_cuda_accelerator(accelerator)
                model, optimizer = accelerator.prepare(model, session.optimizer)
                session.finalize_after_prepare(
                    accelerator=accelerator,
                    optimizer=optimizer,
                    scheduler=None,
                )
                unwrapped = accelerator.unwrap_model(model)
                trainable_dtypes_after_prepare = sorted(
                    {str(parameter.dtype) for parameter in session.trainable_parameters}
                )
                if trainable_dtypes_after_prepare != ["torch.bfloat16"]:
                    raise AssertionError(
                        "trainable parameters were not true BF16 after prepare: "
                        f"{trainable_dtypes_after_prepare!r}"
                    )

                manifest = session.checkpoint_manifest()
                identity = session.execution_contract.execution_identity()
                signature = session.execution_contract.execution_signature()
                if manifest.get("execution_identity") != identity:
                    raise AssertionError(
                        "checkpoint manifest execution identity does not match contract."
                    )
                if manifest.get("execution_signature") != signature:
                    raise AssertionError(
                        "checkpoint manifest execution signature does not match contract."
                    )

                return {
                    "scope": "infrastructure",
                    "promotion_eligible": False,
                    "backend_scaffold": "flux-finetune",
                    "backend_qualification_eligible": False,
                    "optimizer_scaffold": "AdamW",
                    "optimizer_qualification_eligible": False,
                    "qualification_lease": lease,
                    "accelerator": {
                        "device": str(accelerator.device),
                        "num_processes": int(accelerator.num_processes),
                        "distributed_type": distributed_type,
                    },
                    "b2_post_prepare_identity_audit": "pass",
                    "execution_identity": identity,
                    "execution_signature": signature,
                    "trainable_dtypes_before_prepare": trainable_dtypes_before_prepare,
                    "trainable_dtypes_after_prepare": trainable_dtypes_after_prepare,
                    "unwrapped_model_type": type(unwrapped).__name__,
                }
            finally:
                accelerator.end_training()


def _run_worker_case(
    case_id: str,
    *,
    phase: str,
    case_dir: Path,
) -> dict[str, Any]:
    if case_id == "infra:cuda-bf16-capability:v1" and phase == "probe":
        return _case_cuda_bf16_capability()
    if case_id == "infra:full-bf16-session:v1" and phase == "probe":
        return _case_full_bf16_session()
    if case_id in _ADAMW_CASES:
        if phase == "train_save":
            return _adamw_train_save(case_id, case_dir)
        if phase == "resume_second_step":
            return _adamw_resume_second_step(case_id, case_dir)
    if case_id in _MUON_CASES:
        if phase == "train_save":
            return _muon_train_save(case_id, case_dir)
        if phase == "resume_second_step":
            return _muon_resume_second_step(case_id, case_dir)
    raise ExecutionGpuMatrixError(
        f"Unknown execution GPU case/phase {case_id!r}/{phase!r}."
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _worker_main(args: argparse.Namespace) -> int:
    if _BOOTSTRAP_WORKER_RESULT is None or _BOOTSTRAP_WORKER_DIR is None:
        raise ExecutionGpuMatrixBootstrapError(
            "Worker execution is missing validated bootstrap paths."
        )
    result_path = _BOOTSTRAP_WORKER_RESULT
    case_dir = _BOOTSTRAP_WORKER_DIR
    started = time.time()
    payload: dict[str, Any] = {
        "case_id": args.worker_case,
        "phase": args.worker_phase,
        "commit": None,
        "status": "fail",
    }
    try:
        payload["commit"] = _assert_exact_clean_head(args.expected_commit)
        case_dir.mkdir(parents=True, exist_ok=True)
        payload["details"] = _run_worker_case(
            args.worker_case,
            phase=args.worker_phase,
            case_dir=case_dir,
        )
        payload["status"] = "pass"
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    payload["duration_seconds"] = round(time.time() - started, 4)
    _write_json(result_path, payload)
    return 0 if payload["status"] == "pass" else 1


def _subprocess_tail(value: str) -> str:
    return value[-_STDIO_TAIL_LIMIT:]


def _run_phase_subprocess(
    *,
    case_id: str,
    phase: str,
    expected_commit: str,
    case_dir: Path,
) -> dict[str, Any]:
    result_path = case_dir / f"{phase}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--expected-commit",
        expected_commit,
        "--worker-case",
        case_id,
        "--worker-phase",
        phase,
        "--worker-result",
        str(result_path),
        "--worker-dir",
        str(case_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )

    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result = {
                "case_id": case_id,
                "phase": phase,
                "status": "fail",
                "error": f"Invalid worker result: {exc}",
            }
    else:
        result = {
            "case_id": case_id,
            "phase": phase,
            "status": "fail",
            "error": "Worker did not produce an evidence result.",
        }

    result["worker_returncode"] = completed.returncode
    result["worker_stdout_tail"] = _subprocess_tail(completed.stdout)
    result["worker_stderr_tail"] = _subprocess_tail(completed.stderr)
    if completed.returncode != 0 and result.get("status") == "pass":
        result["status"] = "fail"
        result["error"] = (
            "Worker returned non-zero despite reporting pass: "
            f"{completed.returncode}."
        )
    return result


def _coordinator_main(args: argparse.Namespace) -> int:
    commit = _assert_exact_clean_head(args.expected_commit)
    if _BOOTSTRAP_OUTPUT is None:
        raise ExecutionGpuMatrixError(
            "Coordinator execution is missing its validated external output path."
        )
    output_path = _BOOTSTRAP_OUTPUT
    environment = _cuda_environment()

    selected = tuple(args.case) if args.case else _CASES
    if len(set(selected)) != len(selected):
        raise ExecutionGpuMatrixError(
            "Execution GPU qualification case selection contains duplicates."
        )
    unknown = sorted(set(selected).difference(_CASES))
    if unknown:
        raise ExecutionGpuMatrixError(
            "Unknown execution GPU case(s): " + ", ".join(unknown)
        )

    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "version": EVIDENCE_VERSION,
        "commit": commit,
        "expected_commit": args.expected_commit,
        "environment": environment,
        "qualification_snapshot": _qualification_snapshot(),
        "cases": [],
    }

    failed = False
    with tempfile.TemporaryDirectory(prefix="dts-execution-gpu-") as temp_dir:
        work_dir = Path(temp_dir)
        for case_id in selected:
            case_started = time.time()
            safe_case = case_id.replace(":", "_").replace("/", "_")
            case_dir = work_dir / safe_case
            case_dir.mkdir(parents=True, exist_ok=True)
            phases: list[dict[str, Any]] = []
            dependency_failed = False
            for phase in EXECUTION_GPU_CASE_PHASES[case_id]:
                if dependency_failed:
                    phase_row = {
                        "case_id": case_id,
                        "phase": phase,
                        "status": "not_run",
                        "reason": "dependency_failed",
                    }
                else:
                    phase_row = _run_phase_subprocess(
                        case_id=case_id,
                        phase=phase,
                        expected_commit=args.expected_commit,
                        case_dir=case_dir,
                    )
                    if phase_row.get("status") != "pass":
                        dependency_failed = True
                phases.append(phase_row)

            case_status = (
                "pass"
                if phases and all(row.get("status") == "pass" for row in phases)
                else "fail"
            )
            row = {
                "case_id": case_id,
                "status": case_status,
                "phases": phases,
                "coordinator_duration_seconds": round(
                    time.time() - case_started,
                    4,
                ),
            }
            if case_id in _ADAMW_CASES:
                row.update(
                    {
                        "scope": "optimizer",
                        "qualification_target": {
                            "feature": "full_bf16",
                            "kind": "optimizer",
                            "name": "AdamW",
                        },
                        "evidence_bundle_id": ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
                        "qualification_evidence_component": True,
                        "optimizer_qualification_eligible": False,
                        "backend_qualification_eligible": False,
                    }
                )
            if case_id in _MUON_CASES:
                row.update(
                    {
                        "scope": "optimizer",
                        "qualification_target": {
                            "feature": "full_bf16",
                            "kind": "optimizer",
                            "name": "Muon",
                        },
                        "evidence_bundle_id": MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
                        "qualification_evidence_component": True,
                        "optimizer_qualification_eligible": False,
                        "backend_qualification_eligible": False,
                    }
                )
            evidence["cases"].append(row)
            print(f"{case_status.upper():4} {case_id}", flush=True)
            if case_status != "pass":
                failed = True

    evidence_bundles: dict[str, Any] = {}

    adamw_bundle = summarize_adamw_full_bf16_bundle(evidence["cases"])
    if adamw_bundle is not None:
        evidence_bundles[ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID] = adamw_bundle
        if adamw_bundle["status"] != "pass":
            failed = True

    muon_bundle = summarize_muon_full_bf16_bundle(evidence["cases"])
    if muon_bundle is not None:
        evidence_bundles[MUON_FULL_BF16_EVIDENCE_BUNDLE_ID] = muon_bundle
        if muon_bundle["status"] != "pass":
            failed = True

    if evidence_bundles:
        evidence["evidence_bundles"] = evidence_bundles

    _write_json(output_path, evidence)
    print(f"wrote {output_path}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--output",
        default="parameter-policy-execution-gpu-matrix.json",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--worker-case", default="")
    parser.add_argument("--worker-phase", default="")
    parser.add_argument("--worker-result", default="")
    parser.add_argument("--worker-dir", default="")
    args = parser.parse_args()

    worker_mode = bool(
        args.worker_case or args.worker_phase or args.worker_result or args.worker_dir
    )
    if worker_mode:
        if not (
            args.worker_case
            and args.worker_phase
            and args.worker_result
            and args.worker_dir
        ):
            raise SystemExit(
                "Worker mode requires --worker-case, --worker-phase, "
                "--worker-result, and --worker-dir."
            )
        return _worker_main(args)
    return _coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
