#!/usr/bin/env python3
"""Exact-head CUDA evidence harness for Parameter Policy execution qualification.

Phase C0/C1 establishes qualification infrastructure only.  The cases in this
file currently prove CUDA/BF16 availability and that the Phase-B full-BF16
execution contract can reach the real trainer prepare/finalize seam on one
process.  They are deliberately not optimizer- or backend-promotion evidence.

The coordinator always launches each case in a fresh Python subprocess.  Later
Phase C optimizer cases can therefore add train/save and fresh resume phases
without changing the evidence protocol.
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
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


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


import torch
from accelerate import Accelerator


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mikazuki import parameter_policy_execution as execution
from mikazuki.parameter_policy_trainer import create_parameter_policy_session
from tools.parameter_policy_execution_gpu_support import (
    ExecutionGpuMatrixError,
    temporary_execution_qualification,
)


EVIDENCE_SCHEMA = "dts.parameter-policy.execution-gpu-matrix"
EVIDENCE_VERSION = 1
_CASES = (
    "infra:cuda-bf16-capability:v1",
    "infra:full-bf16-session:v1",
)
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
        self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])

    def forward(self, value):
        return self.double_blocks[0](value)


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
                "learning_rate": 1e-3,
            },
        },
    }


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


def _run_worker_case(case_id: str) -> dict[str, Any]:
    if case_id == "infra:cuda-bf16-capability:v1":
        return _case_cuda_bf16_capability()
    if case_id == "infra:full-bf16-session:v1":
        return _case_full_bf16_session()
    raise ExecutionGpuMatrixError(f"Unknown execution GPU case {case_id!r}.")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _worker_main(args: argparse.Namespace) -> int:
    result_path = Path(args.worker_result)
    started = time.time()
    payload: dict[str, Any] = {
        "case_id": args.worker_case,
        "phase": args.worker_phase,
        "commit": None,
        "status": "fail",
    }
    try:
        payload["commit"] = _assert_exact_clean_head(args.expected_commit)
        payload["details"] = _run_worker_case(args.worker_case)
        payload["status"] = "pass"
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    payload["duration_seconds"] = round(time.time() - started, 4)
    _write_json(result_path, payload)
    return 0 if payload["status"] == "pass" else 1


def _subprocess_tail(value: str) -> str:
    return value[-_STDIO_TAIL_LIMIT:]


def _run_case_subprocess(
    *,
    case_id: str,
    expected_commit: str,
    work_dir: Path,
) -> dict[str, Any]:
    result_path = work_dir / (
        case_id.replace(":", "_").replace("/", "_") + ".json"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--expected-commit",
        expected_commit,
        "--worker-case",
        case_id,
        "--worker-phase",
        "probe",
        "--worker-result",
        str(result_path),
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
                "phase": "probe",
                "status": "fail",
                "error": f"Invalid worker result: {exc}",
            }
    else:
        result = {
            "case_id": case_id,
            "phase": "probe",
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
    output_path = _assert_output_outside_repo(args.output)
    environment = _cuda_environment()

    selected = tuple(args.case) if args.case else _CASES
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
            started = time.time()
            row = _run_case_subprocess(
                case_id=case_id,
                expected_commit=args.expected_commit,
                work_dir=work_dir,
            )
            row["coordinator_duration_seconds"] = round(
                time.time() - started,
                4,
            )
            evidence["cases"].append(row)
            print(f"{str(row.get('status', 'fail')).upper():4} {case_id}", flush=True)
            if row.get("status") != "pass":
                failed = True

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
    args = parser.parse_args()

    worker_mode = bool(args.worker_case or args.worker_phase or args.worker_result)
    if worker_mode:
        if not (args.worker_case and args.worker_phase and args.worker_result):
            raise SystemExit(
                "Worker mode requires --worker-case, --worker-phase, and --worker-result."
            )
        return _worker_main(args)
    return _coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
