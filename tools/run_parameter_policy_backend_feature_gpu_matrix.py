#!/usr/bin/env python3
"""Exact-head real-backend execution-feature qualification harness.

D0 establishes the strict execution engine used by Phase D backend qualification.
The runner itself never promotes a backend and never bypasses production feature
gates. A manifest supplies machine-local fresh/resume commands and checkpoint
locations outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
_STDIO_TAIL_LIMIT = 16000


class BackendFeatureBootstrapError(RuntimeError):
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
        raise BackendFeatureBootstrapError(
            f"git {' '.join(args)} failed: {exc.output.strip()}"
        ) from exc


def _git_sha() -> str:
    sha = _git("rev-parse", "HEAD")
    if not sha:
        raise BackendFeatureBootstrapError("git rev-parse HEAD returned an empty SHA.")
    return sha


def _argv_value(argv: list[str], option: str, *, required: bool = True) -> str:
    prefix = option + "="
    found: list[str] = []
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv):
                raise BackendFeatureBootstrapError(
                    f"{option} requires a non-empty value."
                )
            found.append(argv[index + 1].strip())
        elif value.startswith(prefix):
            found.append(value.split("=", 1)[1].strip())
    if len(found) > 1:
        raise BackendFeatureBootstrapError(f"{option} must be supplied at most once.")
    if found and found[0]:
        return found[0]
    if required:
        raise BackendFeatureBootstrapError(f"{option} is required.")
    return ""


def _outside_repo(path: str, *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=False))
    except ValueError:
        return resolved
    raise BackendFeatureBootstrapError(
        f"{field} must be outside the repository; received {resolved}."
    )


def _assert_clean_head(expected_commit: str) -> str:
    expected = str(expected_commit or "").strip()
    actual = _git_sha()
    if actual != expected:
        raise BackendFeatureBootstrapError(
            "Backend feature qualification must run on the requested exact head: "
            f"expected={expected}, actual={actual}."
        )
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:12])
        raise BackendFeatureBootstrapError(
            "Backend feature qualification requires a completely clean workspace, "
            f"including untracked files. git status:\n{preview}"
        )
    return actual


_BOOTSTRAP_EXPECTED_COMMIT = _argv_value(sys.argv[1:], "--expected-commit")
_BOOTSTRAP_OUTPUT = _outside_repo(
    _argv_value(sys.argv[1:], "--output"),
    field="Backend feature --output",
)
_BOOTSTRAP_MANIFEST = _outside_repo(
    _argv_value(sys.argv[1:], "--manifest"),
    field="Backend feature --manifest",
)
_BOOTSTRAP_SHARED_REGRESSION = _outside_repo(
    _argv_value(sys.argv[1:], "--shared-regression-evidence"),
    field="Backend feature --shared-regression-evidence",
)
_BOOTSTRAP_COMMIT = _assert_clean_head(_BOOTSTRAP_EXPECTED_COMMIT)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch

from tools.parameter_policy_gpu_qualification_environment import (
    QualificationEnvironmentError,
    validate_qualification_environment,
)
from mikazuki.parameter_policy_trainer import (
    PARAMETER_POLICY_CHECKPOINT_MANIFEST,
    PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION,
    PARAMETER_POLICY_CHECKPOINT_PROGRESS,
)
from tools.parameter_policy_backend_feature_gpu_support import (
    BACKEND_FEATURE_EVIDENCE_SCHEMA,
    BACKEND_FEATURE_EVIDENCE_VERSION,
    SD_LORA_FULL_BF16_CASE_IDS,
    SDXL_LORA_FULL_BF16_CASE_IDS,
    BackendFeatureGpuMatrixError,
    backend_feature_qualification_snapshot,
    compare_backend_checkpoint_contracts,
    compare_checkpoint_progress,
    compare_sdxl_lora_weight_updates,
    load_backend_feature_manifest,
    summarize_sd_lora_full_bf16_promotion,
    summarize_sdxl_lora_full_bf16_promotion,
    validate_case_input_contract,
    validate_checkpoint_progress,
    validate_full_bf16_checkpoint_manifest,
    validate_shared_full_bf16_regression_evidence,
)


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
    values = sorted({line.strip() for line in output.splitlines() if line.strip()})
    return ",".join(values) if values else None


def _cuda_environment() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise BackendFeatureGpuMatrixError(
            "Backend feature qualification requires a real CUDA device."
        )
    bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(bf16) or not bool(bf16()):
        raise BackendFeatureGpuMatrixError(
            "Backend feature qualification requires CUDA BF16 support."
        )
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": [int(props.major), int(props.minor)],
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return {
        "python": platform.python_version(),
        "requirements_sha256": hashlib.sha256(
            (REPO_ROOT / "requirements.txt").read_bytes()
        ).hexdigest(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": _nvidia_driver_version(),
        "bf16_supported": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "packages": {
            name: _package_version(name)
            for name in (
                "accelerate",
                "bitsandbytes",
                "lion-pytorch",
                "schedulefree",
                "pytorch-optimizer",
                "torchvision",
                "transformers",
                "diffusers",
                "safetensors",
                "huggingface-hub",
                "toml",
                "numpy",
                "opencv-python",
                "imagesize",
            )
        },
    }


def _repo_clean(label: str) -> None:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:12])
        raise BackendFeatureGpuMatrixError(
            f"Repository became dirty {label}; qualification cannot continue:\n{preview}"
        )


def _tail(value: str) -> str:
    return value[-_STDIO_TAIL_LIMIT:]


_SECRET_OPTION_MARKERS = ("token", "key", "secret", "password")


def _redacted_argv(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if item.startswith("--"):
            option, separator, value = item.partition("=")
            normalized = option.lower().replace("-", "_")
            is_secret = any(marker in normalized for marker in _SECRET_OPTION_MARKERS)
            if is_secret:
                if separator:
                    redacted.append(f"{option}=<redacted>")
                else:
                    redacted.append(option)
                    redact_next = True
                continue
        redacted.append(item)
    return redacted


def _qualification_command(command: list[str]) -> list[str]:
    accelerate_config = REPO_ROOT / "config" / "accelerate-gpu.yaml"
    if not accelerate_config.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Missing DTS GPU Accelerate config: {accelerate_config}."
        )
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(accelerate_config),
        "--num_cpu_threads_per_process",
        "2",
        "--quiet",
        *command,
    ]


def _command_contract(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "fresh_argv": _redacted_argv(_qualification_command(case["fresh_command"])),
        "resume_argv": _redacted_argv(_qualification_command(case["resume_command"])),
        "cwd": case["cwd"],
        "entrypoint": case["entrypoint"],
        "environment_keys": sorted(case["environment"]),
        "fresh_checkpoint_dir": case["fresh_checkpoint_dir"],
        "resume_checkpoint_dir": case["resume_checkpoint_dir"],
        "policy_contract": case.get("policy_contract"),
    }


def _run_command(
    command: list[str],
    *,
    cwd: str | None,
    environment: dict[str, str],
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(environment)
    started = time.time()
    completed = subprocess.run(
        _qualification_command(command),
        cwd=cwd or str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "duration_seconds": round(time.time() - started, 4),
    }


def _checkpoint_payload(path: str, *, train_type: str) -> dict[str, Any]:
    checkpoint_dir = Path(path)
    manifest_path = checkpoint_dir / PARAMETER_POLICY_CHECKPOINT_MANIFEST
    if not manifest_path.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Missing Parameter Policy checkpoint manifest: {manifest_path}."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Invalid Parameter Policy checkpoint manifest {manifest_path}: {exc}"
        ) from exc
    return validate_full_bf16_checkpoint_manifest(
        payload,
        train_type=train_type,
        manifest_version=PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION,
    )


def _checkpoint_progress_payload(
    path: str,
    *,
    checkpoint_manifest: dict[str, Any],
) -> dict[str, Any]:
    progress_path = Path(path) / PARAMETER_POLICY_CHECKPOINT_PROGRESS
    if not progress_path.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Missing Parameter Policy checkpoint progress evidence: {progress_path}."
        )
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Invalid Parameter Policy checkpoint progress {progress_path}: {exc}"
        ) from exc
    return validate_checkpoint_progress(
        payload,
        checkpoint_manifest=checkpoint_manifest,
    )


def _assert_checkpoint_dir_absent(path: str, *, label: str) -> None:
    target = Path(path)
    if target.exists():
        raise BackendFeatureGpuMatrixError(
            f"{label} must not exist before its command runs; stale qualification "
            f"evidence is forbidden: {target}."
        )


def _run_case(
    case: dict[str, Any],
    *,
    expected_commit: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "train_type": case["train_type"],
        "feature": case["feature"],
        "scope": "backend",
        "qualification_evidence_component": True,
        "backend_qualification_eligible": False,
        "production_qualification_mutated": False,
        "command_contract": _command_contract(case),
        "policy_contract": case.get("policy_contract"),
        "input_contract": case.get("input_contract"),
        "status": "fail",
    }
    try:
        row["pre_fresh_commit"] = _assert_clean_head(expected_commit)
        validate_case_input_contract(case)
        row["pre_fresh_input_contract_valid"] = True
        _assert_checkpoint_dir_absent(
            case["fresh_checkpoint_dir"],
            label="Fresh checkpoint directory",
        )
        _assert_checkpoint_dir_absent(
            case["resume_checkpoint_dir"],
            label="Resume checkpoint directory",
        )
        fresh = _run_command(
            case["fresh_command"],
            cwd=case["cwd"],
            environment=case["environment"],
        )
        row["fresh"] = fresh
        row["post_fresh_commit"] = _assert_clean_head(expected_commit)
        if fresh["returncode"] != 0:
            raise BackendFeatureGpuMatrixError(
                f"Fresh backend command exited with {fresh['returncode']}."
            )
        fresh_manifest = _checkpoint_payload(
            case["fresh_checkpoint_dir"],
            train_type=case["train_type"],
        )
        row["fresh_checkpoint_manifest"] = fresh_manifest
        fresh_progress = _checkpoint_progress_payload(
            case["fresh_checkpoint_dir"],
            checkpoint_manifest=fresh_manifest,
        )
        row["fresh_checkpoint_progress"] = fresh_progress
        _assert_checkpoint_dir_absent(
            case["resume_checkpoint_dir"],
            label="Resume checkpoint directory",
        )
        row["pre_resume_commit"] = _assert_clean_head(expected_commit)
        validate_case_input_contract(case)
        row["pre_resume_input_contract_valid"] = True

        resume = _run_command(
            case["resume_command"],
            cwd=case["cwd"],
            environment=case["environment"],
        )
        row["resume"] = resume
        row["post_resume_commit"] = _assert_clean_head(expected_commit)
        if resume["returncode"] != 0:
            raise BackendFeatureGpuMatrixError(
                f"Resume backend command exited with {resume['returncode']}."
            )
        resumed_manifest = _checkpoint_payload(
            case["resume_checkpoint_dir"],
            train_type=case["train_type"],
        )
        row["resume_checkpoint_manifest"] = resumed_manifest
        resumed_progress = _checkpoint_progress_payload(
            case["resume_checkpoint_dir"],
            checkpoint_manifest=resumed_manifest,
        )
        row["resume_checkpoint_progress"] = resumed_progress
        compare_backend_checkpoint_contracts(fresh_manifest, resumed_manifest)
        compare_checkpoint_progress(fresh_progress, resumed_progress)
        if case["case_id"] in SDXL_LORA_FULL_BF16_CASE_IDS:
            row["component_update_evidence"] = compare_sdxl_lora_weight_updates(
                case["fresh_checkpoint_dir"],
                case["resume_checkpoint_dir"],
                require_conv_fallback=(
                    case["case_id"] == SDXL_LORA_FULL_BF16_CASE_IDS[1]
                ),
            )
        row["status"] = "pass"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shared-regression-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    commit = _assert_clean_head(args.expected_commit)
    manifest_path = _outside_repo(args.manifest, field="Backend feature --manifest")
    shared_regression_path = _outside_repo(
        args.shared_regression_evidence,
        field="Backend feature --shared-regression-evidence",
    )
    output_path = _outside_repo(args.output, field="Backend feature --output")
    if (
        manifest_path != _BOOTSTRAP_MANIFEST
        or shared_regression_path != _BOOTSTRAP_SHARED_REGRESSION
        or output_path != _BOOTSTRAP_OUTPUT
    ):
        raise BackendFeatureGpuMatrixError(
            "Backend feature bootstrap paths changed after runtime import."
        )

    try:
        qualification_contract = validate_qualification_environment(
            repo_root=REPO_ROOT,
            torch_module=torch,
            require_muon=True,
        )
    except QualificationEnvironmentError as exc:
        raise BackendFeatureGpuMatrixError(str(exc)) from exc

    try:
        shared_regression_payload = json.loads(
            shared_regression_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Invalid shared regression evidence {shared_regression_path}: {exc}"
        ) from exc
    shared_regression_contract = validate_shared_full_bf16_regression_evidence(
        shared_regression_payload,
        expected_commit=args.expected_commit,
        qualification_contract=qualification_contract,
    )

    cases = load_backend_feature_manifest(manifest_path, repo_root=REPO_ROOT)
    by_id = {case["case_id"]: case for case in cases}
    selected = tuple(args.case) if args.case else tuple(by_id)
    if len(selected) != len(set(selected)):
        raise BackendFeatureGpuMatrixError(
            "Backend feature case selection contains duplicates."
        )
    unknown = sorted(set(selected).difference(by_id))
    if unknown:
        raise BackendFeatureGpuMatrixError(
            "Unknown backend feature case(s): " + ", ".join(unknown)
        )

    qualification_snapshot = backend_feature_qualification_snapshot()
    evidence: dict[str, Any] = {
        "schema": BACKEND_FEATURE_EVIDENCE_SCHEMA,
        "version": BACKEND_FEATURE_EVIDENCE_VERSION,
        "commit": commit,
        "expected_commit": args.expected_commit,
        "manifest": str(manifest_path),
        "shared_regression_evidence": str(shared_regression_path),
        "environment": _cuda_environment(),
        "qualification_contract": qualification_contract,
        "shared_regression_contract": shared_regression_contract,
        "qualification_snapshot": qualification_snapshot,
        "cases": [],
    }
    failed = False
    for case_id in selected:
        row = _run_case(
            by_id[case_id],
            expected_commit=args.expected_commit,
        )
        evidence["cases"].append(row)
        if row["status"] != "pass":
            failed = True

    evidence["final_provenance_commit"] = _assert_clean_head(
        args.expected_commit
    )
    promotions: dict[str, dict[str, Any]] = {}
    if set(selected).intersection(SD_LORA_FULL_BF16_CASE_IDS):
        promotions["sd-lora"] = summarize_sd_lora_full_bf16_promotion(
            evidence["cases"],
            qualification_snapshot,
            shared_regression_contract,
        )
    if set(selected).intersection(SDXL_LORA_FULL_BF16_CASE_IDS):
        promotions["sdxl-lora"] = summarize_sdxl_lora_full_bf16_promotion(
            evidence["cases"],
            qualification_snapshot,
            shared_regression_contract,
        )
    if promotions:
        # Preserve the D1 SD1 single-backend JSON ABI; multiple backends can
        # still be audited independently without accepting partial promotion.
        evidence["backend_promotions"] = promotions
        if len(promotions) == 1:
            evidence["backend_promotion"] = next(iter(promotions.values()))
        if any(row["status"] != "pass" for row in promotions.values()):
            failed = True

    _write_json(output_path, evidence)
    print(f"wrote {output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
