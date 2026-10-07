"""Shared fail-closed environment contract for Parameter Policy GPU qualification."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import sys
from typing import Any


QUALIFICATION_ENVIRONMENT_SCHEMA = (
    "dts.parameter-policy.gpu-qualification-environment"
)
QUALIFICATION_ENVIRONMENT_VERSION = 1
QUALIFICATION_PYTHON_MAJOR_MINOR = (3, 11)
QUALIFICATION_TORCH_BASE_VERSION = "2.7.0"
QUALIFICATION_TORCHVISION_VERSION = "0.22.0"
QUALIFICATION_MUON_PROVIDER_VERSION = "3.10.0"


class QualificationEnvironmentError(RuntimeError):
    pass


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _base_version(value: object) -> str:
    return str(value or "").split("+", 1)[0].strip()


def qualification_environment_snapshot(
    *,
    repo_root: Path,
    torch_module: Any,
    require_muon: bool,
) -> dict[str, Any]:
    python_major_minor = (sys.version_info.major, sys.version_info.minor)
    torch_full_version = str(getattr(torch_module, "__version__", "") or "")
    torchvision_version = _package_version("torchvision")
    muon_provider_version = (
        _package_version("pytorch-optimizer")
        if require_muon
        else None
    )

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(
        cuda is not None
        and callable(getattr(cuda, "is_available", None))
        and cuda.is_available()
    )
    bf16_probe = getattr(cuda, "is_bf16_supported", None) if cuda is not None else None
    bf16_supported = bool(
        cuda_available
        and callable(bf16_probe)
        and bf16_probe()
    )

    requirements_path = repo_root / "requirements.txt"
    if not requirements_path.is_file():
        raise QualificationEnvironmentError(
            f"Qualification environment requires {requirements_path}."
        )

    return {
        "python_major_minor": [
            python_major_minor[0],
            python_major_minor[1],
        ],
        "torch_full_version": torch_full_version,
        "torch_base_version": _base_version(torch_full_version),
        "torchvision_version": torchvision_version,
        "pytorch_optimizer_version": muon_provider_version,
        "requirements_sha256": hashlib.sha256(
            requirements_path.read_bytes()
        ).hexdigest(),
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
    }


def validate_qualification_environment_snapshot(
    snapshot: dict[str, Any],
    *,
    require_muon: bool,
) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise QualificationEnvironmentError(
            "GPU qualification environment snapshot must be an object."
        )

    expected_python = list(QUALIFICATION_PYTHON_MAJOR_MINOR)
    mismatches: list[str] = []
    if snapshot.get("python_major_minor") != expected_python:
        mismatches.append(
            "Python "
            f"{QUALIFICATION_PYTHON_MAJOR_MINOR[0]}."
            f"{QUALIFICATION_PYTHON_MAJOR_MINOR[1]} is required; "
            f"found {snapshot.get('python_major_minor')!r}."
        )
    if snapshot.get("torch_base_version") != QUALIFICATION_TORCH_BASE_VERSION:
        mismatches.append(
            f"torch base version {QUALIFICATION_TORCH_BASE_VERSION} is required; "
            f"found {snapshot.get('torch_full_version')!r}."
        )
    if snapshot.get("torchvision_version") != QUALIFICATION_TORCHVISION_VERSION:
        mismatches.append(
            f"torchvision {QUALIFICATION_TORCHVISION_VERSION} is required; "
            f"found {snapshot.get('torchvision_version')!r}."
        )
    if (
        require_muon
        and snapshot.get("pytorch_optimizer_version")
        != QUALIFICATION_MUON_PROVIDER_VERSION
    ):
        mismatches.append(
            "pytorch-optimizer "
            f"{QUALIFICATION_MUON_PROVIDER_VERSION} is required for Muon evidence; "
            f"found {snapshot.get('pytorch_optimizer_version')!r}."
        )
    if snapshot.get("cuda_available") is not True:
        mismatches.append("A real CUDA device is required.")
    if snapshot.get("bf16_supported") is not True:
        mismatches.append("CUDA BF16 support is required.")
    requirements_sha256 = snapshot.get("requirements_sha256")
    if not isinstance(requirements_sha256, str) or not requirements_sha256:
        mismatches.append("requirements.txt SHA256 provenance is required.")

    if mismatches:
        raise QualificationEnvironmentError(
            "GPU qualification environment contract failed: "
            + " ".join(mismatches)
        )

    return {
        "schema": QUALIFICATION_ENVIRONMENT_SCHEMA,
        "version": QUALIFICATION_ENVIRONMENT_VERSION,
        "status": "pass",
        "python_major_minor": (
            f"{QUALIFICATION_PYTHON_MAJOR_MINOR[0]}."
            f"{QUALIFICATION_PYTHON_MAJOR_MINOR[1]}"
        ),
        "torch_base_version": QUALIFICATION_TORCH_BASE_VERSION,
        "torchvision_version": QUALIFICATION_TORCHVISION_VERSION,
        "pytorch_optimizer_version": (
            QUALIFICATION_MUON_PROVIDER_VERSION if require_muon else None
        ),
        "requirements_sha256": requirements_sha256,
        "cuda_available": True,
        "bf16_supported": True,
    }


def validate_qualification_environment(
    *,
    repo_root: Path,
    torch_module: Any,
    require_muon: bool,
) -> dict[str, Any]:
    snapshot = qualification_environment_snapshot(
        repo_root=repo_root,
        torch_module=torch_module,
        require_muon=require_muon,
    )
    return validate_qualification_environment_snapshot(
        snapshot,
        require_muon=require_muon,
    )


__all__ = [
    "QUALIFICATION_ENVIRONMENT_SCHEMA",
    "QUALIFICATION_ENVIRONMENT_VERSION",
    "QUALIFICATION_MUON_PROVIDER_VERSION",
    "QUALIFICATION_PYTHON_MAJOR_MINOR",
    "QUALIFICATION_TORCH_BASE_VERSION",
    "QUALIFICATION_TORCHVISION_VERSION",
    "QualificationEnvironmentError",
    "qualification_environment_snapshot",
    "validate_qualification_environment",
    "validate_qualification_environment_snapshot",
]
