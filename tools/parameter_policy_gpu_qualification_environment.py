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


def validate_qualification_environment(
    *,
    repo_root: Path,
    torch_module: Any,
    require_muon: bool,
) -> dict[str, Any]:
    python_major_minor = (sys.version_info.major, sys.version_info.minor)
    torch_full_version = str(getattr(torch_module, "__version__", "") or "")
    torch_base_version = _base_version(torch_full_version)
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
    requirements_sha256 = hashlib.sha256(
        requirements_path.read_bytes()
    ).hexdigest()

    mismatches: list[str] = []
    if python_major_minor != QUALIFICATION_PYTHON_MAJOR_MINOR:
        mismatches.append(
            "Python "
            f"{QUALIFICATION_PYTHON_MAJOR_MINOR[0]}."
            f"{QUALIFICATION_PYTHON_MAJOR_MINOR[1]} is required; "
            f"found {python_major_minor[0]}.{python_major_minor[1]}."
        )
    if torch_base_version != QUALIFICATION_TORCH_BASE_VERSION:
        mismatches.append(
            f"torch base version {QUALIFICATION_TORCH_BASE_VERSION} is required; "
            f"found {torch_full_version!r}."
        )
    if torchvision_version != QUALIFICATION_TORCHVISION_VERSION:
        mismatches.append(
            f"torchvision {QUALIFICATION_TORCHVISION_VERSION} is required; "
            f"found {torchvision_version!r}."
        )
    if require_muon and muon_provider_version != QUALIFICATION_MUON_PROVIDER_VERSION:
        mismatches.append(
            "pytorch-optimizer "
            f"{QUALIFICATION_MUON_PROVIDER_VERSION} is required for Muon evidence; "
            f"found {muon_provider_version!r}."
        )
    if not cuda_available:
        mismatches.append("A real CUDA device is required.")
    if not bf16_supported:
        mismatches.append("CUDA BF16 support is required.")

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


__all__ = [
    "QUALIFICATION_ENVIRONMENT_SCHEMA",
    "QUALIFICATION_ENVIRONMENT_VERSION",
    "QUALIFICATION_MUON_PROVIDER_VERSION",
    "QUALIFICATION_PYTHON_MAJOR_MINOR",
    "QUALIFICATION_TORCH_BASE_VERSION",
    "QUALIFICATION_TORCHVISION_VERSION",
    "QualificationEnvironmentError",
    "validate_qualification_environment",
]
