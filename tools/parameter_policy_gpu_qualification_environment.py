"""Shared fail-closed environment contract for Parameter Policy GPU qualification."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import re
import sys
from typing import Any


QUALIFICATION_ENVIRONMENT_SCHEMA = (
    "dts.parameter-policy.gpu-qualification-environment"
)
QUALIFICATION_ENVIRONMENT_VERSION = 2
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


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip()).lower()


def _exact_requirement_pins(requirements_path: Path) -> dict[str, str]:
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QualificationEnvironmentError(
            f"Unable to read qualification requirements: {requirements_path}: {exc}"
        ) from exc

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped or "==" not in stripped:
            continue
        if ";" in stripped:
            raise QualificationEnvironmentError(
                "Conditional exact requirement pins are not supported by the GPU "
                f"qualification contract: {requirements_path}:{line_number}: "
                f"{raw_line.strip()!r}."
            )

        left, separator, right = stripped.partition("==")
        if not separator or "==" in right:
            raise QualificationEnvironmentError(
                "Malformed exact requirement pin in GPU qualification requirements: "
                f"{requirements_path}:{line_number}: {raw_line.strip()!r}."
            )

        distribution = left.split("[", 1)[0].strip()
        version = right.strip()
        if (
            not distribution
            or not version
            or any(token in version for token in ("<", ">", "~=", "!=", ","))
        ):
            raise QualificationEnvironmentError(
                "Malformed exact requirement pin in GPU qualification requirements: "
                f"{requirements_path}:{line_number}: {raw_line.strip()!r}."
            )

        canonical = _canonical_distribution_name(distribution)
        previous = pins.get(canonical)
        if previous is not None and previous != version:
            raise QualificationEnvironmentError(
                "Conflicting exact requirement pins in GPU qualification "
                f"requirements for {canonical!r}: {previous!r} vs {version!r}."
            )
        pins[canonical] = version

    if not pins:
        raise QualificationEnvironmentError(
            "GPU qualification requires at least one exact '==' package pin in "
            f"{requirements_path}."
        )
    return dict(sorted(pins.items()))


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
    requirements_exact_pins = _exact_requirement_pins(requirements_path)
    installed_requirement_versions = {
        name: _package_version(name)
        for name in requirements_exact_pins
    }

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
        "requirements_exact_pins": requirements_exact_pins,
        "installed_requirement_versions": installed_requirement_versions,
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

    requirements_exact_pins = snapshot.get("requirements_exact_pins")
    installed_requirement_versions = snapshot.get("installed_requirement_versions")
    if (
        not isinstance(requirements_exact_pins, dict)
        or not requirements_exact_pins
        or not all(
            isinstance(name, str)
            and bool(name)
            and isinstance(version, str)
            and bool(version)
            for name, version in requirements_exact_pins.items()
        )
    ):
        mismatches.append(
            "requirements.txt exact pinned package contract is required."
        )
        requirements_exact_pins = {}
    if not isinstance(installed_requirement_versions, dict):
        mismatches.append(
            "Installed requirements package version evidence is required."
        )
        installed_requirement_versions = {}

    for name, expected_version in sorted(requirements_exact_pins.items()):
        actual_version = installed_requirement_versions.get(name)
        if actual_version != expected_version:
            mismatches.append(
                f"{name} {expected_version} is required by requirements.txt; "
                f"found {actual_version!r}."
            )

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
        "requirements_exact_pins": dict(sorted(requirements_exact_pins.items())),
        "installed_requirement_versions": {
            name: installed_requirement_versions[name]
            for name in sorted(requirements_exact_pins)
        },
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
