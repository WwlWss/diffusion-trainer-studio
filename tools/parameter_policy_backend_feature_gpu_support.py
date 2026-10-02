"""CPU-testable support for exact-head backend feature qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES


BACKEND_FEATURE_MANIFEST_SCHEMA = "dts.parameter-policy.backend-feature-manifest"
BACKEND_FEATURE_MANIFEST_VERSION = 1
BACKEND_FEATURE_EVIDENCE_SCHEMA = "dts.parameter-policy.backend-feature-gpu-matrix"
BACKEND_FEATURE_EVIDENCE_VERSION = 1
BACKEND_FEATURE_NAME = "full_bf16"


class BackendFeatureGpuMatrixError(RuntimeError):
    pass


def _resolved_outside_repo(path: str | Path, *, repo_root: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(repo_root.resolve(strict=False))
    except ValueError:
        return resolved
    raise BackendFeatureGpuMatrixError(
        f"{field} must be outside the repository; received {resolved}."
    )


def _option_values(command: list[str], option: str) -> list[str]:
    values: list[str] = []
    prefix = option + "="
    for index, item in enumerate(command):
        if item == option:
            if index + 1 >= len(command):
                raise BackendFeatureGpuMatrixError(
                    f"{option} requires an explicit value in qualification commands."
                )
            values.append(command[index + 1])
        elif item.startswith(prefix):
            values.append(item.split("=", 1)[1])
    return values


def _require_single_option_value(
    command: list[str],
    option: str,
    *,
    expected: str,
    case_id: str,
    phase: str,
) -> None:
    values = _option_values(command, option)
    if values != [expected]:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must set "
            f"{option} exactly once to {expected!r}; got {values!r}."
        )


def _require_resume_source(
    command: list[str],
    *,
    fresh_checkpoint: Path,
    case_id: str,
) -> None:
    values = _option_values(command, "--resume")
    if len(values) != 1:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} resume_command must set "
            "--resume exactly once."
        )
    try:
        actual = Path(values[0]).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} has invalid --resume path."
        ) from exc
    if actual != fresh_checkpoint.resolve(strict=False):
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} --resume must point exactly to "
            "fresh_checkpoint_dir."
        )
    if "--resume_from_huggingface" in command:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} must use local checkpoint resume "
            "for exact-head qualification."
        )


def _repo_python_entrypoint(
    command: list[str],
    *,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> Path:
    candidates: list[Path] = []
    for item in command:
        raw = str(item or "").strip()
        if not raw.lower().endswith(".py"):
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        resolved = candidate.expanduser().resolve(strict=False)
        try:
            resolved.relative_to(repo_root.resolve(strict=False))
        except ValueError:
            continue
        if resolved.is_file():
            candidates.append(resolved)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must reference "
            "exactly one existing Python entrypoint from this repository."
        )
    return unique[0]


def _validate_lifecycle_command_contract(
    *,
    case_id: str,
    fresh_command: list[str],
    resume_command: list[str],
    fresh_checkpoint: Path,
) -> None:
    if _option_values(fresh_command, "--resume"):
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} fresh_command must not resume."
        )
    _require_single_option_value(
        fresh_command,
        "--max_train_steps",
        expected="2",
        case_id=case_id,
        phase="fresh",
    )
    _require_single_option_value(
        resume_command,
        "--max_train_steps",
        expected="2",
        case_id=case_id,
        phase="resume",
    )
    for phase, command in (
        ("fresh", fresh_command),
        ("resume", resume_command),
    ):
        _require_single_option_value(
            command,
            "--save_every_n_steps",
            expected="1",
            case_id=case_id,
            phase=phase,
        )
        if "--save_state" not in command:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} command must include "
                "--save_state so optimizer/scheduler state is checkpointed."
            )
    _require_resume_source(
        resume_command,
        fresh_checkpoint=fresh_checkpoint,
        case_id=case_id,
    )


def load_backend_feature_manifest(
    path: str | Path,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    manifest_path = _resolved_outside_repo(
        path,
        repo_root=repo_root,
        field="Backend feature --manifest",
    )
    if not manifest_path.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Backend feature manifest does not exist: {manifest_path}."
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Invalid backend feature manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise BackendFeatureGpuMatrixError("Backend feature manifest must be an object.")
    if raw.get("schema") != BACKEND_FEATURE_MANIFEST_SCHEMA:
        raise BackendFeatureGpuMatrixError(
            "Backend feature manifest has unexpected schema."
        )
    if raw.get("version") != BACKEND_FEATURE_MANIFEST_VERSION:
        raise BackendFeatureGpuMatrixError(
            "Backend feature manifest has unexpected version."
        )
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BackendFeatureGpuMatrixError(
            "Backend feature manifest cases must be a non-empty list."
        )

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case #{index} must be an object."
            )
        case_id = str(case.get("case_id") or "").strip()
        if not case_id:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case #{index} requires case_id."
            )
        if case_id in seen_ids:
            raise BackendFeatureGpuMatrixError(
                f"Duplicate backend feature case_id: {case_id!r}."
            )
        seen_ids.add(case_id)

        train_type = str(case.get("train_type") or "").strip().lower()
        if train_type not in PARAMETER_POLICY_RUNTIME_TRAIN_TYPES:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} has unknown train_type "
                f"{train_type!r}."
            )
        feature = str(case.get("feature") or "").strip().lower()
        if feature != BACKEND_FEATURE_NAME:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} must use "
                f"feature={BACKEND_FEATURE_NAME!r}."
            )

        commands: dict[str, list[str]] = {}
        for key in ("fresh_command", "resume_command"):
            value = case.get(key)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} {key} must be a "
                    "non-empty argv list."
                )
            commands[key] = list(value)

        environment = case.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} environment must be "
                "a string mapping."
            )

        forbidden_environment = sorted(
            key
            for key in environment
            if key.upper() in {"PATH", "PYTHONHOME", "PYTHONPATH"}
        )
        if forbidden_environment:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} may not override Python/runtime "
                f"resolution environment keys: {forbidden_environment!r}."
            )

        cwd_raw = case.get("cwd")
        cwd_path = (
            repo_root.resolve(strict=False)
            if cwd_raw in (None, "")
            else Path(str(cwd_raw)).expanduser().resolve(strict=False)
        )
        if cwd_path != repo_root.resolve(strict=False):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} cwd must be the exact "
                "qualification repository root in D0/D1."
            )

        fresh_entrypoint = _repo_python_entrypoint(
            commands["fresh_command"],
            repo_root=repo_root,
            case_id=case_id,
            phase="fresh",
        )
        resume_entrypoint = _repo_python_entrypoint(
            commands["resume_command"],
            repo_root=repo_root,
            case_id=case_id,
            phase="resume",
        )
        if fresh_entrypoint != resume_entrypoint:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} fresh/resume commands must "
                "use the same exact repository trainer entrypoint."
            )

        fresh_checkpoint = _resolved_outside_repo(
            case.get("fresh_checkpoint_dir", ""),
            repo_root=repo_root,
            field=f"{case_id} fresh_checkpoint_dir",
        )
        resume_checkpoint = _resolved_outside_repo(
            case.get("resume_checkpoint_dir", ""),
            repo_root=repo_root,
            field=f"{case_id} resume_checkpoint_dir",
        )
        if fresh_checkpoint == resume_checkpoint:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} requires distinct fresh/resume "
                "checkpoint directories."
            )
        _validate_lifecycle_command_contract(
            case_id=case_id,
            fresh_command=commands["fresh_command"],
            resume_command=commands["resume_command"],
            fresh_checkpoint=fresh_checkpoint,
        )

        normalized.append(
            {
                "case_id": case_id,
                "train_type": train_type,
                "feature": feature,
                "fresh_command": commands["fresh_command"],
                "resume_command": commands["resume_command"],
                "cwd": str(cwd_path),
                "entrypoint": str(
                    fresh_entrypoint.relative_to(repo_root.resolve(strict=False))
                ),
                "environment": dict(environment),
                "fresh_checkpoint_dir": str(fresh_checkpoint),
                "resume_checkpoint_dir": str(resume_checkpoint),
            }
        )
    return normalized


def _identity_signature(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_full_bf16_checkpoint_manifest(
    payload: Any,
    *,
    train_type: str,
    manifest_version: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BackendFeatureGpuMatrixError(
            "Parameter Policy checkpoint manifest must be an object."
        )
    if payload.get("version") != manifest_version:
        raise BackendFeatureGpuMatrixError(
            f"Checkpoint manifest version={payload.get('version')!r}, "
            f"expected={manifest_version}."
        )
    if payload.get("train_type") != train_type:
        raise BackendFeatureGpuMatrixError(
            f"Checkpoint train_type={payload.get('train_type')!r}, "
            f"expected={train_type!r}."
        )

    required = (
        "policy_hash",
        "runtime_topology_fingerprint",
        "trainable_components",
        "frozen_components",
        "trainable_parameter_tensors",
        "trainable_parameter_elements",
        "optimizer_profiles",
        "optimizers",
        "scheduler_identity",
        "scheduler_signature",
        "schedulers",
        "execution_identity",
        "execution_signature",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise BackendFeatureGpuMatrixError(
            f"Checkpoint manifest missing required key(s): {missing!r}."
        )

    identity = payload["execution_identity"]
    expected_identity = {
        "schema": "dts.parameter-policy.execution-identity",
        "version": 1,
        "train_type": train_type,
        "features": {
            "full_bf16": {
                "mixed_precision": "bf16",
                "trainable_parameter_dtype": "bfloat16",
            }
        },
    }
    if identity != expected_identity:
        raise BackendFeatureGpuMatrixError(
            "Checkpoint execution identity is not the expected full-BF16 contract."
        )
    signature = payload.get("execution_signature")
    if not isinstance(signature, str) or not signature:
        raise BackendFeatureGpuMatrixError(
            "Checkpoint execution_signature must be non-empty."
        )
    if signature != _identity_signature(identity):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint execution_signature does not match execution_identity."
        )

    scheduler_identity = payload.get("scheduler_identity")
    if not isinstance(scheduler_identity, dict):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint scheduler_identity must be an object."
        )
    scheduler_signature = payload.get("scheduler_signature")
    if not isinstance(scheduler_signature, str) or not scheduler_signature:
        raise BackendFeatureGpuMatrixError(
            "Checkpoint scheduler_signature must be non-empty."
        )
    if scheduler_signature != _identity_signature(scheduler_identity):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint scheduler_signature does not match scheduler_identity."
        )
    return payload


def compare_backend_checkpoint_contracts(
    fresh: dict[str, Any],
    resumed: dict[str, Any],
) -> None:
    stable_keys = (
        "train_type",
        "policy_hash",
        "runtime_topology_fingerprint",
        "trainable_components",
        "frozen_components",
        "trainable_parameter_tensors",
        "trainable_parameter_elements",
        "optimizer_profiles",
        "optimizers",
        "scheduler_identity",
        "scheduler_signature",
        "schedulers",
        "execution_identity",
        "execution_signature",
    )
    for key in stable_keys:
        if fresh.get(key) != resumed.get(key):
            raise BackendFeatureGpuMatrixError(
                f"Fresh/resume checkpoint contract mismatch for {key!r}."
            )


__all__ = [
    "BACKEND_FEATURE_EVIDENCE_SCHEMA",
    "BACKEND_FEATURE_EVIDENCE_VERSION",
    "BACKEND_FEATURE_MANIFEST_SCHEMA",
    "BACKEND_FEATURE_MANIFEST_VERSION",
    "BACKEND_FEATURE_NAME",
    "BackendFeatureGpuMatrixError",
    "compare_backend_checkpoint_contracts",
    "load_backend_feature_manifest",
    "validate_full_bf16_checkpoint_manifest",
]
