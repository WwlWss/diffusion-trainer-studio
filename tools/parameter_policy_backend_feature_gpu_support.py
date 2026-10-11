"""CPU-testable support for exact-head backend feature qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any

from mikazuki import parameter_policy_execution as execution
from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy import serialize_parameter_policy, validate_parameter_policy
from mikazuki.parameter_policy_matrix import (
    PARAMETER_POLICY_BACKEND_MATRIX,
    PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
)
from tools.parameter_policy_execution_gpu_support import (
    EXECUTION_GPU_CASE_PHASES,
    EXECUTION_GPU_EVIDENCE_SCHEMA,
    EXECUTION_GPU_EVIDENCE_VERSION,
    EXECUTION_INFRA_CASE_IDS,
    SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
)


BACKEND_FEATURE_MANIFEST_SCHEMA = "dts.parameter-policy.backend-feature-manifest"
BACKEND_FEATURE_MANIFEST_VERSION = 1
BACKEND_FEATURE_EVIDENCE_SCHEMA = "dts.parameter-policy.backend-feature-gpu-matrix"
BACKEND_FEATURE_EVIDENCE_VERSION = 2
BACKEND_FEATURE_NAME = "full_bf16"
D0_D1_BACKEND_FEATURE_TRAIN_TYPES = frozenset(
    {
        "sd-lora",
        "sdxl-lora",
        "sdxl-finetune",
        "sd3-lora",
        "flux-lora",
        "chroma-lora",
        "flux-finetune",
    }
)
CHECKPOINT_PROGRESS_SCHEMA = "dts.parameter-policy.checkpoint-progress"
CHECKPOINT_PROGRESS_VERSION = 1

SD_LORA_FULL_BF16_EVIDENCE_ID = execution.FULL_BF16_SD_LORA_EVIDENCE_ID
SHARED_FULL_BF16_OPTIMIZER_EVIDENCE_ID = (
    execution.FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID
)
SD_LORA_FULL_BF16_CASE_IDS = (
    "backend:sd-lora:full-bf16:adamw:v1",
    "backend:sd-lora:full-bf16:muon-adamw-fallback:v1",
)
_SD_LORA_CASE_OPTIMIZER_TYPES = {
    SD_LORA_FULL_BF16_CASE_IDS[0]: frozenset({"AdamW"}),
    SD_LORA_FULL_BF16_CASE_IDS[1]: frozenset({"Muon", "AdamW"}),
}

SDXL_LORA_FULL_BF16_EVIDENCE_ID = execution.FULL_BF16_SDXL_LORA_EVIDENCE_ID
SDXL_LORA_FULL_BF16_CASE_IDS = (
    "backend:sdxl-lora:full-bf16:adamw:v1",
    "backend:sdxl-lora:full-bf16:muon-adamw-fallback:v1",
)
_SDXL_LORA_CASE_OPTIMIZER_TYPES = {
    SDXL_LORA_FULL_BF16_CASE_IDS[0]: frozenset({"AdamW"}),
    SDXL_LORA_FULL_BF16_CASE_IDS[1]: frozenset({"Muon", "AdamW"}),
}

# Only the two reviewed stock-LoRA backends share this qualification contract.
# Every new model family must still define its own explicit coverage and evidence.
D1_STOCK_LORA_SPECS = {
    "sd-lora": {
        "case_ids": SD_LORA_FULL_BF16_CASE_IDS,
        "evidence_id": SD_LORA_FULL_BF16_EVIDENCE_ID,
        "model_family": "sd1",
        "text_components": ("text_encoder.adapter",),
        "optimizer_types": _SD_LORA_CASE_OPTIMIZER_TYPES,
    },
    "sdxl-lora": {
        "case_ids": SDXL_LORA_FULL_BF16_CASE_IDS,
        "evidence_id": SDXL_LORA_FULL_BF16_EVIDENCE_ID,
        "model_family": "sdxl-base",
        "text_components": ("text_encoder_1.adapter", "text_encoder_2.adapter"),
        "optimizer_types": _SDXL_LORA_CASE_OPTIMIZER_TYPES,
    },
}


def _stock_lora_spec_for_case(case_id: str) -> tuple[str, dict[str, Any]] | None:
    for train_type, spec in D1_STOCK_LORA_SPECS.items():
        if case_id in spec["case_ids"]:
            return train_type, spec
    return None


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


def _nargs_option_values(command: list[str], option: str) -> list[str]:
    positions = [index for index, item in enumerate(command) if item == option]
    if len(positions) > 1:
        raise BackendFeatureGpuMatrixError(
            f"{option} must be supplied at most once in qualification commands."
        )
    if any(item.startswith(option + "=") for item in command):
        raise BackendFeatureGpuMatrixError(
            f"{option} must use explicit space-separated values in qualification commands."
        )
    if not positions:
        return []
    values: list[str] = []
    for item in command[positions[0] + 1 :]:
        if item.startswith("--"):
            break
        values.append(item)
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


def _canonical_backend_entrypoint(
    train_type: str,
    *,
    repo_root: Path,
    case_id: str,
) -> Path:
    row = PARAMETER_POLICY_BACKEND_MATRIX.get(train_type)
    if not isinstance(row, dict):
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} has no production backend mapping."
        )
    raw = str(row.get("trainer") or "").strip()
    if not raw:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} has no production trainer entrypoint."
        )
    candidate = (repo_root / raw).resolve(strict=False)
    try:
        candidate.relative_to(repo_root.resolve(strict=False))
    except ValueError as exc:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} production trainer escapes the repository."
        ) from exc
    if not candidate.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} production trainer is unavailable: "
            f"{candidate}."
        )
    return candidate


def _normalize_backend_command(
    command: list[str],
    *,
    canonical_entrypoint: Path,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> list[str]:
    if not command:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command is empty."
        )
    raw_entrypoint = Path(command[0])
    if not raw_entrypoint.is_absolute():
        raw_entrypoint = repo_root / raw_entrypoint
    resolved_entrypoint = raw_entrypoint.expanduser().resolve(strict=False)
    if resolved_entrypoint != canonical_entrypoint:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must start with "
            "the canonical production trainer from PARAMETER_POLICY_BACKEND_MATRIX."
        )
    for extra in command[1:]:
        if str(extra).strip().lower().endswith(".py"):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} command may not inject "
                "an additional Python entrypoint."
            )
    return [
        str(canonical_entrypoint.relative_to(repo_root.resolve(strict=False))),
        *command[1:],
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(
    path: Path,
    *,
    command_path: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Qualification identity file does not exist: {path}."
        )
    identity = {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if command_path is not None:
        identity["command_path"] = str(command_path)
    return identity


def _command_path(raw: str, *, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.absolute()


def _revalidate_file_identity(
    identity: dict[str, Any],
    *,
    label: str,
) -> None:
    command_raw = identity.get("command_path")
    resolved_raw = identity.get("path")
    if not isinstance(resolved_raw, str) or not resolved_raw:
        raise BackendFeatureGpuMatrixError(
            f"D1 qualification input {label!r} lacks a resolved path."
        )
    command = (
        Path(command_raw)
        if isinstance(command_raw, str) and command_raw
        else Path(resolved_raw)
    )
    current_resolved = command.resolve(strict=False)
    if str(current_resolved) != resolved_raw:
        raise BackendFeatureGpuMatrixError(
            f"D1 qualification input {label!r} resolved target changed after "
            "manifest validation."
        )
    current = _file_identity(
        current_resolved,
        command_path=command if command_raw else None,
    )
    if current != identity:
        raise BackendFeatureGpuMatrixError(
            f"D1 qualification input {label!r} changed after manifest validation."
        )


def _command_training_identity(command: list[str]) -> list[str]:
    """Normalize qualification argv while removing lifecycle/output-only fields."""

    normalized: list[str] = []
    index = 0
    value_options_to_ignore = {"--resume", "--output_dir"}
    while index < len(command):
        item = command[index]
        if item in value_options_to_ignore:
            if index + 1 >= len(command):
                raise BackendFeatureGpuMatrixError(
                    f"{item} requires a value in qualification commands."
                )
            index += 2
            continue
        if any(item.startswith(option + "=") for option in value_options_to_ignore):
            index += 1
            continue
        if item.startswith("--") and "=" in item:
            option, value = item.split("=", 1)
            normalized.extend([option, value])
        else:
            normalized.append(item)
        index += 1
    return normalized


def _qualification_output_dir(
    command: list[str],
    *,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> Path:
    values = _option_values(command, "--output_dir")
    if len(values) != 1:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must set "
            "--output_dir exactly once."
        )
    return _resolved_outside_repo(
        values[0],
        repo_root=repo_root,
        field=f"{case_id} {phase} output_dir",
    )


def _trainer_config_file_identity(
    command: list[str],
    *,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> dict[str, Any] | None:
    values = _option_values(command, "--config_file")
    if not values:
        return None
    if len(values) != 1:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must set "
            "--config_file at most once."
        )
    command_path = _command_path(values[0], repo_root=repo_root)
    path = command_path.resolve(strict=False)
    return _file_identity(path, command_path=command_path)


def _sd_lora_base_model_identity(
    command: list[str],
    *,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> dict[str, Any]:
    values = _option_values(command, "--pretrained_model_name_or_path")
    if len(values) != 1:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must set "
            "--pretrained_model_name_or_path exactly once to a local model file."
        )
    command_path = _command_path(values[0], repo_root=repo_root)
    path = _resolved_outside_repo(
        command_path,
        repo_root=repo_root,
        field=f"{case_id} {phase} pretrained_model_name_or_path",
    )
    if not path.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} base model must be "
            f"a local file: {path}."
        )
    return _file_identity(path, command_path=command_path)


def _d1_input_signature(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_stock_lora_d1_input_contract(
    *,
    case_id: str,
    fresh_command: list[str],
    resume_command: list[str],
    repo_root: Path,
    policy_contract: dict[str, Any] | None,
) -> dict[str, Any] | None:
    target = _stock_lora_spec_for_case(case_id)
    if target is None:
        return None
    train_type, spec = target
    if not isinstance(policy_contract, dict):
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} requires a D1 policy contract."
        )

    fresh_effective = _command_training_identity(fresh_command)
    resume_effective = _command_training_identity(resume_command)
    if fresh_effective != resume_effective:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} fresh/resume training-input argv "
            "must match exactly; only --resume and --output_dir may differ."
        )

    fresh_output_dir = _qualification_output_dir(
        fresh_command,
        repo_root=repo_root,
        case_id=case_id,
        phase="fresh",
    )
    resume_output_dir = _qualification_output_dir(
        resume_command,
        repo_root=repo_root,
        case_id=case_id,
        phase="resume",
    )
    if fresh_output_dir == resume_output_dir:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} fresh/resume --output_dir must be "
            "distinct so fresh step-2 state cannot pre-create resumed evidence."
        )

    fresh_model = _sd_lora_base_model_identity(
        fresh_command,
        repo_root=repo_root,
        case_id=case_id,
        phase="fresh",
    )
    fresh_config = _trainer_config_file_identity(
        fresh_command,
        repo_root=repo_root,
        case_id=case_id,
        phase="fresh",
    )

    policy_path = Path(str(policy_contract.get("path") or "")).resolve(strict=False)
    policy_command_path = Path(
        str(policy_contract.get("command_path") or policy_path)
    )
    policy_identity = _file_identity(
        policy_path,
        command_path=policy_command_path,
    )
    if policy_identity["sha256"] != policy_contract.get("raw_sha256"):
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} Parameter Policy sidecar changed "
            "while the qualification manifest was being validated."
        )

    payload = {
        "model_family": spec["model_family"],
        "v2": False,
        "effective_argv": fresh_effective,
        "output_contract": {
            "fresh_output_dir": str(fresh_output_dir),
            "resume_output_dir": str(resume_output_dir),
        },
        "base_model": fresh_model,
        "trainer_config": fresh_config,
        "parameter_policy": policy_identity,
    }
    return {
        **payload,
        "signature": _d1_input_signature(payload),
    }


def validate_case_input_contract(case: dict[str, Any]) -> None:
    contract = case.get("input_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise BackendFeatureGpuMatrixError("D1 input contract must be an object.")

    for key in ("base_model", "parameter_policy"):
        identity = contract.get(key)
        if not isinstance(identity, dict):
            raise BackendFeatureGpuMatrixError(
                f"D1 input contract is missing {key!r} identity."
            )
        _revalidate_file_identity(identity, label=key)

    trainer_config = contract.get("trainer_config")
    if trainer_config is not None:
        if not isinstance(trainer_config, dict):
            raise BackendFeatureGpuMatrixError(
                "D1 trainer_config identity must be an object or null."
            )
        _revalidate_file_identity(trainer_config, label="trainer config")

    payload = {
        "model_family": contract.get("model_family"),
        "v2": contract.get("v2"),
        "effective_argv": contract.get("effective_argv"),
        "output_contract": contract.get("output_contract"),
        "base_model": contract.get("base_model"),
        "trainer_config": contract.get("trainer_config"),
        "parameter_policy": contract.get("parameter_policy"),
    }
    if contract.get("signature") != _d1_input_signature(payload):
        raise BackendFeatureGpuMatrixError(
            "D1 qualification input contract signature is invalid."
        )


def _trainer_config_flattened(
    command: list[str],
    *,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> dict[str, Any]:
    values = _option_values(command, "--config_file")
    if not values:
        return {}
    if len(values) != 1:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} command must set "
            "--config_file at most once."
        )
    path = Path(values[0]).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve(strict=False)
    if not path.is_file():
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} trainer config does not exist: "
            f"{path}."
        )
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} {phase} trainer config is invalid: "
            f"{path}: {exc}"
        ) from exc

    flattened: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[str(nested_key)] = nested_value
        else:
            flattened[str(key)] = value
    return flattened


def _trainer_config_lifecycle_fields(
    command: list[str],
    *,
    repo_root: Path,
    case_id: str,
    phase: str,
) -> set[str]:
    flattened = _trainer_config_flattened(
        command,
        repo_root=repo_root,
        case_id=case_id,
        phase=phase,
    )
    guarded = {
        "max_train_epochs",
        "initial_epoch",
        "initial_step",
        "skip_until_initial_step",
        "resume",
        "resume_from_huggingface",
    }
    return guarded.intersection(flattened)


def _load_stock_lora_d1_policy_contract(
    *,
    case_id: str,
    fresh_command: list[str],
    resume_command: list[str],
    repo_root: Path,
) -> dict[str, Any] | None:
    target = _stock_lora_spec_for_case(case_id)
    if target is None:
        return None
    train_type, spec = target

    policy_paths: list[Path] = []
    policy_command_paths: list[Path] = []
    expected_accumulation = "1" if case_id == spec["case_ids"][0] else "2"
    for phase, command in (("fresh", fresh_command), ("resume", resume_command)):
        hidden = _trainer_config_flattened(
            command,
            repo_root=repo_root,
            case_id=case_id,
            phase=phase,
        )
        hidden_authority = sorted(
            {
                "parameter_policy_config",
                "gradient_accumulation_steps",
                "mixed_precision",
                "full_bf16",
                "network_module",
                "network_args",
                "network_train_unet_only",
                "network_train_text_encoder_only",
                "pretrained_model_name_or_path",
                "output_dir",
                "output_name",
                "cache_text_encoder_outputs",
                "cache_text_encoder_outputs_to_disk",
                "gradient_checkpointing",
                "save_model_as",
            }.intersection(hidden)
        )
        if hidden_authority:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} trainer config may not "
                "hide D1 qualification authority fields: "
                f"{hidden_authority!r}; use explicit command options."
            )
        hidden_v2 = hidden.get("v2")
        if hidden_v2 not in (None, False, 0, "", "false", "False"):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} trainer config enables "
                "SD2.x v2 mode, which is outside the D1 SD1.x qualification scope."
            )

        _require_single_option_value(
            command,
            "--gradient_accumulation_steps",
            expected=expected_accumulation,
            case_id=case_id,
            phase=phase,
        )
        _require_single_option_value(
            command,
            "--mixed_precision",
            expected="bf16",
            case_id=case_id,
            phase=phase,
        )
        if command.count("--full_bf16") != 1 or any(
            item.startswith("--full_bf16=") for item in command
        ):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} command must include "
                "--full_bf16 exactly once as an explicit flag."
            )

        _require_single_option_value(
            command,
            "--network_module",
            expected="networks.lora",
            case_id=case_id,
            phase=phase,
        )
        if "--v2" in command or any(item.startswith("--v2=") for item in command):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} command must remain "
                "SD1.x; --v2 is outside the D1 qualification scope."
            )
        for forbidden_target_flag in (
            "--network_train_unet_only",
            "--network_train_text_encoder_only",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
        ):
            if forbidden_target_flag in command or any(
                item.startswith(forbidden_target_flag + "=") for item in command
            ):
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} {phase} command may not use "
                    f"{forbidden_target_flag}; D1 requires live U-Net and Text Encoder adapters."
                )
        if train_type == "sdxl-lora" and (
            "--gradient_checkpointing" in command
            or any(item.startswith("--gradient_checkpointing=") for item in command)
        ):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} must not enable "
                "gradient_checkpointing until SDXL base-embedding ownership is qualified."
            )
        save_formats = _option_values(command, "--save_model_as")
        if save_formats not in ([], ["safetensors"]):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} must save real "
                "safetensors adapter weights for qualification evidence."
            )

        values = _option_values(command, "--parameter_policy_config")
        if len(values) != 1:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} command must set "
                "--parameter_policy_config exactly once."
            )
        policy_command_path = _command_path(values[0], repo_root=repo_root)
        policy_path = _resolved_outside_repo(
            policy_command_path,
            repo_root=repo_root,
            field=f"{case_id} {phase} parameter_policy_config",
        )
        if not policy_path.is_file():
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} Parameter Policy "
                f"sidecar does not exist: {policy_path}."
            )
        policy_paths.append(policy_path)
        policy_command_paths.append(policy_command_path)

    if policy_paths[0] != policy_paths[1]:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} fresh/resume commands must use "
            "the same Parameter Policy sidecar."
        )

    policy_path = policy_paths[0]
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
        policy = validate_parameter_policy(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} has invalid Parameter Policy "
            f"sidecar {policy_path}: {exc}"
        ) from exc

    expected_components = set(get_model_component_profile(train_type).components)
    actual_components = set(policy["components"])
    if actual_components != expected_components:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} Parameter Policy Component set "
            "does not exactly match the sd-lora model profile."
        )

    profiles = policy["optimizer_profiles"]
    trainable = {
        component_id: route
        for component_id, route in policy["components"].items()
        if bool(route.get("train"))
    }
    if not trainable:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} must train at least one SD LoRA component."
        )
    required_coverage = {
        "unet.attention.adapter",
        "unet.feed_forward.adapter",
        "unet.conv.adapter",
        *spec["text_components"],
    }
    missing_coverage = sorted(required_coverage.difference(trainable))
    if missing_coverage:
        raise BackendFeatureGpuMatrixError(
            f"Backend feature case {case_id!r} must train D1 coverage components: "
            f"missing={missing_coverage!r}."
        )

    def _profile_type(profile_name: object) -> str:
        profile = profiles.get(str(profile_name or ""))
        if not isinstance(profile, dict):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} references unknown optimizer "
                f"profile {profile_name!r}."
            )
        return str(profile.get("type") or "")

    if case_id == spec["case_ids"][0]:
        for component_id, route in trainable.items():
            if _profile_type(route.get("optimizer_profile")) != "AdamW":
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} AdamW case routes "
                    f"{component_id!r} through a non-AdamW primary optimizer."
                )
            if route.get("fallback_optimizer_profile") not in (None, ""):
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} AdamW case must not use "
                    "fallback routing."
                )
        contract_kind = "adamw"
        fallback_component = None
    else:
        conv_route = trainable.get("unet.conv.adapter")
        if not isinstance(conv_route, dict):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} must train "
                "unet.conv.adapter so AdamW fallback is exercised."
            )
        if _profile_type(conv_route.get("optimizer_profile")) != "Muon":
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} unet.conv.adapter primary "
                "optimizer must be Muon."
            )
        if _profile_type(conv_route.get("fallback_optimizer_profile")) != "AdamW":
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} unet.conv.adapter fallback "
                "optimizer must be AdamW."
            )
        if conv_route.get("fallback_learning_rate") in (None, ""):
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} unet.conv.adapter requires "
                "fallback_learning_rate."
            )
        linear_muon = [
            component_id
            for component_id in (
                "unet.attention.adapter",
                "unet.feed_forward.adapter",
                *spec["text_components"],
            )
            if component_id in trainable
            and _profile_type(trainable[component_id].get("optimizer_profile"))
            == "Muon"
        ]
        if set(linear_muon) != {
            "unet.attention.adapter",
            "unet.feed_forward.adapter",
            *spec["text_components"],
        }:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} must train U-Net hidden linear "
                "and every required Text Encoder adapter through Muon."
            )
        network_args = _nargs_option_values(fresh_command, "--network_args")
        resume_network_args = _nargs_option_values(resume_command, "--network_args")
        if network_args != resume_network_args:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} fresh/resume network_args must match."
            )
        parsed_network_args: dict[str, str] = {}
        for item in network_args:
            if "=" not in item:
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} network_args must use key=value."
                )
            key, value = item.split("=", 1)
            key = key.strip()
            if not key or key in parsed_network_args:
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} network_args contain an "
                    "empty or duplicate key."
                )
            parsed_network_args[key] = value.strip()
        try:
            conv_dim = int(parsed_network_args.get("conv_dim", ""))
        except ValueError as exc:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} Muon/fallback case requires "
                "integer conv_dim>0 in explicit --network_args."
            ) from exc
        if conv_dim <= 0:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} Muon/fallback case requires "
                "conv_dim>0 so real Conv LoRA fallback parameters are created."
            )
        contract_kind = "muon_adamw_fallback"
        fallback_component = "unet.conv.adapter"

    _canonical_path, canonical_text = serialize_parameter_policy(policy)
    return {
        "path": str(policy_path),
        "command_path": str(policy_command_paths[0]),
        "raw_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "policy_hash": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
        "model_family": spec["model_family"],
        "v2": False,
        "kind": contract_kind,
        "gradient_accumulation_steps": int(expected_accumulation),
        "mixed_precision": "bf16",
        "full_bf16": True,
        "network_module": "networks.lora",
        "covers_text_encoder_adapter": "text_encoder.adapter" in trainable
        if train_type == "sd-lora" else None,
        "covered_text_components": sorted(
            set(spec["text_components"]).intersection(trainable)
        ),
        "conv_dim": (
            conv_dim
            if case_id == spec["case_ids"][1]
            else None
        ),
        "profile_types": sorted(
            {
                str(profile.get("type") or "")
                for profile in profiles.values()
                if isinstance(profile, dict)
            }
        ),
        "trainable_components": sorted(trainable),
        "fallback_component": fallback_component,
    }


def _validate_lifecycle_command_contract(
    *,
    case_id: str,
    fresh_command: list[str],
    resume_command: list[str],
    fresh_checkpoint: Path,
    repo_root: Path,
) -> None:
    forbidden_options = (
        "--max_train_epochs",
        "--initial_epoch",
        "--initial_step",
        "--skip_until_initial_step",
    )
    for phase, command in (("fresh", fresh_command), ("resume", resume_command)):
        present = [
            option
            for option in forbidden_options
            if option in command or any(item.startswith(option + "=") for item in command)
        ]
        if present:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} command may not use "
                f"step-override option(s): {present!r}."
            )

        hidden = sorted(
            _trainer_config_lifecycle_fields(
                command,
                repo_root=repo_root,
                case_id=case_id,
                phase=phase,
            )
        )
        if hidden:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} {phase} trainer config may not "
                f"override qualification lifecycle fields: {hidden!r}."
            )

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
        if train_type not in D0_D1_BACKEND_FEATURE_TRAIN_TYPES:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} train_type={train_type!r} is "
                "outside the D0/D1 backend qualification scope."
            )
        feature = str(case.get("feature") or "").strip().lower()
        if feature != BACKEND_FEATURE_NAME:
            raise BackendFeatureGpuMatrixError(
                f"Backend feature case {case_id!r} must use "
                f"feature={BACKEND_FEATURE_NAME!r}."
            )

        raw_commands: dict[str, list[str]] = {}
        for key in ("fresh_command", "resume_command"):
            value = case.get(key)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise BackendFeatureGpuMatrixError(
                    f"Backend feature case {case_id!r} {key} must be a "
                    "non-empty trainer argv list."
                )
            raw_commands[key] = list(value)

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

        canonical_entrypoint = _canonical_backend_entrypoint(
            train_type,
            repo_root=repo_root,
            case_id=case_id,
        )
        commands = {
            f"{phase}_command": _normalize_backend_command(
                raw_commands[f"{phase}_command"],
                canonical_entrypoint=canonical_entrypoint,
                repo_root=repo_root,
                case_id=case_id,
                phase=phase,
            )
            for phase in ("fresh", "resume")
        }

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
            repo_root=repo_root,
        )
        policy_contract = _load_stock_lora_d1_policy_contract(
            case_id=case_id,
            fresh_command=commands["fresh_command"],
            resume_command=commands["resume_command"],
            repo_root=repo_root,
        )
        input_contract = _build_stock_lora_d1_input_contract(
            case_id=case_id,
            fresh_command=commands["fresh_command"],
            resume_command=commands["resume_command"],
            repo_root=repo_root,
            policy_contract=policy_contract,
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
                    canonical_entrypoint.relative_to(repo_root.resolve(strict=False))
                ),
                "environment": dict(environment),
                "fresh_checkpoint_dir": str(fresh_checkpoint),
                "resume_checkpoint_dir": str(resume_checkpoint),
                "policy_contract": policy_contract,
                "input_contract": input_contract,
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


def validate_checkpoint_progress(
    payload: Any,
    *,
    checkpoint_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress evidence must be an object."
        )
    if payload.get("schema") != CHECKPOINT_PROGRESS_SCHEMA:
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress evidence has unexpected schema."
        )
    if payload.get("version") != CHECKPOINT_PROGRESS_VERSION:
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress evidence has unexpected version."
        )
    checkpoint_id = payload.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress requires a non-empty checkpoint_id."
        )
    parent = payload.get("resume_source_checkpoint_id")
    if parent is not None and (
        not isinstance(parent, str) or not parent.strip()
    ):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress resume_source_checkpoint_id must be null "
            "or a non-empty string."
        )

    for key in (
        "train_type",
        "policy_hash",
        "runtime_topology_fingerprint",
        "execution_signature",
    ):
        if payload.get(key) != checkpoint_manifest.get(key):
            raise BackendFeatureGpuMatrixError(
                f"Checkpoint progress evidence does not match manifest for {key!r}."
            )

    expected_profiles = checkpoint_manifest.get("optimizer_profiles")
    profiles = payload.get("optimizer_profiles")
    if not isinstance(expected_profiles, dict) or not isinstance(profiles, dict):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress optimizer_profiles must be objects."
        )
    if set(profiles) != set(expected_profiles):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress optimizer Profile set does not match manifest."
        )
    for profile_name, optimizer_type in expected_profiles.items():
        row = profiles.get(profile_name)
        if not isinstance(row, dict) or row.get("optimizer_type") != optimizer_type:
            raise BackendFeatureGpuMatrixError(
                f"Checkpoint progress Profile {profile_name!r} optimizer type mismatch."
            )
        values = row.get("step_values")
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
        ):
            raise BackendFeatureGpuMatrixError(
                f"Checkpoint progress Profile {profile_name!r} requires optimizer "
                "step_values."
            )

    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise BackendFeatureGpuMatrixError(
            "Checkpoint progress scheduler evidence must be an object."
        )
    for key in ("step_count", "last_epoch"):
        value = scheduler.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise BackendFeatureGpuMatrixError(
                f"Checkpoint progress scheduler {key} must be an integer."
            )
    return payload


def compare_checkpoint_progress(
    fresh: dict[str, Any],
    resumed: dict[str, Any],
) -> None:
    fresh_checkpoint_id = fresh.get("checkpoint_id")
    resumed_checkpoint_id = resumed.get("checkpoint_id")
    if fresh.get("resume_source_checkpoint_id") is not None:
        raise BackendFeatureGpuMatrixError(
            "Fresh checkpoint must not claim a resume source checkpoint."
        )
    if resumed.get("resume_source_checkpoint_id") != fresh_checkpoint_id:
        raise BackendFeatureGpuMatrixError(
            "Resumed checkpoint lineage does not reference the fresh checkpoint_id."
        )
    if resumed_checkpoint_id == fresh_checkpoint_id:
        raise BackendFeatureGpuMatrixError(
            "Fresh/resumed checkpoints must use distinct checkpoint_id values."
        )

    fresh_profiles = fresh["optimizer_profiles"]
    resumed_profiles = resumed["optimizer_profiles"]
    if set(fresh_profiles) != set(resumed_profiles):
        raise BackendFeatureGpuMatrixError(
            "Fresh/resume checkpoint progress Profile set mismatch."
        )
    for profile_name in fresh_profiles:
        fresh_values = sorted(set(fresh_profiles[profile_name]["step_values"]))
        resumed_values = sorted(set(resumed_profiles[profile_name]["step_values"]))
        if fresh_values != [1]:
            raise BackendFeatureGpuMatrixError(
                f"Fresh checkpoint Profile {profile_name!r} must represent "
                f"logical optimizer step 1; got {fresh_values!r}."
            )
        if resumed_values != [2]:
            raise BackendFeatureGpuMatrixError(
                f"Resumed checkpoint Profile {profile_name!r} must represent "
                f"logical optimizer step 2; got {resumed_values!r}."
            )

    fresh_scheduler = fresh["scheduler"]
    resumed_scheduler = resumed["scheduler"]
    if resumed_scheduler["step_count"] != fresh_scheduler["step_count"] + 1:
        raise BackendFeatureGpuMatrixError(
            "Resume checkpoint scheduler step_count did not advance exactly once."
        )
    if resumed_scheduler["last_epoch"] != fresh_scheduler["last_epoch"] + 1:
        raise BackendFeatureGpuMatrixError(
            "Resume checkpoint scheduler last_epoch did not advance exactly once."
        )


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


def backend_feature_qualification_snapshot() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "backends": {
            name: {
                "status": row.status,
                "reason": row.reason,
                "evidence_case_id": row.evidence_case_id,
            }
            for name, row in execution.FULL_BF16_BACKEND_QUALIFICATIONS.items()
        },
        "optimizers": {
            name: {
                "status": row.status,
                "reason": row.reason,
                "evidence_case_id": row.evidence_case_id,
            }
            for name, row in execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS.items()
        },
    }


def validate_shared_full_bf16_regression_evidence(
    payload: Any,
    *,
    expected_commit: str,
    qualification_contract: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence must be an object."
        )
    if payload.get("schema") != EXECUTION_GPU_EVIDENCE_SCHEMA:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence has unexpected schema."
        )
    if payload.get("version") != EXECUTION_GPU_EVIDENCE_VERSION:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence has unexpected version."
        )
    expected = str(expected_commit or "").strip()
    for field in ("commit", "expected_commit", "final_provenance_commit"):
        if payload.get(field) != expected:
            raise BackendFeatureGpuMatrixError(
                f"Shared optimizer regression evidence {field} does not match "
                "the D1 exact head."
            )
    if payload.get("qualification_mode") != "regression":
        raise BackendFeatureGpuMatrixError(
            "D1 backend promotion requires shared optimizer evidence produced "
            "with qualification_mode='regression'."
        )
    if payload.get("qualification_contract") != qualification_contract:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression qualification environment does not "
            "match the D1 backend qualification environment."
        )

    rows_raw = payload.get("cases")
    if not isinstance(rows_raw, list):
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence cases must be a list."
        )
    required_cases = tuple(EXECUTION_GPU_CASE_PHASES)
    raw_case_ids = [
        row.get("case_id")
        for row in rows_raw
        if isinstance(row, dict)
    ]
    duplicate_cases = sorted(
        {
            case_id
            for case_id in raw_case_ids
            if raw_case_ids.count(case_id) > 1
        }
    )
    unknown_cases = sorted(
        {
            str(case_id)
            for case_id in raw_case_ids
            if case_id not in required_cases
        }
    )
    rows = {
        row.get("case_id"): row
        for row in rows_raw
        if isinstance(row, dict) and row.get("case_id") in required_cases
    }
    missing_cases = sorted(set(required_cases).difference(rows))
    failed_cases = sorted(
        case_id
        for case_id, row in rows.items()
        if row.get("status") != "pass"
    )
    phase_errors: list[str] = []
    for case_id, expected_phases in EXECUTION_GPU_CASE_PHASES.items():
        row = rows.get(case_id)
        if row is None:
            continue
        phases = row.get("phases")
        if not isinstance(phases, list):
            phase_errors.append(f"{case_id}: missing phases")
            continue
        actual_phases = [
            phase.get("phase")
            for phase in phases
            if isinstance(phase, dict)
        ]
        if actual_phases != list(expected_phases):
            phase_errors.append(
                f"{case_id}: expected phases {list(expected_phases)!r}, "
                f"got {actual_phases!r}"
            )
            continue
        if any(
            not isinstance(phase, dict) or phase.get("status") != "pass"
            for phase in phases
        ):
            phase_errors.append(f"{case_id}: one or more phases did not pass")

    if (
        len(rows_raw) != len(required_cases)
        or duplicate_cases
        or unknown_cases
        or missing_cases
        or failed_cases
        or phase_errors
    ):
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence does not contain an exact "
            "complete PASS C1-C4 matrix: "
            f"duplicates={duplicate_cases!r}, unknown={unknown_cases!r}, "
            f"missing={missing_cases!r}, failed={failed_cases!r}, "
            f"phase_errors={phase_errors!r}."
        )

    summary = payload.get("shared_optimizer_regression")
    if not isinstance(summary, dict):
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence is missing its summary."
        )
    if summary.get("id") != SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence has an unexpected summary id."
        )
    if summary.get("scope") != "shared_optimizer_regression":
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression evidence has an unexpected scope."
        )
    if summary.get("status") != "pass":
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression summary did not pass."
        )
    if summary.get("target_rows_match") is not True:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression source optimizer authority does not match."
        )
    if summary.get("infra_complete") is not True:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression CUDA/BF16 infrastructure is incomplete."
        )
    if summary.get("required_cases") != list(required_cases):
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression required case set does not match source."
        )
    if summary.get("missing_cases") != [] or summary.get("failed_cases") != []:
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression summary reports missing or failed cases."
        )
    if not all(
        rows.get(case_id, {}).get("status") == "pass"
        for case_id in EXECUTION_INFRA_CASE_IDS
    ):
        raise BackendFeatureGpuMatrixError(
            "Shared optimizer regression infrastructure cases did not pass."
        )

    return {
        "id": SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
        "status": "pass",
        "commit": expected,
        "qualification_contract": qualification_contract,
        "required_cases": list(required_cases),
    }


def _checkpoint_optimizer_types(payload: Any) -> frozenset[str] | None:
    if not isinstance(payload, dict):
        return None
    profiles = payload.get("optimizer_profiles")
    optimizers = payload.get("optimizers")
    if not isinstance(profiles, dict) or not isinstance(optimizers, list):
        return None
    profile_types = {
        str(value)
        for value in profiles.values()
        if isinstance(value, str) and value
    }
    optimizer_types = {
        str(row.get("optimizer_type"))
        for row in optimizers
        if isinstance(row, dict)
        and isinstance(row.get("optimizer_type"), str)
        and row.get("optimizer_type")
    }
    if not profile_types or profile_types != optimizer_types:
        return None
    return frozenset(profile_types)


def summarize_sd_lora_full_bf16_promotion(
    case_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    qualification_snapshot: dict[str, dict[str, dict[str, Any]]],
    shared_regression_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = {
        row.get("case_id"): row
        for row in case_rows
        if row.get("case_id") in SD_LORA_FULL_BF16_CASE_IDS
    }
    missing_cases = sorted(set(SD_LORA_FULL_BF16_CASE_IDS).difference(rows))
    failed_cases = sorted(
        case_id
        for case_id, row in rows.items()
        if row.get("status") != "pass"
    )

    case_contract_errors: list[str] = []
    for case_id, expected_optimizers in _SD_LORA_CASE_OPTIMIZER_TYPES.items():
        row = rows.get(case_id)
        if row is None:
            continue
        if row.get("train_type") != "sd-lora" or row.get("feature") != "full_bf16":
            case_contract_errors.append(
                f"{case_id}: expected sd-lora/full_bf16 case identity."
            )
            continue
        input_contract = row.get("input_contract")
        if (
            not isinstance(input_contract, dict)
            or input_contract.get("model_family") != "sd1"
            or input_contract.get("v2") is not False
            or not isinstance(input_contract.get("signature"), str)
            or not input_contract.get("signature")
            or not isinstance(input_contract.get("output_contract"), dict)
            or not input_contract["output_contract"].get("fresh_output_dir")
            or not input_contract["output_contract"].get("resume_output_dir")
            or input_contract["output_contract"].get("fresh_output_dir")
            == input_contract["output_contract"].get("resume_output_dir")
            or not isinstance(input_contract.get("base_model"), dict)
            or not input_contract["base_model"].get("sha256")
            or not isinstance(input_contract.get("parameter_policy"), dict)
            or not input_contract["parameter_policy"].get("sha256")
            or row.get("pre_fresh_input_contract_valid") is not True
            or row.get("pre_resume_input_contract_valid") is not True
        ):
            case_contract_errors.append(
                f"{case_id}: D1 SD1 lifecycle input identity is incomplete or "
                "was not revalidated before fresh/resume."
            )

        policy_contract = row.get("policy_contract")
        expected_policy_kind = (
            "adamw"
            if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
            else "muon_adamw_fallback"
        )
        expected_accumulation = (
            1
            if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
            else 2
        )
        if (
            not isinstance(policy_contract, dict)
            or policy_contract.get("kind") != expected_policy_kind
            or policy_contract.get("gradient_accumulation_steps")
            != expected_accumulation
            or policy_contract.get("mixed_precision") != "bf16"
            or policy_contract.get("full_bf16") is not True
            or policy_contract.get("network_module") != "networks.lora"
            or policy_contract.get("model_family") != "sd1"
            or policy_contract.get("v2") is not False
            or policy_contract.get("covers_text_encoder_adapter") is not True
            or (
                case_id == SD_LORA_FULL_BF16_CASE_IDS[1]
                and (
                    not isinstance(policy_contract.get("conv_dim"), int)
                    or policy_contract.get("conv_dim") <= 0
                )
            )
        ):
            case_contract_errors.append(
                f"{case_id}: expected policy/execution contract kind "
                f"{expected_policy_kind!r}, accumulation={expected_accumulation}, "
                "mixed_precision='bf16', full_bf16=true."
            )
        elif case_id == SD_LORA_FULL_BF16_CASE_IDS[1] and (
            policy_contract.get("fallback_component") != "unet.conv.adapter"
        ):
            case_contract_errors.append(
                f"{case_id}: expected unet.conv.adapter fallback contract."
            )

        fresh_manifest = row.get("fresh_checkpoint_manifest")
        resumed_manifest = row.get("resume_checkpoint_manifest")
        expected_policy_hash = (
            policy_contract.get("policy_hash")
            if isinstance(policy_contract, dict)
            else None
        )
        if (
            not isinstance(expected_policy_hash, str)
            or not expected_policy_hash
            or not isinstance(fresh_manifest, dict)
            or not isinstance(resumed_manifest, dict)
            or fresh_manifest.get("policy_hash") != expected_policy_hash
            or resumed_manifest.get("policy_hash") != expected_policy_hash
        ):
            case_contract_errors.append(
                f"{case_id}: checkpoint policy_hash does not match the validated "
                "Parameter Policy sidecar."
            )

        fresh_types = _checkpoint_optimizer_types(fresh_manifest)
        resumed_types = _checkpoint_optimizer_types(resumed_manifest)
        if fresh_types != expected_optimizers or resumed_types != expected_optimizers:
            case_contract_errors.append(
                f"{case_id}: expected optimizer types "
                f"{sorted(expected_optimizers)!r}, got fresh="
                f"{sorted(fresh_types) if fresh_types is not None else None!r}, "
                f"resume={sorted(resumed_types) if resumed_types is not None else None!r}."
            )

    backends = qualification_snapshot.get("backends", {})
    optimizers = qualification_snapshot.get("optimizers", {})
    backend_row = backends.get("sd-lora", {})
    backend_row_match = (
        backend_row.get("status") == "qualified"
        and backend_row.get("evidence_case_id") == SD_LORA_FULL_BF16_EVIDENCE_ID
    )
    shared_optimizer_authority_match = all(
        optimizers.get(name, {}).get("status") == "qualified"
        and optimizers.get(name, {}).get("evidence_case_id")
        == SHARED_FULL_BF16_OPTIMIZER_EVIDENCE_ID
        for name in ("AdamW", "Muon")
    )
    unexpected_backend_promotions = sorted(
        name
        for name, row in backends.items()
        if name != "sd-lora" and row.get("status") == "qualified"
    )
    unexpected_optimizer_promotions = sorted(
        name
        for name, row in optimizers.items()
        if name not in {"AdamW", "Muon"} and row.get("status") == "qualified"
    )

    shared_regression_match = bool(
        isinstance(shared_regression_contract, dict)
        and shared_regression_contract.get("id")
        == SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID
        and shared_regression_contract.get("status") == "pass"
        and isinstance(shared_regression_contract.get("commit"), str)
        and bool(shared_regression_contract.get("commit"))
    )

    complete = not missing_cases and not failed_cases and not case_contract_errors
    source_scope_valid = (
        backend_row_match
        and shared_optimizer_authority_match
        and shared_regression_match
        and not unexpected_backend_promotions
        and not unexpected_optimizer_promotions
    )
    status = "pass" if complete and source_scope_valid else "fail"
    return {
        "id": SD_LORA_FULL_BF16_EVIDENCE_ID,
        "scope": "backend_promotion",
        "feature": "full_bf16",
        "train_type": "sd-lora",
        "required_cases": list(SD_LORA_FULL_BF16_CASE_IDS),
        "missing_cases": missing_cases,
        "failed_cases": failed_cases,
        "case_contract_errors": case_contract_errors,
        "backend_row_match": backend_row_match,
        "shared_optimizer_authority_match": shared_optimizer_authority_match,
        "shared_regression_match": shared_regression_match,
        "shared_regression_id": (
            shared_regression_contract.get("id")
            if isinstance(shared_regression_contract, dict)
            else None
        ),
        "shared_regression_commit": (
            shared_regression_contract.get("commit")
            if isinstance(shared_regression_contract, dict)
            else None
        ),
        "unexpected_backend_promotions": unexpected_backend_promotions,
        "unexpected_optimizer_promotions": unexpected_optimizer_promotions,
        "status": status,
        "promotion_eligible": status == "pass",
        "backend_qualification_eligible": status == "pass",
        "production_qualification_mutated": False,
    }


__all__ = [
    "BACKEND_FEATURE_EVIDENCE_SCHEMA",
    "BACKEND_FEATURE_EVIDENCE_VERSION",
    "BACKEND_FEATURE_MANIFEST_SCHEMA",
    "BACKEND_FEATURE_MANIFEST_VERSION",
    "BACKEND_FEATURE_NAME",
    "CHECKPOINT_PROGRESS_SCHEMA",
    "CHECKPOINT_PROGRESS_VERSION",
    "D0_D1_BACKEND_FEATURE_TRAIN_TYPES",
    "SD_LORA_FULL_BF16_CASE_IDS",
    "SD_LORA_FULL_BF16_EVIDENCE_ID",
    "SDXL_LORA_FULL_BF16_CASE_IDS",
    "SDXL_LORA_FULL_BF16_EVIDENCE_ID",
    "D1_STOCK_LORA_SPECS",
    "SHARED_FULL_BF16_OPTIMIZER_EVIDENCE_ID",
    "BackendFeatureGpuMatrixError",
    "backend_feature_qualification_snapshot",
    "compare_backend_checkpoint_contracts",
    "compare_checkpoint_progress",
    "load_backend_feature_manifest",
    "summarize_sd_lora_full_bf16_promotion",
    "validate_case_input_contract",
    "validate_checkpoint_progress",
    "validate_full_bf16_checkpoint_manifest",
    "validate_shared_full_bf16_regression_evidence",
]
