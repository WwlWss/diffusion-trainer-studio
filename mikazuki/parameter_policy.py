from __future__ import annotations

"""Host-side contract for optional Parameter Training Policy.

This module defines only configuration semantics.  It deliberately does not
import torch, inspect model parameters, mutate requires_grad, construct
optimizers/schedulers, or touch device placement.  Standard mode is a true
no-op for trainer behavior: stale policy GUI fields are removed and no sidecar
is emitted.
"""

from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mikazuki.optimizer_profiles import (
    get_optimizer_capability,
    normalize_optimizer_profile,
)
from mikazuki.training_gui_args import parse_ui_custom_params


PARAMETER_POLICY_VERSION = 1
PARAMETER_POLICY_DIR = Path("config") / "autosave" / "parameter-policy"

PARAMETER_POLICY_GUI_KEYS = {
    "optimization_mode",
    "parameter_policy_profiles",
    "parameter_policy_components",
}

# These become host-owned only when Component-wise mode is active.  Standard
# mode intentionally keeps historical ui_custom_params last-write-wins behavior.
PARAMETER_POLICY_OWNED_TRAINER_KEYS = {
    # Host contract / GUI-only policy keys.
    "optimization_mode",
    "parameter_policy_config",
    "parameter_policy_profiles",
    "parameter_policy_components",
    # Legacy/global optimizer selectors that must not compete with policy-owned
    # profiles once Component-wise mode is active.
    "optimizer_type",
    "optimizer_args",
    "optimizer_args_custom",
    "use_8bit_adam",
    "use_lion_optimizer",
    "anima_custom_optimizer_type",
    "anima_lora_custom_optimizer_type",
    # Legacy learning-rate controls superseded by Component routes.
    "learning_rate",
    "unet_lr",
    "text_encoder_lr",
    "learning_rate_te1",
    "learning_rate_te2",
    "anima_finetune_learning_rate",
    "anima_lora_text_encoder_lr",
    "block_lr",
    "self_attn_lr",
    "cross_attn_lr",
    "mlp_lr",
    "mod_lr",
    "llm_adapter_lr",
    "qwen3_lr",
}

_COMPONENT_MODE_ALIASES = {"component", "component-wise", "componentwise"}
_STANDARD_MODE_ALIASES = {"", "standard", "off", "false", "0"}


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _as_nonnegative_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Parameter Policy: {field} 必须是非负有限数字。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter Policy: {field} 必须是非负有限数字。") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Parameter Policy: {field} 必须是非负有限数字。")
    return result


def _normalize_profile_name(raw_name: object) -> str:
    name = str(raw_name or "").strip()
    if not name:
        raise ValueError("Parameter Policy: Optimizer Profile Name 不能为空。")
    return name


def _normalize_component_id(raw_id: object) -> str:
    component_id = str(raw_id or "").strip()
    if not component_id:
        raise ValueError("Parameter Policy: Component ID 不能为空。")
    return component_id


def _validate_component_mode_custom_overrides(config: Mapping[str, Any]) -> None:
    overrides = parse_ui_custom_params(config.get("ui_custom_params"))
    nested = config.get("__ui_custom_overrides")
    if isinstance(nested, Mapping):
        overrides = {**overrides, **dict(nested)}
    conflicts = sorted(PARAMETER_POLICY_OWNED_TRAINER_KEYS.intersection(overrides))
    if conflicts:
        raise ValueError(
            "Parameter Policy 与 ui_custom_params 中的 optimizer/LR 托管字段冲突: "
            + ", ".join(conflicts)
            + "。Component-wise 模式下这些值必须只由 Parameter Policy 提供。"
        )


def extract_parameter_policy_gui_state(config: dict) -> dict | None:
    """Remove host GUI fields and return active raw Component-wise state."""

    direct_sidecar = config.pop("parameter_policy_config", None)
    raw = {key: config.pop(key, None) for key in PARAMETER_POLICY_GUI_KEYS}

    if direct_sidecar not in (None, ""):
        raise ValueError(
            "parameter_policy_config 是 DTS 托管字段，不能从 GUI/raw config 手工注入。"
        )

    mode = str(raw.get("optimization_mode") or "standard").strip().lower()
    if mode in _STANDARD_MODE_ALIASES:
        return None
    if mode not in _COMPONENT_MODE_ALIASES:
        raise ValueError(f"Parameter Policy: 未知 Optimization Mode {mode!r}。")

    _validate_component_mode_custom_overrides(config)
    return {
        "optimizer_profiles": raw.get("parameter_policy_profiles"),
        "components": raw.get("parameter_policy_components"),
    }


def _normalize_profiles(raw_profiles: object) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ValueError("Parameter Policy: Component-wise 至少需要一个 Optimizer Profile。")

    result: dict[str, dict[str, Any]] = {}
    normalized_names: dict[str, str] = {}
    for raw_name in sorted(raw_profiles, key=lambda value: str(value).casefold()):
        name = _normalize_profile_name(raw_name)
        folded = name.casefold()
        if folded in normalized_names:
            raise ValueError(
                "Parameter Policy: Optimizer Profile Name 规范化后重复: "
                f"{normalized_names[folded]!r} / {name!r}。"
            )
        raw_profile = raw_profiles[raw_name]
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"Parameter Policy: Optimizer Profile {name!r} 必须是 object。")

        # Structural policy supports registered restricted/planned optimizers as
        # data.  Runnable capability is checked separately by
        # parameter_policy_runtime_blockers().
        profile = normalize_optimizer_profile(raw_profile)
        result[name] = profile
        normalized_names[folded] = name
    return result


def _lookup_profile_name(
    raw_name: object,
    profiles: Mapping[str, dict[str, Any]],
    *,
    field: str,
) -> str:
    name = _normalize_profile_name(raw_name)
    matches = {candidate.casefold(): candidate for candidate in profiles}
    resolved = matches.get(name.casefold())
    if resolved is None:
        raise ValueError(
            f"Parameter Policy: {field} 引用了不存在的 Optimizer Profile {name!r}。"
        )
    return resolved


def _normalize_components(
    raw_components: object,
    profiles: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_components, Mapping) or not raw_components:
        raise ValueError("Parameter Policy: Component-wise 至少需要一个 Component。")

    result: dict[str, dict[str, Any]] = {}
    for raw_component_id in sorted(raw_components, key=lambda value: str(value)):
        component_id = _normalize_component_id(raw_component_id)
        if component_id in result:
            raise ValueError(
                f"Parameter Policy: Component ID 规范化后重复: {component_id!r}。"
            )
        raw = raw_components[raw_component_id]
        if not isinstance(raw, Mapping):
            raise ValueError(f"Parameter Policy: Component {component_id!r} 必须是 object。")

        train = _as_bool(raw.get("train", True), True)
        if not train:
            result[component_id] = {"train": False}
            continue

        primary = _lookup_profile_name(
            raw.get("optimizer_profile"),
            profiles,
            field=f"{component_id}.optimizer_profile",
        )
        learning_rate = _as_nonnegative_float(
            raw.get("learning_rate"),
            f"{component_id}.learning_rate",
        )

        route: dict[str, Any] = {
            "train": True,
            "optimizer_profile": primary,
            "learning_rate": learning_rate,
        }
        capability = get_optimizer_capability(profiles[primary]["type"])
        fallback_raw = raw.get("fallback_optimizer_profile")

        if capability.requires_parameter_eligibility and fallback_raw in (None, ""):
            raise ValueError(
                f"Parameter Policy: Component {component_id!r} 使用 "
                f"{capability.name} 时必须设置 fallback_optimizer_profile。"
            )

        if fallback_raw not in (None, ""):
            fallback = _lookup_profile_name(
                fallback_raw,
                profiles,
                field=f"{component_id}.fallback_optimizer_profile",
            )
            if fallback.casefold() == primary.casefold():
                raise ValueError(
                    f"Parameter Policy: Component {component_id!r} 的 fallback 不能指向主 Profile 自身。"
                )
            fallback_capability = get_optimizer_capability(profiles[fallback]["type"])
            if fallback_capability.requires_parameter_eligibility:
                raise ValueError(
                    f"Parameter Policy: Component {component_id!r} 的 fallback "
                    f"Profile {fallback!r} 仍要求 parameter eligibility；"
                    "v1 fallback 必须是无需二次 eligibility routing 的 optimizer。"
                )
            route["fallback_optimizer_profile"] = fallback
            if raw.get("fallback_learning_rate") not in (None, ""):
                route["fallback_learning_rate"] = _as_nonnegative_float(
                    raw.get("fallback_learning_rate"),
                    f"{component_id}.fallback_learning_rate",
                )
        elif raw.get("fallback_learning_rate") not in (None, ""):
            raise ValueError(
                f"Parameter Policy: Component {component_id!r} 未设置 fallback optimizer，"
                "不能单独设置 fallback_learning_rate。"
            )

        result[component_id] = route

    return result


def canonicalize_parameter_policy(gui_state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(gui_state, Mapping):
        raise ValueError("Parameter Policy: GUI state 必须是 object。")

    profiles = _normalize_profiles(gui_state.get("optimizer_profiles"))
    components = _normalize_components(gui_state.get("components"), profiles)
    return {
        "version": PARAMETER_POLICY_VERSION,
        "optimizer_profiles": profiles,
        "components": components,
    }


def validate_parameter_policy(policy: object) -> dict[str, Any]:
    """Validate sidecar structure and return its canonical representation."""

    if not isinstance(policy, Mapping):
        raise ValueError("Parameter Policy sidecar 必须是 JSON object。")
    if policy.get("version") != PARAMETER_POLICY_VERSION:
        raise ValueError(
            f"Parameter Policy: 不支持 sidecar version={policy.get('version')!r}；"
            f"当前只支持 version={PARAMETER_POLICY_VERSION}。"
        )
    canonical = canonicalize_parameter_policy(
        {
            "optimizer_profiles": policy.get("optimizer_profiles"),
            "components": policy.get("components"),
        }
    )
    return canonical


def serialize_parameter_policy(policy: Mapping[str, Any]) -> tuple[str, str]:
    canonical = validate_parameter_policy(policy)
    content = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    return (PARAMETER_POLICY_DIR / f"{digest}.json").as_posix(), content


def parameter_policy_runtime_blockers(policy: Mapping[str, Any]) -> list[str]:
    """Describe why a structurally valid policy is not runnable yet.

    Step 2 intentionally has no trainer runtime.  Optimizer-specific blockers
    are also reported now so Preview/Export are honest about restricted/planned
    capabilities without making the sidecar format itself unable to represent
    them.
    """

    canonical = validate_parameter_policy(policy)
    blockers = [
        "Parameter Training Policy trainer runtime 尚未实现；当前 Component-wise 仅支持 Preview / Export / Rehydrate。"
    ]
    seen: set[str] = set()
    for profile in canonical["optimizer_profiles"].values():
        capability = get_optimizer_capability(profile["type"])
        if capability.component_support == "supported":
            continue
        message = (
            f"Optimizer {capability.name} 当前为 {capability.component_support}: "
            f"{capability.restriction or '尚未完成 Component-wise runtime 验证。'}"
        )
        if message not in seen:
            seen.add(message)
            blockers.append(message)
    return blockers


def build_parameter_policy_sidecar(
    config: dict,
    page_train_type: str | None = None,
) -> tuple[str | None, dict[str, str], dict | None]:
    del page_train_type  # Model-specific semantics start in Step 3.
    gui_state = extract_parameter_policy_gui_state(config)
    if gui_state is None:
        return None, {}, None

    policy = canonicalize_parameter_policy(gui_state)
    path, content = serialize_parameter_policy(policy)
    config["parameter_policy_config"] = path
    return path, {path: content}, policy


def rehydrate_parameter_policy(policy: object) -> dict[str, Any]:
    canonical = validate_parameter_policy(policy)
    return {
        "optimization_mode": "component",
        "parameter_policy_profiles": deepcopy(canonical["optimizer_profiles"]),
        "parameter_policy_components": deepcopy(canonical["components"]),
    }


def parse_legacy_optimizer_args(raw_args: object) -> dict[str, Any]:
    """Parse sd-scripts' legacy key=literal optimizer_args into a mapping."""

    if raw_args in (None, "", []):
        return {}
    if isinstance(raw_args, str):
        items = [line.strip() for line in raw_args.splitlines() if line.strip()]
    elif isinstance(raw_args, (list, tuple)):
        items = [str(item).strip() for item in raw_args if str(item).strip()]
    else:
        raise ValueError("Legacy optimizer_args 必须是字符串或字符串列表。")

    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Legacy optimizer_args 项缺少 '=': {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Legacy optimizer_args key 不能为空: {item!r}")
        try:
            value = ast.literal_eval(raw_value.strip())
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Legacy optimizer_args 值不是合法 Python literal: {item!r}") from exc
        # Match sd-scripts: later duplicate entries overwrite earlier ones.
        parsed[key] = value
    return parsed


def bootstrap_legacy_optimizer_profile(
    effective_legacy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a profile-only bootstrap from a compiled Standard config.

    Callers must first run the normal backend-specific Standard semantic
    compiler. This keeps bootstrap faithful to Anima/Flux/SDXL and custom-TOML
    normalization without importing the training compiler back into this low-
    level host contract.
    """

    compiled = dict(effective_legacy_config)
    optimizer_type = str(compiled.get("optimizer_type") or "").strip()
    if not optimizer_type:
        if _as_bool(compiled.get("use_8bit_adam")):
            optimizer_type = "AdamW8bit"
        elif _as_bool(compiled.get("use_lion_optimizer")):
            optimizer_type = "Lion"
        else:
            optimizer_type = "AdamW"

    if optimizer_type.casefold() == "custom":
        raise ValueError(
            "当前 Standard 使用 Custom optimizer；Component-wise v1 不能自动迁移任意 Custom optimizer。"
            "Standard 模式仍可继续正常使用。"
        )

    args = parse_legacy_optimizer_args(compiled.get("optimizer_args"))
    try:
        profile = normalize_optimizer_profile({"type": optimizer_type, "args": args})
    except ValueError as exc:
        raise ValueError(
            f"当前 Standard optimizer {optimizer_type!r} 无法自动迁移到 Component-wise Profile: {exc}"
        ) from exc
    return {"legacy_main": profile}


__all__ = [
    "PARAMETER_POLICY_DIR",
    "PARAMETER_POLICY_GUI_KEYS",
    "PARAMETER_POLICY_OWNED_TRAINER_KEYS",
    "PARAMETER_POLICY_VERSION",
    "bootstrap_legacy_optimizer_profile",
    "build_parameter_policy_sidecar",
    "canonicalize_parameter_policy",
    "extract_parameter_policy_gui_state",
    "parameter_policy_runtime_blockers",
    "parse_legacy_optimizer_args",
    "rehydrate_parameter_policy",
    "serialize_parameter_policy",
    "validate_parameter_policy",
]
