"""Host-side canonical contract for optional Multi-Caption training.

The GUI owns several storage-specific editing fields. Trainers never see those
fields directly: enabled Multi-Caption state is compiled into a content-addressed
JSON sidecar and trainers receive only multi_caption_config.

Standard mode is deliberately a no-op. This module removes stale Multi-Caption
GUI state from the outgoing config, but otherwise leaves the historical caption
pipeline untouched.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MULTI_CAPTION_VERSION = 1
MULTI_CAPTION_DIR = Path("config") / "autosave" / "multi-caption"

MULTI_CAPTION_GUI_KEYS = {
    "caption_mode",
    "multi_caption_storage",
    "multi_caption_file_groups",
    "multi_caption_line_extension",
    "multi_caption_line_groups",
    "multi_caption_json_path",
    "multi_caption_json_root",
    "multi_caption_image_key_mode",
    "multi_caption_jsonl_image_key_field",
    "multi_caption_json_groups",
}

_COMMON_PROCESSING_FIELDS = {
    "caption_separator",
    "secondary_separator",
    "enable_wildcard",
    "caption_prefix",
    "caption_suffix",
    "shuffle_caption",
    "keep_tokens",
    "keep_tokens_separator",
    "token_warmup_min",
    "token_warmup_step",
    "caption_dropout_rate",
    "caption_dropout_every_n_epochs",
    "caption_tag_dropout_rate",
}

# Every sd-scripts Dataset implementation used by DTS exposes the same
# process_caption() contract. Keep one capability set across pages so a Group
# does not silently lose wildcard/prefix/warmup merely because the model family
# changed.
PROCESSING_FIELDS = {
    "basic": set(_COMMON_PROCESSING_FIELDS),
    "shared": set(_COMMON_PROCESSING_FIELDS),
    "anima": set(_COMMON_PROCESSING_FIELDS),
}

_BOOL_PROCESSING_FIELDS = {
    "enable_wildcard",
    "shuffle_caption",
}
_INT_PROCESSING_FIELDS = {
    "keep_tokens",
    "caption_dropout_every_n_epochs",
    "token_warmup_min",
}
_FLOAT_PROCESSING_FIELDS = {
    "caption_dropout_rate",
    "caption_tag_dropout_rate",
    "token_warmup_step",
}
_STRING_PROCESSING_FIELDS = {
    "caption_separator",
    "secondary_separator",
    "caption_prefix",
    "caption_suffix",
    "keep_tokens_separator",
}

_FORBIDDEN_GROUP_FIELDS = {
    "weighted_captions",
    "max_token_length",
    "qwen3_max_token_length",
    "t5_max_token_length",
}

_STORAGE_MODES = {"files", "multiline", "json", "jsonl"}
_IMAGE_KEY_MODES = {"relative_path", "filename", "stem"}


def processing_profile_for_page(page_train_type: str | None) -> str:
    page = str(page_train_type or "")
    if page == "lora-basic":
        return "basic"
    if page in {"anima-lora", "anima-finetune"}:
        return "anima"
    return "shared"


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Multi-Caption: {field} 必须是有效数字。") from exc
    if not math.isfinite(result):
        raise ValueError(f"Multi-Caption: {field} 不能是 NaN 或无穷大。")
    return result


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Multi-Caption: {field} 必须是整数。")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Multi-Caption: {field} 必须是整数。") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Multi-Caption: {field} 必须是整数，不能自动截断小数。")
    return int(numeric)


def _normalize_extension(value: object, field: str) -> str:
    extension = str(value or "").strip()
    if not extension:
        raise ValueError(f"Multi-Caption: {field} 不能为空。")
    if "/" in extension or "\\" in extension:
        raise ValueError(f"Multi-Caption: {field} 只能是文件扩展名，不能包含路径分隔符。")
    if not extension.startswith("."):
        extension = "." + extension
    if extension == ".":
        raise ValueError(f"Multi-Caption: {field} 不能为空扩展名。")
    return extension


def _normalize_processing(raw: object, *, profile: str, group_name: str) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Multi-Caption: Group {group_name!r} 的 processing 必须是 object。")

    forbidden = sorted(_FORBIDDEN_GROUP_FIELDS.intersection(raw))
    if forbidden:
        raise ValueError(
            "Multi-Caption: 以下参数是全局 tokenizer/encoding 参数，不能放进 Group processing: "
            + ", ".join(forbidden)
        )

    allowed = PROCESSING_FIELDS[profile]
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Multi-Caption: 当前页面 processing profile={profile!r} 不支持字段: "
            + ", ".join(unknown)
        )

    result: dict[str, Any] = {}
    for key in sorted(raw):
        value = raw[key]
        if value in (None, ""):
            continue
        if key in _BOOL_PROCESSING_FIELDS:
            result[key] = _as_bool(value)
        elif key in _INT_PROCESSING_FIELDS:
            result[key] = _as_int(value, f"{group_name}.{key}")
        elif key in _FLOAT_PROCESSING_FIELDS:
            result[key] = _as_float(value, f"{group_name}.{key}")
        elif key in _STRING_PROCESSING_FIELDS:
            result[key] = str(value)
        else:
            result[key] = value

    if result.get("keep_tokens", 0) < 0:
        raise ValueError(f"Multi-Caption: Group {group_name!r} 的 keep_tokens 不能小于 0。")
    for field in ("caption_dropout_rate", "caption_tag_dropout_rate"):
        if field in result and not 0 <= result[field] <= 1:
            raise ValueError(f"Multi-Caption: Group {group_name!r} 的 {field} 必须在 0..1。")
    if result.get("caption_dropout_every_n_epochs", 0) < 0:
        raise ValueError(
            f"Multi-Caption: Group {group_name!r} 的 caption_dropout_every_n_epochs 不能小于 0。"
        )
    return result


def _normalize_groups(
    raw_groups: object,
    *,
    storage_mode: str,
    profile: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_groups, dict):
        raise ValueError("Multi-Caption: Caption Groups 必须是 object/dict。")

    groups: dict[str, dict[str, Any]] = {}
    positive_enabled = 0
    for raw_name in sorted(raw_groups):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Multi-Caption: Group Name 不能为空。")
        if name in groups:
            raise ValueError(
                f"Multi-Caption: Group Name 规范化后重复: {raw_name!r} -> {name!r}。"
            )
        raw = raw_groups[raw_name]
        if not isinstance(raw, dict):
            raise ValueError(f"Multi-Caption: Group {name!r} 必须是 object。")

        enabled = _as_bool(raw.get("enabled", True), True)
        weight = _as_float(raw.get("weight", 1.0), f"{name}.weight")
        if weight < 0:
            raise ValueError(f"Multi-Caption: Group {name!r} 的 weight 不能小于 0。")
        if enabled and weight > 0:
            positive_enabled += 1

        source_raw = raw.get("source") if isinstance(raw.get("source"), dict) else raw
        if storage_mode == "files":
            source = {
                "extension": _normalize_extension(
                    source_raw.get("extension"), f"{name}.extension"
                )
            }
        elif storage_mode == "multiline":
            line = _as_int(source_raw.get("line"), f"{name}.line")
            if line < 1:
                raise ValueError(f"Multi-Caption: Group {name!r} 的 line 必须从 1 开始。")
            source = {"line": line}
        else:
            key = str(source_raw.get("key") or source_raw.get("json_key") or "").strip()
            if not key:
                raise ValueError(f"Multi-Caption: Group {name!r} 的 JSON key 不能为空。")
            source = {"key": key}

        processing = _normalize_processing(
            raw.get("processing", {}),
            profile=profile,
            group_name=name,
        )
        groups[name] = {
            "enabled": enabled,
            "weight": weight,
            "source": source,
            "processing": processing,
        }

    if not groups:
        raise ValueError("Multi-Caption: 至少需要一个 Caption Group。")
    if positive_enabled == 0:
        raise ValueError("Multi-Caption: 至少需要一个 enabled 且 weight > 0 的 Group。")
    return groups


def _default_groups_for_storage(storage_mode: str) -> dict[str, dict[str, Any]]:
    """Return one immediately valid editable group for a newly enabled Multi mode."""

    base: dict[str, Any] = {
        "enabled": True,
        "weight": 1.0,
        "processing": {},
    }
    if storage_mode == "files":
        base["extension"] = ".txt"
    elif storage_mode == "multiline":
        base["line"] = 1
    elif storage_mode in {"json", "jsonl"}:
        base["key"] = "caption"
    return {"caption": base}

def extract_multi_caption_gui_state(
    config: dict,
    page_train_type: str | None,
) -> dict | None:
    """Remove host GUI fields from config and return active raw Multi state."""
    raw = {key: config.pop(key, None) for key in MULTI_CAPTION_GUI_KEYS}
    mode = str(raw.get("caption_mode") or "standard").strip().lower()
    if mode in {"", "standard", "off", "false", "0"}:
        return None
    if mode not in {"multi", "multi-caption", "multi_caption"}:
        raise ValueError(f"Multi-Caption: 未知 Caption Mode {mode!r}。")

    storage_mode = str(raw.get("multi_caption_storage") or "files").strip().lower()
    state = {
        "storage_mode": storage_mode,
        "profile": processing_profile_for_page(page_train_type),
    }
    if storage_mode == "files":
        groups = raw.get("multi_caption_file_groups")
        if groups is None:
            groups = _default_groups_for_storage(storage_mode)
        elif not isinstance(groups, dict):
            raise ValueError("Multi-Caption: Caption Groups 必须是 object/dict。")
        state["groups"] = groups
    elif storage_mode == "multiline":
        state["extension"] = raw.get("multi_caption_line_extension")
        groups = raw.get("multi_caption_line_groups")
        if groups is None:
            groups = _default_groups_for_storage(storage_mode)
        elif not isinstance(groups, dict):
            raise ValueError("Multi-Caption: Caption Groups 必须是 object/dict。")
        state["groups"] = groups
    elif storage_mode in {"json", "jsonl"}:
        state["path"] = raw.get("multi_caption_json_path")
        state["root"] = raw.get("multi_caption_json_root")
        state["image_key_mode"] = raw.get("multi_caption_image_key_mode")
        state["jsonl_image_key_field"] = raw.get("multi_caption_jsonl_image_key_field")
        groups = raw.get("multi_caption_json_groups")
        if groups is None:
            groups = _default_groups_for_storage(storage_mode)
        elif not isinstance(groups, dict):
            raise ValueError("Multi-Caption: Caption Groups 必须是 object/dict。")
        state["groups"] = groups
    else:
        raise ValueError(f"Multi-Caption: 未知 Storage Mode {storage_mode!r}。")
    return state


def canonicalize_multi_caption_policy(gui_state: dict) -> dict:
    storage_mode = str(gui_state.get("storage_mode") or "").strip().lower()
    if storage_mode not in _STORAGE_MODES:
        raise ValueError(f"Multi-Caption: 未知 Storage Mode {storage_mode!r}。")
    profile = str(gui_state.get("profile") or "shared")
    if profile not in PROCESSING_FIELDS:
        raise ValueError(f"Multi-Caption: 未知 processing profile {profile!r}。")

    storage: dict[str, Any] = {"mode": storage_mode}
    if storage_mode == "multiline":
        storage["extension"] = _normalize_extension(
            gui_state.get("extension") or ".txt",
            "multi_caption_line_extension",
        )
    elif storage_mode in {"json", "jsonl"}:
        path = str(gui_state.get("path") or "").strip()
        if not path:
            raise ValueError("Multi-Caption: JSON/JSONL 模式必须指定 dedicated caption 文件。")
        key_mode = str(gui_state.get("image_key_mode") or "relative_path").strip().lower()
        if key_mode not in _IMAGE_KEY_MODES:
            raise ValueError(
                f"Multi-Caption: image_key_mode 必须是 {sorted(_IMAGE_KEY_MODES)} 之一。"
            )
        storage.update({
            "path": path.replace("\\", "/"),
            "image_key_mode": key_mode,
        })
        root = str(gui_state.get("root") or "").strip()
        if root:
            storage["root"] = root.replace("\\", "/")
        if storage_mode == "jsonl":
            image_field = str(gui_state.get("jsonl_image_key_field") or "image").strip()
            if not image_field:
                raise ValueError("Multi-Caption: JSONL image key field 不能为空。")
            storage["image_key_field"] = image_field

    groups = _normalize_groups(
        gui_state.get("groups"),
        storage_mode=storage_mode,
        profile=profile,
    )
    return {
        "version": MULTI_CAPTION_VERSION,
        "selection": {"mode": "weighted_one"},
        "storage": storage,
        "groups": groups,
    }


def validate_multi_caption_policy(policy: dict) -> None:
    if not isinstance(policy, dict):
        raise ValueError("Multi-Caption sidecar 必须是 JSON object。")
    if policy.get("version") != MULTI_CAPTION_VERSION:
        raise ValueError(
            f"Multi-Caption: 不支持 sidecar version={policy.get('version')!r}；"
            f"当前只支持 version={MULTI_CAPTION_VERSION}。"
        )
    selection = policy.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") != "weighted_one":
        raise ValueError("Multi-Caption: 当前只支持 selection.mode='weighted_one'。")
    storage = policy.get("storage")
    if not isinstance(storage, dict) or storage.get("mode") not in _STORAGE_MODES:
        raise ValueError("Multi-Caption: sidecar storage.mode 无效。")
    groups = policy.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Multi-Caption: sidecar 至少需要一个 Group。")


def serialize_multi_caption_policy(policy: dict) -> tuple[str, str]:
    validate_multi_caption_policy(policy)
    content = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    return (MULTI_CAPTION_DIR / f"{digest}.json").as_posix(), content


def build_multi_caption_sidecar(
    config: dict,
    page_train_type: str | None,
) -> tuple[str | None, dict[str, str], dict | None]:
    """Compile active GUI fields and mutate config to the trainer-facing key."""
    gui_state = extract_multi_caption_gui_state(config, page_train_type)
    if gui_state is None:
        return None, {}, None
    policy = canonicalize_multi_caption_policy(gui_state)
    path, content = serialize_multi_caption_policy(policy)
    config["multi_caption_config"] = path
    return path, {path: content}, policy


def rehydrate_multi_caption_policy(
    policy: dict,
    page_train_type: str | None,
) -> dict:
    """Convert canonical sidecar content back into current GUI semantic fields."""
    validate_multi_caption_policy(policy)
    storage = deepcopy(policy["storage"])
    groups = deepcopy(policy["groups"])
    mode = storage["mode"]

    def gui_group(group: dict) -> dict:
        return {
            "enabled": bool(group.get("enabled", True)),
            "weight": group.get("weight", 1.0),
            **deepcopy(group.get("source") or {}),
            "processing": deepcopy(group.get("processing") or {}),
        }

    gui_groups = {name: gui_group(value) for name, value in groups.items()}
    state: dict[str, Any] = {
        "caption_mode": "multi",
        "multi_caption_storage": mode,
    }
    if mode == "files":
        state["multi_caption_file_groups"] = gui_groups
    elif mode == "multiline":
        state["multi_caption_line_extension"] = storage.get("extension", ".txt")
        state["multi_caption_line_groups"] = gui_groups
    else:
        state["multi_caption_json_path"] = storage.get("path", "")
        state["multi_caption_json_root"] = storage.get("root", "")
        state["multi_caption_image_key_mode"] = storage.get("image_key_mode", "relative_path")
        if mode == "jsonl":
            state["multi_caption_jsonl_image_key_field"] = storage.get("image_key_field", "image")
        state["multi_caption_json_groups"] = gui_groups

    processing_profile_for_page(page_train_type)
    return state


__all__ = [
    "MULTI_CAPTION_DIR",
    "MULTI_CAPTION_GUI_KEYS",
    "MULTI_CAPTION_VERSION",
    "PROCESSING_FIELDS",
    "build_multi_caption_sidecar",
    "canonicalize_multi_caption_policy",
    "extract_multi_caption_gui_state",
    "processing_profile_for_page",
    "rehydrate_multi_caption_policy",
    "serialize_multi_caption_policy",
    "validate_multi_caption_policy",
]
