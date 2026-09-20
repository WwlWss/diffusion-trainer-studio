"""Request-level helpers shared by preview, export, rehydrate and launch."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import mikazuki.app.api as legacy_api
from mikazuki.multi_caption_config import build_multi_caption_sidecar
from mikazuki.parameter_policy import build_parameter_policy_sidecar, parameter_policy_runtime_blockers
from mikazuki.parameter_policy_matrix import (
    PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
    parameter_policy_gpu_selection_blockers,
)
from mikazuki.training_config import PAGE_BACKEND_MAP, prepare_training_config
from mikazuki.training_validation import validate_prepared_config
from mikazuki.utils import train_utils


_PROMPT_DIR = Path("config") / "autosave" / "prompts"
_HOST_PROMPT_KEYS = {
    "positive_prompts", "negative_prompts", "sample_width", "sample_height",
    "sample_cfg", "sample_seed", "sample_steps", "randomly_choice_prompt",
    "prompt_file", "sample_flow_shift",
}
_PROMPT_FLAGS = ("--n", "--s", "--l", "--d", "--w", "--h")

# Step 6F owns the opened backend allow-list in parameter_policy_matrix.py.
# Feature-specific compatibility checks remain fail-closed after this global
# backend gate opens.


def decode_training_request(body: bytes) -> tuple[str | None, dict]:
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("训练配置必须是 JSON object。")
    if isinstance(payload.get("config"), dict):
        return payload.get("train_type"), dict(payload["config"])
    return None, dict(payload)


def _page_backend(page_type: str | None, config: dict) -> str:
    return str(PAGE_BACKEND_MAP.get(page_type or "", page_type) or config.get("model_train_type") or "sd-lora")


def _content_addressed_prompt_path(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    return (_PROMPT_DIR / f"{digest}.txt").as_posix()


def _stage_prompt(content: str, sidecars: dict[str, str]) -> str:
    if not content.endswith("\n"):
        content += "\n"
    path = _content_addressed_prompt_path(content)
    sidecars[path] = content
    return path


def _apply_flow_shift(content: str, flow_shift: object) -> str:
    if flow_shift in (None, ""):
        return content
    try:
        value = float(flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError("Anima: sample_flow_shift 必须是有效数字。") from exc
    rewritten: list[str] = []
    for line in content.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]
        if stripped and not stripped.lstrip().startswith("#") and " --fs " not in stripped:
            stripped = f"{stripped} --fs {value:g}"
        rewritten.append(stripped + newline)
    return "".join(rewritten) if rewritten else content


def _choose_random_prompt(config: dict) -> str:
    train_data_dir = str(config.get("train_data_dir") or "").strip()
    if not train_data_dir:
        raise ValueError("随机预览 Prompt 需要 train_data_dir；dataset_config 模式请使用 Prompt 文件或固定 Prompt。")
    root = Path(train_data_dir)
    if not root.is_dir():
        raise ValueError(f"随机预览 Prompt 的训练目录不存在: {train_data_dir}")
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(subdirs) != 1:
        raise ValueError("随机预览 Prompt 要求 train_data_dir 下恰好一个数据子目录。")
    txt_files = sorted(subdirs[0].glob("*.txt"))
    if not txt_files:
        raise ValueError("随机预览 Prompt 找不到 caption txt 文件。")
    try:
        seed = int(config.get("sample_seed", 2333))
    except (TypeError, ValueError):
        seed = 2333
    return random.Random(seed).choice(txt_files).read_text(encoding="utf-8")


def prepare_prompt_fields(config: dict, page_type: str | None) -> tuple[dict[str, str], list[str]]:
    """Compile host prompt controls into a stable logical trainer prompt file.

    Preview never writes. Export/Start later materialize the same content-addressed
    sidecar, so all surfaces expose an identical ``sample_prompts`` value.
    """
    sidecars: dict[str, str] = {}
    warnings: list[str] = []
    raw_preview = config.pop("enable_preview", False)
    preview_enabled = raw_preview.strip().lower() in {"1", "true", "yes", "on"} if isinstance(raw_preview, str) else bool(raw_preview)
    backend = _page_backend(page_type, config)
    is_anima = backend in {"anima-lora", "anima-finetune"}
    flow_shift = config.get("sample_flow_shift") if is_anima else None
    prompt_file = str(config.get("prompt_file") or "").strip()

    if not preview_enabled and not prompt_file and "sample_prompts" not in config:
        for key in _HOST_PROMPT_KEYS:
            config.pop(key, None)
        return sidecars, warnings

    if prompt_file:
        config.pop("prompt_file", None)
        config.pop("sample_flow_shift", None)
        if flow_shift not in (None, "") and prompt_file.lower().endswith(".txt"):
            if os.path.isfile(prompt_file):
                content = _apply_flow_shift(Path(prompt_file).read_text(encoding="utf-8"), flow_shift)
                config["sample_prompts"] = _stage_prompt(content, sidecars)
            else:
                config["sample_prompts"] = prompt_file
                warnings.append("Prompt 文件尚不存在；Anima Flow Shift 会在文件可读取后编译到派生 Prompt。")
        else:
            config["sample_prompts"] = prompt_file
        for key in _HOST_PROMPT_KEYS - {"prompt_file", "sample_flow_shift"}:
            config.pop(key, None)
        return sidecars, warnings

    # Legacy basic-LoRA stores either an inline prompt line or a file path in
    # sample_prompts. Inline text must become a real sidecar because trainers
    # interpret sample_prompts as a filename.
    if "positive_prompts" not in config and "sample_prompts" in config:
        raw = config.get("sample_prompts")
        config.pop("sample_flow_shift", None)
        if isinstance(raw, str) and any(flag in raw for flag in _PROMPT_FLAGS):
            content = _apply_flow_shift(raw, flow_shift) if is_anima else raw
            config["sample_prompts"] = _stage_prompt(content, sidecars)
        return sidecars, warnings

    if not any(key in config for key in _HOST_PROMPT_KEYS - {"prompt_file"}):
        config.pop("sample_flow_shift", None)
        return sidecars, warnings

    positive = config.get("positive_prompts")
    if config.get("randomly_choice_prompt"):
        positive = _choose_random_prompt(config)
    positive = "" if positive is None else positive
    negative = config.get("negative_prompts", "")
    width = config.get("sample_width", 512)
    height = config.get("sample_height", 512)
    cfg = config.get("sample_cfg", 7)
    seed = config.get("sample_seed", 2333)
    steps = config.get("sample_steps", 24)
    content = f"{positive} --n {negative}  --w {width} --h {height} --l {cfg} --s {steps} --d {seed}"
    if is_anima and flow_shift not in (None, ""):
        content = _apply_flow_shift(content, flow_shift)
    config["sample_prompts"] = _stage_prompt(content, sidecars)
    for key in _HOST_PROMPT_KEYS:
        config.pop(key, None)
    return sidecars, warnings


def materialize_sidecars(sidecars: dict[str, str]) -> None:
    for raw_path, content in sidecars.items():
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)


def prepare_request_config(
    config: dict,
    page_type: str | None,
    stamp: str | None = None,
    launch: bool = False,
    toml_path: str | None = None,
    *,
    materialize: bool = False,
):
    del stamp
    train_utils.fix_config_types(config)
    policy_path, policy_sidecars, policy = build_parameter_policy_sidecar(config, page_type)
    multi_path, multi_sidecars, _multi_policy = build_multi_caption_sidecar(config, page_type)
    sidecars, prompt_warnings = prepare_prompt_fields(config, page_type)

    # Component requests intentionally stay side-effect free here even for
    # Start. Step 6F evaluates all runtime blockers first; launch-only Anima
    # materialization/finalization happens later in the API after read-only
    # asset validation succeeds.
    effective_launch = launch and policy is None
    prepared = prepare_training_config(
        config,
        page_train_type=page_type,
        resolve_backend=legacy_api.resolve_training_backend,
        launch=effective_launch,
        toml_path=toml_path,
    )

    effective_policy_path = prepared.config.get("parameter_policy_config")
    if policy_path is None and effective_policy_path not in (None, ""):
        raise ValueError("parameter_policy_config 是 DTS 托管字段，不能通过 ui_custom_params 手工注入。")
    if policy_path is not None and effective_policy_path != policy_path:
        raise ValueError("parameter_policy_config 是 DTS 托管字段，不能通过 ui_custom_params 覆盖。")

    effective_multi_path = prepared.config.get("multi_caption_config")
    if multi_path is None and effective_multi_path not in (None, ""):
        raise ValueError("multi_caption_config 是 DTS 托管字段，不能通过 ui_custom_params 手工注入。")
    if multi_path is not None and effective_multi_path != multi_path:
        raise ValueError("multi_caption_config 是 DTS 托管字段，不能通过 ui_custom_params 覆盖。")
    prepared.sidecars.update(policy_sidecars)
    prepared.sidecars.update(multi_sidecars)
    prepared.sidecars.update(sidecars)
    prepared.warnings.extend(prompt_warnings)
    if policy is not None:
        blockers = parameter_policy_runtime_blockers(
            policy,
            train_type=prepared.train_type,
            effective_config=prepared.config,
            integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
        )
        blockers.extend(parameter_policy_gpu_selection_blockers(prepared.gpu_ids))
        prepared.runtime_blockers.extend(blockers)
        if launch and blockers:
            raise ValueError(blockers[0])
    if materialize:
        materialize_sidecars(prepared.sidecars)
    return prepared


__all__ = [
    "decode_training_request", "materialize_sidecars", "prepare_prompt_fields",
    "prepare_request_config",
    "validate_prepared_config",
]
