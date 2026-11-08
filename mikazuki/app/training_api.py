"""Unified training preview/export/import/launch endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Literal

import toml
from fastapi import Request

import mikazuki.app.api as legacy_api
from mikazuki.anima_effective_config import materialize_anima_launch_side_effects
from mikazuki.anima_runtime import prepare_runtime_trainer
from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.frontend_training_patch import install_frontend_training_patch
from mikazuki.log import log
from mikazuki.parameter_policy_editor import (
    bootstrap_parameter_policy_editor,
    parameter_policy_editor_metadata,
    parameter_policy_editor_preview,
)
from mikazuki.parameter_policy_admission import ComponentAdmission
from mikazuki.training_config import (
    PAGE_BACKEND_MAP,
    RequestAdmissionIntent,
)
from mikazuki.training_launcher import run_prepared_train
from mikazuki.training_rehydrate import rehydrate_trainer_config
from mikazuki.training_request import (
    decode_training_request,
    materialize_sidecars,
    prepare_request_config,
    validate_prepared_config,
)
from mikazuki.training_schema_overrides import fixed_flux_family_schema, fixed_sd_schema, override_raw_schema
from mikazuki.training_validation import validate_prepared_trainer
from mikazuki.utils import train_utils

install_frontend_training_patch()
legacy_api._fixed_sd_schema = fixed_sd_schema
legacy_api._fixed_flux_family_schema = fixed_flux_family_schema
_original_append_schema = legacy_api._append_schema


def _append_overridden_schema(name, content, lambda_hash):
    return _original_append_schema(name, override_raw_schema(name, content), lambda_hash)


legacy_api._append_schema = _append_overridden_schema
router = legacy_api.router
router.routes[:] = [
    route
    for route in router.routes
    if not (
        getattr(route, "path", None) in {"/run", "/training/preview", "/training/export", "/training/rehydrate"}
        and "POST" in getattr(route, "methods", set())
    )
]


def _decode_training_admission_request(
    body: bytes,
    *,
    purpose: Literal["preview", "export", "start"],
) -> tuple[str | None, dict, RequestAdmissionIntent]:
    """Decode a nested API envelope and its transient, one-job Start intent.

    The acknowledgement is deliberately outside Trainer config. Only the
    server assigns its origin; Preview/Export always receive a neutral intent.
    Legacy flat and nested payloads remain supported when no control is sent.
    """
    if type(purpose) is not str or purpose not in {"preview", "export", "start"}:
        raise ValueError("Unknown training Admission request purpose.")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Training request must contain valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("训练配置必须是 JSON object。")

    if "config" not in payload:
        if "experimental_full_bf16_ack" in payload:
            raise ValueError(
                "Experimental acknowledgement is only accepted as a "
                "top-level field in a nested training request."
            )
        return None, dict(payload), RequestAdmissionIntent()

    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Nested training request config must be a JSON object.")
    if {"experimental_full_bf16_ack", "admission_intent", "admission"}.intersection(raw_config):
        raise ValueError(
            "Admission controls are API-owned and cannot be supplied as Trainer config."
        )
    if {"admission_intent", "admission"}.intersection(payload):
        raise ValueError("Admission diagnostics and intent are server-owned fields.")

    page_type = payload.get("train_type")
    if page_type is not None and type(page_type) is not str:
        raise ValueError("Training request train_type must be a string.")

    supplied_ack = payload.get("experimental_full_bf16_ack", False)
    if type(supplied_ack) is not bool:
        raise ValueError("experimental_full_bf16_ack must be a strict boolean.")

    # Preview and Export are intentionally incapable of consuming consent.
    # The positive bit is ignored and not copied into any returned config.
    if purpose != "start" or not supplied_ack:
        intent = RequestAdmissionIntent()
    else:
        intent = RequestAdmissionIntent(True, "webui_start")

    return page_type, dict(raw_config), intent


def _prepared_parameter_policy_preview(prepared) -> dict | None:
    path = prepared.config.get("parameter_policy_config")
    if not path:
        return None

    content = prepared.sidecars.get(str(path))
    if content is None:
        raise RuntimeError(
            "Prepared Component request is missing its host-owned Parameter Policy sidecar."
        )
    try:
        policy = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Prepared Component request contains an invalid Parameter Policy sidecar."
        ) from exc

    return parameter_policy_editor_preview(
        policy,
        prepared.train_type,
        prepared.runtime_blockers,
    )


def _prepared_payload(prepared) -> dict:
    toml_text = toml.dumps(prepared.config)
    sidecars = [{"path": path, "content": content} for path, content in prepared.sidecars.items()]
    bundle = {
        "format": "dts-training-bundle-v1",
        "train_type": prepared.train_type,
        "toml": toml_text,
        "sidecars": {path: content for path, content in prepared.sidecars.items()},
    }
    payload = {
        "train_type": prepared.train_type,
        "trainer": prepared.trainer_file,
        "effective_config": prepared.config,
        "toml": toml_text,
        "warnings": prepared.warnings,
        "sidecars": sidecars,
        "bundle": json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    }
    policy_preview = _prepared_parameter_policy_preview(prepared)
    admission = getattr(prepared, "admission", None)
    if policy_preview is not None:
        if type(admission) is not ComponentAdmission:
            raise RuntimeError(
                "Prepared Component payload requires a first-party typed Admission diagnostic."
            )
        readiness = admission.readiness
        payload["admission"] = {
            "status": admission.status,
            "purpose": admission.purpose,
            "requires_experimental_ack": admission.requires_experimental_ack,
            "host_admissible": admission.host_admissible,
            "evidence_tier": readiness.evidence_tier if readiness is not None else None,
            "warnings": list(admission.warnings),
            "runtime_blockers": list(admission.runtime_blockers),
        }
        payload["runtime_ready"] = policy_preview["runtime_ready"]
        payload["runtime_blockers"] = list(policy_preview["runtime_blockers"])
        payload["parameter_policy_preview"] = {
            "version": policy_preview["version"],
            "train_type": policy_preview["train_type"],
            "profiles": policy_preview["profiles"],
            "components": policy_preview["components"],
        }
    elif admission is not None:
        raise RuntimeError(
            "Standard payload cannot carry a caller-provided Admission diagnostic."
        )
    return payload


def _write_text_atomic(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)


_ALLOWED_BUNDLE_SIDECAR_ROOTS = (
    (Path("config") / "autosave" / "multi-caption").as_posix() + "/",
    (Path("config") / "autosave" / "parameter-policy").as_posix() + "/",
    (Path("config") / "autosave" / "prompts").as_posix() + "/",
)


def _validated_bundle_sidecars(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("Training bundle sidecars 必须是 object。")
    sidecars: dict[str, str] = {}
    for raw_path, raw_content in raw.items():
        path = str(raw_path).replace("\\", "/")
        pure = Path(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Training bundle sidecar 路径不安全: {raw_path!r}")
        if not any(path.startswith(prefix) for prefix in _ALLOWED_BUNDLE_SIDECAR_ROOTS):
            raise ValueError(f"Training bundle sidecar 路径不属于 DTS 托管目录: {raw_path!r}")
        if not isinstance(raw_content, str):
            raise ValueError(f"Training bundle sidecar 内容必须是文本: {raw_path!r}")
        expected = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()[:24]
        if pure.stem != expected:
            raise ValueError(
                f"Training bundle sidecar 内容 hash 与路径不匹配: {raw_path!r}"
            )
        sidecars[path] = raw_content
    return sidecars


@router.get("/training/parameter-policy/metadata")
async def parameter_policy_metadata(request: Request):
    try:
        train_type = request.query_params.get("train_type")
        if not train_type:
            raise ValueError("Parameter Policy metadata requires train_type.")
        data = parameter_policy_editor_metadata(str(train_type))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "parameter-policy-metadata"})
    return APIResponseSuccess(message="parameter policy metadata ready", data=data)


@router.post("/training/parameter-policy/bootstrap")
async def bootstrap_parameter_policy(request: Request):
    try:
        page_type, config = decode_training_request(await request.body())
        if not page_type:
            raise ValueError("Parameter Policy bootstrap requires train_type.")
        gui_state = bootstrap_parameter_policy_editor(config, str(page_type))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "parameter-policy-bootstrap"})
    return APIResponseSuccess(
        message="parameter policy bootstrap ready",
        data={"gui_state": gui_state},
    )


@router.post("/training/preview")
async def preview_training_config(request: Request):
    try:
        page_type, config, admission_intent = _decode_training_admission_request(
            await request.body(), purpose="preview",
        )
        prepared = prepare_request_config(
            config, page_type, launch=False,
            admission_intent=admission_intent,
            admission_purpose="preview",
        )
        validate_prepared_config(prepared, False)
        payload = _prepared_payload(prepared)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "prepare"})
    return APIResponseSuccess(message="preview ready", data=payload)


@router.post("/training/export")
async def export_training_config(request: Request):
    try:
        page_type, config, admission_intent = _decode_training_admission_request(
            await request.body(), purpose="export",
        )
        prepared = prepare_request_config(
            config, page_type, launch=False,
            admission_intent=admission_intent,
            admission_purpose="export",
        )
        validate_prepared_config(prepared, False)
        payload = _prepared_payload(prepared)
        materialize_sidecars(prepared.sidecars)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "export"})
    return APIResponseSuccess(message="export ready", data=payload)


@router.post("/training/rehydrate")
async def rehydrate_training_config(request: Request):
    try:
        page_type, effective = decode_training_request(await request.body())
        if not page_type:
            raise ValueError("导入 Trainer TOML 时必须提供当前页面 train_type。")
        sidecars = None
        if effective.get("format") == "dts-training-bundle-v1":
            bundle = effective
            bundle_train_type = str(bundle.get("train_type") or "")
            bundle_backend = str(PAGE_BACKEND_MAP.get(bundle_train_type, bundle_train_type))
            page_backend = str(PAGE_BACKEND_MAP.get(str(page_type), str(page_type)))
            if bundle_backend and bundle_backend != page_backend:
                raise ValueError(
                    f"Training bundle backend={bundle_backend!r}，不能导入当前 backend={page_backend!r} 页面。"
                )
            toml_text = bundle.get("toml")
            if not isinstance(toml_text, str):
                raise ValueError("Training bundle 缺少 toml 文本。")
            sidecars = _validated_bundle_sidecars(bundle.get("sidecars", {}))
            effective = toml.loads(toml_text)
            # Import is the one place where unpacking a portable bundle is an
            # explicit user action. Materialize only validated DTS-owned
            # content-addressed paths so generated prompt sidecars are portable
            # too; Preview remains side-effect free.
            materialize_sidecars(sidecars)
        gui_state = rehydrate_trainer_config(effective, str(page_type), sidecars=sidecars)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "rehydrate"})
    return APIResponseSuccess(message="rehydrate ready", data={"gui_state": gui_state})


@router.post("/run")
async def create_toml_file(request: Request):
    run_id = uuid.uuid4().hex
    toml_path = os.path.join(os.getcwd(), "config", "autosave", f"{run_id}.toml")
    try:
        page_type, config, admission_intent = _decode_training_admission_request(
            await request.body(), purpose="start",
        )
        prepared = prepare_request_config(
            config, page_type, launch=True, toml_path=toml_path,
            admission_intent=admission_intent,
            admission_purpose="start",
        )
        component_start = bool(prepared.config.get("parameter_policy_config"))
        if component_start:
            # Step 6F keeps blocked/invalid Component Start requests free of
            # launch side effects. Validate model/assets/dataset first, then
            # materialize any staged Anima trainer and launch-only directories.
            validate_prepared_config(prepared, True, check_trainer=False)
            prepare_runtime_trainer(prepared)
            validate_prepared_trainer(prepared)
            materialize_anima_launch_side_effects(prepared.config, prepared.train_type)
        else:
            # Preserve the Standard-mode launch ordering.
            prepare_runtime_trainer(prepared)
            validate_prepared_config(prepared, True)
        materialize_sidecars(prepared.sidecars)
        _write_text_atomic(toml_path, toml.dumps(prepared.config))
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        log.error(f"Training config preparation failed: {exc}")
        return APIResponseFail(message=str(exc))

    train_dir = prepared.config.get("train_data_dir")
    threads = 8 if train_dir and len(train_utils.get_total_images(train_dir)) > 200 else 2
    result = run_prepared_train(
        toml_path,
        prepared.trainer_file,
        prepared.gpu_ids,
        threads,
        page_train_type=str(page_type or prepared.train_type),
        run_id=run_id,
    )
    if result.status == "success" and prepared.warnings:
        result.data = dict(result.data or {})
        result.data["warnings"] = prepared.warnings
    return result
