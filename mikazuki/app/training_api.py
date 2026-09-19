"""Unified training preview/export/import/launch endpoints."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import toml
from fastapi import Request

import mikazuki.app.api as legacy_api
from mikazuki.anima_runtime import prepare_runtime_trainer
from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.frontend_training_patch import install_frontend_training_patch
from mikazuki.log import log
from mikazuki.training_launcher import run_prepared_train
from mikazuki.training_rehydrate import rehydrate_trainer_config
from mikazuki.training_request import (
    decode_training_request,
    materialize_sidecars,
    prepare_request_config,
    validate_prepared_config,
)
from mikazuki.training_schema_overrides import fixed_flux_family_schema, fixed_sd_schema, override_raw_schema
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


def _prepared_payload(prepared) -> dict:
    toml_text = toml.dumps(prepared.config)
    sidecars = [{"path": path, "content": content} for path, content in prepared.sidecars.items()]
    bundle = {
        "format": "dts-training-bundle-v1",
        "train_type": prepared.train_type,
        "toml": toml_text,
        "sidecars": {path: content for path, content in prepared.sidecars.items()},
    }
    return {
        "train_type": prepared.train_type,
        "trainer": prepared.trainer_file,
        "effective_config": prepared.config,
        "toml": toml_text,
        "warnings": prepared.warnings,
        "sidecars": sidecars,
        "bundle": json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    }


def _write_text_atomic(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)


@router.post("/training/preview")
async def preview_training_config(request: Request):
    try:
        page_type, config = decode_training_request(await request.body())
        prepared = prepare_request_config(config, page_type, launch=False)
        validate_prepared_config(prepared, False)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "prepare"})
    return APIResponseSuccess(message="preview ready", data=_prepared_payload(prepared))


@router.post("/training/export")
async def export_training_config(request: Request):
    try:
        page_type, config = decode_training_request(await request.body())
        prepared = prepare_request_config(config, page_type, launch=False)
        validate_prepared_config(prepared, False)
        materialize_sidecars(prepared.sidecars)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "export"})
    return APIResponseSuccess(message="export ready", data=_prepared_payload(prepared))


@router.post("/training/rehydrate")
async def rehydrate_training_config(request: Request):
    try:
        page_type, effective = decode_training_request(await request.body())
        if not page_type:
            raise ValueError("导入 Trainer TOML 时必须提供当前页面 train_type。")
        sidecars = None
        if effective.get("format") == "dts-training-bundle-v1":
            bundle_train_type = str(effective.get("train_type") or "")
            if bundle_train_type and bundle_train_type != str(page_type):
                raise ValueError(
                    f"Training bundle 属于 {bundle_train_type!r} 页面，不能导入当前 {page_type!r} 页面。"
                )
            toml_text = effective.get("toml")
            if not isinstance(toml_text, str):
                raise ValueError("Training bundle 缺少 toml 文本。")
            effective = toml.loads(toml_text)
            raw_sidecars = effective_sidecars = json.loads(json.dumps(
                json.loads((await request.body()).decode("utf-8")).get("config", {}).get("sidecars", {})
            ))
            if not isinstance(raw_sidecars, dict):
                raise ValueError("Training bundle sidecars 必须是 object。")
            sidecars = {str(key): str(value) for key, value in raw_sidecars.items()}
        gui_state = rehydrate_trainer_config(effective, str(page_type), sidecars=sidecars)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "rehydrate"})
    return APIResponseSuccess(message="rehydrate ready", data={"gui_state": gui_state})


@router.post("/run")
async def create_toml_file(request: Request):
    run_id = uuid.uuid4().hex
    toml_path = os.path.join(os.getcwd(), "config", "autosave", f"{run_id}.toml")
    try:
        page_type, config = decode_training_request(await request.body())
        prepared = prepare_request_config(config, page_type, launch=True, toml_path=toml_path)
        # Qwen3 joint training deliberately leaves the pinned sd-scripts
        # submodule pristine. At Start only, materialize the reviewed trainer
        # patch into an isolated cache tree, then validate that concrete trainer
        # exactly like any other launch asset.
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
