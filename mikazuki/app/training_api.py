"""Unified training preview/launch endpoints."""

from __future__ import annotations

import os
from datetime import datetime

import toml
from fastapi import Request

import mikazuki.app.api as legacy_api
import mikazuki.process as process
from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.log import log
from mikazuki.training_request import (
    decode_training_request,
    prepare_request_config,
    validate_prepared_config,
)
from mikazuki.training_schema_factory import fixed_flux_family_schema, fixed_sd_schema
from mikazuki.utils import train_utils

legacy_api._fixed_sd_schema = fixed_sd_schema
legacy_api._fixed_flux_family_schema = fixed_flux_family_schema
router = legacy_api.router
router.routes[:] = [
    route for route in router.routes
    if not (getattr(route, "path", None) in {"/run", "/training/preview"}
            and "POST" in getattr(route, "methods", set()))
]


@router.post("/training/preview")
async def preview_training_config(request: Request):
    try:
        page_type, config = decode_training_request(await request.body())
        prepared = prepare_request_config(config, page_type, "preview", False)
        validate_prepared_config(prepared, False)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return APIResponseFail(message=str(exc), data={"stage": "prepare"})
    return APIResponseSuccess(message="preview ready", data={
        "train_type": prepared.train_type,
        "trainer": prepared.trainer_file,
        "effective_config": prepared.config,
        "toml": toml.dumps(prepared.config),
        "warnings": prepared.warnings,
    })


@router.post("/run")
async def create_toml_file(request: Request):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_path = os.path.join(os.getcwd(), "config", "autosave", f"{stamp}.toml")
    try:
        page_type, config = decode_training_request(await request.body())
        prepared = prepare_request_config(config, page_type, stamp, True, toml_path)
        validate_prepared_config(prepared, True)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        log.error(f"Training config preparation failed: {exc}")
        return APIResponseFail(message=str(exc))

    train_dir = prepared.config.get("train_data_dir")
    threads = 8 if train_dir and len(train_utils.get_total_images(train_dir)) > 200 else 2
    with open(toml_path, "w", encoding="utf-8") as handle:
        handle.write(toml.dumps(prepared.config))

    result = process.run_train(toml_path, prepared.trainer_file, prepared.gpu_ids, threads)
    if result.status == "success" and prepared.warnings:
        result.data = dict(result.data or {})
        result.data["warnings"] = prepared.warnings
    return result
