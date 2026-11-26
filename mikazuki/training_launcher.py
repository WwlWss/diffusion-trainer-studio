"""Launch a configuration that has already passed effective-config preparation."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.gpu_runtime import validate_cuda_training_runtime
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.training_config import RequestAdmissionIntent


ACCELERATE_GPU_CONFIG = Path(__file__).resolve().parents[1] / "config" / "accelerate-gpu.yaml"


def run_prepared_train(
    toml_path: str,
    trainer_file: str,
    gpu_ids: Optional[list] = None,
    cpu_threads: Optional[int] = 2,
    *,
    page_train_type: str | None = None,
    run_id: str | None = None,
    admission_intent: RequestAdmissionIntent | None = None,
):
    """Start Accelerate without mutating trainer configuration again.

    The optional request intent is one invocation's *confirmation*, never
    authorization to skip the upstream Request or downstream Trainer gates.
    """
    # Fail closed on forged, mutated or non-typed Host controls before GPU
    # validation, task creation, logging, or any other observable side effect.
    if admission_intent is not None:
        if type(admission_intent) is not RequestAdmissionIntent:
            raise ValueError("Request admission intent must be an exact first-party type.")
        if (
            type(admission_intent.experimental_full_bf16_ack) is not bool
            or type(admission_intent.origin) is not str
            or admission_intent.origin not in ("default", "webui_start")
            or (
                admission_intent.experimental_full_bf16_ack is True
                and admission_intent.origin != "webui_start"
            )
        ):
            raise ValueError("Invalid Request admission intent for this Start invocation.")

    try:
        selected_gpu_ids, gpu_names = validate_cuda_training_runtime(gpu_ids)
    except RuntimeError as exc:
        log.error(f"GPU preflight failed / GPU 启动检查失败: {exc}")
        return APIResponse(status="error", message=str(exc))

    if not ACCELERATE_GPU_CONFIG.is_file():
        message = f"Missing DTS GPU Accelerate config: {ACCELERATE_GPU_CONFIG}"
        log.error(message)
        return APIResponse(status="error", message=message)

    log.info(f"Training started with effective config / 使用最终配置启动训练: {toml_path}")
    log.info(f"CUDA training preflight passed / GPU 检查通过: {', '.join(gpu_names)}")
    args = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(ACCELERATE_GPU_CONFIG),
        "--num_cpu_threads_per_process",
        str(cpu_threads),
        "--quiet",
        trainer_file,
        "--config_file",
        toml_path,
    ]
    if (
        admission_intent is not None
        and admission_intent.origin == "webui_start"
        and admission_intent.experimental_full_bf16_ack is True
    ):
        # Child trainer argument ONLY, after its own --config_file. Not an
        # Accelerate parent option, TOML value, environment or task metadata.
        args.append("--experimental_full_bf16_ack")

    env = os.environ.copy()
    env["ACCELERATE_DISABLE_RICH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if selected_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {selected_gpu_ids}")
        if len(selected_gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(selected_gpu_ids))]
            if sys.platform == "win32":
                env["USE_LIBUV"] = "0"
                args[3:3] = ["--rdzv_backend", "c10d"]

    task = tm.create_task(
        args,
        env,
        metadata={
            "page_train_type": page_train_type,
            "run_id": run_id,
            "toml_path": toml_path,
        },
    )
    if not task:
        active = tm.find_active_task()
        data = {"active_task_id": active.task_id} if active else None
        return APIResponse(status="error", message="已有训练任务正在启动或运行，不能重复 Start。", data=data)

    def _run():
        try:
            task.execute()
            result = task.communicate()
            if result.returncode != 0:
                log.error("Training failed / 训练失败")
            else:
                log.info("Training finished / 训练完成")
        except Exception as exc:
            log.error(f"An error occurred when training / 训练出现致命错误: {exc}")

    asyncio.create_task(asyncio.to_thread(_run))
    return APIResponse(
        status="success",
        message=f"Training started / 训练开始 ID: {task.task_id}",
        data={"task_id": task.task_id, "run_id": run_id, "page_train_type": page_train_type},
    )
