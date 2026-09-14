"""Launch a configuration that has already passed effective-config preparation."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm


def run_prepared_train(
    toml_path: str,
    trainer_file: str,
    gpu_ids: Optional[list] = None,
    cpu_threads: Optional[int] = 2,
    *,
    page_train_type: str | None = None,
    run_id: str | None = None,
):
    """Start Accelerate without mutating trainer configuration again."""
    log.info(f"Training started with effective config / 使用最终配置启动训练: {toml_path}")
    args = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        str(cpu_threads),
        "--quiet",
        trainer_file,
        "--config_file",
        toml_path,
    ]

    env = os.environ.copy()
    env["ACCELERATE_DISABLE_RICH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")
        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
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
