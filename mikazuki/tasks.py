import os
import subprocess
import threading
import uuid
from enum import Enum
from subprocess import CompletedProcess, TimeoutExpired
from typing import Dict, List

import psutil

from mikazuki.log import log

try:
    import msvcrt  # noqa: F401
    import _winapi  # noqa: F401
    _mswindows = True
except ModuleNotFoundError:
    _mswindows = False


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


class TaskStatus(Enum):
    CREATED = 0
    RUNNING = 1
    FINISHED = 2
    TERMINATED = 3


ACTIVE_STATUSES = {TaskStatus.CREATED, TaskStatus.RUNNING}


class Task:
    def __init__(self, task_id, command, environ=None, metadata=None):
        self.task_id = task_id
        self.lock = threading.Lock()
        self.command = command
        self.status = TaskStatus.CREATED
        self.environ = environ or os.environ
        self.metadata = dict(metadata or {})
        self.process = None

    def communicate(self, input=None, timeout=None):
        if self.process is None:
            raise RuntimeError("Task process has not been started")
        try:
            stdout, stderr = self.process.communicate(input, timeout=timeout)
        except TimeoutExpired as exc:
            self.process.kill()
            if _mswindows:
                exc.stdout, exc.stderr = self.process.communicate()
            else:
                self.process.wait()
            raise
        except Exception:
            self.process.kill()
            raise
        retcode = self.process.poll()
        with self.lock:
            if self.status != TaskStatus.TERMINATED:
                self.status = TaskStatus.FINISHED
        return CompletedProcess(self.process.args, retcode, stdout, stderr)

    def wait(self):
        if self.process is None:
            return
        self.process.wait()
        with self.lock:
            if self.status != TaskStatus.TERMINATED:
                self.status = TaskStatus.FINISHED

    def execute(self):
        with self.lock:
            if self.status == TaskStatus.TERMINATED:
                return
            if self.status != TaskStatus.CREATED:
                raise RuntimeError(f"Task {self.task_id} is already {self.status.name}")
            self.process = subprocess.Popen(self.command, env=self.environ)
            self.status = TaskStatus.RUNNING

    def terminate(self):
        with self.lock:
            if self.status not in ACTIVE_STATUSES:
                return
            process = self.process
            self.status = TaskStatus.TERMINATED
        if process is None:
            return
        try:
            kill_proc_tree(process.pid, False)
        except Exception as exc:
            log.error(f"Error when killing process: {exc}")


class TaskManager:
    def __init__(self, max_concurrent=1) -> None:
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()

    def find_active_task(self, page_train_type: str | None = None):
        with self.lock:
            for task in self.tasks.values():
                if task.status not in ACTIVE_STATUSES:
                    continue
                if page_train_type is not None and task.metadata.get("page_train_type") != page_train_type:
                    continue
                return task
        return None

    def create_task(self, command: List[str], environ, metadata=None):
        with self.lock:
            active = [task for task in self.tasks.values() if task.status in ACTIVE_STATUSES]
            if len(active) >= self.max_concurrent:
                log.error(
                    "Unable to create a task because there are already "
                    f"{len(active)} tasks starting/running, reaching the maximum concurrent limit. / "
                    f"无法创建任务，因为已有 {len(active)} 个任务正在启动或运行。"
                )
                return None
            task_id = str(uuid.uuid4())
            task = Task(task_id=task_id, command=command, environ=environ, metadata=metadata)
            self.tasks[task_id] = task
        log.info(f"Task {task_id} created")
        return task

    def add_task(self, task_id: str, task: Task):
        with self.lock:
            self.tasks[task_id] = task

    def terminate_task(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
        if task:
            task.terminate()

    def wait_for_process(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
        if task:
            task.wait()

    def dump(self) -> List[Dict]:
        with self.lock:
            tasks = list(self.tasks.values())
        return [{"id": task.task_id, "status": task.status.name, **task.metadata} for task in tasks]


tm = TaskManager()
