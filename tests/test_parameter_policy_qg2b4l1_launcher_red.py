"""QG-2-B4-L1: exact dynamic Launcher transport Tests-only RED.

Execute the checked-in run_prepared_train AST with inert GPU/task/async
collaborators. This does not launch Accelerate or import any GPU runtime.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import unittest

from mikazuki.training_config import RequestAdmissionIntent

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "mikazuki" / "training_launcher.py"
MARKER = "QG-2-B4-L1 LAUNCHER TRANSPORT CONTRACT ABSENT"
ACK = "--experimental_full_bf16_ack"
_MISSING = object()
TOML = "config/autosave/run-1.toml"
TRAINER = "scripts/dev/flux_train_network.py"


def _launcher_node():
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"), filename=str(LAUNCHER))
    return next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "run_prepared_train"
    )


class _FakeConfigPath:
    def __init__(self, exists=True):
        self.exists = exists

    def is_file(self):
        return self.exists

    def __str__(self):
        return "config/accelerate-gpu.yaml"


class _FakeTaskManager:
    def __init__(self, exists=True):
        self.exists = exists
        self.submissions = []

    def create_task(self, command, env, metadata=None):
        self.submissions.append(
            (list(command), dict(env), dict(metadata or {}))
        )
        return SimpleNamespace(task_id="task-1") if self.exists else None

    def find_active_task(self):
        return SimpleNamespace(task_id="already-active")


class _LauncherCase:
    def __init__(
        self, *, platform="linux", gpu_error=None,
        config_present=True, task_available=True,
    ):
        self.platform = platform
        self.gpu_error = gpu_error
        self.gpu_calls = []
        self.scheduled = []
        self.task_manager = _FakeTaskManager(exists=task_available)
        self.config = _FakeConfigPath(exists=config_present)
        self.os = SimpleNamespace(environ={"BASE_ENV": "retained"})
        self.sys = SimpleNamespace(executable="/usr/bin/python3", platform=platform)
        self.log = SimpleNamespace(info=lambda *_: None, error=lambda *_: None)
        self.asyncio = SimpleNamespace(
            # Do not call _run, task.execute, task.communicate, asyncio.to_thread,
            # subprocess.Popen, or the actual event loop.
            to_thread=lambda function: ("DEFERRED", function),
            create_task=lambda awaitable: self.scheduled.append(awaitable),
        )
        self.fn = self._load_real_launcher()

    def _gpu(self, selection):
        self.gpu_calls.append(selection)
        if self.gpu_error:
            raise RuntimeError(self.gpu_error)
        selected = [str(v) for v in (selection or [])]
        return selected, [f"Fake GPU {id}" for id in selected] or ["Fake GPU"]

    def _load_real_launcher(self):
        node = copy.deepcopy(_launcher_node())
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                node,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        globals_ = {
            "APIResponse": SimpleNamespace,
            "ACCELERATE_GPU_CONFIG": self.config,
            "validate_cuda_training_runtime": self._gpu,
            "tm": self.task_manager,
            "os": self.os,
            "sys": self.sys,
            "log": self.log,
            "asyncio": self.asyncio,
        }
        exec(compile(module, str(LAUNCHER), "exec"), globals_)
        return globals_["run_prepared_train"]

    def invoke(self, *, intent=_MISSING, gpu_ids=None, run_id="run-1"):
        kwargs = {"page_train_type": "flux-lora", "run_id": run_id}
        if intent is not _MISSING:
            kwargs["admission_intent"] = intent
        return self.fn(
            TOML, TRAINER, ["0"] if gpu_ids is None else gpu_ids, 2, **kwargs
        )

    @property
    def submissions(self):
        return self.task_manager.submissions


class QG2B4L1LauncherRed(unittest.TestCase):
    def missing(self, detail):
        self.fail(f"{MARKER}: {detail}")

    def launch_with_intent(self, fixture, intent, detail, **kw):
        try:
            return fixture.invoke(intent=intent, **kw)
        except TypeError as exc:
            if "admission_intent" in str(exc) and "unexpected keyword" in str(exc):
                self.missing(detail + ": " + str(exc))
            raise

    def test_b4l101_public_launcher_typed_kwonly_neutral_default(self):
        node = _launcher_node()
        keyword_only = [arg.arg for arg in node.args.kwonlyargs]
        if "admission_intent" not in keyword_only:
            self.missing("Launcher does not accept one-job admission_intent")
        position = keyword_only.index("admission_intent")
        self.assertIsInstance(node.args.kw_defaults[position], ast.Constant)
        self.assertIsNone(node.args.kw_defaults[position].value)
        self.assertIn("RequestAdmissionIntent", ast.unparse(node.args.kwonlyargs[position].annotation))

    def test_b4l102_positive_typed_start_appends_one_child_cli_flag(self):
        fixture = _LauncherCase()
        result = self.launch_with_intent(
            fixture, RequestAdmissionIntent(True, "webui_start"),
            "typed positive Start must reach the child trainer only",
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(len(fixture.submissions), 1)
        argv, _, _ = fixture.submissions[0]
        self.assertEqual(argv.count(ACK), 1)
        self.assertEqual(argv[-4:], [TRAINER, "--config_file", TOML, ACK])
        self.assertEqual(len(fixture.scheduled), 1)

    def test_b4l103_typed_false_preserves_exact_standard_argv(self):
        baseline = _LauncherCase()
        original = baseline.invoke()
        self.assertEqual(original.status, "success")
        fixture = _LauncherCase()
        response = self.launch_with_intent(
            fixture, RequestAdmissionIntent(False, "default"),
            "neutral typed Request intent must retain legacy Standard argv",
        )
        self.assertEqual(response.status, "success")
        self.assertEqual(len(fixture.submissions), 1)
        self.assertEqual(fixture.submissions, baseline.submissions)
        self.assertNotIn(ACK, fixture.submissions[0][0])

    def _verify_bad_intents_before_gpu(self, values):
        # Aggregate every variant into ONE test-method failure, so the strict
        # CI can freeze eight exact RED identities without losing coverage.
        wrong = []
        for value in values:
            fixture = _LauncherCase()
            descriptor = repr(value)
            try:
                fixture.invoke(intent=value)
            except (TypeError, ValueError) as exc:
                if "unexpected keyword argument 'admission_intent'" in str(exc):
                    wrong.append(f"{descriptor}: Launcher lacks admission_intent parameter")
                elif "intent" not in str(exc).lower():
                    wrong.append(f"{descriptor}: rejection did not identify invalid intent: {exc}")
            except Exception as exc:
                wrong.append(f"{descriptor}: wrong exception {type(exc).__name__}: {exc}")
            else:
                wrong.append(f"{descriptor}: improperly accepted invalid intent")
            if fixture.gpu_calls or fixture.submissions or fixture.scheduled:
                wrong.append(f"{descriptor}: invoked GPU/task/async before rejecting intent")
        if wrong:
            self.missing("; ".join(wrong))

    def test_b4l104_structural_impostors_fail_before_gpu_or_task(self):
        self._verify_bad_intents_before_gpu((
            True, {"experimental_full_bf16_ack": True, "origin": "webui_start"},
            SimpleNamespace(experimental_full_bf16_ack=True, origin="webui_start"),
            object(),
        ))

    def test_b4l105_corrupted_typed_origin_or_bool_fail_before_side_effects(self):
        wrong_origin = RequestAdmissionIntent(True, "webui_start")
        object.__setattr__(wrong_origin, "origin", "default")
        wrong_boolean = RequestAdmissionIntent(True, "webui_start")
        object.__setattr__(wrong_boolean, "experimental_full_bf16_ack", 1)
        self._verify_bad_intents_before_gpu((wrong_origin, wrong_boolean))

    def test_b4l106_positive_then_absent_or_false_never_replays_or_persists(self):
        positive = _LauncherCase()
        self.launch_with_intent(
            positive, RequestAdmissionIntent(True, "webui_start"),
            "positive intent is a single invocation, not stored state",
        )
        argv, env, metadata = positive.submissions[0]
        self.assertEqual(argv.count(ACK), 1)
        self.assertFalse(any("admission" in str(key).lower() or "bf16_ack" in str(key).lower()
                             for key in env))
        self.assertFalse(any("admission" in str(key).lower() or "bf16_ack" in str(key).lower()
                             for key in metadata))
        self.assertNotIn("admission_intent", str(metadata))
        self.assertNotIn("experimental_full_bf16_ack", str(env))
        for intent in (_MISSING, RequestAdmissionIntent(False, "webui_start")):
            neutral = _LauncherCase()
            if intent is _MISSING:
                neutral.invoke(run_id="run-2")
            else:
                self.launch_with_intent(
                    neutral, intent, "following neutral invocation must not replay",
                    run_id="run-2",
                )
            self.assertEqual(len(neutral.submissions), 1)
            self.assertNotIn(ACK, neutral.submissions[0][0])
            self.assertNotIn("admission_intent", str(neutral.submissions[0][2]))

    def test_b4l107_multigpu_windows_keeps_parent_flags_and_child_position(self):
        fixture = _LauncherCase(platform="win32")
        self.launch_with_intent(
            fixture, RequestAdmissionIntent(True, "webui_start"),
            "multi-GPU Windows requires flag after trainer not parent flags",
            gpu_ids=["0", "1"],
        )
        argv, env, metadata = fixture.submissions[0]
        self.assertEqual(argv.count(ACK), 1)
        self.assertEqual(argv[-4:], [TRAINER, "--config_file", TOML, ACK])
        trainer_index = argv.index(TRAINER)
        for token in ("--multi_gpu", "--num_processes", "--rdzv_backend", "c10d"):
            self.assertIn(token, argv[:trainer_index])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["USE_LIBUV"], "0")
        self.assertEqual(metadata["run_id"], "run-1")

    def test_b4l108_busy_positive_start_does_not_schedule_or_taint_retry(self):
        busy = _LauncherCase(task_available=False)
        result = self.launch_with_intent(
            busy, RequestAdmissionIntent(True, "webui_start"),
            "busy launcher must neither schedule nor leak intent",
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.data, {"active_task_id": "already-active"})
        self.assertEqual(len(busy.submissions), 1)
        self.assertEqual(busy.submissions[0][0].count(ACK), 1)
        self.assertEqual(busy.scheduled, [])
        retry = _LauncherCase()
        retry.invoke(run_id="retry-2")
        self.assertEqual(len(retry.submissions), 1)
        self.assertNotIn(ACK, retry.submissions[0][0])


class QG2B4L1LauncherBaselines(unittest.TestCase):
    def test_b4l901_existing_standard_launch_argv_env_metadata(self):
        fixture = _LauncherCase()
        response = fixture.invoke()
        self.assertEqual(response.status, "success")
        self.assertEqual(len(fixture.submissions), 1)
        argv, env, metadata = fixture.submissions[0]
        self.assertEqual(
            argv, [
                "/usr/bin/python3", "-m", "accelerate.commands.launch",
                "--config_file", "config/accelerate-gpu.yaml",
                "--num_cpu_threads_per_process", "2", "--quiet",
                TRAINER, "--config_file", TOML,
            ],
        )
        self.assertNotIn(ACK, argv)
        self.assertEqual(env["BASE_ENV"], "retained")
        self.assertEqual(env["ACCELERATE_DISABLE_RICH"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(metadata, {
            "page_train_type": "flux-lora", "run_id": "run-1",
            "toml_path": TOML,
        })
        self.assertEqual(len(fixture.scheduled), 1)

    def test_b4l902_existing_gpu_rejection_has_no_task_or_env_side_effect(self):
        fixture = _LauncherCase(gpu_error="DTS fake CUDA not available")
        response = fixture.invoke()
        self.assertEqual(response.status, "error")
        self.assertIn("fake CUDA", response.message)
        self.assertEqual(len(fixture.gpu_calls), 1)
        self.assertEqual(fixture.submissions, [])
        self.assertEqual(fixture.scheduled, [])
        self.assertEqual(fixture.os.environ, {"BASE_ENV": "retained"})

    def test_b4l903_missing_accelerate_gpu_yaml_fails_closed(self):
        fixture = _LauncherCase(config_present=False)
        response = fixture.invoke()
        self.assertEqual(response.status, "error")
        self.assertIn("Missing DTS GPU Accelerate config", response.message)
        self.assertEqual(len(fixture.gpu_calls), 1)
        self.assertEqual(fixture.submissions, [])
        self.assertEqual(fixture.scheduled, [])

    def test_b4l904_existing_busy_task_returns_active_id_without_schedule(self):
        fixture = _LauncherCase(task_available=False)
        response = fixture.invoke()
        self.assertEqual(response.status, "error")
        self.assertEqual(response.data, {"active_task_id": "already-active"})
        self.assertEqual(len(fixture.submissions), 1)
        self.assertEqual(fixture.scheduled, [])

    def test_b4l905_legacy_windows_multi_gpu_parent_argv_is_preserved(self):
        fixture = _LauncherCase(platform="win32")
        response = fixture.invoke(gpu_ids=["0", "1"])
        self.assertEqual(response.status, "success")
        argv, env, _metadata = fixture.submissions[0]
        self.assertEqual(argv[-3:], [TRAINER, "--config_file", TOML])
        before_trainer = argv[:argv.index(TRAINER)]
        self.assertEqual(before_trainer.count("--multi_gpu"), 1)
        self.assertEqual(before_trainer.count("--rdzv_backend"), 1)
        self.assertIn("c10d", before_trainer)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["USE_LIBUV"], "0")
        self.assertNotIn(ACK, argv)

    def test_b4l906_actual_immutable_request_intent_requires_explicit_webui_origin(self):
        positive = RequestAdmissionIntent(True, "webui_start")
        neutral = RequestAdmissionIntent()
        self.assertIs(positive.experimental_full_bf16_ack, True)
        self.assertEqual(positive.origin, "webui_start")
        self.assertIs(neutral.experimental_full_bf16_ack, False)
        self.assertEqual(neutral.origin, "default")
        with self.assertRaises(FrozenInstanceError):
            positive.origin = "default"
        for kwargs in (
            {"experimental_full_bf16_ack": True},
            {"experimental_full_bf16_ack": "true", "origin": "webui_start"},
            {"experimental_full_bf16_ack": 1, "origin": "webui_start"},
            {"experimental_full_bf16_ack": True, "origin": "untrusted"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RequestAdmissionIntent(**kwargs)


if __name__ == "__main__":
    unittest.main()
