"""QG-2-B4-A1: actual API-to-Launcher one-shot Intent RED-to-GREEN suite.

The original Tests-only RED exposed the missing API keyword; the unchanged
eight target tests now require GREEN. Real API and Launcher AST are executed
with inert Host collaborators. Synthetic positive preparation proves only
transport; experimental candidate Start remains held in the real gates.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from mikazuki.training_config import PreparedTrainingConfig, RequestAdmissionIntent
from tests.test_parameter_policy_qg2a2_api_red import (
    _Request, _api_function, _helper_function, _response, _source_function,
)
from tests.test_parameter_policy_qg2b4l1_launcher_red import (
    ACK, _LauncherCase,
)

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "mikazuki" / "app" / "training_api.py"
MARKER = "QG-2-B4-A1 API HANDOFF CONTRACT ABSENT"
ABSENT = object()


class _ApiRunCase:
    """Actual create_toml_file() with observable, no-I/O host collaborators."""

    def __init__(
        self, *, component=True, intent_mode="mirror", fixed_intent=None,
        fail_assets=False, image_count=0, real_launcher=None,
    ):
        self.component = component
        self.intent_mode = intent_mode
        self.fixed_intent = fixed_intent
        self.fail_assets = fail_assets
        self.image_count = image_count
        self.real_launcher = real_launcher
        self.events = []
        self.prepared = []
        self.request_intents = []
        self.launch_calls = []
        self.toml_writes = []
        self.sidecar_writes = []
        self.handler = _api_function(
            "create_toml_file",
            _decode_training_admission_request=_helper_function(),
            prepare_request_config=self._prepare,
            validate_prepared_config=self._validate_assets,
            prepare_runtime_trainer=self._stage_trainer,
            validate_prepared_trainer=self._validate_trainer,
            materialize_anima_launch_side_effects=self._stage_anima,
            materialize_sidecars=self._write_sidecars,
            _write_text_atomic=self._write_toml,
            run_prepared_train=self._launch,
            train_utils=SimpleNamespace(get_total_images=self._images),
            toml=SimpleNamespace(dumps=lambda obj: json.dumps(obj, sort_keys=True)),
            log=SimpleNamespace(error=lambda *args: None),
        )

    def _prepare(self, config, page_type, **kwargs):
        self.events.append(("prepare", kwargs["launch"], kwargs["admission_purpose"]))
        assert kwargs["admission_purpose"] == "start"
        request_intent = kwargs["admission_intent"]
        assert type(request_intent) is RequestAdmissionIntent
        self.request_intents.append(request_intent)
        if self.intent_mode == "mirror":
            intent = request_intent
        elif self.intent_mode == "fixed":
            intent = self.fixed_intent
        else:
            raise AssertionError("Unknown test preparation mode")
        assert type(intent) is RequestAdmissionIntent
        effective = (
            {
                "parameter_policy_config": "config/autosave/policy.json",
                "train_data_dir": "test-dataset",
                "output_name": "component-case",
            }
            if self.component else {"output_name": "standard-case"}
        )
        prepared = PreparedTrainingConfig(
            train_type="sd-lora",
            trainer_file="scripts/stable/train_network.py",
            config=effective,
            gpu_ids=["0"],
            sidecars={"config/autosave/policy.json": "{\"version\":1}\n"}
            if self.component else {"config/autosave/prompt.txt": "hello"},
            admission_intent=intent,
        )
        self.prepared.append(prepared)
        return prepared

    def _validate_assets(self, prepared, require_assets, **kwargs):
        self.events.append(("assets", require_assets, kwargs))
        if self.fail_assets:
            raise ValueError("deliberately refused asset preflight")

    def _stage_trainer(self, prepared):
        self.events.append(("stage",))

    def _validate_trainer(self, prepared):
        self.events.append(("trainer",))

    def _stage_anima(self, config, train_type):
        self.events.append(("anima", train_type))

    def _write_sidecars(self, sidecars):
        self.events.append(("sidecars",))
        self.sidecar_writes.append(dict(sidecars))

    def _write_toml(self, path, contents):
        self.events.append(("toml",))
        self.toml_writes.append((path, contents))

    def _images(self, path):
        return range(self.image_count)

    def _launch(self, *args, **kwargs):
        self.events.append(("launcher",))
        self.launch_calls.append((args, dict(kwargs)))
        if self.real_launcher is not None:
            return self.real_launcher.fn(*args, **kwargs)
        return _response("success", "started", {"task_id": "fake-task"})

    def start(self, *, ack=ABSENT, payload=None):
        if payload is None:
            payload = {
                "train_type": "sd-lora",
                "config": {"output_name": "one-request"},
            }
            if ack is not ABSENT:
                payload["experimental_full_bf16_ack"] = ack
        return asyncio.run(self.handler(_Request(payload)))

    @property
    def steps(self):
        return [event[0] for event in self.events]


class QG2B4A1ApiHandoffRed(unittest.TestCase):
    """Eight original target RED cases now require real API handoff PASS."""

    def _absent(self, detail):
        self.fail(f"{MARKER}: {detail}")

    def _handoff(self, case, index=0):
        self.assertGreater(len(case.launch_calls), index)
        _args, kwargs = case.launch_calls[index]
        if "admission_intent" not in kwargs:
            self._absent(
                "checked-in /run omitted prepared.admission_intent "
                "from run_prepared_train"
            )
        self.assertIs(
            kwargs["admission_intent"], case.prepared[index].admission_intent,
            "API must forward the independently prepared object by identity",
        )
        self.assertIs(type(kwargs["admission_intent"]), RequestAdmissionIntent)
        return kwargs["admission_intent"]

    def test_b4a101_source_has_exact_prepared_intent_handoff_keyword(self):
        node = _source_function("create_toml_file")
        self.assertIsInstance(node, ast.AsyncFunctionDef)
        calls = [
            item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "run_prepared_train"
        ]
        self.assertEqual(len(calls), 1, "Expected one actual /run Launcher call")
        keywords = {
            keyword.arg: keyword.value
            for keyword in calls[0].keywords if keyword.arg is not None
        }
        if "admission_intent" not in keywords:
            self._absent("actual /run Launcher call has no admission_intent keyword")
        self.assertEqual(
            ast.unparse(keywords["admission_intent"]),
            "prepared.admission_intent",
            "API must not use the raw request or construct a substitute intent",
        )

    def test_b4a102_typed_positive_prepared_intent_reaches_launcher_by_identity(self):
        case = _ApiRunCase()
        response = case.start(ack=True)
        self.assertEqual(response.status, "success")
        self.assertEqual(len(case.launch_calls), 1)
        value = self._handoff(case)
        self.assertIs(value.experimental_full_bf16_ack, True)
        self.assertEqual(value.origin, "webui_start")
        self.assertIs(value, case.request_intents[0])

    def test_b4a103_neutral_start_forwards_fresh_typed_default_explicitly(self):
        case = _ApiRunCase()
        response = case.start()
        self.assertEqual(response.status, "success")
        value = self._handoff(case)
        self.assertIs(value.experimental_full_bf16_ack, False)
        self.assertEqual(value.origin, "default")
        self.assertIs(value, case.request_intents[0])

    def test_b4a104_prepared_neutral_downgrade_defeats_positive_raw_start(self):
        neutral = RequestAdmissionIntent()
        case = _ApiRunCase(intent_mode="fixed", fixed_intent=neutral)
        self.assertEqual(case.start(ack=True).status, "success")
        self.assertTrue(case.request_intents[0].experimental_full_bf16_ack)
        value = self._handoff(case)
        self.assertIs(value, neutral)
        self.assertFalse(value.experimental_full_bf16_ack)
        self.assertIsNot(value, case.request_intents[0])

    def test_b4a105_standard_launch_transports_only_neutral_prepared_intent(self):
        case = _ApiRunCase(component=False)
        response = case.start()
        self.assertEqual(response.status, "success")
        value = self._handoff(case)
        self.assertFalse(value.experimental_full_bf16_ack)
        self.assertEqual(
            case.steps, ["prepare", "stage", "assets", "sidecars", "toml", "launcher"],
        )
        self.assertEqual(case.launch_calls[0][0][3], 2)

    def test_b4a106_two_starts_do_not_replay_previous_positive_intent(self):
        case = _ApiRunCase()
        self.assertEqual(case.start(ack=True).status, "success")
        self.assertEqual(case.start().status, "success")
        positive = self._handoff(case, 0)
        neutral = self._handoff(case, 1)
        self.assertTrue(positive.experimental_full_bf16_ack)
        self.assertFalse(neutral.experimental_full_bf16_ack)
        self.assertIsNot(positive, neutral)
        self.assertEqual(len(case.launch_calls), 2)

    def test_b4a107_ack_remains_out_of_toml_sidecars_and_task_metadata(self):
        case = _ApiRunCase(image_count=201)
        self.assertEqual(case.start(ack=True).status, "success")
        value = self._handoff(case)
        self.assertTrue(value.experimental_full_bf16_ack)
        self.assertEqual(case.launch_calls[0][0][3], 8)
        self.assertEqual(case.steps, [
            "prepare", "assets", "stage", "trainer", "anima",
            "sidecars", "toml", "launcher",
        ])
        self.assertEqual(len(case.toml_writes), 1)
        self.assertEqual(len(case.sidecar_writes), 1)
        for item in (
            case.prepared[0].config, case.sidecar_writes[0],
            case.toml_writes[0][1], case.launch_calls[0][1]["run_id"],
        ):
            text = str(item)
            self.assertNotIn("experimental_full_bf16_ack", text)
            self.assertNotIn("admission_intent", text)
            self.assertNotIn("webui_start", text)

    def test_b4a108_real_api_to_real_launcher_child_argv_once_only(self):
        launcher = _LauncherCase()
        case = _ApiRunCase(real_launcher=launcher)
        self.assertEqual(case.start(ack=True).status, "success")
        self._handoff(case)
        self.assertEqual(len(launcher.submissions), 1)
        argv, env, metadata = launcher.submissions[0]
        self.assertEqual(argv.count(ACK), 1)
        self.assertEqual(argv[-1], ACK)
        self.assertEqual(argv[-4:-1], [
            "scripts/stable/train_network.py", "--config_file",
            case.launch_calls[0][0][0],
        ])
        self.assertFalse(any("admission" in key or "bf16_ack" in key for key in env))
        self.assertFalse(any("admission" in key or "bf16_ack" in key for key in metadata))
        self.assertEqual(case.start().status, "success")
        self._handoff(case, 1)
        self.assertEqual(len(launcher.submissions), 2)
        self.assertNotIn(ACK, launcher.submissions[1][0])


class QG2B4A1ApiHandoffBaselines(unittest.TestCase):
    """Six independent, already-working contracts that must remain GREEN."""

    def test_b4a901_actual_decoder_stamps_only_strict_start_boolean(self):
        parse = _helper_function()
        self.assertIsNotNone(parse)
        base = {"train_type": "sd-lora", "config": {}}
        positive = dict(base, experimental_full_bf16_ack=True)
        page, config, intent = parse(json.dumps(positive).encode(), purpose="start")
        self.assertEqual((page, config), ("sd-lora", {}))
        self.assertIs(type(intent), RequestAdmissionIntent)
        self.assertTrue(intent.experimental_full_bf16_ack)
        self.assertEqual(intent.origin, "webui_start")
        for purpose in ("preview", "export"):
            *_, neutral = parse(json.dumps(positive).encode(), purpose=purpose)
            self.assertFalse(neutral.experimental_full_bf16_ack)
        for bad in ("true", 1, None, {}, []):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse(
                        json.dumps(dict(base, experimental_full_bf16_ack=bad)).encode(),
                        purpose="start",
                    )

    def test_b4a902_component_assets_precede_staging_sidecars_and_launch(self):
        case = _ApiRunCase()
        response = case.start()
        self.assertEqual(response.status, "success")
        self.assertEqual(case.steps, [
            "prepare", "assets", "stage", "trainer", "anima",
            "sidecars", "toml", "launcher",
        ])
        self.assertEqual(case.events[1][1], True)
        self.assertFalse(case.events[0][1] is False)
        self.assertEqual(case.events[0][2], "start")
        self.assertEqual(len(case.launch_calls), 1)

    def test_b4a903_standard_stage_precedes_asset_check_without_component_hooks(self):
        case = _ApiRunCase(component=False)
        self.assertEqual(case.start().status, "success")
        self.assertEqual(case.steps, [
            "prepare", "stage", "assets", "sidecars", "toml", "launcher",
        ])
        self.assertEqual(case.events[2][1], True)
        self.assertEqual(len(case.toml_writes), 1)

    def test_b4a904_invalid_api_control_rejected_before_any_side_effect(self):
        cases = (
            {"train_type": "sd-lora", "config": {}, "experimental_full_bf16_ack": 1},
            {"train_type": "sd-lora", "config": {"experimental_full_bf16_ack": True}},
            {"train_type": "sd-lora", "config": {}, "admission_intent": {"origin": "webui_start"}},
            {"full_bf16": True, "experimental_full_bf16_ack": True},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                case = _ApiRunCase()
                response = case.start(payload=payload)
                self.assertEqual(response.status, "fail")
                self.assertEqual(case.events, [])
                self.assertEqual(case.launch_calls, [])

    def test_b4a905_failed_component_asset_preflight_has_no_staging_or_writes(self):
        case = _ApiRunCase(fail_assets=True)
        response = case.start()
        self.assertEqual(response.status, "fail")
        self.assertIn("refused asset", response.message)
        self.assertEqual(case.steps, ["prepare", "assets"])
        self.assertEqual(case.toml_writes, [])
        self.assertEqual(case.sidecar_writes, [])
        self.assertEqual(case.launch_calls, [])

    def test_b4a906_actual_launcher_accepted_positive_and_neutral_argv(self):
        positive = _LauncherCase()
        result = positive.invoke(intent=RequestAdmissionIntent(True, "webui_start"))
        self.assertEqual(result.status, "success")
        self.assertEqual(positive.submissions[0][0].count(ACK), 1)
        self.assertEqual(positive.submissions[0][0][-1], ACK)
        neutral = _LauncherCase()
        result = neutral.invoke(intent=RequestAdmissionIntent())
        self.assertEqual(result.status, "success")
        self.assertNotIn(ACK, neutral.submissions[0][0])


if __name__ == "__main__":
    unittest.main()
