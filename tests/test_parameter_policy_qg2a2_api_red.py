"""QG-2-A2: new API+frontend per-job Admission Tests-only RED, 12+4.

No FastAPI/toml/Pydantic import requirement: extract the actual API AST
function bodies and inject observable Host-only collaborators. New contract
tests first check their *specific absent* source interface; when that interface
is implemented they continue to behavior / safety / ordering assertions.
"""
from __future__ import annotations

import ast
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

from mikazuki.parameter_policy_admission import (
    ComponentAdmission,
    assess_component_training_admission,
)
from mikazuki.parameter_policy_execution import FULL_BF16_BACKEND_QUALIFICATIONS
from mikazuki.training_config import PreparedTrainingConfig, RequestAdmissionIntent
from tests.test_parameter_policy_admission_red import _config, _pending, _policy
from tests.test_parameter_policy_qg2a1_request_red import _make_preparation

ROOT = Path(__file__).resolve().parents[1]
API_FILE = ROOT / "mikazuki" / "app" / "training_api.py"
UI_FILE = ROOT / "mikazuki" / "frontend_training_patch.py"
API = API_FILE.read_text(encoding="utf-8")
UI = UI_FILE.read_text(encoding="utf-8")
TREE = ast.parse(API, filename=str(API_FILE))
MARKER = "QG-2-A2 API/UI CONTRACT ABSENT"

# training_request imports the full FastAPI application transitively. Keep
# minimal Host CI dependency-free by extracting its *actual* decoder AST,
# exactly as the existing tests/test_parameter_policy_request_contract does.
REQUEST_FILE = ROOT / "mikazuki" / "training_request.py"
REQUEST_TREE = ast.parse(
    REQUEST_FILE.read_text(encoding="utf-8"), filename=str(REQUEST_FILE),
)
decoder_node = next(
    node for node in REQUEST_TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "decode_training_request"
)
decoder_module = ast.Module(
    body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")],
        level=0,
    ), decoder_node],
    type_ignores=[],
)
ast.fix_missing_locations(decoder_module)
_decoder_globals = {"json": json}
exec(compile(decoder_module, str(REQUEST_FILE), "exec"), _decoder_globals)
decode_training_request = _decoder_globals["decode_training_request"]


def _source_function(name: str):
    node = next(
        (node for node in TREE.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == name),
        None,
    )
    return node


def _body(name: str) -> str:
    node = _source_function(name)
    return ast.get_source_segment(API, node) if node is not None else ""


def _called_names(name: str) -> set[str]:
    node = _source_function(name)
    if node is None:
        return set()
    return {
        n.func.id for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def _decode_v1(body: bytes, *, purpose: str):
    page, cfg = decode_training_request(body)
    return page, cfg, RequestAdmissionIntent()


def _fake_prepared(component: bool = False, admission=None):
    return PreparedTrainingConfig(
        train_type="sd-lora", trainer_file="trainer.py",
        config=({"parameter_policy_config": "policy.json"} if component else {}),
        sidecars=({"policy.json": "{}"} if component else {}),
        admission=admission,
    )


def _response(status: str, message="", data=None):
    return SimpleNamespace(status=status, message=message, data=data)


class _Request:
    def __init__(self, payload):
        self.payload = payload

    async def body(self):
        return json.dumps(self.payload).encode("utf-8")


def _api_function(name, **overrides):
    """Run real function AST without importing the FastAPI application."""
    import builtins
    import copy
    node = _source_function(name)
    if node is None:
        raise AssertionError(f"Missing existing endpoint: {name}")
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")],
            level=0,
        ), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = {
        "__builtins__": builtins.__dict__,
        "Request": object,
        "json": json,
        "os": os,
        "uuid": uuid,
        "decode_training_request": decode_training_request,
        "_decode_training_admission_request": _decode_v1,
        "RequestAdmissionIntent": RequestAdmissionIntent,
        "ComponentAdmission": ComponentAdmission,
        "APIResponseSuccess": lambda message="", data=None: _response(
            "success", message, data,
        ),
        "APIResponseFail": lambda message="", data=None: _response(
            "fail", message, data,
        ),
        "log": SimpleNamespace(error=lambda value: None),
        "prepare_request_config": lambda *a, **kw: _fake_prepared(),
        "validate_prepared_config": lambda *a, **kw: None,
        "_prepared_payload": lambda prepared: {"toml": "", "bundle": "{}"},
        "materialize_sidecars": lambda *a: None,
        "prepare_runtime_trainer": lambda *a: None,
        "validate_prepared_trainer": lambda *a: None,
        "materialize_anima_launch_side_effects": lambda *a: None,
        "_write_text_atomic": lambda *a: None,
        "run_prepared_train": lambda *a, **kw: _response(
            "success", "started", {"task_id": "test"},
        ),
        "train_utils": SimpleNamespace(get_total_images=lambda p: []),
        "toml": SimpleNamespace(dumps=lambda data: json.dumps(data, sort_keys=True)),
        "_prepared_parameter_policy_preview": lambda prepared: None,
    }
    env.update(overrides)
    exec(compile(module, str(API_FILE), "exec"), env)
    return env[name]


def _helper_function():
    node = _source_function("_decode_training_admission_request")
    if node is None:
        return None
    import copy
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")],
            level=0,
        ), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = {
        "json": json,
        "decode_training_request": decode_training_request,
        "RequestAdmissionIntent": RequestAdmissionIntent,
    }
    exec(compile(module, str(API_FILE), "exec"), env)
    return env[node.name]


class QG2A2TargetedRed(unittest.TestCase):
    def _need(self, condition, detail: str):
        if not condition:
            self.fail(f"{MARKER}: {detail}")

    def _helper(self):
        self._need(
            _source_function("_decode_training_admission_request") is not None,
            "dedicated API request control-plane parser is absent",
        )
        return _helper_function()

    def _api_wired(self, endpoint: str, *, purpose: str):
        source = _body(endpoint)
        self._need(
            "_decode_training_admission_request" in source
            and "prepare_request_config" in source
            and "admission_purpose" in source
            and f'"{purpose}"' in source,
            f"{endpoint} lacks typed, explicit {purpose} Admission boundary",
        )

    def test_a201_dedicated_typed_envelope_parser_and_legacy_decoder(self):
        self._helper()
        self.assertEqual(
            decode_training_request(b'{"train_type":"sd-lora","config":{"x":2}}'),
            ("sd-lora", {"x": 2}),
        )
        self.assertEqual(
            decode_training_request(b'{"x":2}'), (None, {"x": 2}),
        )

    def test_a202_nested_start_ack_exact_bool_and_server_owned_origin(self):
        parse = self._helper()
        cases = (
            (True, True, "webui_start"),
            (False, False, "default"),
        )
        for val, expected, origin in cases:
            raw = {"train_type": "sd-lora", "config": {"full_bf16": True},
                   "experimental_full_bf16_ack": val}
            page, config, intent = parse(
                json.dumps(raw).encode(), purpose="start",
            )
            self.assertEqual(page, "sd-lora")
            self.assertEqual(config, {"full_bf16": True})
            self.assertIs(type(intent), RequestAdmissionIntent)
            self.assertIs(intent.experimental_full_bf16_ack, expected)
            self.assertEqual(intent.origin, origin)
        _, _, absent = parse(
            b'{"train_type":"sd-lora","config":{}}', purpose="start",
        )
        self.assertFalse(absent.experimental_full_bf16_ack)

    def test_a203_control_plane_forgery_and_pseudo_bools_are_rejected(self):
        parse = self._helper()
        invalid = [
            {"train_type": "sd-lora", "config": {}, "experimental_full_bf16_ack": "true"},
            {"train_type": "sd-lora", "config": {}, "experimental_full_bf16_ack": 1},
            {"train_type": "sd-lora", "config": {}, "experimental_full_bf16_ack": None},
            {"train_type": "sd-lora", "config": {}, "experimental_full_bf16_ack": {}},
            {"train_type": "sd-lora", "config": {"experimental_full_bf16_ack": True}},
            {"full_bf16": True, "experimental_full_bf16_ack": True},
            {"train_type": "sd-lora", "config": "not a dictionary",
             "experimental_full_bf16_ack": True},
        ]
        for raw in invalid:
            with self.assertRaises(ValueError):
                parse(json.dumps(raw).encode(), purpose="start")
        page, config, intent = parse(
            b'{"train_type":"sd-lora","config":{},"can_start":true,'
            b'"host_admissible":true}',
            purpose="start",
        )
        self.assertFalse(intent.experimental_full_bf16_ack)
        self.assertEqual(config, {})

    def test_a204_preview_is_neutral_typed_model_free_and_never_writes(self):
        self._api_wired("preview_training_config", purpose="preview")
        events = []
        endpoint = _api_function(
            "preview_training_config",
            _decode_training_admission_request=lambda body, *, purpose: (
                decode_training_request(body)[0],
                decode_training_request(body)[1],
                RequestAdmissionIntent(),
            ),
            prepare_request_config=lambda config, page, **kw: (
                events.append(("prepare", kw)) or _fake_prepared()
            ),
            validate_prepared_config=lambda prepared, paths: events.append(
                ("validate", paths),
            ),
            _prepared_payload=lambda prepared: (
                events.append(("payload", None)) or {"toml": "x"}
            ),
            materialize_sidecars=lambda _: events.append(("WRITE", None)),
        )
        result = asyncio.run(endpoint(_Request({
            "train_type": "sd-lora", "config": {},
            "experimental_full_bf16_ack": True,
        })))
        self.assertEqual(result.status, "success")
        self.assertEqual([x[0] for x in events], ["prepare", "validate", "payload"])
        self.assertIs(events[0][1]["launch"], False)
        self.assertEqual(events[0][1]["admission_purpose"], "preview")
        self.assertFalse(events[0][1]["admission_intent"].experimental_full_bf16_ack)

    def test_a205_export_has_export_purpose_and_preserves_sidecar_order(self):
        self._api_wired("export_training_config", purpose="export")
        events = []
        prepared = _fake_prepared()
        prepared.sidecars = {"config/autosave/prompts/content.txt": "hello"}
        endpoint = _api_function(
            "export_training_config",
            prepare_request_config=lambda config, page, **kw: (
                events.append(("prepare", kw)) or prepared
            ),
            validate_prepared_config=lambda prepared, paths: events.append(
                ("validate", paths),
            ),
            _prepared_payload=lambda prepared: (
                events.append(("payload", None)) or {"bundle": "{}"}
            ),
            materialize_sidecars=lambda sc: events.append(("write", dict(sc))),
        )
        result = asyncio.run(endpoint(_Request({
            "train_type": "sd-lora", "config": {},
        })))
        self.assertEqual(result.status, "success")
        self.assertEqual(
            [e[0] for e in events], ["prepare", "validate", "payload", "write"],
        )
        self.assertEqual(events[0][1]["admission_purpose"], "export")
        self.assertFalse(events[0][1]["admission_intent"].experimental_full_bf16_ack)

    def test_a206_payload_includes_real_first_party_typed_diagnostics(self):
        payload_source = _body("_prepared_payload")
        self._need(
            '"admission"' in payload_source and
            "prepared.admission" in payload_source,
            "public payload lacks separate trusted typed Admission",
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {
            "sd-lora": _pending(),
        }):
            admission = assess_component_training_admission(
                _policy(), train_type="sd-lora",
                effective_config=_config(), purpose="preview",
            )
        prepared = _fake_prepared(True, admission)
        fn = _api_function("_prepared_payload")
        result = fn(prepared)
        self.assertEqual(result["admission"]["status"], "ack_required")
        self.assertEqual(result["admission"]["evidence_tier"], "unverified")
        self.assertTrue(result["admission"]["requires_experimental_ack"])
        self.assertFalse(result["admission"]["host_admissible"])
        self.assertTrue(result["admission"]["warnings"])
        self.assertNotIn("can_start", result)

    def test_a207_payload_does_not_serialize_ack_or_promote_host_only_ready(self):
        self._need(
            "prepared.admission" in _body("_prepared_payload"),
            "typed API diagnostics missing; unable to verify boundary",
        )
        prepared = _fake_prepared(True)
        prepared.admission_intent = RequestAdmissionIntent(
            True, "webui_start",
        )
        prepared.runtime_blockers = ["GPU blocked"]
        prepared.admission = assess_component_training_admission(
            _policy(), train_type="sd-lora",
            effective_config=_config(), purpose="preview",
        )
        fn = _api_function(
            "_prepared_payload",
            _prepared_parameter_policy_preview=lambda obj: {
                "version": 1, "train_type": "sd-lora",
                "profiles": [], "components": [],
                "runtime_ready": False, "runtime_blockers": ["GPU blocked"],
            },
        )
        result = fn(prepared)
        self.assertTrue(result["admission"]["host_admissible"])
        self.assertFalse(result["runtime_ready"])
        for field in ("effective_config", "toml", "bundle", "sidecars"):
            self.assertNotIn("experimental_full_bf16_ack",
                             json.dumps(result[field]))
            self.assertNotIn("admission_intent", json.dumps(result[field]))
        self.assertNotIn("can_start", result)

    def test_a208_start_uses_one_job_typed_intent_before_writes(self):
        self._api_wired("create_toml_file", purpose="start")
        self._need(
            "admission_intent" in _body("create_toml_file"),
            "Start lacks explicit typed one-job intention transport",
        )
        records = []
        endpoint = _api_function(
            "create_toml_file",
            _decode_training_admission_request=lambda body, *, purpose: (
                decode_training_request(body)[0],
                decode_training_request(body)[1],
                RequestAdmissionIntent(True, "webui_start"),
            ),
            prepare_request_config=lambda config, page, **kw: (
                records.append(("prepare", kw)) or
                (_ for _ in ()).throw(ValueError("stage-held candidate"))
            ),
            validate_prepared_config=lambda *a, **kw: records.append(("assets", kw)),
            materialize_sidecars=lambda s: records.append(("write", s)),
            run_prepared_train=lambda *a, **kw: records.append(("LAUNCH", kw)),
        )
        result = asyncio.run(endpoint(_Request({
            "train_type": "sd-lora", "config": {},
            "experimental_full_bf16_ack": True,
        })))
        self.assertEqual(result.status, "fail")
        self.assertEqual([i[0] for i in records], ["prepare"])
        self.assertTrue(records[0][1]["admission_intent"].experimental_full_bf16_ack)
        self.assertEqual(records[0][1]["admission_purpose"], "start")

    def test_a209_true_candidate_still_stops_before_assets_or_launch(self):
        self._api_wired("create_toml_file", purpose="start")
        records = []
        actual_prepare = _make_preparation()
        endpoint = _api_function(
            "create_toml_file",
            _decode_training_admission_request=lambda body, *, purpose: (
                decode_training_request(body)[0],
                decode_training_request(body)[1],
                RequestAdmissionIntent(True, "webui_start"),
            ),
            prepare_request_config=lambda c, p, **kw: actual_prepare(
                c, p, launch=kw["launch"],
                admission_intent=kw["admission_intent"],
                admission_purpose=kw["admission_purpose"],
            ),
            validate_prepared_config=lambda *args, **kw: records.append("assets"),
            prepare_runtime_trainer=lambda *args: records.append("trainer"),
            materialize_sidecars=lambda *args: records.append("sidecar"),
            _write_text_atomic=lambda *args: records.append("toml"),
            run_prepared_train=lambda *args, **kw: records.append("launch"),
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS,
                        {"sd-lora": _pending()}):
            result = asyncio.run(endpoint(_Request({
                "train_type": "sd-lora", "config": {},
                "experimental_full_bf16_ack": True,
            })))
        self.assertEqual(result.status, "fail")
        self.assertIn("QG-2-A2", result.message)
        self.assertEqual(records, [])

    def test_a210_qualified_d1_keeps_assets_then_staging_then_launch_order(self):
        self._api_wired("create_toml_file", purpose="start")
        stages = []
        prepared = _fake_prepared(component=True)
        endpoint = _api_function(
            "create_toml_file",
            prepare_request_config=lambda c,p,**kw: (
                stages.append("prepare") or prepared
            ),
            validate_prepared_config=lambda state, paths, **kw: stages.append(
                "assets" if paths and not kw.get("check_trainer", True)
                else "wrong-assets",
            ),
            prepare_runtime_trainer=lambda state: stages.append("trainer"),
            validate_prepared_trainer=lambda state: stages.append("trainer-check"),
            materialize_anima_launch_side_effects=lambda *a: stages.append("anima"),
            materialize_sidecars=lambda s: stages.append("sidecars"),
            _write_text_atomic=lambda *a: stages.append("toml"),
            run_prepared_train=lambda *a, **kw: (
                stages.append("launch") or _response("success", "started", {})
            ),
        )
        result = asyncio.run(endpoint(_Request({
            "train_type": "sd-lora", "config": {},
        })))
        self.assertEqual(result.status, "success")
        self.assertEqual(stages, [
            "prepare", "assets", "trainer", "trainer-check", "anima",
            "sidecars", "toml", "launch",
        ])

    def test_a211_frontend_requires_visible_transient_start_only_ack(self):
        self._need(
            "__experimentalFullBf16Ack" in UI
            and "experimental_full_bf16_ack" in UI
            and "checkbox" in UI.lower(),
            "frontend lacks visible nonpersisted per-Start acknowledgement control",
        )
        self.assertIn('"/api/run"', UI)
        self.assertIn("__runtimeReady", UI)
        self.assertIn("localStorage.setItem", UI)
        self.assertNotIn('a.value.experimental_full_bf16_ack', UI)
        self.assertNotIn('localStorage.setItem("experimental_full_bf16_ack"', UI)

    def test_a212_ui_ready_is_not_host_admissible_or_gui_only_authority(self):
        self._need(
            "__experimentalFullBf16Ack" in UI
            and "experimental_full_bf16_ack" in UI,
            "frontend lacks separate A2 consent/readiness boundary",
        )
        self.assertIn("__runtimeBlockers", UI)
        self.assertIn("__runtimeReady", UI)
        self.assertIn("!__runtimeReady.value", UI)
        self.assertNotIn("__hostAdmissible.value&&", UI)
        self.assertNotIn("can_start", _body("create_toml_file"))


class QG2A2ExistingBaselines(unittest.TestCase):
    def test_a901_unchanged_nested_and_flat_json_request_decoder(self):
        self.assertEqual(
            decode_training_request(b'{"train_type":"sd-lora","config":{"x":3}}'),
            ("sd-lora", {"x": 3}),
        )
        self.assertEqual(
            decode_training_request(b'{"x":3}'),
            (None, {"x": 3}),
        )
        with self.assertRaises(ValueError):
            decode_training_request(b'["invalid list"]')

    def test_a902_existing_preview_compiles_validates_and_never_writes(self):
        events = []
        fn = _api_function(
            "preview_training_config",
            prepare_request_config=lambda *a, **kw: (
                events.append("prepare") or _fake_prepared()
            ),
            validate_prepared_config=lambda *a, **kw: events.append("validate"),
            _prepared_payload=lambda *a: (events.append("payload") or {"toml": ""}),
            materialize_sidecars=lambda *a: events.append("WRITE"),
        )
        response = asyncio.run(fn(_Request({
            "train_type": "sd-lora", "config": {},
        })))
        self.assertEqual(response.status, "success")
        self.assertEqual(events, ["prepare", "validate", "payload"])

    def test_a903_existing_export_validates_payload_before_deliberate_write(self):
        events = []
        f = _api_function(
            "export_training_config",
            prepare_request_config=lambda *a, **kw: (
                events.append("prepare") or _fake_prepared()
            ),
            validate_prepared_config=lambda *a, **kw: events.append("validate"),
            _prepared_payload=lambda *a: (events.append("payload") or {"toml": ""}),
            materialize_sidecars=lambda *a: events.append("sidecars"),
        )
        response = asyncio.run(f(_Request({
            "train_type": "sd-lora", "config": {},
        })))
        self.assertEqual(response.status, "success")
        self.assertEqual(events,
                         ["prepare", "validate", "payload", "sidecars"])

    def test_a904_existing_blocked_start_rejects_before_writes_and_launch(self):
        events = []
        def blocked(*a, **kw):
            events.append("preflight")
            raise ValueError("blocked by Step6F")
        f = _api_function(
            "create_toml_file",
            prepare_request_config=blocked,
            validate_prepared_config=lambda *a, **kw: events.append("assets"),
            prepare_runtime_trainer=lambda *a: events.append("trainer"),
            materialize_sidecars=lambda *a: events.append("sidecars"),
            _write_text_atomic=lambda *a: events.append("toml"),
            run_prepared_train=lambda *a, **kw: events.append("LAUNCH"),
        )
        response = asyncio.run(f(_Request({
            "train_type": "sd-lora", "config": {},
        })))
        self.assertEqual(response.status, "fail")
        self.assertIn("Step6F", response.message)
        self.assertEqual(events, ["preflight"])


if __name__ == "__main__":
    unittest.main()
