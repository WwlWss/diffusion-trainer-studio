"""QG-2-A1 Request Admission + per-job intent: 12 unmet contracts, 4 real baselines.

The original A101-A112 Tests-only RED has been followed by scoped Request
GREEN. These tests exercise the implemented contract as well as its previous
unmet-interface RED marker. No API/Trainer/CUDA authorization is granted.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mikazuki.training_config as config_module
from mikazuki.parameter_policy import parameter_policy_runtime_blockers
from mikazuki.parameter_policy_admission import (
    ComponentAdmission, assess_component_training_admission,
)
from mikazuki.parameter_policy_execution import (
    FULL_BF16_BACKEND_QUALIFICATIONS,
    active_parameter_policy_execution_features,
    parameter_policy_execution_blockers,
)
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.training_gui_args import parse_ui_custom_params
from tests.test_parameter_policy_admission_red import _config, _pending, _policy
from tests.test_parameter_policy_request_contract import (
    _component_gui_config, _load_prepare_request_config,
    _prepared, _real_prepare_request,
)

MARKER = "QG-2-A1 REQUEST INTENT CONTRACT ABSENT"
ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "mikazuki" / "training_request.py"
CONFIG = ROOT / "mikazuki" / "training_config.py"


def _request_tree():
    source = REQ.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=REQ.as_posix())
    func = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "prepare_request_config")
    return source, func


def _owner_calls(func):
    return {node.func.id for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)}


def _fake_prepared(*, backend="sd-lora", **effective):
    return _prepared(
        {"parameter_policy_config": "policy.json", **_config(**effective)},
        train_type=backend,
    )


def _make_preparation(*, backend="sd-lora", prepared=None,
                      source_admission=None, runtime_blockers=None,
                      materializations=None, policy=None, launch_log=None):
    """Real trusted policy/admission + unchanged legacy Step6F, AST-isolated API.

    This follows the existing Request regression style; it does not import the
    FastAPI application. The new implementation must retain AST-fixture
    compatibility (or explicitly review all dependent baseline fixtures).
    """
    p = _policy(backend) if policy is None else policy
    events = [] if materializations is None else materializations
    launch_calls = [] if launch_log is None else launch_log

    def make_prepared(raw, **kw):
        launch_calls.append(kw["launch"])
        return (
            _fake_prepared(backend=backend) if prepared is None else prepared
        )

    return _load_prepare_request_config(
        normalize_parameter_policy_editor_state=lambda raw: None,
        build_parameter_policy_sidecar=lambda raw, page: (
            "policy.json", {"policy.json": '{"version":1}'}, p,
        ),
        build_multi_caption_sidecar=lambda raw, page: (None, {}, None),
        prepare_prompt_fields=lambda raw, page: ({}, []),
        prepare_training_config=make_prepared,
        legacy_api=SimpleNamespace(
            resolve_training_backend=lambda cfg, tp: (tp, "trainer.py")
        ),
        parameter_policy_runtime_blockers=(
            parameter_policy_runtime_blockers
            if runtime_blockers is None else runtime_blockers
        ),
        parameter_policy_gpu_selection_blockers=lambda gpu_ids: [],
        PARAMETER_POLICY_RUNTIME_TRAIN_TYPES=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
        materialize_sidecars=lambda sidecars: events.append(dict(sidecars)),
        assess_component_training_admission=(
            assess_component_training_admission
            if source_admission is None else source_admission
        ),
        RequestAdmissionIntent=getattr(config_module, "RequestAdmissionIntent", None),
        parse_ui_custom_params=parse_ui_custom_params,
    )


class QG2A1TargetedRed(unittest.TestCase):
    def _contract(self):
        intent = getattr(config_module, "RequestAdmissionIntent", None)
        config_fields = {item.name for item in fields(config_module.PreparedTrainingConfig)}
        source, function = _request_tree()
        keyword_args = {arg.arg for arg in function.args.kwonlyargs}
        missing = []
        if intent is None or not is_dataclass(intent):
            missing.append("RequestAdmissionIntent frozen type")
        if not {"admission", "admission_intent"}.issubset(config_fields):
            missing.append("PreparedTrainingConfig admission fields")
        if not {"admission_intent", "admission_purpose"}.issubset(keyword_args):
            missing.append("prepare_request_config keyword-only request ABI")
        if "assess_component_training_admission" not in _owner_calls(function):
            missing.append("post-override first-party Admission call")
        if missing:
            self.fail(f"{MARKER}: {', '.join(missing)}")
        return intent

    def test_a101_immutable_strict_intent_origin_and_boolean(self):
        Intent = self._contract()
        self.assertTrue(Intent.__dataclass_params__.frozen)
        self.assertIn("experimental_full_bf16_ack", {x.name for x in fields(Intent)})
        self.assertIn("origin", {x.name for x in fields(Intent)})
        self.assertEqual(Intent().experimental_full_bf16_ack, False)
        self.assertEqual(Intent().origin, "default")
        self.assertTrue(Intent(True, "webui_start").experimental_full_bf16_ack)
        with self.assertRaises(FrozenInstanceError):
            original = Intent()
            original.origin = "webui_start"
        for invalid in ("true", "false", 1, 0, None, {}, object()):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                Intent(invalid, "webui_start")
        for invalid_origin in ("direct_cli", "worker", "bundle", "", None, 1):
            with self.subTest(origin=repr(invalid_origin)), self.assertRaises(ValueError):
                Intent(False, invalid_origin)
        with self.assertRaises(ValueError):
            Intent(True, "default")

    def test_a102_prepared_admission_stays_outside_trainer_config(self):
        Intent = self._contract()
        prepared = config_module.PreparedTrainingConfig(
            "sd-lora", "trainer.py", {"full_bf16": True}, None, [],
        )
        self.assertIsNone(prepared.admission)
        self.assertIsNone(prepared.admission_intent)
        prepared.admission_intent = Intent(False, "default")
        self.assertEqual(prepared.config, {"full_bf16": True})
        self.assertNotIn("experimental_full_bf16_ack", prepared.config)
        self.assertNotIn("admission_intent", prepared.config)
        self.assertNotIn("admission", prepared.config)

    def test_a103_request_purpose_and_typed_keyword_intent_fail_closed(self):
        Intent = self._contract()
        source, function = _request_tree()
        self.assertEqual(function.args.kwarg, None)
        kwonly = {a.arg for a in function.args.kwonlyargs}
        self.assertTrue({"materialize", "admission_intent",
                         "admission_purpose"}.issubset(kwonly))
        prepare = _make_preparation()
        for launch, purpose in ((True, "preview"), (True, "export"),
                                (False, "start"), (False, "unknown")):
            with self.subTest(launch=launch, purpose=purpose):
                with self.assertRaises(ValueError):
                    prepare({}, "sd-lora", launch=launch,
                            admission_purpose=purpose)
        for invalid in (True, "true", {}, 1):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                prepare({}, "sd-lora", admission_intent=invalid)
        expected = prepare({}, "sd-lora", launch=False, admission_purpose="export")
        self.assertEqual(expected.admission.purpose, "export")

    def test_a104_admission_uses_actual_post_override_config_not_raw(self):
        self._contract()
        # Raw true, normalized false: the real compiled config wins.
        calls = []
        def monitor(policy, **kw):
            calls.append(dict(kw))
            return assess_component_training_admission(policy, **kw)
        prepared = _fake_prepared(full_bf16=False)
        prepare = _make_preparation(prepared=prepared, source_admission=monitor)
        result = prepare({"full_bf16": True}, "sd-lora",
                         launch=False, admission_purpose="preview")
        self.assertEqual(result.admission.status, "inactive")
        self.assertFalse(result.config["full_bf16"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["effective_config"], result.config)
        self.assertEqual(calls[0]["purpose"], "preview")
        self.assertIs(calls[0]["experimental_ack"], False)

        # Real end-to-end Host compilation: custom TOML is applied only by
        # prepare_training_config after raw GUI normalization.  Admission
        # MUST see this *final* value in both directions.
        real_prepare = _real_prepare_request()
        for raw_feature, overlay_feature, expected_status in (
            (True, False, "inactive"),
            (False, True, "legacy_approved"),
        ):
            with self.subTest(
                raw=raw_feature, post_override=overlay_feature,
            ):
                raw = _component_gui_config(
                    "sd-lora", ["unet.attention.adapter"],
                    full_bf16=raw_feature,
                    mixed_precision="bf16",
                    ui_custom_params=(
                        "full_bf16 = true" if overlay_feature
                        else "full_bf16 = false"
                    ),
                )
                actual = real_prepare(
                    raw, "sd-lora", launch=False,
                    admission_purpose="preview",
                )
                self.assertIs(actual.config["full_bf16"], overlay_feature)
                self.assertEqual(actual.admission.status, expected_status)
                self.assertEqual(actual.runtime_blockers, [])

    def test_a105_pending_certificate_has_typed_preview_ack_blocker(self):
        self._contract()
        prepared = _fake_prepared(full_bf16=True)
        prepare = _make_preparation(prepared=prepared)
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            result = prepare({}, "sd-lora", launch=False,
                             admission_purpose="preview")
        self.assertIsInstance(result.admission, ComponentAdmission)
        self.assertEqual(result.admission.status, "ack_required")
        self.assertFalse(result.admission.host_admissible)
        self.assertEqual(result.admission.readiness.evidence_tier, "unverified")
        self.assertTrue(any("ack" in b.lower() for b in result.runtime_blockers))
        self.assertNotIn("experimental_full_bf16_ack", result.config)

    def test_a106_true_start_intent_is_diagnostic_only_no_launch_until_qg2(self):
        Intent = self._contract()
        seen, materialized = [], []
        def monitor(policy, **kw):
            result = assess_component_training_admission(policy, **kw)
            seen.append(result)
            return result
        prepare = _make_preparation(
            source_admission=monitor, materializations=materialized
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            with self.assertRaises(ValueError):
                prepare({}, "sd-lora", launch=True, materialize=True)
            self.assertEqual(seen[-1].status, "ack_required")
            with self.assertRaises(ValueError):
                prepare({}, "sd-lora", launch=True, materialize=True,
                        admission_intent=Intent(True, "webui_start"))
            self.assertEqual(seen[-1].status, "experimental_host_candidate")
        self.assertEqual(materialized, [])

    def test_a107_raw_custom_and_nested_ack_forgery_all_rejected(self):
        self._contract()
        probes = (
            {"experimental_full_bf16_ack": True},
            {"ui_custom_params": "experimental_full_bf16_ack = true"},
            {"__ui_custom_overrides": {"experimental_full_bf16_ack": True}},
            {"experimental_full_bf16_ack": "true", "can_start": True},
        )
        events = []
        prepare = _make_preparation(materializations=events)
        for raw in probes:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare(dict(raw), "sd-lora", launch=True, materialize=True)
        self.assertEqual(events, [])

    def test_a108_sd3_and_real_precision_incompatibility_block_even_with_ack(self):
        Intent = self._contract()
        cases = (
            ("sd3-lora", _fake_prepared(backend="sd3-lora"), _policy("sd3-lora")),
            ("sd-lora", _fake_prepared(mixed_precision="fp16"), _policy()),
        )
        found = []
        for backend, prepared, policy in cases:
            seen = []
            def monitor(p, **kwargs):
                result = assess_component_training_admission(p, **kwargs)
                seen.append(result)
                return result
            prep = _make_preparation(
                backend=backend, prepared=prepared, policy=policy,
                source_admission=monitor,
            )
            with self.assertRaises(ValueError):
                prep({}, backend, launch=True,
                     admission_intent=Intent(True, "webui_start"))
            found.append((backend, seen[-1].status))
        self.assertEqual(found, [("sd3-lora", "blocked"),
                                 ("sd-lora", "blocked")])

    def test_a109_previously_qualified_sd1_without_new_ack_still_prepares(self):
        self._contract()
        prepare = _make_preparation()
        result = prepare({}, "sd-lora", launch=True, materialize=False)
        self.assertEqual(result.admission.status, "legacy_approved")
        self.assertTrue(result.admission.host_admissible)
        self.assertEqual(result.runtime_blockers, [])
        self.assertNotIn("experimental_full_bf16_ack", result.config)

    def test_a110_preview_export_never_capture_or_serialize_positive_ack(self):
        Intent = self._contract()
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            for purpose in ("preview", "export"):
                with self.subTest(purpose=purpose):
                    result = _make_preparation()(
                        {}, "sd-lora", launch=False,
                        admission_purpose=purpose,
                        admission_intent=Intent(True, "webui_start"),
                    )
                    self.assertEqual(result.admission.status, "ack_required")
                    self.assertFalse(result.admission_intent.experimental_full_bf16_ack)
                    self.assertNotIn("experimental_full_bf16_ack", result.config)
                    self.assertFalse(any("experimental_full_bf16_ack" in s
                                         for s in result.sidecars.values()))

    def test_a111_consecutive_jobs_cannot_replay_request_consent(self):
        Intent = self._contract()
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            first = _make_preparation()(
                {}, "sd-lora", launch=False,
                admission_intent=Intent(True, "webui_start"),
            )
            second = _make_preparation()({}, "sd-lora", launch=False)
        self.assertIsNot(first.admission_intent, second.admission_intent)
        self.assertFalse(first.admission_intent.experimental_full_bf16_ack)
        self.assertFalse(second.admission_intent.experimental_full_bf16_ack)
        self.assertEqual(second.admission.status, "ack_required")

    def test_a112_standard_inactive_and_original_step6f_remain_unchanged(self):
        Intent = self._contract()
        standard_calls = []
        standard = _load_prepare_request_config(
            prepare_training_config=lambda config, **kw: (
                standard_calls.append(kw["launch"]) or _prepared()
            )
        )
        result = standard({}, "sd-lora", launch=True)
        self.assertEqual(standard_calls, [True])
        self.assertFalse(getattr(result, "admission", None))
        inactive = _make_preparation(
            prepared=_fake_prepared(full_bf16=False)
        )({}, "sd-lora", launch=False)
        self.assertEqual(inactive.admission.status, "inactive")
        with self.assertRaises(ValueError):
            _make_preparation(prepared=_fake_prepared(full_bf16=False))(
                {}, "sd-lora", launch=True,
                admission_intent=Intent(True, "webui_start"),
            )


class QG2A1ExistingBaselines(unittest.TestCase):
    def test_a901_real_d1_admission_and_existing_sd3_pending_still_distinct(self):
        self.assertEqual(
            parameter_policy_execution_blockers(
                _policy(), train_type="sd-lora",
                effective_config=_config(), require_assets=False,
            ), [],
        )
        blockers = parameter_policy_execution_blockers(
            _policy("sd3-lora"), train_type="sd3-lora",
            effective_config=_config(), require_assets=False,
        )
        self.assertTrue(blockers)
        self.assertIn("pending", blockers[0])

    def test_a902_true_false_and_quoted_boolean_legacy_gate_still_strict(self):
        self.assertEqual(active_parameter_policy_execution_features(
            {"full_bf16": False}
        ), ())
        with self.assertRaisesRegex(ValueError, "full_bf16"):
            active_parameter_policy_execution_features({"full_bf16": "false"})

    def test_a903_existing_step6f_block_precedes_all_materialization(self):
        events = []
        def make(raw, **kwargs):
            events.append(("prepare", kwargs["launch"]))
            return _prepared({"parameter_policy_config": "policy.json"})
        prepare = _load_prepare_request_config(
            build_parameter_policy_sidecar=lambda raw, page: (
                "policy.json", {"policy.json": "{}"}, {"version": 1},
            ),
            prepare_training_config=make,
            parameter_policy_runtime_blockers=lambda policy, **kw: [
                "existing Step6F hard blocker"
            ],
            materialize_sidecars=lambda sidecars: events.append(("write", sidecars)),
        )
        with self.assertRaisesRegex(ValueError, "existing Step6F"):
            prepare({}, "sd-lora", launch=True, materialize=True)
        self.assertEqual(events, [("prepare", False)])

    def test_a904_standard_request_launch_and_return_shape_are_preserved(self):
        events = []
        def make(raw, **kw):
            events.append(kw["launch"])
            return _prepared()
        prepare = _load_prepare_request_config(
            prepare_training_config=make,
            materialize_sidecars=lambda _: events.append("write"),
        )
        result = prepare({}, "sd-lora", launch=True, materialize=True)
        self.assertEqual(events, [True, "write"])
        self.assertEqual(result.runtime_blockers, [])


if __name__ == "__main__":
    unittest.main()
