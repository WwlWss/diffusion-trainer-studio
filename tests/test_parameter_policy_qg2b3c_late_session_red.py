"""QG-2-B3-C Late Session identity Tests-only RED. Torch-free source and real IO."""
from __future__ import annotations

import ast
from dataclasses import fields
import json
from pathlib import Path
import tempfile
import unittest

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy import load_parameter_policy_sidecar
from mikazuki.parameter_policy_admission import (
    DirectTrainerAdmissionSnapshot, fingerprint_effective_training_config,
)
from tests.test_parameter_policy_qg2b3a_snapshot_red import _policy, _write_policy, _config

ROOT = Path(__file__).resolve().parents[1]
MARKER = "QG-2-B3-C LATE SESSION CONTRACT ABSENT"
LATE = "mikazuki/parameter_policy_trainer.py"
TRAINERS = ("scripts/stable/train_network.py", "scripts/dev/train_network.py")


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def node(path, name, cls=None):
    body = ast.parse(source(path), filename=path).body
    if cls:
        body = next(x for x in body if isinstance(x, ast.ClassDef) and x.name == cls).body
    return next(x for x in body if isinstance(x, ast.FunctionDef) and x.name == name)


def function_source(path, name, cls=None):
    return ast.get_source_segment(source(path), node(path, name, cls))


def calls(name, method):
    found = []
    for item in ast.walk(node(LATE, name)):
        if isinstance(item, ast.Call):
            target = item.func
            if isinstance(target, ast.Name) and target.id == method:
                found.append((item.lineno, item))
            elif isinstance(target, ast.Attribute) and target.attr == method:
                found.append((item.lineno, item))
    return found


class QG2B3CLateTargetedRed(unittest.TestCase):
    def need(self, value, reason):
        if not value:
            self.fail(f"{MARKER}: {reason}")

    def test_b3c301_stable_dev_pass_final_snapshot_and_typed_intent(self):
        for path in TRAINERS:
            calls_found = [x for x in ast.walk(node(path, "train", "NetworkTrainer"))
                           if isinstance(x, ast.Call)
                           and isinstance(x.func, ast.Attribute)
                           and x.func.attr == "create_parameter_policy_session"]
            self.assertEqual(len(calls_found), 1, path)
            keys = {x.arg for x in calls_found[0].keywords}
            self.need({"admission_snapshot", "admission_intent"} <= keys,
                      path + " late call does not pass second-gate snapshot and one-shot intent")

    def test_b3c302_late_session_requires_two_typed_control_inputs(self):
        fun = node(LATE, "create_parameter_policy_session")
        keys = {x.arg for x in fun.args.kwonlyargs}
        self.need({"admission_snapshot", "admission_intent"} <= keys,
                  "late public ABI does not require explicit typed identity inputs")
        # No default can silently authorize missing identity.
        indexes = [i for i, arg in enumerate(fun.args.kwonlyargs)
                   if arg.arg in {"admission_snapshot", "admission_intent"}]
        for i in indexes:
            self.assertIsNone(fun.args.kw_defaults[i])

    def test_b3c303_late_comparison_covers_all_four_coordinates_before_scan(self):
        fun = node(LATE, "create_parameter_policy_session")
        matches = calls("create_parameter_policy_session", "assert_matches")
        self.need(bool(matches), "no early snapshot.assert_matches in late session")
        scan = calls("create_parameter_policy_session", "scan_parameter_roots")
        self.assertEqual(len(scan), 1)
        self.assertLess(matches[0][0], scan[0][0])
        self.assertTrue({
            "policy_hash", "effective_config_fingerprint",
            "train_type", "experimental_ack"
        } <= {x.arg for x in matches[0][1].keywords})

    def test_b3c304_first_party_reclassification_and_candidate_hold(self):
        body = function_source(LATE, "create_parameter_policy_session")
        self.need("assess_component_training_admission(" in body,
                  "late gate does not rerun first-party Direct Trainer classifier")
        self.assertIn('purpose="direct_trainer"', body)
        self.assertIn("experimental_host_candidate", body)
        self.assertIn("ack_required", body)
        self.assertIn("blocked", body)

    def test_b3c305_late_loader_delegates_strict_shared_policy_parser(self):
        body = function_source(LATE, "load_parameter_policy_file")
        self.need("load_parameter_policy_sidecar(" in body,
                  "late JSON parser diverges from strict Early loader")

    def test_b3c306_late_gate_rechecks_policy_config_before_root_side_effects(self):
        body = function_source(LATE, "create_parameter_policy_session")
        self.need("fingerprint_effective_training_config(" in body,
                  "late gate does not independently hash live effective config")
        self.assertLess(body.index("assert_matches("), body.index("scan_parameter_roots("))
        self.assertLess(body.index("assess_component_training_admission("),
                        body.index("scan_parameter_roots("))


class QG2B3CLateBaselines(unittest.TestCase):
    def test_b3c901_real_frozen_five_fields_and_canonical_hash(self):
        self.assertEqual(
            [x.name for x in fields(DirectTrainerAdmissionSnapshot)],
            ["policy_hash", "effective_config_fingerprint", "train_type",
             "experimental_ack", "admission"],
        )
        self.assertTrue(DirectTrainerAdmissionSnapshot.__dataclass_params__.frozen)
        self.assertTrue(hasattr(DirectTrainerAdmissionSnapshot, "__slots__"))
        h = fingerprint_effective_training_config({"nested": {"b": [2, 3], "a": True}})
        self.assertRegex(h, "^[0-9a-f]{64}$")
        self.assertEqual(
            h, fingerprint_effective_training_config({"nested": {"a": True, "b": (2, 3)}}),
        )

    def test_b3c902_actual_strict_loader_rejects_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            canonical, full = load_parameter_policy_sidecar(path)
            self.assertRegex(full, "^[0-9a-f]{64}$")
            self.assertEqual(canonical["version"], 1)
            for raw in (
                '{"version":1,"version":1,"optimizer_profiles":{},"components":{}}',
                '{"version":1,"optimizer_profiles":{"main":{"type":"AdamW","args":{"x":NaN}}},"components":{}}',
                '{"version":1,"optimizer_profiles":{"main":{"type":"AdamW","args":{"x":Infinity}}},"components":{}}',
            ):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_parameter_policy_sidecar(path)

    def test_b3c903_existing_semantic_execution_assets_and_scan_sequence(self):
        body = function_source(LATE, "create_parameter_policy_session")
        required = (
            "parameter_policy_v1_semantic_blockers(",
            "parameter_policy_execution_blockers(",
            "build_parameter_policy_execution_contract(",
            "scan_parameter_roots(",
            "build_parameter_routing_plan(",
        )
        for part in required:
            self.assertIn(part, body)
        positions = [body.index(x) for x in required]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("require_assets=True", body)
        self.assertIn("structural_frozen_parameters", body)

    def test_b3c904_stable_dev_sr1_identity_guard_and_standard_laziness(self):
        for path in TRAINERS:
            body = function_source(path, "train", "NetworkTrainer")
            self.assertIn("fingerprint_effective_training_config(vars(args))", body)
            self.assertIn("admission_snapshot.effective_config_fingerprint", body)
            self.assertIn("if parameter_policy_requested:", body)
            self.assertIn("preflight_direct_trainer_admission(", body)
            self.assertLess(body.index("preflight_direct_trainer_admission("),
                            body.index("train_util.prepare_accelerator("))


if __name__ == "__main__":
    unittest.main()
