"""QG-2-B3-B: Stable/Dev pure early bridge Tests-only RED.

No torch, Accelerator, tokenizer, model, dataset, submodule or CUDA needed.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

from mikazuki.direct_trainer_admission_intent import (
    DirectTrainerAdmissionIntent,
    extract_direct_trainer_admission_intent,
)
from mikazuki.parameter_policy_admission import (
    DirectTrainerAdmissionSnapshot,
    prepare_direct_trainer_admission,
)

ROOT = Path(__file__).resolve().parents[1]
MARKER = "QG-2-B3-B EARLY BRIDGE CONTRACT ABSENT"
BRIDGES = (
    "scripts/stable/library/dts_parameter_policy_bridge.py",
    "scripts/dev/library/dts_parameter_policy_bridge.py",
)
TRAINERS = (
    "scripts/stable/train_network.py",
    "scripts/dev/train_network.py",
)
PURE_API = "preflight_direct_trainer_admission"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _tree(path: str) -> ast.Module:
    return ast.parse(_source(path), filename=path)


def _function(path: str, name: str, *, cls: str | None = None):
    root = _tree(path)
    body = root.body
    if cls is not None:
        match = next(
            (node for node in body
             if isinstance(node, ast.ClassDef) and node.name == cls),
            None,
        )
        if match is None:
            raise AssertionError(f"Missing expected class {cls} in {path}")
        body = match.body
    return next(
        (node for node in body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == name),
        None,
    )


def _method_source(path: str, name: str = "train") -> str:
    node = _function(path, name, cls="NetworkTrainer")
    if node is None:
        raise AssertionError(f"Missing pre-existing NetworkTrainer.{name}: {path}")
    return ast.get_source_segment(_source(path), node) or ""


class QG2B3BEarlyBridgeTargetedRed(unittest.TestCase):
    def _need(self, ok: bool, message: str) -> None:
        if not ok:
            self.fail(f"{MARKER}: {message}")

    def _bridge(self, path: str):
        node = _function(path, PURE_API)
        self._need(node is not None, f"{path} lacks pure early bridge API")
        return node

    def _two_gates(self, path: str) -> str:
        body = _method_source(path)
        self._need(
            body.count(f".{PURE_API}(") >= 2,
            f"{path} lacks initial and final pre-model admission gates",
        )
        return body

    def test_b3b201_both_bridges_export_same_pure_early_api(self):
        for path in BRIDGES:
            node = self._bridge(path)
            self.assertEqual(node.name, PURE_API)
            self.assertIn(f'"{PURE_API}"', _source(path).split("__all__ =", 1)[-1])
            params = {arg.arg for arg in node.args.kwonlyargs}
            self.assertTrue(
                {"policy_path", "train_type", "effective_config", "admission_intent"}
                <= params
            )

    def test_b3b202_stable_early_gate_before_tokenizer_and_dataset(self):
        body = self._two_gates(TRAINERS[0])
        first = body.index(f".{PURE_API}(")
        for operation in (
            "setup_logging(", "self.load_tokenizer(", "train_util.prepare_dataset_args(",
            "train_util.prepare_accelerator(", "self.load_target_model(",
        ):
            self.assertLess(first, body.index(operation), operation)

    def test_b3b203_dev_early_gate_before_tokenization_and_dataset(self):
        body = self._two_gates(TRAINERS[1])
        first = body.index(f".{PURE_API}(")
        for operation in (
            "setup_logging(", "self.get_tokenize_strategy(",
            "train_util.prepare_dataset_args(", "train_util.prepare_accelerator(",
            "self.load_target_model(",
        ):
            self.assertLess(first, body.index(operation), operation)

    def test_b3b204_both_bridge_paths_delegate_to_first_party_without_torch(self):
        for path in BRIDGES:
            node = self._bridge(path)
            segment = ast.get_source_segment(_source(path), node) or ""
            self.assertIn("prepare_direct_trainer_admission", segment)
            self.assertNotIn("load_parameter_policy_trainer(", segment)
            self.assertNotIn("mikazuki.parameter_policy_trainer", segment)
            self.assertNotIn("import torch", segment)
            self.assertNotIn("import accelerate", segment)

    def test_b3b205_host_admission_statuses_are_enforced_before_side_effects(self):
        for path in TRAINERS:
            body = self._two_gates(path)
            self.assertIn("ack_required", body)
            self.assertIn("blocked", body)
            self.assertIn("experimental_host_candidate", body)
            self.assertIn("legacy_approved", body)
            self.assertLess(body.index(f".{PURE_API}("), body.index("setup_logging("))
            self.assertNotIn("if snapshot.host_admissible:", body)

    def test_b3b206_both_trainers_recompute_after_args_normalization(self):
        for path in TRAINERS:
            body = self._two_gates(path)
            first = body.index(f".{PURE_API}(")
            second = body.index(f".{PURE_API}(", first + 1)
            self.assertLess(first, body.index("self.assert_extra_args("))
            self.assertLess(body.index("self.assert_extra_args("), second)
            self.assertLess(second, body.index("train_util.prepare_accelerator("))

    def test_b3b207_typed_intent_and_policy_identity_are_carried_through_both_gates(self):
        for path in TRAINERS:
            body = self._two_gates(path)
            self.assertIn("admission_intent", body)
            self.assertIn("parameter_policy_config", body)
            self.assertIn("parameter_policy_train_type", body)
            self.assertIn("effective_config", body)
            self.assertNotIn("experimental_full_bf16_ack =", body)

    def test_b3b208_final_snapshot_remains_local_and_no_late_session_migration(self):
        for path in TRAINERS:
            body = self._two_gates(path)
            self.assertIn("admission_snapshot", body)
            self.assertLess(
                body.index(f".{PURE_API}("),
                body.index("create_parameter_policy_session("),
            )
            self.assertNotIn("args.admission_snapshot =", body)
            self.assertNotIn("args.experimental_full_bf16_ack =", body)
        # All late Session, Launcher and API code is frozen outside B3-B.
        for path in BRIDGES:
            self.assertNotIn("load_parameter_policy_trainer().preflight_direct_trainer_admission", _source(path))


class QG2B3BEarlyBridgeBaselines(unittest.TestCase):
    def test_b3b901_b3a_pure_factory_and_frozen_snapshot_remain_present(self):
        self.assertTrue(callable(prepare_direct_trainer_admission))
        self.assertTrue(DirectTrainerAdmissionSnapshot.__dataclass_params__.frozen)
        self.assertTrue(hasattr(DirectTrainerAdmissionSnapshot, "__slots__"))
        for path in (
            "mikazuki/parameter_policy.py",
            "mikazuki/parameter_policy_admission.py",
            "mikazuki/direct_trainer_admission_intent.py",
        ):
            source = _tree(path)
            imports = [
                node for node in source.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            names = set()
            for node in imports:
                if isinstance(node, ast.Import):
                    names.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif node.module:
                    names.add(node.module.split(".", 1)[0])
            self.assertNotIn("torch", names)
            self.assertNotIn("accelerate", names)
            self.assertNotIn("parameter_policy_trainer", names)

    def test_b3b902_existing_early_boolean_guards_precede_heavy_work(self):
        for path in TRAINERS:
            body = _method_source(path)
            check = body.index(".preflight_component_runtime_booleans(")
            self.assertLess(check, body.index("train_util.prepare_dataset_args("))
            self.assertLess(check, body.index("train_util.prepare_accelerator("))
            self.assertLess(check, body.index("self.load_target_model("))

    def test_b3b903_existing_cli_one_shot_intent_is_strict(self):
        args = type("Args", (), {})()
        args.experimental_full_bf16_ack = True
        intent = extract_direct_trainer_admission_intent(args)
        self.assertIs(type(intent), DirectTrainerAdmissionIntent)
        self.assertIs(intent.experimental_full_bf16_ack, True)
        self.assertFalse(hasattr(args, "experimental_full_bf16_ack"))
        for value in ("true", 1, None):
            bad = type("Args", (), {})()
            bad.experimental_full_bf16_ack = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    extract_direct_trainer_admission_intent(bad)

    def test_b3b904_bridges_keep_heavy_loader_lazy_and_standard_not_importing(self):
        for path in BRIDGES:
            source = _source(path)
            tree = _tree(path)
            self.assertIsNotNone(_function(path, "load_parameter_policy_trainer"))
            self.assertIsNotNone(_function(path, "preflight_component_runtime_booleans"))
            self.assertIn("importlib.import_module(\"mikazuki.parameter_policy_trainer\")", source)
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "mikazuki.parameter_policy_trainer")
        for path in TRAINERS:
            body = _method_source(path)
            self.assertIn("if parameter_policy_requested:", body)
            self.assertIn("from library import dts_parameter_policy_bridge", body)


if __name__ == "__main__":
    unittest.main()
