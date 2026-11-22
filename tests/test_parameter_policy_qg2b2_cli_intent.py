"""QG-2-B2 CLI intent + TOML isolation tests (Host-only, no CUDA)."""
from __future__ import annotations

import argparse
import ast
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from mikazuki.direct_trainer_admission_intent import (
    DirectTrainerAdmissionIntent,
    add_direct_trainer_admission_argument,
    extract_direct_trainer_admission_intent,
    reject_direct_trainer_ack_toml,
)

ROOT = Path(__file__).resolve().parents[1]
ACK = "experimental_full_bf16_ack"
FLAG = "--experimental_full_bf16_ack"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _function(path: str, name: str):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return source, node


def _class_method(path: str, class_name: str, method_name: str):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    owner = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    node = next(
        item for item in owner.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == method_name
    )
    return source, node


def _main_guard(path: str):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if (
            isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and any(
                isinstance(item, ast.Constant) and item.value == "__main__"
                for item in node.test.comparators
            )
        ):
            return source, node
    raise AssertionError(f"missing executable main guard: {path}")


class QG2B2DirectTrainerCliIntentTests(unittest.TestCase):
    def test_b2_01_intent_is_frozen_and_accepts_only_real_boolean(self):
        intent = DirectTrainerAdmissionIntent()
        self.assertFalse(intent.experimental_full_bf16_ack)
        self.assertTrue(DirectTrainerAdmissionIntent(True).experimental_full_bf16_ack)
        self.assertTrue(DirectTrainerAdmissionIntent.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            intent.experimental_full_bf16_ack = True
        for invalid in ("true", "false", 1, 0, None, {}, []):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):
                    DirectTrainerAdmissionIntent(invalid)

    def test_b2_02_extraction_is_one_shot_and_removes_raw_namespace_field(self):
        args = SimpleNamespace(experimental_full_bf16_ack=True, learning_rate=1e-4)
        intent = extract_direct_trainer_admission_intent(args)
        self.assertIs(intent.experimental_full_bf16_ack, True)
        self.assertFalse(hasattr(args, ACK))
        self.assertEqual(args.learning_rate, 1e-4)

        second = extract_direct_trainer_admission_intent(args)
        self.assertFalse(second.experimental_full_bf16_ack)
        self.assertFalse(hasattr(args, ACK))

        for invalid in ("true", 1, None):
            bad = SimpleNamespace(experimental_full_bf16_ack=invalid)
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):
                    extract_direct_trainer_admission_intent(bad)
                # Invalid data is not silently consumed or normalized.
                self.assertTrue(hasattr(bad, ACK))

    def test_b2_03_parser_registers_explicit_store_true_default_false(self):
        parser = argparse.ArgumentParser()
        add_direct_trainer_admission_argument(parser)
        default = parser.parse_args([])
        positive = parser.parse_args([FLAG])
        self.assertIs(default.experimental_full_bf16_ack, False)
        self.assertIs(positive.experimental_full_bf16_ack, True)
        self.assertIs(type(default.experimental_full_bf16_ack), bool)
        self.assertIs(type(positive.experimental_full_bf16_ack), bool)

    def test_b2_04_toml_reserved_key_is_rejected_after_flattening(self):
        reject_direct_trainer_ack_toml({"learning_rate": 1e-4})
        for flattened in (
            {ACK: True},
            {ACK: False},
            {ACK: "true"},
        ):
            with self.subTest(value=flattened[ACK]):
                with self.assertRaisesRegex(ValueError, ACK):
                    reject_direct_trainer_ack_toml(flattened)
        with self.assertRaises(TypeError):
            reject_direct_trainer_ack_toml(None)

    def test_b2_05_all_network_family_parsers_inherit_the_same_cli_flag(self):
        base_paths = ("scripts/stable/train_network.py", "scripts/dev/train_network.py")
        for path in base_paths:
            source, setup = _function(path, "setup_parser")
            calls = [
                node for node in ast.walk(setup)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "add_direct_trainer_admission_argument"
            ]
            self.assertEqual(len(calls), 1, path)
            # A single stdlib-only first-party Intent module is loaded through
            # the pre-existing DTS-root bridge, not an eager top-level import.
            self.assertIn("load_direct_trainer_intent_module()", source, path)
            self.assertIn("from library import dts_parameter_policy_bridge", source, path)
            self.assertNotIn("from mikazuki.direct_trainer_admission_intent import", source, path)
            self.assertIn("add_direct_trainer_admission_argument(parser)", source)

        # All model-family NetworkTrainer entrypoints derive their parser from
        # one of the two base train_network modules, so the flag must be
        # inherited exactly once (not re-registered with conflicting defaults).
        derived = (
            "scripts/stable/sdxl_train_network.py",
            "scripts/dev/sdxl_train_network.py",
            "scripts/dev/sd3_train_network.py",
            "scripts/dev/flux_train_network.py",
            "scripts/dev/lumina_train_network.py",
        )
        for path in derived:
            source, setup = _function(path, "setup_parser")
            calls = [
                node for node in ast.walk(setup)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setup_parser"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "train_network"
            ]
            self.assertEqual(len(calls), 1, path)
            # Derived models use the already-bootstrapped base trainer identity.
            self.assertIn(
                "train_network.extract_direct_trainer_admission_intent(args)",
                source, path,
            )
            self.assertNotIn("from mikazuki.direct_trainer_admission_intent import", source, path)

    def test_b2_06_toml_rejection_precedes_namespace_merge_in_both_train_utils(self):
        for path in (
            "scripts/stable/library/train_util.py",
            "scripts/dev/library/train_util.py",
        ):
            source, read_config = _function(path, "read_config_from_file")
            # Full Trainers share this reader and must not need DTS on PYTHONPATH.
            # Require a real fail-closed reserved-key guard before Namespace.
            guards = [
                node for node in ast.walk(read_config)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Constant)
                and node.test.left.value == ACK
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.In)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Name)
                and node.test.comparators[0].id == "ignore_nesting_dict"
                and any(isinstance(stmt, ast.Raise) for stmt in node.body)
            ]
            namespace_calls = [
                node for node in ast.walk(read_config)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Namespace"
            ]
            self.assertEqual(len(guards), 1, path)
            self.assertEqual(len(namespace_calls), 1, path)
            self.assertLess(guards[0].lineno, namespace_calls[0].lineno, path)
            body = ast.get_source_segment(source, read_config)
            self.assertIn("ignore_nesting_dict", body, path)
            self.assertNotIn("from mikazuki.direct_trainer_admission_intent import", body, path)

    def test_b2_07_output_config_drops_ack_even_if_called_without_extraction(self):
        # Historical acceptance of positive ACK was a bug: consent only
        # authorizes one training invocation, never an export operation.
        for path in (
            "scripts/stable/library/train_util.py",
            "scripts/dev/library/train_util.py",
        ):
            source, read_config = _function(path, "read_config_from_file")
            body = ast.get_source_segment(source, read_config)
            self.assertIn('"experimental_full_bf16_ack"', body, path)
            self.assertIn('"output_config"', body, path)
            self.assertIn("args_dict", body, path)

            for ack in (False, True):
                with self.subTest(path=path, ack=ack), tempfile.TemporaryDirectory() as directory:
                    config_path = str(Path(directory) / "out.toml")
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--config_file", default=None)
                    parser.add_argument("--output_config", action="store_true")
                    parser.add_argument("--experimental_full_bf16_ack", action="store_true")
                    parser.add_argument("--learning_rate", type=float, default=None)
                    args = SimpleNamespace(
                        config_file=config_path,
                        output_config=True,
                        experimental_full_bf16_ack=ack,
                        learning_rate=1e-4,
                    )
                    captured = {}

                    class _Toml:
                        @staticmethod
                        def dump(values, stream):
                            captured.update(values)

                    class _Logger:
                        @staticmethod
                        def info(_message):
                            pass

                        @staticmethod
                        def error(_message):
                            pass

                    module = ast.Module(body=[read_config], type_ignores=[])
                    ast.fix_missing_locations(module)
                    namespace = {
                        "argparse": argparse,
                        "os": os,
                        "pathlib": __import__("pathlib"),
                        "toml": _Toml,
                        "logger": _Logger,
                        "exit": lambda code: (_ for _ in ()).throw(SystemExit(code)),
                    }
                    exec(compile(module, path, "exec"), namespace)
                    if ack:
                        with self.assertRaisesRegex(ValueError, "acknowledgement"):
                            namespace["read_config_from_file"](args, parser)
                        self.assertFalse(Path(config_path).exists(), path)
                        self.assertEqual(captured, {}, path)
                    else:
                        with self.assertRaises(SystemExit) as stopped:
                            namespace["read_config_from_file"](args, parser)
                        self.assertEqual(stopped.exception.code, 0)
                        self.assertTrue(Path(config_path).exists(), path)
                        self.assertNotIn(ACK, captured, path)
                        self.assertEqual(captured.get("learning_rate"), 1e-4)

    def test_b2_08_all_direct_trainer_entrypoints_extract_and_forward_intent(self):
        entrypoints = (
            ("scripts/stable/train_network.py", "NetworkTrainer"),
            ("scripts/dev/train_network.py", "NetworkTrainer"),
            ("scripts/stable/sdxl_train_network.py", "SdxlNetworkTrainer"),
            ("scripts/dev/sdxl_train_network.py", "SdxlNetworkTrainer"),
            ("scripts/dev/sd3_train_network.py", "Sd3NetworkTrainer"),
            ("scripts/dev/flux_train_network.py", "FluxNetworkTrainer"),
            ("scripts/dev/lumina_train_network.py", "LuminaNetworkTrainer"),
        )
        for path, trainer_class in entrypoints:
            source, main = _main_guard(path)
            body = ast.get_source_segment(source, main)
            # Preserve the legacy CLI verification before TOML reading, but
            # consume the acknowledgement only after the FINAL argparse parse.
            self.assertLess(
                body.index("verify_command_line_training_args(args)"),
                body.index("read_config_from_file(args, parser)"),
                path,
            )
            self.assertLess(
                body.index("read_config_from_file(args, parser)"),
                body.index("extract_direct_trainer_admission_intent(args)"),
                path,
            )
            self.assertIn(
                "trainer.train(args, admission_intent=admission_intent)",
                body,
                path,
            )
            if trainer_class == "NetworkTrainer":
                self.assertIn("load_direct_trainer_intent_module()", source, path)
            else:
                self.assertIn(
                    "train_network.extract_direct_trainer_admission_intent(args)",
                    body, path,
                )
            self.assertNotIn(
                "from mikazuki.direct_trainer_admission_intent import",
                source, path,
            )

            # The two base trainers own the optional train() ABI. Derived
            # model-family trainers inherit it and must not override it.
            if trainer_class == "NetworkTrainer":
                _train_source, train = _class_method(path, "NetworkTrainer", "train")
                arguments = {
                    item.arg
                    for item in (
                        list(train.args.posonlyargs)
                        + list(train.args.args)
                        + list(train.args.kwonlyargs)
                    )
                }
                self.assertIn("admission_intent", arguments, path)
                train_text = ast.get_source_segment(_train_source, train)
                self.assertIn("admission_intent=None", train_text, path)
            else:
                class_tree = ast.parse(source, filename=path)
                owner = next(
                    item for item in class_tree.body
                    if isinstance(item, ast.ClassDef) and item.name == trainer_class
                )
                self.assertTrue(
                    any(
                        isinstance(base, ast.Attribute)
                        and base.attr == "NetworkTrainer"
                        and isinstance(base.value, ast.Name)
                        and base.value.id == "train_network"
                        for base in owner.bases
                    ),
                    path,
                )
                self.assertFalse(
                    any(
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == "train"
                        for item in owner.body
                    ),
                    f"{path} must inherit the base train() intent ABI",
                )


if __name__ == "__main__":
    unittest.main()
