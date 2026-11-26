"""QG-2-MR-SR1-A: 26 real source/CLI/subprocess failures + 4 true baselines.

Tests-only RED: do not alter trainers, entitlement or CI assertions to force
failure. Real Stable/Dev TOML parsers are executed, and import prelude probes
run in new Python -S processes with no repository-root PYTHONPATH.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import tomllib
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from mikazuki.direct_trainer_admission_intent import (
    ACK_ARGUMENT, DirectTrainerAdmissionIntent,
    add_direct_trainer_admission_argument,
    extract_direct_trainer_admission_intent,
)

ROOT = Path(__file__).resolve().parents[1]
MARKER = "QG-2-MR-SR1 REAL SOURCE DEFECT"
NETWORKS = (
    "scripts/stable/train_network.py",
    "scripts/dev/train_network.py",
    "scripts/stable/sdxl_train_network.py",
    "scripts/dev/sdxl_train_network.py",
    "scripts/dev/sd3_train_network.py",
    "scripts/dev/flux_train_network.py",
    "scripts/dev/lumina_train_network.py",
)
READERS = (
    "scripts/stable/library/train_util.py",
    "scripts/dev/library/train_util.py",
)
QG1B_WORKFLOWS = (
    ".github/workflows/qg1b-trusted-producer-host.yml",
    ".github/workflows/qg1b-sr1-host.yml",
)


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _node(path: str, name: str, *, class_name: str | None = None):
    tree = ast.parse(_source(path), filename=path)
    nodes = tree.body
    if class_name:
        cls = next(x for x in nodes if isinstance(x, ast.ClassDef) and x.name == class_name)
        nodes = cls.body
    return next(x for x in nodes if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                and x.name == name)


class _Toml:
    @staticmethod
    def load(stream):
        return tomllib.loads(stream.read())

    @staticmethod
    def dump(values, stream):
        # Only used by the legacy output_config branch; an actual file is
        # created, which lets the RED assert the invalid ACK was rejected
        # before irreversible output.
        stream.write("learning_rate = 0.0003\n")


class _Logger:
    @staticmethod
    def info(_message):
        pass

    @staticmethod
    def error(_message):
        pass


def _reader(path: str):
    fn = _node(path, "read_config_from_file")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {
        "argparse": argparse, "os": os, "pathlib": __import__("pathlib"),
        "logger": _Logger, "toml": _Toml,
        "exit": lambda code: (_ for _ in ()).throw(SystemExit(code)),
    }
    exec(compile(module, path, "exec"), env)
    return env["read_config_from_file"]


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--output_config", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=None)
    add_direct_trainer_admission_argument(parser)
    return parser


def _child(program: str, script_dir: Path, cwd: Path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    prelude = (
        "import sys\n"
        "sys.path.insert(0, " + repr(str(script_dir.resolve())) + ")\n"
        "assert " + repr(str(ROOT.resolve())) + " not in sys.path, sys.path\n"
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", prelude + program],
        env=env, cwd=str(cwd), capture_output=True, text=True,
        timeout=20, check=False,
    )


class QG2MRSR1Targeted(unittest.TestCase):
    """26 frozen identities: MR-01/02 turn GREEN in C; MR-03/04 remain RED."""

    def _toml_final_namespace(self, reader_path: str, *, ack: bool):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actual.toml"
            path.write_text("[training]\nlearning_rate = 0.0003\n", encoding="utf-8")
            cli = ["--config_file", str(path)] + (
                ["--experimental_full_bf16_ack"] if ack else []
            )
            parser = _parser()
            args = parser.parse_args(cli)
            with patch.object(sys, "argv", ["trainer", *cli]):
                effective = _reader(reader_path)(args, parser)
            self.assertEqual(effective.learning_rate, 0.0003)
            # The raw flag may exist only between the final argparse parse
            # and its single typed consumption. Check value AND non-persistence.
            intent = extract_direct_trainer_admission_intent(effective)
            self.assertIs(type(intent), DirectTrainerAdmissionIntent)
            if intent.experimental_full_bf16_ack is not ack:
                self.fail(
                    f"{MARKER}: {reader_path} lost/changed final CLI "
                    f"acknowledgement during TOML reparse (expected {ack}, "
                    f"got {intent.experimental_full_bf16_ack})."
                )
            if ACK_ARGUMENT in vars(effective):
                self.fail(
                    f"{MARKER}: {reader_path} retained reserved {ACK_ARGUMENT} "
                    "after the final typed one-shot extraction."
                )

    def _output_ack_rejected_before_write(self, reader_path: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.toml"
            parser = _parser()
            cli = ["--config_file", str(path), "--output_config",
                   "--experimental_full_bf16_ack"]
            args = parser.parse_args(cli)
            try:
                _reader(reader_path)(args, parser)
            except ValueError as exc:
                self.assertIn("acknowledgement", str(exc).lower())
                self.assertFalse(path.exists())
                return
            except SystemExit as exc:
                self.fail(
                    f"{MARKER}: {reader_path} accepted positive CLI ACK "
                    f"when output_config exited {exc.code}; "
                    f"file was written={path.exists()}."
                )
            self.fail(f"{MARKER}: {reader_path} accepted positive ACK for export.")

    def _assert_main_reads_before_consuming(self, path: str):
        source = _source(path)
        tree = ast.parse(source, filename=path)
        main = next(
            x for x in tree.body if isinstance(x, ast.If)
            and isinstance(x.test, ast.Compare)
            and isinstance(x.test.left, ast.Name)
            and x.test.left.id == "__name__"
        )
        segment = ast.get_source_segment(source, main) or ""
        token_read = "read_config_from_file(args, parser)"
        token_extract = "extract_direct_trainer_admission_intent(args)"
        if (
            token_read not in segment or token_extract not in segment or
            segment.count(token_read) != 1 or segment.count(token_extract) != 1 or
            segment.index(token_extract) < segment.index(token_read)
        ):
            self.fail(
                f"{MARKER}: {path} extracts CLI acknowledgement before "
                "the final TOML parser can recreate it."
            )

    def _assert_network_clean_import(self, path: str):
        source = _source(path)
        tree = ast.parse(source, filename=path)
        direct = [
            node for node in tree.body if isinstance(node, ast.ImportFrom)
            and node.module == "mikazuki.direct_trainer_admission_intent"
        ]
        # Execute the exact currently checked-in early import in a fresh
        # child process, without installed DTS package or PYTHONPATH. After
        # GREEN the bridge is the controlled bootstrap to the same first-
        # party identity: the subprocess exercises its actual helper.
        if direct:
            statement = "\n".join(
                ast.get_source_segment(source, node) for node in direct
            )
        else:
            if path in NETWORKS[:2]:
                if "load_direct_trainer_intent_module" not in source:
                    self.fail(f"{MARKER}: {path} has no first-party bridge bootstrap")
            elif not (
                "train_network.extract_direct_trainer_admission_intent" in source
                or "load_direct_trainer_intent_module" in source
            ):
                self.fail(f"{MARKER}: {path} does not reuse a bootstrapped intent")
            statement = (
                "from library import dts_parameter_policy_bridge as bridge\n"
                "module = bridge.load_direct_trainer_intent_module()\n"
                "assert module.DirectTrainerAdmissionIntent.__module__ == "
                "'mikazuki.direct_trainer_admission_intent'\n"
            )
        with tempfile.TemporaryDirectory() as tmp:
            script_dir = ROOT / Path(path).parent
            result = _child(statement, script_dir, Path(tmp))
        if result.returncode != 0:
            self.fail(
                f"{MARKER}: {path} first-party import fails in clean child "
                f"process (exit={result.returncode}): {result.stderr[-900:]}"
            )

    def _assert_full_trainer_toml_cold_process(self, path: str):
        code = ast.get_source_segment(_source(path), _node(path, "read_config_from_file"))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "full.toml"
            cfg.write_text("[training]\nlearning_rate = 0.0003\n", encoding="utf-8")
            # Execute the actual stored TOML Reader in a fresh stdlib-only
            # child, with the same sys.path behavior as a path-run Full
            # Trainer but without Torch/Accelerate model initialization.
            program = (
                "import argparse, os, pathlib, sys, tomllib\n"
                "from types import SimpleNamespace\n"
                "class Toml:\n"
                " @staticmethod\n"
                " def load(stream): return tomllib.loads(stream.read())\n"
                "class Log:\n"
                " @staticmethod\n"
                " def info(msg): pass\n"
                " @staticmethod\n"
                " def error(msg): pass\n"
                "scope={'argparse':argparse,'os':os,'pathlib':pathlib,"
                "'toml':Toml,'logger':Log}\n"
                "exec(compile(" + repr(code) + ", 'real_reader.py', 'exec'), scope)\n"
                "parser=argparse.ArgumentParser()\n"
                "parser.add_argument('--config_file')\n"
                "parser.add_argument('--output_config', action='store_true')\n"
                "parser.add_argument('--learning_rate',type=float)\n"
                "argv=['--config_file'," + repr(str(cfg)) + "]\n"
                "sys.argv=['full_trainer', *argv]\n"
                "out=scope['read_config_from_file'](parser.parse_args(argv),parser)\n"
                "assert out.learning_rate == 0.0003\n"
            )
            root_branch = "scripts/stable" if "/stable/" in path else "scripts/dev"
            result = _child(program, ROOT / root_branch, Path(tmp))
        if result.returncode != 0:
            self.fail(
                f"{MARKER}: {path} ordinary Full Trainer TOML reader "
                "requires DTS root on PYTHONPATH: "
                + result.stderr[-900:]
            )

    def _assert_standard_rejects_ack_before_validation(self, path: str):
        train = _node(path, "train", class_name="NetworkTrainer")
        idx = next(
            i for i, node in enumerate(train.body)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "verify_training_args"
        )
        # Run the real production train() prefix, through the point where
        # actual training verification would begin. It must reject an
        # invalid Standard acknowledgement without preparing the dataset.
        isolated = ast.FunctionDef(
            name="preflight_only", args=train.args, body=train.body[:idx]
            + [ast.Pass()], decorator_list=[], returns=None, type_comment=None
        )
        module = ast.Module(body=[isolated], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "random": random, "time": time,
            "DirectTrainerAdmissionIntent": DirectTrainerAdmissionIntent,
        }
        exec(compile(module, path, "exec"), namespace)
        try:
            namespace["preflight_only"](
                SimpleNamespace(get_parameter_policy_train_type=lambda args: "sd-lora"),
                SimpleNamespace(parameter_policy_config=None),
                DirectTrainerAdmissionIntent(True),
            )
        except ValueError as exc:
            self.assertIn("Component", str(exc))
            return
        self.fail(
            f"{MARKER}: {path} Standard Direct Trainer silently accepted "
            "positive experimental ACK instead of rejecting before validation."
        )

    def _assert_main_push_has_producer_tests(self, path: str):
        source = _source(path)
        lines = source.splitlines()
        start = lines.index("  push:")
        stop = next((i for i in range(start + 1, len(lines))
                     if lines[i].startswith("  pull_request:")), len(lines))
        event = lines[start:stop]
        if not any(item.strip() in ("- main", '- "main"', "- 'main'")
                   for item in event):
            self.fail(
                f"{MARKER}: {path} protects QG1B source only on an old "
                "feature branch, not on future main push."
            )

    def test_mr101_stable_toml_no_ack_must_not_recreate_control(self):
        self._toml_final_namespace("scripts/stable/library/train_util.py", ack=False)

    def test_mr103_stable_toml_positive_ack_must_be_consumed(self):
        self._toml_final_namespace("scripts/stable/library/train_util.py", ack=True)

    def test_mr105_stable_export_positive_ack_must_fail_before_write(self):
        self._output_ack_rejected_before_write("scripts/stable/library/train_util.py")

    def test_mr102_dev_toml_no_ack_must_not_recreate_control(self):
        self._toml_final_namespace("scripts/dev/library/train_util.py", ack=False)

    def test_mr104_dev_toml_positive_ack_must_be_consumed(self):
        self._toml_final_namespace("scripts/dev/library/train_util.py", ack=True)

    def test_mr106_dev_export_positive_ack_must_fail_before_write(self):
        self._output_ack_rejected_before_write("scripts/dev/library/train_util.py")

    def test_mr201_final_parser_order_scripts_stable_train_network(self):
        self._assert_main_reads_before_consuming("scripts/stable/train_network.py")

    def test_mr301_cold_import_scripts_stable_train_network(self):
        self._assert_network_clean_import("scripts/stable/train_network.py")

    def test_mr202_final_parser_order_scripts_dev_train_network(self):
        self._assert_main_reads_before_consuming("scripts/dev/train_network.py")

    def test_mr302_cold_import_scripts_dev_train_network(self):
        self._assert_network_clean_import("scripts/dev/train_network.py")

    def test_mr203_final_parser_order_scripts_stable_sdxl_train_network(self):
        self._assert_main_reads_before_consuming("scripts/stable/sdxl_train_network.py")

    def test_mr303_cold_import_scripts_stable_sdxl_train_network(self):
        self._assert_network_clean_import("scripts/stable/sdxl_train_network.py")

    def test_mr204_final_parser_order_scripts_dev_sdxl_train_network(self):
        self._assert_main_reads_before_consuming("scripts/dev/sdxl_train_network.py")

    def test_mr304_cold_import_scripts_dev_sdxl_train_network(self):
        self._assert_network_clean_import("scripts/dev/sdxl_train_network.py")

    def test_mr205_final_parser_order_scripts_dev_sd3_train_network(self):
        self._assert_main_reads_before_consuming("scripts/dev/sd3_train_network.py")

    def test_mr305_cold_import_scripts_dev_sd3_train_network(self):
        self._assert_network_clean_import("scripts/dev/sd3_train_network.py")

    def test_mr206_final_parser_order_scripts_dev_flux_train_network(self):
        self._assert_main_reads_before_consuming("scripts/dev/flux_train_network.py")

    def test_mr306_cold_import_scripts_dev_flux_train_network(self):
        self._assert_network_clean_import("scripts/dev/flux_train_network.py")

    def test_mr207_final_parser_order_scripts_dev_lumina_train_network(self):
        self._assert_main_reads_before_consuming("scripts/dev/lumina_train_network.py")

    def test_mr307_cold_import_scripts_dev_lumina_train_network(self):
        self._assert_network_clean_import("scripts/dev/lumina_train_network.py")

    def test_mr401_stable_full_trainer_toml_cold_process(self):
        self._assert_full_trainer_toml_cold_process("scripts/stable/library/train_util.py")

    def test_mr501_stable_standard_positive_ack_rejected(self):
        self._assert_standard_rejects_ack_before_validation("scripts/stable/train_network.py")

    def test_mr402_dev_full_trainer_toml_cold_process(self):
        self._assert_full_trainer_toml_cold_process("scripts/dev/library/train_util.py")

    def test_mr502_dev_standard_positive_ack_rejected(self):
        self._assert_standard_rejects_ack_before_validation("scripts/dev/train_network.py")

    def test_mr601_qg1b_main_branch_check_1(self):
        self._assert_main_push_has_producer_tests(".github/workflows/qg1b-trusted-producer-host.yml")

    def test_mr602_qg1b_main_branch_check_2(self):
        self._assert_main_push_has_producer_tests(".github/workflows/qg1b-sr1-host.yml")


class QG2MRSR1Baselines(unittest.TestCase):
    def test_mr901_one_shot_extraction_strict_and_never_replayed(self):
        parser = _parser()
        args = parser.parse_args(["--experimental_full_bf16_ack"])
        intent = extract_direct_trainer_admission_intent(args)
        self.assertIs(type(intent), DirectTrainerAdmissionIntent)
        self.assertIs(intent.experimental_full_bf16_ack, True)
        self.assertNotIn(ACK_ARGUMENT, vars(args))
        self.assertIs(
            extract_direct_trainer_admission_intent(args).experimental_full_bf16_ack,
            False,
        )

    def test_mr902_toml_rejects_reserved_key_on_both_reader_paths(self):
        for path in READERS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                cfg = Path(tmp) / "injected.toml"
                cfg.write_text(
                    "[training]\nexperimental_full_bf16_ack = false\n",
                    encoding="utf-8",
                )
                parser = _parser()
                cli = ["--config_file", str(cfg)]
                with patch.object(sys, "argv", ["trainer", *cli]):
                    with self.assertRaisesRegex(ValueError, ACK_ARGUMENT):
                        _reader(path)(parser.parse_args(cli), parser)

    def test_mr903_existing_bridge_finds_dts_root_without_torch(self):
        for branch in ("scripts/stable", "scripts/dev"):
            with self.subTest(branch=branch), tempfile.TemporaryDirectory() as tmp:
                result = _child(
                    "from library import dts_parameter_policy_bridge as bridge\n"
                    "root = bridge._find_dts_root()\n"
                    "assert (root / 'mikazuki' / 'parameter_policy_trainer.py').is_file()\n",
                    ROOT / branch, Path(tmp),
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_mr904_standard_neutral_preflight_has_no_new_rejection(self):
        for path in NETWORKS[:2]:
            with self.subTest(path=path):
                train = _node(path, "train", class_name="NetworkTrainer")
                idx = next(i for i, node in enumerate(train.body)
                           if isinstance(node, ast.Expr)
                           and isinstance(node.value, ast.Call)
                           and isinstance(node.value.func, ast.Attribute)
                           and node.value.func.attr == "verify_training_args")
                isolated = ast.FunctionDef(
                    name="preflight_only", args=train.args, body=train.body[:idx]
                    + [ast.Pass()], decorator_list=[], returns=None, type_comment=None
                )
                mod = ast.Module(body=[isolated], type_ignores=[])
                ast.fix_missing_locations(mod)
                ns = {"random": random, "time": time,
                      "DirectTrainerAdmissionIntent": DirectTrainerAdmissionIntent}
                exec(compile(mod, path, "exec"), ns)
                ns["preflight_only"](
                    SimpleNamespace(
                        get_parameter_policy_train_type=lambda args: "sd-lora"
                    ),
                    SimpleNamespace(parameter_policy_config=None),
                    DirectTrainerAdmissionIntent(False),
                )
