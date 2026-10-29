"""Component-only runtime boolean fail-closed contract (host, no Torch)."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import tomllib
import unittest

from mikazuki import parameter_policy_execution as execution
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers
from mikazuki.parameter_policy_runtime_bools import (
    _COMPONENT_RUNTIME_BOOLEAN_FIELDS,
    component_runtime_boolean_type_blockers,
    require_component_runtime_booleans,
    strict_component_runtime_bool,
)


ROOT = Path(__file__).resolve().parents[1]


class ComponentRuntimeBooleanContractTests(unittest.TestCase):
    def test_strict_type_contract_never_coerces_quoted_toml_strings(self):
        for input_, expected in (
            (None, False), (False, False), (True, True),
            (0, False), (0.0, False), (1, True), (1.0, True),
        ):
            with self.subTest(value=input_):
                self.assertIs(
                    strict_component_runtime_bool(input_, field="full_bf16"),
                    expected,
                )
        for bad in (
            "", "false", "0", "off", "no", "true", "1", "on", "yes",
            "maybe", 2, -1, 0.5, float("nan"), float("inf"),
            float("-inf"), [], {}, object(),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                ValueError, "full_bf16",
            ):
                strict_component_runtime_bool(bad, field="full_bf16")

    def test_all_reviewed_runtime_switches_reject_pseudo_booleans(self):
        for field in sorted(_COMPONENT_RUNTIME_BOOLEAN_FIELDS):
            for raw in ("false", "0", "off", "no", "", "true", "1", "on"):
                with self.subTest(field=field, raw=raw):
                    issues = component_runtime_boolean_type_blockers(
                        {field: raw}, train_type="sd3-lora",
                    )
                    self.assertEqual(len(issues), 1, issues)
                    self.assertIn(field, issues[0])
        self.assertEqual(
            component_runtime_boolean_type_blockers(
                {field: False for field in _COMPONENT_RUNTIME_BOOLEAN_FIELDS},
                train_type="sd3-lora",
            ),
            [],
        )

    def test_backend_specific_v2_and_numeric_optimizer_fields(self):
        self.assertIn(
            "v2", " ".join(component_runtime_boolean_type_blockers(
                {"v2": "false"}, train_type="sd-lora",
            )),
        )
        self.assertEqual(
            component_runtime_boolean_type_blockers(
                {"v2": "false"}, train_type="sd3-lora",
            ),
            [],
        )
        self.assertEqual(
            component_runtime_boolean_type_blockers(
                {"blocks_to_swap": 2, "fused_optimizer_groups": 4},
                train_type="sd3-lora",
            ),
            [],
        )

    def test_toml_to_argparse_namespace_reproduces_actual_truthiness_mismatch(self):
        raw = tomllib.loads('full_bf16 = "false"\nfp8_base = "false"\n')
        args = argparse.Namespace(**raw)
        self.assertIs(type(args.full_bf16), str)
        self.assertTrue(bool(args.full_bf16))
        self.assertTrue(bool(args.fp8_base))
        before = copy.deepcopy(vars(args))
        with self.assertRaisesRegex(ValueError, "full_bf16"):
            require_component_runtime_booleans(
                vars(args), train_type="sd3-lora",
            )
        self.assertEqual(vars(args), before)

    def test_early_bridge_preflight_is_lazy_and_rejects_unsafe_flags(self):
        for family in ("dev", "stable"):
            with self.subTest(family=family):
                path = ROOT / "scripts" / family / "library" / "dts_parameter_policy_bridge.py"
                spec = importlib.util.spec_from_file_location(
                    "test_" + family + "_boolean_bridge", path,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                args = argparse.Namespace(fp8_base="false")
                with self.assertRaisesRegex(ValueError, "fp8_base"):
                    module.preflight_component_runtime_booleans(
                        args=args, train_type="sd3-lora",
                    )
                module.preflight_component_runtime_booleans(
                    args=argparse.Namespace(fp8_base=False),
                    train_type="sd3-lora",
                )

    def test_trainer_entrypoints_gate_only_component_before_other_work(self):
        for path in (
            "scripts/dev/train_network.py",
            "scripts/dev/flux_train.py",
            "scripts/stable/train_network.py",
            "scripts/stable/train_db.py",
            "scripts/stable/sdxl_train.py",
        ):
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                tree = ast.parse(text, filename=path)
                calls = [
                    n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "preflight_component_runtime_booleans"
                ]
                self.assertEqual(len(calls), 1)
                early = calls[0].lineno
                verify = [
                    n.lineno for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "verify_training_args"
                ]
                self.assertTrue(verify, path)
                self.assertLess(early, min(verify), path)
                self.assertIn("if parameter_policy_requested:", text)

    def test_shared_compatibility_gate_reports_blockers_before_existing_semantics(self):
        for flag in ("full_bf16", "full_fp16", "fp8_base", "deepspeed", "torch_compile"):
            with self.subTest(flag=flag):
                problems = parameter_policy_v1_semantic_blockers(
                    {flag: "false"}, "flux-finetune",
                )
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(flag, problems[0])
        self.assertEqual(
            parameter_policy_v1_semantic_blockers(
                {"full_bf16": False}, "flux-finetune",
            ),
            [],
        )

    def test_execution_feature_guard_rejects_string_instead_of_silently_inactive(self):
        for raw in ('false', '0', 'off', 'true', '1'):
            with self.subTest(raw=raw):
                blockers = execution.parameter_policy_execution_blockers(
                    {"not": "canonical"},
                    train_type="sd3-lora",
                    effective_config={"full_bf16": raw},
                )
                self.assertEqual(len(blockers), 1)
                self.assertIn("full_bf16", blockers[0])
                with self.assertRaisesRegex(ValueError, "full_bf16"):
                    execution.build_parameter_policy_execution_contract(
                        train_type="sd3-lora",
                        effective_config={"full_bf16": raw},
                    )

    def test_real_booleans_preserve_existing_feature_and_pending_semantics(self):
        self.assertEqual(
            execution.active_parameter_policy_execution_features(
                {"full_bf16": False},
            ),
            (),
        )
        self.assertEqual(
            execution.active_parameter_policy_execution_features(
                {"full_bf16": True},
            ),
            ("full_bf16",),
        )
        self.assertEqual(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"].status,
            "pending",
        )


if __name__ == "__main__":
    unittest.main()
