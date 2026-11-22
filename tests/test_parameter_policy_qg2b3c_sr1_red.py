"""QG-2-B3-C-SR1 tests-only RED: historical real CPU ABI and pre-apply identity.

No Torch import in this suite; real Torch CPU methods are exercised by the
independent SR1 workflow job with explicit runtime prerequisites.
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_admission import (
    DirectTrainerAdmissionSnapshot,
    assess_component_training_admission,
    prepare_direct_trainer_admission,
)
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
    FULL_BF16_OPTIMIZER_QUALIFICATIONS,
)
from tests.test_parameter_policy_qg2b3a_snapshot_red import _config, _policy, _write_policy

ROOT = Path(__file__).resolve().parents[1]
MARKER = "QG-2-B3-C-SR1 HISTORICAL CPU CONTRACT ABSENT"
RUNTIME = "tests/test_parameter_policy_trainer_runtime_smoke.py"
FULL = "tests/test_parameter_policy_full_runtime_smoke.py"
NETWORK = "tests/test_parameter_policy_network_runtime_smoke.py"
ANIMA = "tests/test_parameter_policy_anima_runtime_smoke.py"
TRAINERS = ("scripts/stable/train_network.py", "scripts/dev/train_network.py")
SESSION_NAME = "create_parameter_policy_session"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _tree(path: str) -> ast.Module:
    return ast.parse(_read(path), filename=path)


def _calls(path: str, method: str = SESSION_NAME) -> list[ast.Call]:
    return sorted(
        (node for node in ast.walk(_tree(path))
         if isinstance(node, ast.Call)
         and isinstance(node.func, ast.Name)
         and node.func.id == method),
        key=lambda node: node.lineno,
    )


def _named_method(path: str, method_name: str) -> ast.FunctionDef:
    candidates = (
        node for node in ast.walk(_tree(path))
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return next(candidates)


def _method_calls(path: str, method_name: str, called_name: str = SESSION_NAME) -> list[ast.Call]:
    return [
        node for node in ast.walk(_named_method(path, method_name))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == called_name
    ]


def _has_typed_kwargs(call: ast.Call) -> bool:
    keys = {arg.arg for arg in call.keywords}
    return {"admission_snapshot", "admission_intent"} <= keys


def _trainer_body(path: str) -> str:
    trainer = next(
        node for node in _tree(path).body
        if isinstance(node, ast.ClassDef) and node.name == "NetworkTrainer"
    )
    method = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    return ast.get_source_segment(_read(path), method) or ""


class QG2B3CSR1TargetedRed(unittest.TestCase):
    def need(self, actual: bool, detail: str) -> None:
        if not actual:
            self.fail(f"{MARKER}: {detail}")

    def test_sr101_trainer_runtime_history_uses_required_typed_late_identity(self):
        calls = _calls(RUNTIME)
        self.need(bool(calls), "original Trainer Torch runtime smoke calls have disappeared")
        missing = [call.lineno for call in calls if not _has_typed_kwargs(call)]
        self.need(not missing, f"Trainer-runtime historical Session calls missing typed identity at {missing}")

    def test_sr102_full_model_history_uses_required_typed_late_identity(self):
        calls = _calls(FULL)
        self.need(bool(calls), "original full-model Torch smoke coverage has disappeared")
        missing = [call.lineno for call in calls if not _has_typed_kwargs(call)]
        self.need(not missing, f"full-model historical Session calls missing typed identity at {missing}")

    def test_sr103_network_and_anima_history_uses_required_typed_late_identity(self):
        for path in (NETWORK, ANIMA):
            calls = _calls(path)
            self.need(bool(calls), f"{path} historical CPU Torch Session assertion was lost")
            missing = [call.lineno for call in calls if not _has_typed_kwargs(call)]
            self.need(not missing, f"{path} missing typed first-party snapshot and one-shot intent at {missing}")

    def test_sr104_mock_full_bf16_tests_do_not_use_public_training_admission(self):
        # Old mocked-release Flux Full-BF16 tests protect low-level runtime and
        # checkpoint behavior. They must not promote this unreviewed backend.
        for method in (
            "test_mock_qualified_full_bf16_session_captures_contract_and_weak_roots",
            "test_dead_execution_root_fails_closed_without_strong_module_reference",
            "_build_mock_qualified_full_bf16_session",
        ):
            public = _method_calls(RUNTIME, method)
            self.need(not public,
                      f"mock-qualified low-level {method} still invokes real Direct Trainer admission "
                      f"at {[item.lineno for item in public]}")
        # The rest of the tests still run; this is not permission to skip or
        # delete their optimizer, dtype, scheduler, root and resume assertions.

    def test_sr105_true_full_bf16_rejection_has_typed_identity_and_no_scan(self):
        method = "test_full_bf16_direct_trainer_fails_before_parameter_scan"
        statements = _named_method(RUNTIME, method)
        segment = ast.get_source_segment(_read(RUNTIME), statements) or ""
        calls = _method_calls(RUNTIME, method)
        self.need(len(calls) == 1,
                  "genuine Direct Trainer Full-BF16 rejection assertion was removed")
        self.need(_has_typed_kwargs(calls[0]),
                  "genuine Full-BF16 rejection must pass actual blocked snapshot + typed intent")
        self.assertIn("scan.assert_not_called()", segment)
        self.assertIn("assertRaisesRegex", segment)

    def test_sr106_both_trainers_revalidate_snapshot_before_lora_apply(self):
        # B3-C Late remains AFTER apply_to for actual LoRA parameter roots;
        # this only requires an additional read-only fail-closed Pre-Apply gate.
        for path in TRAINERS:
            source = _trainer_body(path)
            pre = source.index("network.prepare_network(args)")
            apply = source.index("network.apply_to(", pre)
            late = source.index("create_parameter_policy_session(", apply)
            self.assertLess(pre, apply)
            self.assertLess(apply, late)
            segment = source[pre:apply]
            self.need(
                "preflight_direct_trainer_admission(" in segment
                and "assert_matches(" in segment
                and "admission_snapshot.admission" in segment,
                f"{path} lacks pure four-coordinate, first-party Pre-Apply "
                "identity recheck before mutation of the real LoRA model",
            )


class QG2B3CSR1Baselines(unittest.TestCase):
    def test_sr901_real_typed_first_party_snapshot_is_strict_and_inactive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = prepare_direct_trainer_admission(
                path,
                train_type="sd-lora",
                effective_config=_config(full_bf16=False),
                admission_intent=intent,
            )
            self.assertIs(type(snapshot), DirectTrainerAdmissionSnapshot)
            self.assertEqual(snapshot.admission.status, "inactive")
            self.assertEqual(len(snapshot.policy_hash), 64)
            self.assertEqual(len(snapshot.effective_config_fingerprint), 64)

    def test_sr902_mock_release_rows_cannot_unlock_flux_full_bf16(self):
        fake = ExecutionFeatureQualification(
            "qualified", "test-only evidence", "test:flux",
        )
        with patch.dict(
            FULL_BF16_BACKEND_QUALIFICATIONS,
            {"flux-finetune": fake},
        ), patch.dict(
            FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": fake},
        ):
            for ack in (False, True):
                with self.subTest(ack=ack):
                    admission = assess_component_training_admission(
                        _policy("flux-finetune"),
                        train_type="flux-finetune",
                        effective_config=_config(full_bf16=True),
                        purpose="direct_trainer",
                        experimental_ack=ack,
                    )
                    self.assertEqual(admission.status, "blocked")
                    self.assertFalse(admission.host_admissible)

    def test_sr903_real_late_session_identity_checks_precede_scan_and_optimizer(self):
        code = _read("mikazuki/parameter_policy_trainer.py")
        body = code[code.index("def create_parameter_policy_session("):]
        ordered = (
            "admission_snapshot.assert_matches(",
            "assess_component_training_admission(",
            "parameter_policy_execution_blockers(",
            "scan_parameter_roots(",
            "build_parameter_policy_optimizer(",
        )
        position = [body.index(item) for item in ordered]
        self.assertEqual(position, sorted(position))
        self.assertIn("require_assets=True", body)
        self.assertIn("experimental_host_candidate", body)
        self.assertIn("if type(admission_intent) is not DirectTrainerAdmissionIntent:", body)

    def test_sr904_late_runtime_scheduler_identity_remains_resolved_after_loader(self):
        # These are distinct identity layers; no premature Scheduler computation.
        module = _read("mikazuki/parameter_policy_trainer.py")
        self.assertIn("def _scheduler_identity_from_args(", module)
        self.assertIn('config.get("max_train_steps")', module)
        self.assertIn("self.scheduler_signature = signature", module)
        for path in TRAINERS:
            body = _trainer_body(path)
            self.assertLess(
                body.index("create_parameter_policy_session("),
                body.index("args.max_train_steps ="),
            )
            self.assertLess(
                body.index("args.max_train_steps ="),
                body.index("args.save_every_n_epochs ="),
            )

    def test_sr905_standard_does_not_require_typed_component_intent(self):
        # MR-SR1-D added a Standard-only guard before the Component guard.
        # Do not infer the Component contract from the FIRST textual occurrence
        # of the same strict type check; assert each guarded path independently.
        for path in TRAINERS:
            body = _trainer_body(path)
            self.assertIn("def train(self, args, admission_intent=None):", body)
            standard = "if not parameter_policy_requested and admission_intent is not None:"
            component = (
                "if parameter_policy_requested:\n"
                "            if type(admission_intent) is not DirectTrainerAdmissionIntent:"
            )
            self.assertIn(standard, body)
            self.assertIn("if admission_intent.experimental_full_bf16_ack:", body)
            self.assertIn(component, body)
            self.assertLess(
                body.index(standard),
                body.index("train_util.verify_training_args("),
            )
            self.assertLess(
                body.index("train_util.verify_training_args("),
                body.index(component),
            )


if __name__ == "__main__":
    unittest.main()
