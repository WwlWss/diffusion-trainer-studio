"""QG-2-B4-0 B312 contract-rebase baselines, Host-only and fail-closed.

These tests prove the previous five B312 diagnostics were stale. The cumulative
B312 case remains ONE intentional RED for precisely the two missing live
API -> Launcher -> child CLI links. Do not change production authorization.
"""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy import load_parameter_policy_sidecar
from mikazuki.parameter_policy_admission import (
    DirectTrainerAdmissionSnapshot,
    assess_component_training_admission,
    prepare_direct_trainer_admission,
)
from mikazuki.parameter_policy_execution import FULL_BF16_BACKEND_QUALIFICATIONS
from tests.test_parameter_policy_admission_red import (
    _call_lines, _config, _pending, _policy, _source, _source_fn, _source_method,
)
from tests.test_parameter_policy_qg2b3a_snapshot_red import _write_policy


ROOT = Path(__file__).resolve().parents[1]
TRAINERS = ("scripts/stable/train_network.py", "scripts/dev/train_network.py")
BRIDGES = (
    "scripts/stable/library/dts_parameter_policy_bridge.py",
    "scripts/dev/library/dts_parameter_policy_bridge.py",
)


def _real_torch_imports(path):
    tree = ast.parse(_source(path), filename=path)
    return [
        node.lineno for node in ast.walk(tree)
        if (
            isinstance(node, ast.Import)
            and any(
                item.name == "torch" or item.name.startswith("torch.")
                for item in node.names
            )
        ) or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "torch" or node.module.startswith("torch."))
        )
    ]


class QG2B40RebasedBaselines(unittest.TestCase):
    def test_b4001_real_strict_sidecar_loader_is_torch_free_ast(self):
        self.assertFalse(_real_torch_imports("mikazuki/parameter_policy.py"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            _write_policy(path)
            canonical, digest = load_parameter_policy_sidecar(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(canonical["version"], 1)
            self.assertTrue(canonical["components"])

    def test_b4002_stable_dev_have_real_initial_and_final_host_gates(self):
        for path in TRAINERS:
            with self.subTest(path=path):
                _, method = _source_method(path, "NetworkTrainer", "train")
                preflight = _call_lines(method, "preflight_direct_trainer_admission")
                dataset = _call_lines(method, "prepare_dataset_args")
                extra = _call_lines(method, "assert_extra_args")
                accelerator = _call_lines(method, "prepare_accelerator")
                self.assertGreaterEqual(len(preflight), 3)
                self.assertEqual(len(dataset), 1)
                self.assertEqual(len(extra), 1)
                self.assertEqual(len(accelerator), 1)
                self.assertLess(preflight[0], dataset[0])
                self.assertLess(extra[0], preflight[1])
                self.assertLess(preflight[1], accelerator[0])
                self.assertGreater(preflight[2], accelerator[0])

    def test_b4003_both_pure_bridges_call_actual_first_party_factory(self):
        for path in BRIDGES:
            with self.subTest(path=path):
                text, method = _source_fn(path, "preflight_direct_trainer_admission")
                self.assertEqual(
                    len(_call_lines(method, "prepare_direct_trainer_admission")), 1
                )
                self.assertIn('importlib.import_module("mikazuki.parameter_policy_admission")', text)
                self.assertNotIn("load_parameter_policy_trainer(", text)
                self.assertNotIn("mikazuki.parameter_policy_trainer", text)
                module = ast.parse(_source(path), filename=path)
                self.assertFalse(
                    any(isinstance(n, (ast.Import, ast.ImportFrom))
                        and "torch" in ast.unparse(n) for n in module.body)
                )

    def test_b4004_cumulative_b312_uses_real_api_names_and_ast_imports(self):
        source, method = _source_method(
            "tests/test_parameter_policy_admission_red.py",
            "QG1B3QG2TargetedRed",
            "test_b312_direct_trainer_uses_same_admission_and_explicit_cli_intent",
        )
        self.assertNotIn("preflight_component_training_admission", source)
        self.assertIn('"preflight_direct_trainer_admission"', source)
        self.assertIn("torch_imports = any(", source)
        self.assertIn("ast.ImportFrom", source)
        self.assertIn("ast.Import", source)
        self.assertIn('"prepare_direct_trainer_admission"', source)
        self.assertIn("test_b312_direct_trainer_uses_same_admission", method.name)

    def test_b4005_first_party_statuses_are_checked_without_short_circuit(self):
        old = assess_component_training_admission(
            _policy(), train_type="sd-lora", effective_config=_config(),
            purpose="direct_trainer", experimental_ack=False,
        )
        self.assertEqual(old.status, "legacy_approved")
        self.assertTrue(old.host_admissible)
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            no_ack = assess_component_training_admission(
                _policy(), train_type="sd-lora", effective_config=_config(),
                purpose="direct_trainer", experimental_ack=False,
            )
            with_ack = assess_component_training_admission(
                _policy(), train_type="sd-lora", effective_config=_config(),
                purpose="direct_trainer", experimental_ack=True,
            )
        self.assertEqual(no_ack.status, "ack_required")
        self.assertFalse(no_ack.host_admissible)
        self.assertEqual(with_ack.status, "experimental_host_candidate")
        self.assertTrue(with_ack.host_admissible)
        unsupported = assess_component_training_admission(
            _policy("sd3-lora"), train_type="sd3-lora",
            effective_config=_config(), purpose="direct_trainer",
            experimental_ack=True,
        )
        self.assertEqual(unsupported.status, "blocked")
        self.assertFalse(unsupported.host_admissible)

    def test_b4006_frozen_late_snapshot_and_held_candidate_remain(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            _write_policy(path)
            snapshot = prepare_direct_trainer_admission(
                path, train_type="sd-lora",
                effective_config=_config(full_bf16=False),
                admission_intent=DirectTrainerAdmissionIntent(False),
            )
            self.assertIs(type(snapshot), DirectTrainerAdmissionSnapshot)
            self.assertEqual(snapshot.admission.status, "inactive")
        session_source, method = _source_fn(
            "mikazuki/parameter_policy_trainer.py", "create_parameter_policy_session"
        )
        self.assertIn("admission_snapshot.assert_matches(", session_source)
        self.assertIn('purpose="direct_trainer"', session_source)
        self.assertIn('"experimental_host_candidate"', session_source)
        self.assertIn("require_assets=True", session_source)
        self.assertLess(
            session_source.index("admission_snapshot.assert_matches("),
            session_source.index("scan_parameter_roots("),
        )
        self.assertLess(
            session_source.index("assess_component_training_admission("),
            session_source.index("scan_parameter_roots("),
        )
        self.assertTrue(
            {"admission_snapshot", "admission_intent"}.issubset(
                {arg.arg for arg in method.args.kwonlyargs}
            )
        )


if __name__ == "__main__":
    unittest.main()
