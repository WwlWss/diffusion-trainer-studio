"""QG-2-MR-R2-2-SR1-A: frozen-HEAD GPU evidence migration Tests-only RED.

The seven targeted obligations must fail on a317f39b (each with RED_MARKER).
The six independent, already-implemented safety baselines must pass.  This is
source-contract evidence: it does not launch CUDA, import Torch or certify a
Full-BF16 production backend.  See docs/qg2-mr-r2-2-sr1-a-tests-only-red.md.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_RUNNER = ROOT / "tools" / "run_parameter_policy_execution_gpu_matrix.py"
BASELINE_RUNNER = ROOT / "tools" / "run_parameter_policy_gpu_matrix.py"
GPU_FIXTURE = ROOT / "tools" / "_parameter_policy_gpu_low_level_fixture.py"
GPU_SUPPORT = ROOT / "tools" / "parameter_policy_execution_gpu_support.py"
CPU_FIXTURE = ROOT / "tests" / "_parameter_policy_low_level_fixtures.py"
CPU_SMOKE = ROOT / "tests" / "test_parameter_policy_trainer_runtime_smoke.py"
SESSION = ROOT / "mikazuki" / "parameter_policy_trainer.py"
ADMISSION = ROOT / "mikazuki" / "parameter_policy_admission.py"
READINESS = ROOT / "mikazuki" / "parameter_policy_readiness_producer.py"
INTENT = ROOT / "mikazuki" / "direct_trainer_admission_intent.py"
RED_MARKER = "QG-2-MR-R2-2-SR1 GPU HARNESS MIGRATION ABSENT"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(_read(path), filename=str(path))
    matches = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise AssertionError(f"Expected exactly one top-level function {name!r} in {path}")
    return matches[0]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child) == name
    ]


class QG2MRR22SR1TargetRed(unittest.TestCase):
    """Seven real missing obligations; do not mark expected failures as xfail."""

    def _assert_isolated_gpu_session(self, function_name: str) -> None:
        function = _function(EXECUTION_RUNNER, function_name)
        legacy = _calls(function, "create_parameter_policy_session")
        isolated = _calls(function, "build_gpu_evidence_full_bf16_session")
        self.assertEqual(
            len(legacy), 0,
            f"{RED_MARKER}: {function_name} still enters production Late Admission "
            "for the unqualified synthetic flux-finetune Full-BF16 backend.",
        )
        self.assertEqual(
            len(isolated), 1,
            f"{RED_MARKER}: {function_name} must construct exactly one "
            "isolated, test-only low-level GPU evidence session.",
        )

    def test_R201_infra_full_bf16_session_uses_isolated_evidence_builder(self):
        self._assert_isolated_gpu_session("_case_full_bf16_session")

    def test_R202_adamw_full_bf16_session_uses_isolated_evidence_builder(self):
        self._assert_isolated_gpu_session("_build_adamw_runtime")

    def test_R203_muon_full_bf16_session_uses_isolated_evidence_builder(self):
        self._assert_isolated_gpu_session("_build_muon_runtime")

    def test_R204_explicit_fallback_session_uses_isolated_evidence_builder(self):
        self._assert_isolated_gpu_session("_build_c4_runtime")

    def test_R205_baseline_device_audit_uses_real_typed_first_party_admission(self):
        function = _function(BASELINE_RUNNER, "_accelerator_device_audit")
        sessions = _calls(function, "create_parameter_policy_session")
        self.assertEqual(len(sessions), 1, "Device audit must retain one real Session call")
        keywords = {kw.arg for kw in sessions[0].keywords if kw.arg is not None}
        self.assertTrue(
            {"admission_snapshot", "admission_intent"}.issubset(keywords),
            f"{RED_MARKER}: baseline GPU device audit lacks both mandatory "
            "first-party Late Admission keyword arguments.",
        )
        self.assertEqual(
            len(_calls(function, "prepare_direct_trainer_admission")), 1,
            f"{RED_MARKER}: baseline GPU device audit must create its snapshot "
            "through the real Host factory, not an invented admission status.",
        )
        self.assertEqual(
            len(_calls(function, "DirectTrainerAdmissionIntent")), 1,
            f"{RED_MARKER}: baseline GPU device audit needs an explicit typed "
            "false acknowledgement; never serialize or manufacture an ACK.",
        )

    def test_R206_gpu_low_level_fixture_is_reviewed_and_restricted(self):
        self.assertTrue(
            GPU_FIXTURE.is_file(),
            f"{RED_MARKER}: reviewed GPU-only low-level fixture module is absent.",
        )
        source = _read(GPU_FIXTURE)
        required = (
            "def build_gpu_evidence_full_bf16_session(",
            "flux-finetune", "full_bf16", "mixed_precision",
            "build_parameter_policy_execution_contract",
            "scan_parameter_roots", "build_parameter_routing_plan",
            "compile_parameter_policy_runtime_spec",
            "build_parameter_policy_optimizer", "ParameterPolicyTrainerSession",
            "apply_requires_grad_contract", "assert_requires_grad_contract",
        )
        for token in required:
            with self.subTest(required=token):
                self.assertIn(
                    token, source,
                    f"{RED_MARKER}: isolated fixture lost strict scope or "
                    f"a real low-level runtime invariant: {token}",
                )
        self.assertNotIn(
            "create_parameter_policy_session(", source,
            f"{RED_MARKER}: isolated fixture must not call production Late Admission "
            "using forged authorization.",
        )

    def test_R207_gpu_json_distinguishes_isolated_evidence_from_host_admission(self):
        source = _read(EXECUTION_RUNNER)
        self.assertIn(
            '"session_construction_scope": "isolated_gpu_low_level_evidence_only"',
            source,
            f"{RED_MARKER}: evidence must disclose that synthetic flux-finetune "
            "sessions bypass production Admission only inside test-only low-level "
            "construction, never assert backend qualification.",
        )
        self.assertIn('"backend_qualification_eligible": False', source)
        self.assertIn('"optimizer_qualification_eligible": False', source)


class QG2MRR22SR1ExistingBaselines(unittest.TestCase):
    """These established safeguards must stay GREEN during Tests-only RED."""

    def test_R901_production_late_session_keeps_mandatory_typed_arguments(self):
        function = _function(SESSION, "create_parameter_policy_session")
        kwonly = function.args.kwonlyargs
        defaults = function.args.kw_defaults
        names = {arg.arg: default for arg, default in zip(kwonly, defaults)}
        self.assertIn("admission_snapshot", names)
        self.assertIn("admission_intent", names)
        self.assertIsNone(names["admission_snapshot"])
        self.assertIsNone(names["admission_intent"])
        source = _read(SESSION)
        self.assertIn("type(admission_snapshot) is not DirectTrainerAdmissionSnapshot", source)
        self.assertIn("type(admission_intent) is not DirectTrainerAdmissionIntent", source)
        self.assertIn("admission_snapshot.assert_matches(", source)
        self.assertIn('require_assets=True', source)

    def test_R902_flux_full_bf16_remains_real_host_blocked(self):
        source = _read(READINESS)
        self.assertIn('_AUDITED_BACKENDS = frozenset({"sd-lora", "sdxl-lora"})', source)
        self.assertIn("if normalized_backend not in _AUDITED_BACKENDS:", source)
        self.assertIn('"capability.backend.audit_required"', source)
        admission = _read(ADMISSION)
        self.assertIn('"experimental_host_candidate"', admission)
        self.assertIn('"blocked"', admission)
        self.assertIn('"direct_trainer"', admission)
        self.assertIn('late_admission.status in ("blocked", "ack_required", "experimental_host_candidate")', _read(SESSION))

    def test_R903_existing_cpu_only_fixture_preserves_real_low_level_invariants(self):
        source = _read(CPU_FIXTURE)
        for token in (
            'train_type != "flux-finetune"',
            'config.get("full_bf16") is not True',
            'config.get("mixed_precision") != "bf16"',
            "build_parameter_policy_execution_contract(",
            "scan_parameter_roots(",
            "build_parameter_routing_plan(",
            "compile_parameter_policy_runtime_spec(",
            "build_parameter_policy_optimizer(",
            "ParameterPolicyTrainerSession(",
            "apply_requires_grad_contract()",
            "assert_requires_grad_contract()",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn("create_parameter_policy_session(", source)

    def test_R904_real_rejection_test_keeps_pre_scan_protection(self):
        function = _function(CPU_SMOKE, "test_full_bf16_direct_trainer_fails_before_parameter_scan")
        self.assertEqual(len(_calls(function, "create_parameter_policy_session")), 1)
        self.assertEqual(len(_calls(function, "make_test_admission_kwargs")), 1)
        src = ast.get_source_segment(_read(CPU_SMOKE), function) or ""
        self.assertIn("expected_status=\"blocked\"", src)
        self.assertIn("scan.assert_not_called()", src)

    def test_R905_exact_head_gpu_worker_and_evidence_provenance_are_unchanged(self):
        source = _read(EXECUTION_RUNNER)
        self.assertIn("_BOOTSTRAP_COMMIT = _assert_exact_clean_head", source)
        self.assertLess(source.index("_BOOTSTRAP_COMMIT = _assert_exact_clean_head"), source.index("import torch"))
        for token in (
            '"status", "--porcelain=v1", "--untracked-files=all"',
            "_assert_output_outside_repo(",
            "_assert_worker_dir_outside_repo(",
            "_assert_worker_result_in_dir(",
            '"--expected-commit"',
            "sys.executable", "shell=False", "subprocess.run(",
            '"final_provenance_commit"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertIn('"production_qualification_mutated": False', _read(GPU_SUPPORT))

    def test_R906_frozen_eight_gpu_cases_and_fourteen_phases_remain(self):
        source = _read(GPU_SUPPORT)
        for case_id in (
            "infra:cuda-bf16-capability:v1",
            "infra:full-bf16-session:v1",
            "optimizer:adamw:full-bf16:accum1:v1",
            "optimizer:adamw:full-bf16:accum2:v1",
            "optimizer:muon:full-bf16:accum1:v1",
            "optimizer:muon:full-bf16:accum2:v1",
            "optimizer:muon-adamw-explicit-fallback:full-bf16:accum1:v1",
            "optimizer:muon-adamw-explicit-fallback:full-bf16:accum2:v1",
        ):
            with self.subTest(case_id=case_id):
                self.assertIn(f'"{case_id}"', source)
        self.assertIn("EXECUTION_GPU_CASE_PHASES: dict[str, tuple[str, ...]]", source)
        self.assertIn('"train_save"', source)
        self.assertIn('"resume_second_step"', source)
        self.assertIn('"probe"', source)
        self.assertIn("EXECUTION_GPU_EVIDENCE_VERSION = 3", source)


if __name__ == "__main__":
    unittest.main()
