from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_parameter_policy_execution_gpu_matrix.py"
BASELINE_RUNNER = ROOT / "tools" / "run_parameter_policy_gpu_matrix.py"
SUPPORT = ROOT / "tools" / "parameter_policy_execution_gpu_support.py"
GPU_WORKFLOW = ROOT / ".github" / "workflows" / "parameter-policy-gpu-matrix.yml"
HOST_WORKFLOW = ROOT / ".github" / "workflows" / "anima-qwen3-review.yml"


class ParameterPolicyExecutionGpuHarnessContractTests(unittest.TestCase):
    def test_runner_is_exact_head_subprocess_evidence_harness(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('EVIDENCE_SCHEMA = "dts.parameter-policy.execution-gpu-matrix"', source)
        self.assertIn('EVIDENCE_VERSION = 2', source)
        self.assertIn('"infra:cuda-bf16-capability:v1"', source)
        self.assertIn('"infra:full-bf16-session:v1"', source)
        self.assertIn('"--expected-commit"', source)
        self.assertIn('"status", "--porcelain=v1", "--untracked-files=all"', source)
        self.assertIn("_assert_output_outside_repo", source)
        self.assertLess(
            source.index("_BOOTSTRAP_COMMIT = _assert_exact_clean_head"),
            source.index("import torch"),
        )
        self.assertIn("subprocess.run(", source)
        self.assertIn("sys.executable", source)
        self.assertIn("shell=False", source)
        self.assertIn('"promotion_eligible": False', source)
        self.assertIn('"backend_qualification_eligible": False', source)
        self.assertIn('"optimizer_qualification_eligible": False', source)

    def test_strict_runners_reject_untracked_workspace_before_runtime_imports(self):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        probe = ROOT / "phase-c-provenance-untracked.tmp"
        try:
            probe.write_text("probe\n", encoding="utf-8")
            cases = (
                (
                    RUNNER,
                    [
                        "--expected-commit",
                        head,
                        "--output",
                        str(ROOT.parent / "execution-probe.json"),
                    ],
                ),
                (
                    BASELINE_RUNNER,
                    [
                        "--expected-commit",
                        head,
                        "--output",
                        str(ROOT.parent / "baseline-probe.json"),
                    ],
                ),
            )
            for runner, args in cases:
                with self.subTest(runner=runner.name):
                    completed = subprocess.run(
                        [sys.executable, str(runner), *args],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    combined = completed.stdout + completed.stderr
                    self.assertIn("completely clean workspace", combined)
                    self.assertIn(probe.name, combined)
        finally:
            probe.unlink(missing_ok=True)
            (ROOT.parent / "execution-probe.json").unlink(missing_ok=True)
            (ROOT.parent / "baseline-probe.json").unlink(missing_ok=True)

    def test_strict_runners_reject_repository_internal_output_before_cuda(self):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        cases = (
            (RUNNER, ROOT / "execution-evidence.json"),
            (BASELINE_RUNNER, ROOT / "baseline-evidence.json"),
        )
        for runner, output in cases:
            with self.subTest(runner=runner.name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "--expected-commit",
                        head,
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                combined = completed.stdout + completed.stderr
                self.assertIn("--output outside", combined)
                self.assertFalse(output.exists())

    def test_execution_worker_rejects_repository_internal_worker_dir_before_cuda(self):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        result_path = ROOT.parent / "worker-result.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--expected-commit",
                head,
                "--worker-case",
                "infra:cuda-bf16-capability:v1",
                "--worker-phase",
                "probe",
                "--worker-result",
                str(result_path),
                "--worker-dir",
                str(ROOT / "worker-temp"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--worker-dir outside", completed.stdout + completed.stderr)
        self.assertFalse(result_path.exists())

    def test_execution_worker_result_must_stay_inside_worker_dir(self):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        worker_dir = ROOT.parent / "c2-worker-temp"
        bad_result = ROOT.parent / "c2-worker-result-outside.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--expected-commit",
                head,
                "--worker-case",
                "infra:cuda-bf16-capability:v1",
                "--worker-phase",
                "probe",
                "--worker-result",
                str(bad_result),
                "--worker-dir",
                str(worker_dir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "--worker-result inside --worker-dir",
            completed.stdout + completed.stderr,
        )
        self.assertFalse(bad_result.exists())

    def test_baseline_runner_keeps_runtime_simplenamespace_import(self):
        source = BASELINE_RUNNER.read_text(encoding="utf-8")
        self.assertIn("from types import SimpleNamespace", source)
        self.assertIn("SimpleNamespace(", source)

    def test_baseline_runner_strict_mode_checks_provenance_before_torch_import(self):
        source = BASELINE_RUNNER.read_text(encoding="utf-8")
        self.assertIn('"--expected-commit"', source)
        self.assertIn('"status", "--porcelain=v1", "--untracked-files=all"', source)
        self.assertLess(
            source.index("_STRICT_COMMIT = ("),
            source.index("import torch"),
        )

    def test_adamw_cases_use_phased_fresh_subprocess_protocol(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"optimizer:adamw:full-bf16:accum1:v1"', source)
        self.assertIn('"optimizer:adamw:full-bf16:accum2:v1"', source)
        self.assertIn('"train_save"', source)
        self.assertIn('"resume_second_step"', source)
        self.assertIn('"dependency_failed"', source)
        support = SUPPORT.read_text(encoding="utf-8")
        self.assertIn('"incomplete"', support)
        self.assertIn("summarize_adamw_full_bf16_bundle", source)
        self.assertIn("ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID", source)
        self.assertIn('"qualification_evidence_component": True', source)
        self.assertIn('"production_qualification_mutated": False', support)
        self.assertIn("--worker-dir", source)
        self.assertIn("_run_phase_subprocess", source)
        self.assertIn("sys.executable", source)

    def test_adamw_evidence_does_not_modify_production_qualification_table(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn('FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"] =', source)
        self.assertNotIn("FULL_BF16_OPTIMIZER_QUALIFICATIONS.update", source)

    def test_runner_has_no_environment_bypass_or_production_promotion(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("DTS_FULL_BF16_BYPASS", source)
        self.assertNotIn("os.environ[", source)
        self.assertNotIn('ExecutionFeatureQualification(\n        "qualified"', source)
        support = SUPPORT.read_text(encoding="utf-8")
        self.assertIn("temporary_execution_qualification", source)
        self.assertIn("temporary_execution_qualification", support)
        self.assertIn("marked qualified without an", support)
        self.assertIn("evidence_case_id", support)
        self.assertIn("finally:", support)

    def test_execution_qualification_is_not_coupled_to_github_gpu_workflow(self):
        workflow = GPU_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn(
            "run_parameter_policy_execution_gpu_matrix.py",
            workflow,
        )
        self.assertNotIn(
            "parameter-policy-execution-gpu-matrix.json",
            workflow,
        )

    def test_host_review_tracks_and_compiles_execution_gpu_runner(self):
        workflow = HOST_WORKFLOW.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            workflow.count("tools/run_parameter_policy_execution_gpu_matrix.py"),
            2,
        )

    def test_runner_remains_standalone_provider_independent(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--expected-commit", required=True)', source)
        self.assertIn('default="parameter-policy-execution-gpu-matrix.json"', source)
        self.assertNotIn("GITHUB_SHA", source)
        self.assertNotIn("GITHUB_ACTIONS", source)


if __name__ == "__main__":
    unittest.main()
