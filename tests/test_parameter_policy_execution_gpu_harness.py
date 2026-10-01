from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_parameter_policy_execution_gpu_matrix.py"
GPU_WORKFLOW = ROOT / ".github" / "workflows" / "parameter-policy-gpu-matrix.yml"
HOST_WORKFLOW = ROOT / ".github" / "workflows" / "anima-qwen3-review.yml"


class ParameterPolicyExecutionGpuHarnessContractTests(unittest.TestCase):
    def test_runner_is_exact_head_subprocess_evidence_harness(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('EVIDENCE_SCHEMA = "dts.parameter-policy.execution-gpu-matrix"', source)
        self.assertIn('EVIDENCE_VERSION = 1', source)
        self.assertIn('"infra:cuda-bf16-capability:v1"', source)
        self.assertIn('"infra:full-bf16-session:v1"', source)
        self.assertIn('"--expected-commit"', source)
        self.assertIn('["git", "diff", "--quiet"]', source)
        self.assertIn('["git", "diff", "--cached", "--quiet"]', source)
        self.assertIn("subprocess.run(", source)
        self.assertIn("sys.executable", source)
        self.assertIn("shell=False", source)
        self.assertIn('"promotion_eligible": False', source)
        self.assertIn('"backend_qualification_eligible": False', source)
        self.assertIn('"optimizer_qualification_eligible": False', source)

    def test_runner_has_no_environment_bypass_or_production_promotion(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("DTS_FULL_BF16_BYPASS", source)
        self.assertNotIn("os.environ[", source)
        self.assertNotIn('ExecutionFeatureQualification(\n        "qualified"', source)
        self.assertIn("_temporary_execution_qualification", source)
        self.assertIn("marked qualified without an", source)
        self.assertIn("evidence_case_id", source)
        self.assertIn("finally:", source)

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
