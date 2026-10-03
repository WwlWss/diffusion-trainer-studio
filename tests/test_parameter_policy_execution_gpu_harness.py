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
        self.assertIn(
            'payload["post_run_commit"] = _assert_exact_clean_head',
            source,
        )
        self.assertIn(
            'evidence["final_provenance_commit"] = _assert_exact_clean_head',
            source,
        )

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

    def test_execution_worker_rejects_duplicate_worker_dir_before_cuda(self):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        worker_dir = ROOT.parent / "c2-worker-duplicate-dir"
        result_path = worker_dir / "result.json"
        unsafe_dir = ROOT / "worker-temp-duplicate"
        try:
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
                    str(worker_dir),
                    "--worker-dir",
                    str(unsafe_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "--worker-dir exactly once",
                completed.stdout + completed.stderr,
            )
            self.assertFalse(result_path.exists())
            self.assertFalse(unsafe_dir.exists())
        finally:
            result_path.unlink(missing_ok=True)
            try:
                worker_dir.rmdir()
            except OSError:
                pass

    def test_execution_worker_rejects_duplicate_worker_result_before_cuda(self):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        worker_dir = ROOT.parent / "c2-worker-duplicate-result"
        result_path = worker_dir / "result.json"
        unsafe_result = ROOT / "worker-result-duplicate.json"
        try:
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
                    "--worker-result",
                    str(unsafe_result),
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
                "--worker-result exactly once",
                completed.stdout + completed.stderr,
            )
            self.assertFalse(result_path.exists())
            self.assertFalse(unsafe_result.exists())
        finally:
            result_path.unlink(missing_ok=True)
            unsafe_result.unlink(missing_ok=True)
            try:
                worker_dir.rmdir()
            except OSError:
                pass

    def test_execution_worker_uses_validated_bootstrap_paths(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("result_path = _BOOTSTRAP_WORKER_RESULT", source)
        self.assertIn("case_dir = _BOOTSTRAP_WORKER_DIR", source)
        self.assertNotIn("result_path = Path(args.worker_result)", source)
        self.assertNotIn("case_dir = Path(args.worker_dir)", source)

    def test_c2_scheduler_uses_production_provider(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('importlib.import_module("library.train_util")', source)
        self.assertIn("library.train_util.get_scheduler_fix", source)
        self.assertIn("_load_production_get_scheduler_fix()", source)
        self.assertNotIn(
            "def get_scheduler_fix(child_args, optimizer, num_processes):",
            source,
        )
        self.assertNotIn("torch.optim.lr_scheduler.LambdaLR(", source)

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
        self.assertGreaterEqual(
            source.count('"qualification_evidence_component": True'),
            3,
        )
        self.assertGreaterEqual(
            source.count('"optimizer_qualification_eligible": False'),
            3,
        )
        self.assertNotIn(
            '"optimizer_qualification_eligible": case_status == "pass"',
            source,
        )
        self.assertIn('"optimizer_evidence_complete": status == "pass"', support)
        self.assertIn('"promotion_eligible": False', support)
        self.assertIn('"optimizer_qualification_eligible": False', support)
        self.assertIn('"production_qualification_mutated": False', support)
        self.assertIn("case selection contains duplicates", source)
        self.assertIn("--worker-dir", source)
        self.assertIn("_run_phase_subprocess", source)
        self.assertIn("sys.executable", source)

    def test_muon_cases_use_pure_phased_lifecycle_protocol(self):
        source = RUNNER.read_text(encoding="utf-8")
        support = SUPPORT.read_text(encoding="utf-8")
        self.assertIn('"optimizer:muon:full-bf16:accum1:v1"', support)
        self.assertIn('"optimizer:muon:full-bf16:accum2:v1"', support)
        self.assertIn("MUON_FULL_BF16_EVIDENCE_BUNDLE_ID", source)
        self.assertIn("summarize_muon_full_bf16_bundle", source)
        self.assertIn("def _run_muon_logical_step(", source)
        self.assertIn("def _muon_train_save(", source)
        self.assertIn("def _muon_resume_second_step(", source)
        self.assertIn("torch.nn.Linear(8, 12, bias=False)", source)
        self.assertIn("torch.nn.Linear(12, 8, bias=False)", source)
        self.assertIn('"matrix_shapes"', source)
        self.assertIn('"covers_transpose_true"', source)
        self.assertIn('"covers_transpose_false"', source)
        self.assertIn('"fallback_count": sum(', source)
        self.assertIn('"state_parameter_count"', source)
        self.assertIn('"internal_adamw_state_present"', source)
        self.assertIn('"momentum_buffer"', source)
        self.assertIn('"exp_avg"', source)
        self.assertIn('"use_muon"', source)
        self.assertIn("def _muon_qualification_contract(", source)
        self.assertIn(
            '"dts.parameter-policy.muon-full-bf16-qualification-family"',
            source,
        )
        self.assertIn('"reference_arguments"', source)
        self.assertIn('"argument_family"', source)
        self.assertIn("MUON_FULL_BF16_ARGUMENT_FAMILY", source)
        self.assertIn('_PINNED_MUON_PROVIDER_VERSION = "3.10.0"', source)

    def test_c4_explicit_fallback_uses_phased_multi_child_lifecycle(self):
        source = RUNNER.read_text(encoding="utf-8")
        support = SUPPORT.read_text(encoding="utf-8")
        self.assertIn(
            '"optimizer:muon-adamw-explicit-fallback:full-bf16:accum1:v1"',
            support,
        )
        self.assertIn(
            '"optimizer:muon-adamw-explicit-fallback:full-bf16:accum2:v1"',
            support,
        )
        self.assertIn(
            "MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID",
            source,
        )
        self.assertIn("class _TinyMuonAdamWFallbackFlux", source)
        self.assertIn("torch.nn.Linear(8, 12, bias=True)", source)
        self.assertIn("torch.nn.Linear(12, 8, bias=True)", source)
        self.assertIn('"fallback_optimizer_profile": "adamw_fallback"', source)
        self.assertIn('"fallback_learning_rate": 1e-2', source)
        self.assertIn("def _c4_routing_evidence(", source)
        self.assertIn("def _c4_child_ownership_evidence(", source)
        self.assertIn("def _c4_profile_parameter_evidence(", source)
        self.assertIn("def _c4_muon_state_evidence(", source)
        self.assertIn("def _c4_adamw_state_evidence(", source)
        self.assertIn("def _c4_scheduler_evidence(", source)
        self.assertIn("def _run_c4_logical_step(", source)
        self.assertIn("def _c4_train_save(", source)
        self.assertIn("def _c4_resume_second_step(", source)
        self.assertIn('optimizers=("Muon", "AdamW")', source)
        self.assertIn('"scope": "optimizer_topology"', source)

    def test_c4_hardening_fails_closed_on_nonfinite_state_and_invalid_steps(self):
        source = RUNNER.read_text(encoding="utf-8")
        c4_step = source.split("def _run_c4_logical_step(", 1)[1].split(
            "\ndef _build_c4_runtime", 1
        )[0]
        self.assertGreaterEqual(
            c4_step.count("_assert_tensor_evidence_finite("),
            9,
        )
        self.assertIn('label="C4 routed parameter state"', c4_step)
        self.assertIn('label="C4 optimizer state"', c4_step)
        self.assertIn('label="C4 scheduler state"', c4_step)

        self.assertIn("def _c4_exact_nonnegative_step(", source)
        muon_state = source.split("def _c4_muon_state_evidence(", 1)[1].split(
            "\ndef _c4_adamw_state_evidence", 1
        )[0]
        adamw_state = source.split("def _c4_adamw_state_evidence(", 1)[1].split(
            "\ndef _c4_scheduler_evidence", 1
        )[0]
        fresh_state = source.split("def _c4_fresh_state_is_empty(", 1)[1].split(
            "\ndef _c4_qualification_contract", 1
        )[0]
        self.assertIn("_c4_exact_nonnegative_step(", muon_state)
        self.assertIn("_c4_exact_nonnegative_step(", adamw_state)
        self.assertIn("_c4_exact_nonnegative_step(", fresh_state)
        self.assertNotIn("int(raw_step)", muon_state)
        self.assertNotIn("int(raw_step)", adamw_state)

    def test_c4_actual_muon_child_matches_pinned_provider_class(self):
        source = RUNNER.read_text(encoding="utf-8")
        build = source.split("def _build_c4_runtime(", 1)[1].split(
            "\ndef _c4_handoff_path", 1
        )[0]
        self.assertIn('muon_path = _c4_muon_state_evidence(session)', build)
        self.assertIn('muon_path["child_class"] != provider["class"]', build)
        self.assertIn(
            "C4 Muon runtime resolved a different provider class",
            build,
        )

    def test_c4_evidence_does_not_add_production_topology_qualification(self):
        source = RUNNER.read_text(encoding="utf-8")
        production = (ROOT / "mikazuki" / "parameter_policy_execution.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("FULL_BF16_OPTIMIZER_TOPOLOGY_QUALIFICATIONS", production)
        self.assertNotIn(
            'FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"] =',
            source,
        )
        self.assertNotIn(
            'FULL_BF16_OPTIMIZER_QUALIFICATIONS["AdamW"] =',
            source,
        )

    def test_muon_evidence_does_not_mutate_production_qualification(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn('FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"] =', source)
        self.assertNotIn("FULL_BF16_OPTIMIZER_QUALIFICATIONS.update", source)
        self.assertIn(
            'optimizers=("Muon",)',
            source,
        )

    def test_optimizer_evidence_bundles_are_additive(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("evidence_bundles: dict[str, Any] = {}", source)
        self.assertIn(
            "evidence_bundles[ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID]",
            source,
        )
        self.assertIn(
            "evidence_bundles[MUON_FULL_BF16_EVIDENCE_BUNDLE_ID]",
            source,
        )
        self.assertIn(
            'evidence["evidence_bundles"] = evidence_bundles',
            source,
        )
        self.assertNotIn(
            'evidence["evidence_bundles"] = {\n            ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID',
            source,
        )

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

    def test_d0_execution_qualification_is_wired_to_gpu_workflow(self):
        workflow = GPU_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "run_parameter_policy_execution_gpu_matrix.py",
            workflow,
        )
        self.assertIn(
            "parameter-policy-execution-gpu-matrix.json",
            workflow,
        )
        self.assertIn("--expected-commit", workflow)
        self.assertIn("${{ github.sha }}", workflow)
        self.assertIn("DTS_EVIDENCE_DIR", workflow)
        self.assertIn("runner.temp", workflow)

    def test_host_review_tracks_and_compiles_execution_and_backend_feature_runners(self):
        workflow = HOST_WORKFLOW.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            workflow.count("tools/run_parameter_policy_execution_gpu_matrix.py"),
            2,
        )
        self.assertGreaterEqual(
            workflow.count("tools/run_parameter_policy_backend_feature_gpu_matrix.py"),
            2,
        )
        self.assertGreaterEqual(
            workflow.count("tools/parameter_policy_backend_feature_gpu_support.py"),
            2,
        )

    def test_d0_runner_emits_shared_optimizer_promotion_summary(self):
        source = RUNNER.read_text(encoding="utf-8")
        support = SUPPORT.read_text(encoding="utf-8")
        self.assertIn("summarize_shared_full_bf16_promotion", source)
        self.assertIn('evidence["shared_optimizer_promotion"] = promotion', source)
        self.assertIn("SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID", support)
        self.assertIn(
            "execution.FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID",
            support,
        )
        production = (
            ROOT / "mikazuki" / "parameter_policy_execution.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "phase-d0:shared-adamw-muon-full-bf16:v1",
            production,
        )

    def test_runner_remains_standalone_provider_independent(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--expected-commit", required=True)', source)
        self.assertIn('default="parameter-policy-execution-gpu-matrix.json"', source)
        self.assertNotIn("GITHUB_SHA", source)
        self.assertNotIn("GITHUB_ACTIONS", source)


if __name__ == "__main__":
    unittest.main()
