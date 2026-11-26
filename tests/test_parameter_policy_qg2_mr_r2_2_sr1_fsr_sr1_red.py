"""QG-2-MR-R2-2-SR1-B-FSR-SR1-A Tests-only RED; no CUDA is launched."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import torch
from test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper import (
    _TinyFlux as ApprovedTinyFlux,
    _TinyMuonFlux as ApprovedTinyMuonFlux,
    _TinyMuonAdamWFallbackFlux as ApprovedTinyFallbackFlux,
    _args, _policy_for,
)
from tools._parameter_policy_gpu_low_level_fixture import (
    build_gpu_evidence_full_bf16_session,
)

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "test_parameter_policy_execution_gpu_harness.py"
MARKER = "QG-2-MR-R2-2-SR1-B-FSR-SR1 missing isolation guard"

def _probe_source() -> str:
    source = HARNESS.read_text(encoding="utf-8")
    functions = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "test_strict_runners_reject_untracked_workspace_before_runtime_imports"]
    if len(functions) != 1:
        raise AssertionError("Expected one pre-import probe method.")
    return ast.get_source_segment(source, functions[0]) or ""


class SR101FileOwnershipRed(unittest.TestCase):
    def test_SR101_T1_probe_created_exclusively(self):
        self.assertIn('probe.open("x"', _probe_source(), MARKER)

    def test_SR101_T2_runner_outputs_in_private_external_directory(self):
        source = _probe_source()
        self.assertIn("TemporaryDirectory(", source, MARKER)
        self.assertIn("dir=ROOT.parent", source)
        self.assertIn("output_path.exists()", source)

    def test_SR101_T3_never_unlinks_fixed_unowned_outputs(self):
        source = _probe_source()
        self.assertNotIn('(ROOT.parent / "execution-probe.json").unlink', source, MARKER)
        self.assertNotIn('(ROOT.parent / "baseline-probe.json").unlink', source, MARKER)

    def test_SR101_T4_existing_external_files_survive_harness_rejection(self):
        # Run the *real Harness test method* against an isolated fake repo;
        # all Git and child process invocations are mocked, never CUDA.
        import test_parameter_policy_execution_gpu_harness as harness

        with TemporaryDirectory(prefix="qg2-fsr-external-") as temp_dir:
            outer = Path(temp_dir)
            fake_repo = outer / "repo"
            fake_repo.mkdir()
            prior_files = [
                outer / "execution-probe.json",
                outer / "baseline-probe.json",
            ]
            for file in prior_files:
                file.write_bytes(b"preexisting user evidence\n")

            def fake_git(argv, **kwargs):
                if argv == ["git", "rev-parse", "HEAD"]:
                    return "f" * 40
                if argv == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
                    probes = list(fake_repo.glob("phase-c-provenance-untracked*.json"))
                    if len(probes) != 1:
                        raise AssertionError("Expected exactly one Git-visible probe")
                    return f"?? {probes[0].name}\n"
                raise AssertionError(f"Unexpected Git command: {argv!r}")

            def fake_runner(argv, **kwargs):
                probes = list(fake_repo.glob("phase-c-provenance-untracked*.json"))
                if len(probes) != 1:
                    raise AssertionError("Missing exclusive untracked probe")
                return subprocess.CompletedProcess(
                    argv, 1,
                    stdout=f"completely clean workspace: {probes[0].name}",
                    stderr="",
                )

            with (
                mock.patch.object(harness, "ROOT", fake_repo),
                mock.patch.object(harness, "RUNNER", fake_repo / "exec.py"),
                mock.patch.object(harness, "BASELINE_RUNNER", fake_repo / "base.py"),
                mock.patch.object(harness.subprocess, "check_output", side_effect=fake_git),
                mock.patch.object(harness.subprocess, "run", side_effect=fake_runner) as run,
            ):
                case = harness.ParameterPolicyExecutionGpuHarnessContractTests(
                    "test_strict_runners_reject_untracked_workspace_before_runtime_imports"
                )
                case.test_strict_runners_reject_untracked_workspace_before_runtime_imports()
                self.assertEqual(run.call_count, 2)

            for file in prior_files:
                self.assertEqual(
                    file.read_bytes(),
                    b"preexisting user evidence\n",
                    MARKER + ": external evidence was deleted or altered",
                )

    def test_SR101_B1_existing_git_guard_and_timeout_are_retained(self):
        source = _probe_source()
        self.assertIn('"git", "status", "--porcelain=v1", "--untracked-files=all"', source)
        self.assertIn("timeout=25", source)
        self.assertIn("completely clean workspace", source)


class _TinyFlux(torch.nn.Module):
    """An intentionally unreviewed file declaring an otherwise valid tiny class."""
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])


class SR102SyntheticScopeRed(unittest.TestCase):
    def _reject_before_scan(self, model, *, mutation=None, message: str):
        policy = _policy_for(model if mutation is not None else ApprovedTinyFlux())
        if mutation is not None:
            mutation(policy["optimizer_profiles"])
        with TemporaryDirectory(prefix="qg2-fsr-sr1-") as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            with mock.patch(
                "mikazuki.parameter_policy_trainer.scan_parameter_roots",
                side_effect=RuntimeError("UNEXPECTED_PARAMETER_ROOT_SCAN"),
            ) as scan:
                try:
                    build_gpu_evidence_full_bf16_session(
                        args=_args(path),
                        train_type="flux-finetune",
                        roots={"transformer": model},
                    )
                except ValueError as exc:
                    self.assertIn(message, str(exc), MARKER)
                except RuntimeError as exc:
                    if str(exc) == "UNEXPECTED_PARAMETER_ROOT_SCAN":
                        self.fail(MARKER + ": unapproved input reached parameter scan")
                    raise
                else:
                    self.fail(MARKER + ": unapproved input accepted")
                scan.assert_not_called()

    def test_SR102_T1_adamw_extra_argument(self):
        self._reject_before_scan(
            ApprovedTinyFlux(),
            mutation=lambda p: p["main"]["args"].update(weight_decay=0.125),
            message="unreviewed optimizer profile arguments",
        )

    def test_SR102_T2_muon_momentum_changed(self):
        self._reject_before_scan(
            ApprovedTinyMuonFlux(),
            mutation=lambda p: p["main"]["args"].update(momentum=0.8),
            message="unreviewed optimizer profile arguments",
        )

    def test_SR102_T3_muon_ns_steps_changed(self):
        self._reject_before_scan(
            ApprovedTinyMuonFlux(),
            mutation=lambda p: p["main"]["args"].update(ns_steps=4),
            message="unreviewed optimizer profile arguments",
        )

    def test_SR102_T4_muon_field_removed(self):
        self._reject_before_scan(
            ApprovedTinyMuonFlux(),
            mutation=lambda p: p["main"]["args"].pop("weight_decouple"),
            message="unreviewed optimizer profile arguments",
        )

    def test_SR102_T5_fallback_adamw_extra_argument(self):
        self._reject_before_scan(
            ApprovedTinyFallbackFlux(),
            mutation=lambda p: p["adamw_fallback"]["args"].update(weight_decay=0.125),
            message="unreviewed optimizer profile arguments",
        )

    def test_SR102_T6_same_name_and_shape_but_external_source(self):
        self._reject_before_scan(
            _TinyFlux(),
            message="unreviewed model source",
        )


if __name__ == "__main__":
    unittest.main()
