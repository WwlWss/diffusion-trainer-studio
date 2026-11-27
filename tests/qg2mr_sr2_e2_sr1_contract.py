"""E2-D-SR1 Tests-only RED; actual C4 implementation, isolated counterexamples.

No production or existing test/CI changes. Fixture Git repos and unittest
cases are independent, never evidence of actual model/runtime qualification.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from tests import qg2mr_sr2_e2_contract as frozen
from tests import qg2mr_sr2_e2_c1_contract as c1
from tests import qg2mr_sr2_e2_c2_contract as c2
from tests import qg2mr_sr2_e2_c3_contract as c3
from tests import qg2mr_sr2_e2_c4_contract as c4
from tests import qg2mr_sr2_e2_runner as runner

ROOT = frozen.ROOT
PARENT = "34db5cae04774cdf32da49080f1ecff12c0727d0"
MARKER = "QG2 E2-D-SR1 PERSISTENT CONTRACT ABSENT"
GITLINKS = ("frontend", "sd-scripts", "mikazuki/dataset-tag-editor")
FRONTEND_IDS = ["test_parameter_policy_frontend_behavior.ParameterPolicyFrontendBehaviorTests.test_ack_checkbox_is_visible_transient_and_start_consumes_once","test_parameter_policy_frontend_behavior.ParameterPolicyFrontendBehaviorTests.test_optimizer_type_switch_reconciles_fallback_semantics","test_parameter_policy_frontend_behavior.ParameterPolicyFrontendBehaviorTests.test_patched_layout_is_valid_javascript","test_parameter_policy_frontend_behavior.ParameterPolicyFrontendBehaviorTests.test_profile_names_hook_exposes_live_keys","test_parameter_policy_frontend_behavior.ParameterPolicyFrontendBehaviorTests.test_profile_rename_delete_and_empty_name_behavior","test_parameter_policy_frontend_behavior.ParameterPolicyFrontendBehaviorTests.test_request_envelope_carries_ack_only_for_explicit_start"]
# Actual identical C4 Ubuntu/Windows result records; not estimated counts.
HISTORICAL_SKIPS = json.loads(r'''
[
  [
    "test_parameter_policy_anima_runtime_smoke.ParameterPolicyAnimaRuntimeSmokeTests.test_anima_full_dit_and_qwen_ownership",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_execution_gpu_runtime.ExecutionGpuRuntimeEvidenceTests.test_bfloat16_tensor_fingerprint_is_stable_and_sensitive",
    "torch is not installed"
  ],
  [
    "test_parameter_policy_execution_gpu_runtime.ExecutionGpuRuntimeEvidenceTests.test_gradient_evidence_records_missing_and_present_gradients",
    "torch is not installed"
  ],
  [
    "test_parameter_policy_execution_gpu_runtime.ExecutionGpuRuntimeEvidenceTests.test_muon_state_fingerprint_survives_fresh_optimizer_load",
    "torch is not installed"
  ],
  [
    "test_parameter_policy_execution_gpu_runtime.ExecutionGpuRuntimeEvidenceTests.test_state_fingerprint_is_mapping_order_independent",
    "torch is not installed"
  ],
  [
    "test_parameter_policy_execution_gpu_runtime.ExecutionGpuRuntimeEvidenceTests.test_zero_tensor_records_nonzero_false",
    "torch is not installed"
  ],
  [
    "test_parameter_policy_full_runtime_smoke.ParameterPolicyFullRuntimeSmokeTests.test_dreambooth_text_encoder_only_backpropagates_through_frozen_unet",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_full_runtime_smoke.ParameterPolicyFullRuntimeSmokeTests.test_flux_partial_component_owns_only_double_stream",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_full_runtime_smoke.ParameterPolicyFullRuntimeSmokeTests.test_sdxl_structural_freeze_never_enters_optimizer",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_network_runtime_smoke.ParameterPolicyNetworkRuntimeSmokeTests.test_component_lr_logging_survives_accelerate_scheduler_wrapper",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_network_runtime_smoke.ParameterPolicyNetworkRuntimeSmokeTests.test_pre_routing_train_flags_use_validated_policy_loader",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_network_runtime_smoke.ParameterPolicyNetworkRuntimeSmokeTests.test_sd3_clip_pair_survives_real_accelerator_prepare",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_qg2b3c_sr1_real_late_cpu.QG2B3CSR1RealLateCPUTests.test_cpu01_real_late_rejects_policy_content_replacement_before_scan",
    "CPU Torch runtime is mandatory in SR1 CI"
  ],
  [
    "test_parameter_policy_qg2b3c_sr1_real_late_cpu.QG2B3CSR1RealLateCPUTests.test_cpu02_real_late_rejects_live_effective_config_drift_before_scan",
    "CPU Torch runtime is mandatory in SR1 CI"
  ],
  [
    "test_parameter_policy_qg2b3c_sr1_real_late_cpu.QG2B3CSR1RealLateCPUTests.test_cpu03_real_late_rejects_backend_and_ack_drift_before_scan",
    "CPU Torch runtime is mandatory in SR1 CI"
  ],
  [
    "test_parameter_policy_qg2b3c_sr1_real_late_cpu.QG2B3CSR1RealLateCPUTests.test_cpu04_real_late_shared_loader_rejects_ambiguous_json_before_scan",
    "CPU Torch runtime is mandatory in SR1 CI"
  ],
  [
    "test_parameter_policy_qg2b3c_sr1_real_late_cpu.QG2B3CSR1RealLateCPUTests.test_cpu05_real_late_held_full_bf16_rejects_before_scan",
    "CPU Torch runtime is mandatory in SR1 CI"
  ],
  [
    "test_parameter_policy_qg2b3c_sr1_real_late_cpu.QG2B3CSR1RealLateCPUTests.test_cpu06_real_late_reclassification_drift_rejects_before_scan",
    "CPU Torch runtime is mandatory in SR1 CI"
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_accelerate_prepare_accumulation_and_state_round_trip",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_all_supported_schedulefree_optimizers_run_without_external_scheduler",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_mixed_runtime_steps_and_scheduler_groups_share_live_children",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_muon_accumulation_suppresses_unsynced_step_on_cpu",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_muon_adamw_explicit_fallback_accumulation_on_cpu",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_muon_adamw_explicit_fallback_state_round_trip",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_muon_adamw_explicit_fallback_uses_real_production_routing",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_muon_state_round_trip_restores_group_step_and_coefficients",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_optimizer_and_scheduler_metadata_mismatch_fail_closed",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_real_pinned_muon_uses_only_muon_groups_and_steps_on_cpu",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_runtime_job_uses_exact_pinned_versions",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_torch_runtime_smoke.ParameterPolicyTorchRuntimeSmokeTests.test_schedulefree_eval_checkpoint_resumes_in_train_mode_and_matches_reference",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job."
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_accelerate_accumulation_two_suppresses_first_physical_step",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_accelerator_load_state_accepts_same_scheduler_identity",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_baseline_execution_environment_is_noop_for_multi_process",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_baseline_manifest_metadata_diagnostics_and_log_shape_remain_exact",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_baseline_session_has_no_execution_contract_or_root_refs",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_checkpoint_save_hook_rejects_mutated_runtime_before_manifest_write",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_dead_execution_root_fails_closed_without_strong_module_reference",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_explicit_or_non_cuda_device_comparison_does_not_query_current_cuda",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_checkpoint_progress_tracks_resume_optimizer_step",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_checkpoint_rejects_mixed_bf16_before_state_mutation",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_direct_trainer_fails_before_parameter_scan",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_dtype_contract_accepts_trainable_bf16_and_frozen_fp32",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_dtype_contract_rejects_trainable_fp32",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_epoch_start_does_not_rescan_live_roots",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_live_root_identity_accepts_unchanged_model",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_live_root_identity_rejects_alias_topology_mutation",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_live_root_identity_rejects_in_place_shape_change",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_live_root_identity_rejects_parameter_replacement",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_manifest_contains_execution_identity_and_validates_same_identity",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_metadata_diagnostics_and_log_include_execution_identity",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_multi_process_environment_fails_closed",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_non_no_distributed_type_fails_closed",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_optimizer_managed_scheduler_persists_execution_identity",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_post_resume_rechecks_live_root_identity",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_post_resume_rejects_progress_without_optimizer_restore",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_real_module_cast_prepare_and_finalize_passes",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_requested_resume_requires_loaded_progress",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_full_bf16_runtime_contract_checks_dtype_without_optimizer_audit",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_implicit_cuda_device_uses_current_cuda_device",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_malformed_execution_manifest_pair_gets_execution_specific_error",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_manifest_v2_metadata_diagnostics_and_finalize_contract",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_mixed_bf16_checkpoint_rejects_full_bf16_before_state_mutation",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_mock_qualified_full_bf16_session_captures_contract_and_weak_roots",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_optimizer_managed_scheduler_ignores_unused_global_scheduler_fields",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_runtime_contract_detects_frozen_optimizer_leak",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_runtime_contract_detects_requires_grad_mutation_with_phase",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_runtime_contract_detects_trainable_device_mismatch",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_runtime_device_comparison_resolves_only_implicit_cuda_alias",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_scheduler_mismatch_fails_before_optimizer_or_scheduler_state_mutation",
    "runtime dependencies are not installed"
  ],
  [
    "test_parameter_policy_trainer_runtime_smoke.ParameterPolicyTrainerRuntimeSmokeTests.test_session_owns_freeze_scheduler_prepare_device_and_manifest",
    "runtime dependencies are not installed"
  ]
]
''')
TARGET_NUMBERS = (101, 201, 202, 203, 301, 302, 303, 304, 401, 402,
                  501, 502, 503, 601, 602, 603, 701, 702, 703)
BASE_NUMBERS = tuple(range(901, 913))


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=SR1 isolated fixture", "-c",
         "user.email=sr1@example.invalid", "-c", "commit.gpgsign=false",
         "-c", "protocol.file.allow=always", "-C", str(root), *args],
        text=True, encoding="utf-8", errors="strict", timeout=90).strip()


@contextmanager
def git_fixture():
    # A real child commit update and real parent gitlinks, not mocked Git.
    with tempfile.TemporaryDirectory(prefix="dts-e2-sr1-git-") as directory:
        base = Path(directory)
        origin, repo = base / "origin", base / "superproject"
        origin.mkdir()
        repo.mkdir()
        git(origin, "init", "-q")
        asset = origin / "dist/assets/layout.96d49288.js"
        asset.parent.mkdir(parents=True)
        asset.write_text("// isolated fixture\n", encoding="utf-8")
        git(origin, "add", ".")
        git(origin, "commit", "-qm", "first child")
        first = git(origin, "rev-parse", "HEAD")
        git(repo, "init", "-q")
        for rel in GITLINKS:
            git(repo, "submodule", "add", "-q", "--", origin.as_posix(), rel)
        git(repo, "commit", "-qam", "first gitlinks")
        (origin / "update.txt").write_text("new valid child commit\n", encoding="utf-8")
        git(origin, "add", ".")
        git(origin, "commit", "-qm", "new child")
        second = git(origin, "rev-parse", "HEAD")
        if first == second:
            raise RuntimeError("Fixture child did not advance")
        for rel in GITLINKS:
            child = repo / rel
            git(child, "fetch", "-q", "origin")
            git(child, "checkout", "-q", "--detach", second)
        git(repo, "add", *GITLINKS)
        git(repo, "commit", "-qm", "updated valid gitlinks")
        for rel in GITLINKS:
            expected = git(repo, "ls-tree", "HEAD", "--", rel)
            if expected != f"160000 commit {second}\t{rel}" or git(repo / rel, "rev-parse", "HEAD") != second:
                raise RuntimeError(f"Invalid actual fixture gitlink {rel}")
        yield repo, first, second


class ChildBoundaryReached(Exception):
    """The actual checkout gate returned and reached downstream discovery."""


def host_checkout_gate(repo):
    # Only replace downstream child execution with a sentinel. Git and Node
    # commands in actual full_host execute normally, including on Windows.
    with mock.patch.object(c1, "ROOT", repo), mock.patch.dict(
            os.environ, {"GITHUB_SHA": git(repo, "rev-parse", "HEAD")}), \
            mock.patch.object(c1, "checked_child", side_effect=ChildBoundaryReached):
        try:
            c1.full_host()
        except ChildBoundaryReached:
            return True, ""
        except RuntimeError as exc:
            return False, str(exc)
    raise RuntimeError("Actual checkout gate did not reach child boundary")


def replace_once(text, old, new):
    if text.count(old) != 1 or old == new:
        raise RuntimeError(f"Fixture mutation anchor drift: {old!r}")
    return text.replace(old, new, 1)


@contextmanager
def source_override(name, changed):
    original = frozen.source
    with mock.patch.object(frozen, "source",
                           side_effect=lambda n: changed if n == name else original(n)):
        yield


def lifecycle_accepts(name, text, rejection):
    # The live C4 entry calls c3.run_behavior. Exercise that actual suite
    # with one real workflow changed; do not call historical hash alone.
    with source_override(name, text), redirect_stdout(io.StringIO()) as out, \
            redirect_stderr(io.StringIO()) as err:
        ok = c3.run_behavior()
    if not ok and rejection not in out.getvalue() + err.getvalue():
        raise RuntimeError("Unrelated lifecycle rejection:\n" + err.getvalue())
    return ok


def covered(name, paths):
    ev = frozen.events(frozen.source(name))
    return {event: frozen.covers(ev.get(event), paths)
            for event in ("push", "pull_request")}


def direct_imports(name):
    path = ROOT / "tests" / name
    imports = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                   else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
        imports.update(m.replace(".", "/") + ".py" for m in modules
                       if m and m.startswith("mikazuki."))
    return imports


class CallbackProbe(unittest.TestCase):
    def test_body(self):
        self.assertTrue(True)


class NoTerminalResult(unittest.TestCase):
    def test_body(self):
        raise AssertionError("This body must not execute in the counterexample")
    def run(self, result=None):
        result.startTest(self)
        result.stopTest(self)
        return result


class UnauthorizedSkip(unittest.TestCase):
    @unittest.skip("runtime dependencies are not installed")
    def test_body(self):
        raise AssertionError("This body must not execute in the counterexample")


class ObservedResult(runner.CpuIdentityResult):
    # Independent observation of actual success callbacks for the fixture.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_successes = []
    def addSuccess(self, test):
        self.observed_successes.append(test.id())
        super().addSuccess(test)


def actual_record(suite):
    ids = runner.test_ids(suite)
    result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=ObservedResult).run(suite)
    record = runner.cpu_result_record(result, ids)
    # Supply independently observed identities at the result boundary;
    # never invent successes from count minus skip.
    record.setdefault("discovered_ids", list(ids))
    record.setdefault("started_ids", list(result.executed_ids))
    record.setdefault("passed_ids", list(result.observed_successes))
    return record, result, ids


class FixtureIdentity(unittest.TestCase):
    def __init__(self, identity):
        super().__init__("test_body")
        self.identity = identity
    def id(self):
        return self.identity
    def test_body(self):
        self.assertTrue(True)


def host_correlation_gate(discovered, actual):
    # Two independently executed real fixture suites, not repository frontend
    # test execution. Replace only subprocess transport; run actual full_host.
    reference, _, ids = actual_record(unittest.TestSuite(FixtureIdentity(i) for i in discovered))
    executed, _, actual_ids = actual_record(unittest.TestSuite(FixtureIdentity(i) for i in actual))
    if ids != discovered or actual_ids != actual:
        raise RuntimeError("Invalid correlation fixture identities")
    def child(module, args, marker):
        if "--discover-only" in args:
            return {"ids": ids, "errors": [], "loaded_cpu": []}
        if module.endswith("qg2mr_sr2_e2_runner") and args == ["host"]:
            return executed
        if args == ["legacy-reference"] or "reference" in args[0]:
            return reference
        raise RuntimeError(f"Unexpected fixture child call: {module} {args}")
    with mock.patch.object(c1, "checked_child", side_effect=child):
        try:
            return c1.full_host() == 0
        except RuntimeError as exc:
            if "identity" not in str(exc).lower() and "outcome" not in str(exc).lower():
                raise
            return False


def parsed(text):
    # Test-only YAML validity witness; BaseLoader keeps Actions 'on' literal.
    result = yaml.load(text, Loader=yaml.BaseLoader)
    if not isinstance(result, dict) or "on" not in result or "jobs" not in result:
        raise RuntimeError("Invalid mutation YAML")
    return result


def integrated_accepts(text):
    try:
        c4.require_integrated_workflow(text)
    except AssertionError:
        return False
    return True


def cpu_fixture_source(filename, eager):
    stem = Path(filename).stem
    groups = {}
    for ident in runner.expected_cpu_ids("all"):
        module, cls, method = ident.split(".")
        if module == stem:
            groups.setdefault(cls, []).append(method)
    if not groups:
        raise RuntimeError(f"Unknown fixture module {filename}")
    text = "import unittest\n" + ("import torch\n" if eager else "")
    for cls, methods in groups.items():
        text += "\nclass " + cls + "(unittest.TestCase):\n"
        for method in methods:
            text += "    def " + method + "(self):\n        import torch\n        self.assertIsNotNone(torch)\n"
    return text


@contextmanager
def discovery_fixture(eager_count):
    with tempfile.TemporaryDirectory(prefix="dts-e2-sr1-discovery-") as directory:
        root = Path(directory)
        tests = root / "tests"
        tests.mkdir()
        for i, filename in enumerate(sorted(frozen.CPU_FILES)):
            (tests / filename).write_text(cpu_fixture_source(filename, i < eager_count), encoding="utf-8")
        (tests / "test_parameter_policy_sr1_host_fixture.py").write_text(
            "import unittest\nclass Host(unittest.TestCase):\n"
            "    def test_actual(self):\n        self.assertTrue(True)\n", encoding="utf-8")
        with mock.patch.object(frozen, "ROOT", root), mock.patch.object(frozen, "TESTS", tests):
            yield root


def raw_fixture_discovery(root):
    code = """
import importlib.util, json, unittest
assert importlib.util.find_spec('torch') is None
loader = unittest.TestLoader()
suite = loader.discover(start_dir='tests', pattern='test_parameter_policy*.py')
print(json.dumps({'errors': loader.errors, 'count': suite.countTestCases()}))
"""
    return json.loads(subprocess.check_output([sys.executable, "-B", "-c", code],
                     cwd=root, text=True, encoding="utf-8", timeout=90))


class Targets(unittest.TestCase):
    def require(self, condition, detail):
        print(f"SR1 OBSERVED {self._testMethodName}: {detail}", flush=True)
        if not condition:
            self.fail(MARKER + ": " + detail)

    def test_d101_current_updated_gitlink_accepted(self):
        with git_fixture() as (repo, first, second):
            ok, reason = host_checkout_gate(repo)
            if not ok and "Pinned submodule mismatch" not in reason:
                raise RuntimeError("Unrelated checkout rejection: " + reason)
            self.require(ok, f"actual parent and all child gitlinks match {second}; prior {first}; actual gate={ok} {reason}")

    def lifecycle(self, kind, new, rejection):
        name = "qg2a1-request-intent-host.yml"
        text = frozen.source(name)
        if not lifecycle_accepts(name, text, rejection):
            raise RuntimeError("Unmodified lifecycle is not a valid control")
        parsed(new)
        self.require(lifecycle_accepts(name, new, rejection),
                     "valid " + kind + " mutation rejected by active lifecycle suite")

    def test_d201_valid_comment_change(self):
        text = frozen.source("qg2a1-request-intent-host.yml")
        changed = replace_once(text, "\njobs:\n", "\njobs:\n  # SR1 valid documentation change\n")
        self.lifecycle("comment", changed, "Historical job body changed")

    def test_d202_valid_timeout_change(self):
        text = frozen.source("qg2a1-request-intent-host.yml")
        import re
        m = re.search(r"(?m)^    timeout-minutes: (\d+)$", text)
        if not m:
            raise RuntimeError("Missing actual timeout")
        changed = replace_once(text, m[0], f"    timeout-minutes: {int(m[1]) + 1}")
        self.lifecycle("timeout", changed, "Historical job body changed")

    def test_d203_valid_additive_trigger_change(self):
        text = frozen.source("qg2a1-request-intent-host.yml")
        extra = '      - "mikazuki/parameter_policy_execution.py"\n'
        if extra in text:
            # After Fix E, use another actual, currently uncovered path.
            extra = '      - "docs/qualification-gate-qg2-mr-sr2-e2-c4-permanent-green.md"\n'
        if extra in text:
            raise RuntimeError("No additive fixture mutation")
        changed = replace_once(text, "  pull_request:\n", extra + "  pull_request:\n")
        changed = replace_once(changed, "  workflow_dispatch:\n", extra + "  workflow_dispatch:\n")
        self.lifecycle("additive trigger", changed, "Reviewed event/header snapshot changed")

    def paths(self, workflow, test_file, paths):
        self.assertLessEqual(set(paths), direct_imports(test_file))
        for path in paths:
            self.assertTrue((ROOT / path).is_file(), path)
        missing = covered(workflow, paths)
        self.require(not any(missing.values()), f"{workflow} actual import input coverage: {missing}")

    def test_d301_a1_execution_and_matrix(self):
        self.paths("qg2a1-request-intent-host.yml", "test_parameter_policy_qg2a1_request_red.py",
                   {"mikazuki/parameter_policy_execution.py", "mikazuki/parameter_policy_matrix.py"})

    def test_d302_a1_sr1_admission_and_execution(self):
        self.paths("qg2a1-sr1-host.yml", "test_parameter_policy_qg2a1_sr1.py",
                   {"mikazuki/parameter_policy_admission.py", "mikazuki/parameter_policy_execution.py"})

    def test_d303_b3a_profile_and_execution(self):
        self.paths("qg2b3a-policy-snapshot-host.yml", "test_parameter_policy_qg2b3a_snapshot_red.py",
                   {"mikazuki/model_component_profiles.py", "mikazuki/parameter_policy_execution.py"})

    def test_d304_integrated_frontend_request_config(self):
        self.assertIn("mikazuki/frontend_training_patch.py", direct_imports("test_parameter_policy_frontend_behavior.py"))
        self.assertIn("mikazuki/training_config.py", direct_imports("test_parameter_policy_request_contract.py"))
        tree = ast.parse((ROOT / "tests/test_parameter_policy_request_contract.py").read_text(encoding="utf-8"))
        self.assertTrue(any(isinstance(n, ast.BinOp) and ast.unparse(n) == "ROOT / 'mikazuki' / 'training_request.py'"
                            for n in ast.walk(tree)))
        missing = covered(frozen.E2, {"mikazuki/frontend_training_patch.py", "mikazuki/training_request.py",
                                     "mikazuki/training_config.py"})
        self.require(not any(missing.values()), f"actual E2 Host source dependencies: {missing}")

    def test_d401_new_unauthorized_skip_rejected(self):
        record, result, ids = actual_record(unittest.TestSuite([UnauthorizedSkip("test_body")]))
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.observed_successes, [])
        self.assertNotIn(ids[0], {i for i, _ in HISTORICAL_SKIPS})
        self.require(not runner.strict_host_success(record), "actual unauthorized Skip with old reason accepted")

    def test_d402_removed_runtime_owner_semantics_rejected(self):
        # The old C3 digest may incidentally reject this edit. This test
        # specifically exercises the active E2B908 semantic owner check.
        name = frozen.ANIMA
        text = frozen.source(name)
        line = next((l for l in text.splitlines(keepends=True)
                     if l.strip() == "tests.test_parameter_policy_execution_gpu_runtime " + "\\"), None)
        if line is None:
            raise RuntimeError("Original actual owner command drift")
        changed = replace_once(text, line, "")
        parsed(changed)
        check = frozen.E2Baselines("test_e2b908_preserve_unfiltered_pr_cpu_and_manual_gpu")
        check.test_e2b908_preserve_unfiltered_pr_cpu_and_manual_gpu()
        with source_override(name, changed):
            try:
                check.test_e2b908_preserve_unfiltered_pr_cpu_and_manual_gpu()
            except AssertionError:
                accepted = False
            else:
                accepted = True
        self.require(not accepted, "actual Runtime Owner command removed; E2B908 semantic contract still accepts")

    def test_d501_cpu_start_without_success_rejected(self):
        record, result, ids = actual_record(unittest.TestSuite([NoTerminalResult("test_body")]))
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.executed_ids, ids)
        self.assertEqual(result.observed_successes, [])
        self.require(not runner.strict_cpu_success(record, ids), "actual CPU start without terminal success accepted")

    def test_d502_host_child_identity_mismatch_rejected(self):
        discovered = FRONTEND_IDS + ["sr1_fixture.original.test_body"]
        actual = FRONTEND_IDS + ["sr1_fixture.replacement.test_body"]
        if not host_correlation_gate(discovered, discovered):
            raise RuntimeError("Valid identity correlation control rejected")
        self.require(not host_correlation_gate(discovered, actual),
                     "actual equal-count but different child execution IDs accepted by full_host")

    def test_d503_host_start_without_result_rejected(self):
        record, result, ids = actual_record(unittest.TestSuite([NoTerminalResult("test_body")]))
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.executed_ids, ids)
        self.assertEqual(result.observed_successes, [])
        self.require(not runner.strict_host_success(record), "actual Host start without terminal outcome accepted")

    def integration_mutation(self, changed, detail):
        if not integrated_accepts(frozen.source(frozen.E2)):
            raise RuntimeError("Unmodified integrated workflow rejected")
        self.require(not integrated_accepts(changed), detail)

    def test_d601_effective_matrix_windows_exclusion(self):
        text = frozen.source(frozen.E2)
        prefix, job = text.split("\n  r2-cpu:\n")
        changed = prefix + "\n  r2-cpu:\n" + replace_once(job,
            "        os: [ubuntu-latest, windows-latest]\n",
            "        os: [ubuntu-latest, windows-latest]\n        exclude:\n          - os: windows-latest\n")
        matrix = parsed(changed)["jobs"]["r2-cpu"]["strategy"]["matrix"]
        self.assertEqual(matrix["exclude"], [{"os": "windows-latest"}])
        self.assertEqual([os for os in matrix["os"] if {"os": os} not in matrix["exclude"]], ["ubuntu-latest"])
        self.integration_mutation(changed, "valid YAML effective Matrix removes Windows")

    def test_d602_checkout_ref_override(self):
        text = frozen.source(frozen.E2)
        prefix, job = text.split("\n  r2-cpu:\n")
        changed = prefix + "\n  r2-cpu:\n" + replace_once(job,
            "          submodules: false\n",
            "          submodules: false\n          ref: e930c1397eb65f7be39a9a81970265a2800b0dbf\n")
        self.assertEqual(parsed(changed)["jobs"]["r2-cpu"]["steps"][0]["with"]["ref"],
                         "e930c1397eb65f7be39a9a81970265a2800b0dbf")
        self.integration_mutation(changed, "valid YAML Checkout ref replaces event commit")

    def test_d603_wrong_working_directory(self):
        text = frozen.source(frozen.E2)
        command = "        run: " + frozen.CPU_COMMAND + "fsr"
        changed = replace_once(text, command, "        working-directory: tests\n" + command)
        step = next(s for s in parsed(changed)["jobs"]["r2-cpu"]["steps"]
                    if s.get("run") == frozen.CPU_COMMAND + "fsr")
        self.assertEqual(step["working-directory"], "tests")
        self.integration_mutation(changed, "valid YAML changes actual execution directory without changing command")

    def delayed_import(self, eager_count):
        with discovery_fixture(eager_count) as root:
            raw = raw_fixture_discovery(root)
            self.assertEqual(len(raw["errors"]), eager_count)
            self.assertTrue(all("No module named 'torch'" in e for e in raw["errors"]))
            try:
                ids = frozen.reference_discovery()
            except RuntimeError as exc:
                if "Invalid frozen no-Torch discovery" not in str(exc):
                    raise
                ids = []
            self.require(ids == ["test_parameter_policy_sr1_host_fixture.Host.test_actual"],
                         f"real fixture with {eager_count} actual eager Torch import errors; permanent reference={ids}")

    def test_d701_both_cpu_files_delayed_import(self):
        self.delayed_import(0)

    def test_d702_one_cpu_file_delayed_import(self):
        self.delayed_import(1)

    def test_d703_legacy_baseline_rejects_delayed_import(self):
        with tempfile.TemporaryDirectory(prefix="dts-e2-sr1-source-") as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            for filename in frozen.FROZEN_FILES:
                shutil.copyfile(ROOT / "tests" / filename, tests / filename)
            for filename in frozen.CPU_FILES:
                (tests / filename).write_text(cpu_fixture_source(filename, False), encoding="utf-8")
            fixture = root / "tools/_parameter_policy_gpu_low_level_fixture.py"
            fixture.parent.mkdir()
            shutil.copyfile(ROOT / "tools/_parameter_policy_gpu_low_level_fixture.py", fixture)
            check = frozen.E2Baselines("test_e2b901_original_census_and_cpu_paths")
            check.test_e2b901_original_census_and_cpu_paths()
            with mock.patch.object(frozen, "ROOT", root), mock.patch.object(frozen, "TESTS", tests):
                try:
                    check.test_e2b901_original_census_and_cpu_paths()
                except AssertionError as exc:
                    if not any(filename in str(exc) for filename in frozen.CPU_FILES):
                        raise
                    accepted = False
                else:
                    accepted = True
            self.require(accepted, "actual E2B901 rejects structurally valid delayed-import identity fixture")


class Baselines(unittest.TestCase):
    def test_d901_actual_current_checkout_and_gitlinks(self):
        ok, reason = host_checkout_gate(ROOT)
        self.assertTrue(ok, reason)
        for rel in GITLINKS:
            actual = git(ROOT / rel, "rev-parse", "HEAD")
            self.assertEqual(git(ROOT, "ls-tree", "HEAD", "--", rel),
                             f"160000 commit {actual}\t{rel}")

    def test_d902_actual_wrong_and_missing_submodule_rejected(self):
        with git_fixture() as (repo, first, second):
            git(repo / GITLINKS[0], "checkout", "-q", "--detach", first)
            self.assertNotEqual(git(repo / GITLINKS[0], "rev-parse", "HEAD"), second)
            self.assertFalse(host_checkout_gate(repo)[0])
            git(repo / GITLINKS[0], "checkout", "-q", "--detach", second)
            git(repo, "submodule", "deinit", "-f", "--", GITLINKS[0])
            self.assertFalse((repo / GITLINKS[0] / "dist/assets/layout.96d49288.js").exists())
            self.assertFalse(host_checkout_gate(repo)[0])

    def test_d903_original_migration_snapshot_evidence(self):
        # Actual historical parent source, not regenerated from mutation.
        for name in c3.SNAPSHOT:
            text = subprocess.check_output(
                ["git", "show", f"{PARENT}:.github/workflows/{name}"],
                cwd=ROOT, text=True, encoding="utf-8", timeout=30)
            c3.require_snapshot(name, text)
        for path, blob in c3.UNTOUCHED.items():
            self.assertEqual(git(ROOT, "rev-parse", f"{PARENT}:{path}"), blob)

    def test_d904_actual_preimport_exclusion(self):
        result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite([
            c1.PartitionBehavior("test_c1101_exact_preimport_exclusion_and_new_host")]))
        self.assertTrue(result.wasSuccessful() and result.testsRun == 1 and not result.skipped)

    def test_d905_original_eighteen_cpu_identities(self):
        result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite([
            c2.CpuContractBehavior("test_c2101_frozen_original_cpu_method_identities")]))
        self.assertTrue(result.wasSuccessful() and result.testsRun == 1 and not result.skipped)
        self.assertEqual(len(set(runner.expected_cpu_ids("all"))), 18)

    def test_d906_actual_seventy_historical_skip_identities(self):
        expected = sorted(HISTORICAL_SKIPS)
        script = r"""
import json, unittest
from tests import qg2mr_sr2_e2_runner as runner
expected = json.loads(__import__('sys').argv[1])
wanted = {i for i, _ in expected}
suite, ids, _ = runner.discover_host()
def flatten(suite):
    for case in suite:
        if isinstance(case, unittest.TestCase):
            yield case
        else:
            yield from flatten(case)
selected = unittest.TestSuite(case for case in flatten(suite) if case.id() in wanted)
assert selected.countTestCases() == len(wanted) == 70
result = unittest.TextTestRunner(verbosity=1).run(selected)
assert not (result.failures or result.errors or result.expectedFailures or result.unexpectedSuccesses)
print("SR1_ACTUAL_SKIPS_JSON " + json.dumps({
    "tests_run": result.testsRun,
    "skipped": sorted([[case.id(), reason] for case, reason in result.skipped]),
}))
"""
        observed = frozen._child_json(["-c", script, json.dumps(expected)], "SR1_ACTUAL_SKIPS_JSON ")
        self.assertEqual(observed["tests_run"], 70)
        self.assertEqual(observed["skipped"], expected)
        print("SR1 HISTORICAL HOST SKIP OWNERSHIP EVIDENCE: 70 exact actual identities", flush=True)

    def test_d907_actual_fail_error_block(self):
        class Bad(unittest.TestCase):
            def test_fail(self):
                self.fail("fixture fail")
            def test_error(self):
                raise RuntimeError("fixture error")
        record, result, ids = actual_record(unittest.TestLoader().loadTestsFromTestCase(Bad))
        self.assertEqual((len(result.failures), len(result.errors)), (1, 1))
        self.assertFalse(runner.strict_host_success(record))
        self.assertFalse(runner.strict_cpu_success(record, ids))

    def test_d908_actual_cpu_skip_xfail_xpass_block(self):
        class Bad(unittest.TestCase):
            @unittest.skip("fixture skip")
            def test_skip(self):
                pass
            @unittest.expectedFailure
            def test_xfail(self):
                self.fail("fixture xfail")
            @unittest.expectedFailure
            def test_xpass(self):
                pass
        record, result, ids = actual_record(unittest.TestLoader().loadTestsFromTestCase(Bad))
        self.assertEqual((len(result.skipped), len(result.expectedFailures), len(result.unexpectedSuccesses)), (1, 1, 1))
        self.assertFalse(runner.strict_cpu_success(record, ids))

    def test_d909_actual_sr2_six_plus_five(self):
        scope = c4.load_actual_program(frozen.source(frozen.SR2))
        self.assertEqual(scope["main"](), 0)

    def test_d910_preserve_three_unfiltered_prs(self):
        for name in c3.UNFILTERED:
            self.assertEqual(frozen.events(frozen.source(name))["pull_request"], {})

    def test_d911_preserve_manual_gpu_boundary(self):
        text = frozen.source("parameter-policy-gpu-matrix.yml")
        self.assertEqual(frozen.events(text), {"workflow_dispatch": {}}) if "  workflow_dispatch:\n" in text and not "    inputs:" in text else None
        self.assertRegex(text, r"(?m)^  workflow_dispatch:$")
        self.assertNotRegex(text, r"(?m)^  (push|pull_request):")
        self.assertIn("self-hosted", text)

    def test_d912_actual_inherited_integrated_contract(self):
        for run in (c1.run_behavior, c2.run_behavior, c3.run_behavior, c4.run_behavior):
            self.assertTrue(run())
        self.assertEqual(frozen.run_exact("green"), 0)


class OuterResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = []
        self.passed = []
    def startTest(self, case):
        self.started.append(case.id())
        super().startTest(case)
    def addSuccess(self, case):
        self.passed.append(case.id())
        super().addSuccess(case)


def execute(cls, numbers):
    names = unittest.TestLoader().getTestCaseNames(cls)
    if len(names) != len(numbers) or any(sum(n.startswith(f"test_d{i}_") for n in names) != 1 for i in numbers):
        raise RuntimeError(f"Unexpected outer test identities: {names}")
    suite = unittest.TestSuite(cls(n) for n in names)
    ids = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2, resultclass=OuterResult).run(suite)
    failed = {case.id(): trace for case, trace in result.failures}
    errors = [case.id() for case, _ in result.errors]
    record = {
        "suite": cls.__name__, "expected_ids": ids, "started_ids": result.started,
        "passed_ids": result.passed, "failed_ids": sorted(failed), "error_ids": errors,
        "skipped": [[case.id(), reason] for case, reason in result.skipped],
        "xfail": len(result.expectedFailures), "xpass": len(result.unexpectedSuccesses),
        "tests_run": result.testsRun,
        "marked_failed_ids": sorted(i for i, trace in failed.items()
                                   if "AssertionError: " + MARKER + ":" in trace),
    }
    print("DTS_E2_SR1_RESULT_JSON " + json.dumps(record), flush=True)
    valid = (result.started == ids and result.testsRun == len(ids)
             and not errors and not result.skipped and not result.expectedFailures
             and not result.unexpectedSuccesses
             and len(result.passed) == len(set(result.passed))
             and not set(result.passed) & set(failed)
             and set(result.passed) | set(failed) == set(ids))
    return valid, record


def main():
    if sys.argv[1:] != ["red"]:
        raise SystemExit("usage: python -m tests.qg2mr_sr2_e2_sr1_contract red")
    if importlib.util.find_spec("torch") is not None or yaml.__version__ != "6.0.2":
        raise SystemExit("Invalid Tests-only RED environment")
    t_ok, t = execute(Targets, TARGET_NUMBERS)
    b_ok, b = execute(Baselines, BASE_NUMBERS)
    if (t_ok and b_ok and not t["passed_ids"]
            and set(t["marked_failed_ids"]) == set(t["expected_ids"])
            and b["passed_ids"] == b["expected_ids"] and not b["failed_ids"]):
        print("E2-D-SR1-B VALID TESTS-ONLY RED: 19/19 exact target FAIL + "
              "12/12 actual protection baseline PASS; 0 ERROR/SKIP/xfail/xpass", flush=True)
        return 1
    print("E2-D-SR1-B INVALID OR PARTIAL RED; inspect exact records", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
