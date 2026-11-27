"""C2 staged evidence, strict CPU behavior checks and actual full discovery parity.

Original 36+8 E2 and C1 test sources remain frozen. No production mutation.
"""
from __future__ import annotations

import ast
import io
import json
import sys
import unittest
from pathlib import Path

from tests import qg2mr_sr2_e2_contract as frozen
from tests import qg2mr_sr2_e2_test_results as results
from tests import qg2mr_sr2_e2_c1_contract as c1
from tests import qg2mr_sr2_e2_runner as runner

ROOT = frozen.ROOT
GREEN_PREFIXES = ("test_e2h101_", "test_e2h102_", "test_e2h103_",
                  "test_e2h104_", "test_e2h106_")
FULL_MARKER = "DTS_E2_FULL_CPU_DISCOVERY_JSON "
EXTRA_INPUTS = frozenset((
    "mikazuki/training_gui_args.py",
    "mikazuki/sd3_full_bf16_scope.py",
    "mikazuki/sd3_architecture_contract.py",
    "tools/parameter_policy_execution_gpu_runtime.py",
    "tools/parameter_policy_gpu_qualification_environment.py",
    "tests/_parameter_policy_admission_fixtures.py",
    "tests/_parameter_policy_low_level_fixtures.py",
    "tests/test_parameter_policy_trainer_runtime_smoke.py",
    "tools/run_parameter_policy_backend_feature_gpu_matrix.py",
    "tools/run_parameter_policy_backend_gpu_matrix.py",
    ".github/workflows/parameter-policy-gpu-matrix.yml",
    "tests/qg2mr_sr2_e2_c1_contract.py",
    "tests/qg2mr_sr2_e2_c2_contract.py",
    "docs/qualification-gate-qg2-mr-sr2-e2-c2-r2-cpu-runtime.md",
    "docs/qualification-gate-qg2-mr-sr2-e2-c1-host-partition.md",
))
WORKFLOW = ".github/workflows/qg2-mr-sr2-e2-ci-integration.yml"


class CpuContractBehavior(unittest.TestCase):
    def test_c2101_frozen_original_cpu_method_identities(self):
        self.assertEqual(set(runner.CPU_SUITE_FILES.values()), frozen.CPU_FILES)
        self.assertEqual(len(runner.CPU_SUITE_IDS["helper"]), 7)
        self.assertEqual(len(runner.CPU_SUITE_IDS["fsr"]), 11)
        for key, filename in runner.CPU_SUITE_FILES.items():
            tree = ast.parse((ROOT / "tests" / filename).read_text(encoding="utf-8"))
            methods = {
                f"{Path(filename).stem}.{cls.name}.{method.name}"
                for cls in tree.body if isinstance(cls, ast.ClassDef)
                for method in cls.body if isinstance(method, ast.FunctionDef)
                and method.name.startswith("test_")
            }
            self.assertEqual(methods, set(runner.expected_cpu_ids(key)))
        self.assertEqual(runner.loaded_cpu_modules(), [])

    def test_c2102_missing_duplicate_wrong_cpu_identities_rejected(self):
        expected = runner.expected_cpu_ids("helper")
        for actual in (expected[:-1], expected[:-1] + (expected[0],),
                       expected[:-1] + ("wrong.identity",)):
            with self.subTest(actual=actual), self.assertRaises(runner.PartitionError):
                runner.validate_cpu_ids(actual, expected)

    @staticmethod
    def run_suite(cls):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        ids = runner.test_ids(suite)
        result = unittest.TextTestRunner(stream=io.StringIO(),
                                         resultclass=runner.CpuIdentityResult).run(suite)
        return runner.cpu_result_record(result, ids), ids

    def test_c2103_real_success_and_actual_start_id_recording(self):
        class Good(unittest.TestCase):
            def test_one(self):
                self.assertEqual(1 + 1, 2)
            def test_two(self):
                self.assertTrue(True)
        record, ids = self.run_suite(Good)
        self.assertEqual(record["executed_ids"], ids)
        self.assertEqual(record["passed_ids"], ids)
        self.assertEqual(record["stopped_ids"], ids)
        self.assertTrue(runner.strict_cpu_success(record, ids))

    def test_c2104_actual_failure_error_skip_xfail_xpass_all_rejected(self):
        class Bad(unittest.TestCase):
            def test_failure(self):
                self.fail("real failure")
            def test_error(self):
                raise RuntimeError("real error")
            @unittest.skip("real skip")
            def test_skip(self):
                pass
            @unittest.expectedFailure
            def test_xfail(self):
                self.fail("real expected failure")
            @unittest.expectedFailure
            def test_xpass(self):
                pass
        record, ids = self.run_suite(Bad)
        self.assertEqual(record["failures"], 1)
        self.assertEqual(record["errors"], 1)
        self.assertEqual(len(record["skipped"]), 1)
        self.assertEqual(record["expected_failures"], 1)
        self.assertEqual(record["unexpected_successes"], 1)
        self.assertFalse(runner.strict_cpu_success(record, ids))

    def test_c2105_count_and_execution_identity_mutations_rejected(self):
        class Good(unittest.TestCase):
            def test_a(self):
                pass
            def test_b(self):
                pass
        record, ids = self.run_suite(Good)
        for field, value in (
            ("discovered", 0), ("tests_run", 1),
            ("executed_ids", list(reversed(ids))),
            ("executed_ids", [ids[0], ids[0]]),
            ("executed_ids", [ids[0], "different.id"]),
            ("skipped", [[ids[0], "silent skip"]]),
            ("passed_ids", []),
            ("passed_ids", [ids[0], ids[0]]),
            ("stopped_ids", ids[:-1]),
        ):
            with self.subTest(field=field, value=value):
                self.assertFalse(runner.strict_cpu_success(dict(record, **{field: value}), ids))

    def test_c2106_real_extra_dependency_trigger_contract(self):
        text = (ROOT / WORKFLOW).read_text(encoding="utf-8")
        ev = frozen.events(text)
        required = frozen.R2_INPUTS | EXTRA_INPUTS
        for path in required:
            self.assertTrue((ROOT / path).is_file(), path)
        for event in ("push", "pull_request"):
            self.assertFalse(frozen.covers(ev.get(event), required), event)
        self.assertIn(frozen.DEV, ev["push"]["branches"])


def run_behavior() -> bool:
    names = unittest.defaultTestLoader.getTestCaseNames(CpuContractBehavior)
    prefixes = [f"test_c2{i}_" for i in range(101, 107)]
    if len(names) != 6 or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
        raise RuntimeError(f"Wrong C2 behavior identities: {names}")
    result = results.run_recorded_suite(
        unittest.TestSuite(CpuContractBehavior(n) for n in names))
    ok = c1.exact_result(result, 6) and not result.failures
    print(f"E2-C2 BEHAVIOR: {'6/6 PASS' if ok else 'HOLD'}", flush=True)
    return ok


def stage() -> int:
    c1.require_no_torch()
    if not c1.run_behavior() or not run_behavior():
        return 2
    targets = unittest.defaultTestLoader.getTestCaseNames(frozen.E2Targets)
    baselines = unittest.defaultTestLoader.getTestCaseNames(frozen.E2Baselines)
    for names, prefixes in ((targets, frozen.TARGET_PREFIXES), (baselines, frozen.BASE_PREFIXES)):
        if len(names) != len(prefixes) or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
            raise RuntimeError(f"Frozen E2 identity mismatch: {names}")
    green = {n for n in targets if n.startswith(GREEN_PREFIXES)}
    if len(green) != 5:
        raise RuntimeError(f"Wrong C2 scoped identities: {green}")
    target_result = results.run_recorded_suite(
        unittest.TestSuite(frozen.E2Targets(n) for n in targets))
    base_result = results.run_recorded_suite(
        unittest.TestSuite(frozen.E2Baselines(n) for n in baselines))
    failed = {case.id().rsplit(".", 1)[-1]: trace for case, trace in target_result.failures}
    if (not c1.exact_result(target_result, 36) or not c1.exact_result(base_result, 8)
            or base_result.failures or len(target_result.failures) != 31
            or set(failed) != set(targets) - green
            or not all(frozen.MARKER in trace for trace in failed.values())):
        print("E2-C2 INVALID STAGE SHAPE", flush=True)
        return 2
    print("E2-C2 SCOPED GREEN: 5/5 actual target PASS + 31/31 retained RED + "
          "8/8 baseline PASS + C1 10/10 + C2 6/6 behavior PASS; "
          "0 ERROR/SKIP/xfail/xpass; integrated E2 HOLD", flush=True)
    return 1


def full_reference() -> int:
    runner.verify_pinned_cpu()
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_parameter_policy*.py")
    if loader.errors:
        raise RuntimeError("Full CPU legacy discovery errors:\n" + "\n".join(loader.errors))
    ids = runner.test_ids(suite)
    runner.verify_cpu_sources("all")
    print(FULL_MARKER + json.dumps({"ids": ids, "errors": loader.errors}), flush=True)
    return 0


def partition_parity() -> int:
    # Three fresh subprocesses in the SAME pinned CPU environment. Do not
    # compare a dependency-free ID census to a dependency-enabled census.
    ref = c1.checked_child("tests.qg2mr_sr2_e2_c2_contract", ["full-reference"], FULL_MARKER)
    host = c1.checked_child("tests.qg2mr_sr2_e2_runner", ["host", "--discover-only"],
                            runner.DISCOVERY_MARKER)
    cpu = c1.checked_child("tests.qg2mr_sr2_e2_runner", ["r2-cpu", "--discover-only"],
                           runner.CPU_DISCOVERY_MARKER)
    full_ids, host_ids, cpu_ids = ref["ids"], host["ids"], cpu["ids"]
    for record in (ref, host, cpu):
        if record["errors"]:
            raise RuntimeError("Unexpected partition discovery errors")
    runner.validate_cpu_ids(cpu_ids, runner.expected_cpu_ids("all"))
    if (host["loaded_cpu"] or set(host_ids) & set(cpu_ids)
            or len(full_ids) != len(set(full_ids))
            or len(host_ids) != len(set(host_ids))
            or len(full_ids) != len(host_ids) + len(cpu_ids)
            or set(full_ids) != set(host_ids) | set(cpu_ids)
            or [i for i in full_ids if i not in set(cpu_ids)] != host_ids):
        raise RuntimeError("Pinned CPU partition lost, duplicated or reordered Host identities")
    print(f"E2-C2 FULL PARTITION PARITY: {len(full_ids)} original IDs = "
          f"{len(host_ids)} Host + {len(cpu_ids)} R2 CPU; no missing/duplicate/overlap; "
          "Host ordered IDs unchanged", flush=True)
    return 0


if __name__ == "__main__":
    modes = {"stage": stage, "full-reference": full_reference, "partition-parity": partition_parity}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("usage: python -m tests.qg2mr_sr2_e2_c2_contract stage|full-reference|partition-parity")
    raise SystemExit(modes[sys.argv[1]]())
