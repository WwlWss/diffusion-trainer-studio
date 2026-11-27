"""Independent C1 scope checks. Do not edit the accepted E2-B test identities.

stage: one real GREEN, 35 known RED, eight actual baselines; exit 1.
host: current-commit gitlink checks and full Host identity/outcome parity.
legacy-reference: retained command/log identity; independently execute the
positive Host reference without importing CPU-owned modules.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests import qg2mr_sr2_e2_contract as frozen
from tests import qg2mr_sr2_e2_runner as runner
from tests import qg2mr_sr2_e2_test_results as results
from tests import qg2mr_sr2_e2_checkout as checkout
from tests import qg2mr_sr2_e2_positive_reference as positive_reference

ROOT = frozen.ROOT
HOST_ID = "test_e2h101_real_no_torch_discovery_partition"
LEGACY_RESULT = "DTS_E2_C1_LEGACY_RESULT_JSON "
KNOWN_SKIP_REASONS = frozenset({
    "runtime dependencies are not installed",
    "torch is not installed",
    "CPU Torch runtime is mandatory in SR1 CI",
    "Step 5D runtime dependencies are installed only in the dedicated smoke job.",
})
FIXTURE_PREFIX = "test_parameter_policy_c1_partition_"


@contextmanager
def fixture_directory():
    old_path = sys.path[:]
    old_modules = set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="qg2-e2-c1-") as directory:
        path = Path(directory)
        for filename in runner.R2_CPU_ONLY_FILES:
            (path / filename).write_text(
                'raise AssertionError("CPU MODULE MUST NOT BE IMPORTED")\n',
                encoding="utf-8",
            )
        try:
            yield path
        finally:
            sys.path[:] = old_path
            for name in set(sys.modules) - old_modules:
                if name.startswith(FIXTURE_PREFIX):
                    sys.modules.pop(name, None)


def write_fixture(path: Path, suffix: str, content: str):
    (path / (FIXTURE_PREFIX + suffix + ".py")).write_text(content, encoding="utf-8")


SINGLE = """import unittest
class Probe(unittest.TestCase):
    def test_real(self):
        self.assertEqual(2 + 2, 4)
"""


class PartitionBehavior(unittest.TestCase):
    def test_c1101_exact_preimport_exclusion_and_new_host(self):
        with fixture_directory() as path:
            write_fixture(path, "first", SINGLE)
            write_fixture(path, "later", SINGLE)
            suite, ids, files = runner.discover_host(path)
            self.assertEqual(files, tuple(FIXTURE_PREFIX + n + ".py" for n in ("first", "later")))
            self.assertEqual(ids, [FIXTURE_PREFIX + n + ".Probe.test_real" for n in ("first", "later")])
            self.assertEqual(runner.loaded_cpu_modules(), [])
            result = results.run_recorded_suite(suite, verbosity=1)
            record = runner.result_record(result, ids)
            self.assertTrue(runner.strict_host_success(record))
            self.assertEqual(record["passed_ids"], ids)

    def test_c1102_missing_cpu_file_fails_closed(self):
        with fixture_directory() as path:
            (path / sorted(runner.R2_CPU_ONLY_FILES)[0]).unlink()
            write_fixture(path, "missing", SINGLE)
            with self.assertRaisesRegex(runner.PartitionError, "CPU files missing"):
                runner.discover_host(path)

    def test_c1103_empty_host_selection_fails_closed(self):
        with fixture_directory() as path:
            with self.assertRaisesRegex(runner.PartitionError, "Empty Host file selection"):
                runner.discover_host(path)

    def test_c1104_unreviewed_package_rejected_before_import(self):
        with fixture_directory() as path:
            write_fixture(path, "package_host", SINGLE)
            package = path / "new_test_package"
            package.mkdir()
            (package / "__init__.py").write_text(
                'raise AssertionError("UNREVIEWED PACKAGE IMPORTED")\n', encoding="utf-8")
            with self.assertRaisesRegex(runner.PartitionError, "needs review for test packages"):
                runner.discover_host(path)

    def test_c1105_real_import_error_not_counted(self):
        with fixture_directory() as path:
            write_fixture(path, "broken", 'raise ImportError("C1 actual import regression")\n')
            with self.assertRaisesRegex(runner.PartitionError, "C1 actual import regression"):
                runner.discover_host(path)

    def test_c1106_duplicate_actual_load_tests_ids_rejected(self):
        with fixture_directory() as path:
            write_fixture(path, "duplicates", SINGLE + '''
def load_tests(loader, tests, pattern):
    return unittest.TestSuite((Probe("test_real"), Probe("test_real")))
''')
            with self.assertRaisesRegex(runner.PartitionError, "Duplicate Host test identities"):
                runner.discover_host(path)

    def test_c1107_standard_load_tests_is_respected(self):
        with fixture_directory() as path:
            write_fixture(path, "hook", SINGLE + '''
class Second(unittest.TestCase):
    def test_other(self):
        raise AssertionError("load_tests did not select the suite")

def load_tests(loader, tests, pattern):
    if pattern != "test_parameter_policy*.py":
        raise AssertionError("legacy load_tests pattern was not preserved")
    return unittest.TestSuite((Probe("test_real"),))
''')
            suite, ids, _ = runner.discover_host(path)
            self.assertEqual(ids, [FIXTURE_PREFIX + "hook.Probe.test_real"])
            result = results.run_recorded_suite(suite, verbosity=1)
            self.assertTrue(runner.strict_host_success(runner.result_record(result, ids)))

    def test_c1108_cpu_alias_already_loaded_fails_closed(self):
        with fixture_directory() as path:
            name = "tests." + Path(sorted(runner.R2_CPU_ONLY_FILES)[0]).stem
            with mock.patch.dict(sys.modules, {name: types.ModuleType(name)}):
                with self.assertRaisesRegex(runner.PartitionError, "CPU-only modules loaded"):
                    runner.discover_host(path)

    def test_c1109_real_failedtest_identity_rejected(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName("qg2_e2_c1_nonexistent_test_module")
        self.assertTrue(loader.errors)
        with self.assertRaisesRegex(runner.PartitionError, "_FailedTest"):
            runner.test_ids(suite)

    def test_c1110_strict_host_outcomes_and_explicit_skip_accounting(self):
        class Actual(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

            @unittest.skip("runtime dependencies are not installed")
            def test_deferred(self):
                self.fail("Deferred fixture must not execute")

        suite = unittest.TestLoader().loadTestsFromTestCase(Actual)
        result = results.run_recorded_suite(suite)
        record = runner.result_record(result, result.discovered_ids)
        self.assertTrue(runner.strict_host_success(record))
        self.assertEqual(len(record["skipped"]), 1)
        self.assertEqual(len(record["passed_ids"]), 1)
        self.assertEqual(record["started_ids"], record["discovered_ids"])
        for key, value in (("tests_run", 1), ("discovered", 3), ("failures", 1),
                           ("errors", 1), ("expected_failures", 1),
                           ("unexpected_successes", 1), ("loaded_cpu", ["cpu"])):
            with self.subTest(key=key):
                self.assertFalse(runner.strict_host_success(dict(record, **{key: value})))


def exact_result(result, count):
    return results.exact_result(result, count, allow_failures=True)


def run_behavior() -> bool:
    names = unittest.defaultTestLoader.getTestCaseNames(PartitionBehavior)
    prefixes = [f"test_c1{i}_" for i in range(101, 111)]
    if len(names) != 10 or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
        raise RuntimeError(f"Wrong C1 behavior test identities: {names}")
    result = results.run_recorded_suite(
        unittest.TestSuite(PartitionBehavior(n) for n in names))
    ok = exact_result(result, 10) and not result.failures
    print(f"E2-C1 BEHAVIOR: {'10/10 PASS' if ok else 'HOLD'}", flush=True)
    return ok


def require_no_torch():
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("C1 qualification requires a real no-Torch environment")


def stage() -> int:
    require_no_torch()
    if not run_behavior():
        return 2
    target_names = unittest.defaultTestLoader.getTestCaseNames(frozen.E2Targets)
    base_names = unittest.defaultTestLoader.getTestCaseNames(frozen.E2Baselines)
    for names, prefixes in ((target_names, frozen.TARGET_PREFIXES),
                            (base_names, frozen.BASE_PREFIXES)):
        if len(names) != len(prefixes) or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
            raise RuntimeError(f"Frozen test identity mismatch: {names}")
    targets = results.run_recorded_suite(
        unittest.TestSuite(frozen.E2Targets(n) for n in target_names))
    baselines = results.run_recorded_suite(
        unittest.TestSuite(frozen.E2Baselines(n) for n in base_names))
    failed = {case.id().rsplit(".", 1)[-1]: trace for case, trace in targets.failures}
    if (not exact_result(targets, 36) or not exact_result(baselines, 8)
            or baselines.failures or len(targets.failures) != 35
            or set(failed) != set(target_names) - {HOST_ID}
            or not all(frozen.MARKER in trace for trace in failed.values())):
        print("E2-C1 INVALID STAGE SHAPE", flush=True)
        return 2
    print("E2-C1 SCOPED GREEN: E2H101 actual PASS + 35/35 retained RED + "
          "8/8 baseline PASS + 10/10 partition behavior PASS; "
          "0 ERROR/SKIP/xfail/xpass; integrated E2 HOLD", flush=True)
    return 1


def legacy_reference() -> int:
    # Retain CLI and log identity; the permanent reference is positive.
    require_no_torch()
    suite, ids = positive_reference.discover_reference(ROOT / "tests", frozen.CPU_FILES)
    result = results.run_recorded_suite(suite)
    record = runner.result_record(result, ids)
    print(LEGACY_RESULT + json.dumps(record), flush=True)
    return 0 if runner.strict_host_success(record) else 1


def checked_child(module: str, args: list[str], marker: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", "-m", module, *args],
        cwd=ROOT, text=True, capture_output=True, timeout=480,
        encoding="utf-8", errors="replace",
    )
    print(proc.stderr, end="", flush=True)
    print(proc.stdout, end="", flush=True)
    rows = [line[len(marker):] for line in proc.stdout.splitlines() if line.startswith(marker)]
    if proc.returncode != 0 or len(rows) != 1:
        raise RuntimeError(f"Invalid actual {module} {args}: rc={proc.returncode}, records={len(rows)}")
    return json.loads(rows[0])


def full_host() -> int:
    require_no_torch()
    identity = checkout.verify_current_checkout(ROOT)
    print(checkout.CHECKOUT_MARKER + json.dumps(identity), flush=True)
    for rel, sha in identity["actual_gitlinks"].items():
        print(f"E2-C1 SUBMODULE {rel}: {sha}", flush=True)
    if not (ROOT / "frontend/dist/assets/layout.96d49288.js").is_file():
        raise RuntimeError("Missing required frontend asset; Host execution incomplete")
    subprocess.run(["node", "--version"], check=True, timeout=30, shell=False)
    reference = checked_child("tests.qg2mr_sr2_e2_c1_contract", ["legacy-reference"], LEGACY_RESULT)
    discovered = checked_child("tests.qg2mr_sr2_e2_runner", ["host", "--discover-only"], runner.DISCOVERY_MARKER)
    actual = checked_child("tests.qg2mr_sr2_e2_runner", ["host"], runner.RESULT_MARKER)
    ids = discovered["ids"]
    if (discovered["errors"] or discovered["loaded_cpu"]
            or ids != reference["ids"] or len(ids) != len(set(ids))
            or actual.get("discovered_ids") != ids
            or actual.get("started_ids") != ids
            or reference.get("discovered_ids") != ids
            or reference.get("started_ids") != ids
            or actual.get("passed_ids") != reference.get("passed_ids")
            or not runner.strict_host_success(actual)
            or not runner.strict_host_success(reference)
            or sorted(actual["skipped"]) != sorted(reference["skipped"])):
        raise RuntimeError("Host execution identity/outcome/skip parity failed")
    if any(reason not in KNOWN_SKIP_REASONS for _, reason in actual["skipped"]):
        raise RuntimeError(f"Unexpected Host skip reasons: {actual['skipped']}")
    frontend = [i for i in ids if i.startswith("test_parameter_policy_frontend_behavior.")]
    if len(frontend) != 6 or not set(frontend) <= set(actual["passed_ids"]):
        raise RuntimeError("Six real Frontend tests were not exercised")
    print(f"E2-C1 FULL HOST GREEN: {actual['tests_run']} actual test outcomes; "
          f"{len(actual['passed_ids'])} PASS, "
          f"{len(actual['skipped'])} inherited dependency SKIP, 0 FAIL/ERROR/xfail/xpass; "
          "independent positive ordered identity and skip parity; Frontend 6/6 real PASS", flush=True)
    return 0


if __name__ == "__main__":
    modes = {"stage": stage, "host": full_host, "legacy-reference": legacy_reference}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("usage: python -m tests.qg2mr_sr2_e2_c1_contract stage|host|legacy-reference")
    raise SystemExit(modes[sys.argv[1]]())
