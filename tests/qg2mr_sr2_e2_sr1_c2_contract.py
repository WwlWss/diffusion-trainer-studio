"""SR1-C2 scoped current checkout and positive partition acceptance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests import qg2mr_sr2_e2_checkout as checkout
from tests import qg2mr_sr2_e2_contract as contract
from tests import qg2mr_sr2_e2_c1_contract as c1
from tests import qg2mr_sr2_e2_runner as runner
from tests import qg2mr_sr2_e2_test_results as results
from tests import qg2mr_sr2_e2_sr1_contract as frozen
from tests import qg2mr_sr2_e2_sr1_c1_contract as fix_a


def actual_reference_execution(root):
    # Actual retained reference CLI entry in a fresh process; no mocked
    # discovery or unittest result.
    script = """
from pathlib import Path
import sys
from tests import qg2mr_sr2_e2_c1_contract as c1
c1.ROOT = Path(sys.argv[1])
raise SystemExit(c1.legacy_reference())
"""
    proc = subprocess.run([sys.executable, "-B", "-c", script, str(root)],
                          cwd=c1.ROOT, text=True, encoding="utf-8", errors="strict",
                          capture_output=True, timeout=120)
    rows = [line[len(c1.LEGACY_RESULT):] for line in proc.stdout.splitlines()
            if line.startswith(c1.LEGACY_RESULT)]
    if proc.returncode or len(rows) != 1:
        raise RuntimeError(f"Actual reference execution failed: {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return json.loads(rows[0])


def actual_gate(repo):
    return frozen.host_checkout_gate(repo)


class CheckoutPartitionBehavior(unittest.TestCase):
    def test_b101_actual_current_gitlinks_and_advanced_commit(self):
        with frozen.git_fixture() as (repo, first, second):
            head = frozen.git(repo, "rev-parse", "HEAD")
            tree = frozen.git(repo, "rev-parse", "HEAD^{tree}")
            self.assertNotEqual(first, second)
            with mock.patch.dict(os.environ, {"GITHUB_SHA": head}):
                record = checkout.verify_current_checkout(repo)
            self.assertEqual(record["head"], head)
            self.assertEqual(record["tree"], tree)
            self.assertEqual(record["event_sha"], head)
            self.assertEqual(record["expected_gitlinks"], dict.fromkeys(checkout.REQUIRED_GITLINKS, second))
            self.assertEqual(record["actual_gitlinks"], record["expected_gitlinks"])
            self.assertEqual(actual_gate(repo), (True, ""))

    def test_b102_event_identity_and_root_rejected(self):
        with frozen.git_fixture() as (repo, first, second):
            head = frozen.git(repo, "rev-parse", "HEAD")
            with mock.patch.dict(os.environ, {"GITHUB_SHA": first}):
                with self.assertRaisesRegex(checkout.CheckoutError, "Event checkout identity mismatch"):
                    checkout.verify_current_checkout(repo)
            with mock.patch.dict(os.environ, {"GITHUB_SHA": "HEAD"}):
                with self.assertRaisesRegex(checkout.CheckoutError, "Invalid full Git object"):
                    checkout.verify_current_checkout(repo)
            with mock.patch.dict(os.environ, {"GITHUB_SHA": head}):
                self.assertEqual(checkout.verify_current_checkout(repo)["head"], head)
                nested = repo / "untracked-directory"
                nested.mkdir()
                with self.assertRaisesRegex(checkout.CheckoutError, "Not the actual Git worktree root"):
                    checkout.verify_current_checkout(nested)

    def test_b103_wrong_nonfrontend_checkout_and_recovery(self):
        with frozen.git_fixture() as (repo, first, second):
            child = repo / checkout.REQUIRED_GITLINKS[1]
            self.assertEqual(actual_gate(repo), (True, ""))
            frozen.git(child, "checkout", "-q", "--detach", first)
            ok, reason = actual_gate(repo)
            self.assertFalse(ok)
            self.assertIn("Current submodule mismatch", reason)
            frozen.git(child, "checkout", "-q", "--detach", second)
            self.assertEqual(actual_gate(repo), (True, ""))

    def test_b104_deinitialized_nonfrontend_cannot_fall_back_to_parent(self):
        with frozen.git_fixture() as (repo, first, second):
            rel = checkout.REQUIRED_GITLINKS[2]
            self.assertEqual(actual_gate(repo), (True, ""))
            frozen.git(repo, "submodule", "deinit", "-f", "--", rel)
            ok, reason = actual_gate(repo)
            self.assertFalse(ok)
            self.assertTrue("Not the actual Git worktree root" in reason
                            or "Missing submodule checkout" in reason
                            or "Git checkout command failed" in reason, reason)
            self.assertTrue((repo / "frontend/dist/assets/layout.96d49288.js").is_file())
            frozen.git(repo, "submodule", "update", "--init", "--", rel)
            self.assertEqual(actual_gate(repo), (True, ""))

    def test_b105_committed_non_gitlink_and_missing_path_rejected(self):
        with frozen.git_fixture() as (repo, first, second):
            head = frozen.git(repo, "rev-parse", "HEAD")
            self.assertEqual(actual_gate(repo), (True, ""))
            ordinary = repo / "ordinary.txt"
            ordinary.write_text("ordinary committed blob\n", encoding="utf-8")
            frozen.git(repo, "add", "--", "ordinary.txt")
            frozen.git(repo, "commit", "-qm", "ordinary file")
            new = frozen.git(repo, "rev-parse", "HEAD")
            self.assertNotEqual(new, head)
            with self.assertRaisesRegex(checkout.CheckoutError, "non-gitlink"):
                checkout.read_gitlinks(repo, new, ("ordinary.txt",))
            with self.assertRaisesRegex(checkout.CheckoutError, "gitlinks missing"):
                checkout.read_gitlinks(repo, new, ("frontend", "not-present"))
            self.assertEqual(actual_gate(repo), (True, ""))

    def test_b106_actual_reference_execution_zero_one_two_eager_cpu_modules(self):
        expected = ["test_parameter_policy_sr1_host_fixture.Host.test_actual"]
        for eager_count in range(3):
            with self.subTest(eager=eager_count), frozen.discovery_fixture(eager_count) as root:
                raw = frozen.raw_fixture_discovery(root)
                self.assertEqual(len(raw["errors"]), eager_count)
                self.assertEqual(contract.reference_discovery(), expected)
                row = actual_reference_execution(root)
                self.assertTrue(runner.strict_host_success(row))
                self.assertEqual(row["discovered_ids"], expected)
                self.assertEqual(row["passed_ids"], expected)
                self.assertEqual(row["loaded_cpu"], [])

    def test_b107_exclude_cpu_before_import_and_include_new_load_tests(self):
        expected = ["test_parameter_policy_sr1_host_fixture.Host.test_actual",
                    "test_parameter_policy_sr1_new_host.Selected.test_new"]
        with frozen.discovery_fixture(0) as root:
            tests = root / "tests"
            # Active tripwires, not mocked import failures.
            for filename in contract.CPU_FILES:
                (tests / filename).write_text(
                    'raise AssertionError("CPU-owned source was imported")\n', encoding="utf-8")
            (tests / "test_parameter_policy_sr1_new_host.py").write_text(
                """import unittest
class Selected(unittest.TestCase):
    def test_new(self):
        self.assertEqual(2 + 2, 4)
class Unselected(unittest.TestCase):
    def test_must_not_run(self):
        raise AssertionError("load_tests ignored")
def load_tests(loader, tests, pattern):
    assert pattern == "test_parameter_policy*.py"
    return unittest.TestSuite([Selected("test_new")])
""", encoding="utf-8")
            self.assertEqual(contract.reference_discovery(), expected)
            row = actual_reference_execution(root)
            self.assertTrue(runner.strict_host_success(row))
            self.assertEqual(row["passed_ids"], expected)

    def test_b108_reference_rejects_real_host_import_error_and_missing_cpu_file(self):
        with frozen.discovery_fixture(0) as root:
            host = root / "tests/test_parameter_policy_sr1_host_fixture.py"
            host.write_text('raise ImportError("actual host regression")\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "actual host regression"):
                contract.reference_discovery()
        with frozen.discovery_fixture(0) as root:
            (root / "tests" / sorted(contract.CPU_FILES)[0]).unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing exact CPU ownership"):
                contract.reference_discovery()

    def test_b109_actual_cpu_identity_baseline_rejects_missing_method(self):
        with tempfile.TemporaryDirectory(prefix="dts-sr1-c2-cpu-census-") as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            for filename in contract.FROZEN_FILES:
                shutil.copyfile(c1.ROOT / "tests" / filename, tests / filename)
            fixture = root / "tools/_parameter_policy_gpu_low_level_fixture.py"
            fixture.parent.mkdir()
            shutil.copyfile(c1.ROOT / "tools/_parameter_policy_gpu_low_level_fixture.py", fixture)
            check = contract.E2Baselines("test_e2b901_original_census_and_cpu_paths")
            filename = runner.CPU_SUITE_FILES["helper"]
            source = frozen.cpu_fixture_source(filename, False)
            for name in contract.CPU_FILES:
                (tests / name).write_text(frozen.cpu_fixture_source(name, False), encoding="utf-8")
            with mock.patch.object(contract, "ROOT", root), mock.patch.object(contract, "TESTS", tests):
                check.test_e2b901_original_census_and_cpu_paths()
                method = runner.expected_cpu_ids("helper")[0].rsplit(".", 1)[-1]
                self.assertEqual(source.count("def " + method + "("), 1)
                (tests / filename).write_text(
                    source.replace("def " + method + "(", "def test_unreviewed_replacement(", 1),
                    encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, filename):
                    check.test_e2b901_original_census_and_cpu_paths()


def run_behavior(cls, prefix, count):
    names = unittest.TestLoader().getTestCaseNames(cls)
    if (len(names) != count or any(sum(n.startswith(f"test_{prefix}{i}_") for n in names) != 1
                                  for i in range(101, 101 + count))):
        raise RuntimeError(f"Unexpected behavior identities: {cls.__name__}")
    result = results.run_recorded_suite(unittest.TestSuite(cls(n) for n in names))
    ok = results.exact_result(result, count)
    print(f"SR1-C2 {cls.__name__}: {count}/{count} actual PASS" if ok
          else f"SR1-C2 {cls.__name__}: HOLD", flush=True)
    return ok


def main():
    c1.require_no_torch()
    if frozen.yaml.__version__ != "6.0.2":
        raise RuntimeError("Require the original test-only YAML witness version")
    if not run_behavior(fix_a.ResultBehavior, "a", 8):
        return 2
    if not run_behavior(CheckoutPartitionBehavior, "b", 9):
        return 2
    t_ok, t = frozen.execute(frozen.Targets, frozen.TARGET_NUMBERS)
    b_ok, b = frozen.execute(frozen.Baselines, frozen.BASE_NUMBERS)
    green = [i for i in t["expected_ids"] if i.rsplit(".", 1)[-1].startswith(
        ("test_d101_", "test_d501_", "test_d502_", "test_d503_",
         "test_d701_", "test_d702_", "test_d703_"))]
    red = set(t["expected_ids"]) - set(green)
    if (t_ok and b_ok and len(green) == 7 and len(red) == 12
            and t["passed_ids"] == green
            and set(t["failed_ids"]) == set(t["marked_failed_ids"]) == red
            and b["passed_ids"] == b["expected_ids"] and not b["failed_ids"]):
        print("E2-D-SR1-C2 SCOPED GREEN: 7/7 target PASS + 12/12 retained RED + "
              "12/12 baseline PASS + 8/8 result behavior PASS + 9/9 checkout/partition behavior PASS; "
              "0 ERROR/SKIP/xfail/xpass; integrated SR1 HOLD", flush=True)
        return 1
    print("E2-D-SR1-C2 INVALID STAGE SHAPE", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
