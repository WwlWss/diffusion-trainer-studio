"""SR1-C1 scoped result closure, preserving all 19 frozen RED identities."""
from __future__ import annotations

import copy
import io
import unittest

from tests import qg2mr_sr2_e2_runner as runner
from tests import qg2mr_sr2_e2_test_results as results
from tests import qg2mr_sr2_e2_sr1_contract as frozen
from tests import qg2mr_sr2_e2_c1_contract as c1


def record(cls):
    suite = unittest.TestLoader().loadTestsFromTestCase(cls)
    outcome = results.run_recorded_suite(suite, stream=io.StringIO())
    return runner.result_record(outcome, outcome.discovered_ids)


class Good(unittest.TestCase):
    def test_a(self):
        self.assertTrue(True)

    def test_b(self):
        with self.subTest(value=1):
            self.assertEqual(1, 1)


class ResultBehavior(unittest.TestCase):
    def test_a101_actual_success_and_subtest_success(self):
        row = record(Good)
        ids = row["ids"]
        self.assertEqual(len(ids), 2)
        for field in ("started_ids", "stopped_ids", "passed_ids", "executed_ids", "discovered_ids"):
            self.assertEqual(row[field], ids)
        self.assertTrue(runner.strict_host_success(row))
        self.assertTrue(runner.strict_cpu_success(row, ids))

    def test_a102_real_authorized_shape_skip_without_owner_change(self):
        class Mixed(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)
            @unittest.skip("runtime dependencies are not installed")
            def test_skip(self):
                self.fail("Should be skipped")
        row = record(Mixed)
        self.assertEqual(len(row["passed_ids"]), 1)
        self.assertEqual(len(row["skipped"]), 1)
        self.assertTrue(runner.strict_host_success(row))
        self.assertFalse(runner.strict_cpu_success(row, row["ids"]))

    def test_a103_actual_no_terminal_and_duplicate_success_rejected(self):
        class Duplicate(unittest.TestCase):
            def test_body(self):
                pass
            def run(self, result=None):
                result.startTest(self)
                result.addSuccess(self)
                result.addSuccess(self)
                result.stopTest(self)
                return result
        for cls in (frozen.NoTerminalResult, Duplicate):
            row = record(cls)
            self.assertFalse(runner.strict_host_success(row))
            self.assertFalse(runner.strict_cpu_success(row, row["ids"]))

    def test_a104_actual_class_setup_error_and_class_setup_skip_rejected(self):
        class Broken(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("actual setup error")
            def test_body(self):
                pass
        class Deferred(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("runtime dependencies are not installed")
            def test_body(self):
                pass
        for cls in (Broken, Deferred):
            row = record(cls)
            self.assertEqual(row["started_ids"], [])
            self.assertFalse(runner.strict_host_success(row))
            self.assertFalse(runner.strict_cpu_success(row, row["ids"]))

    def test_a105_actual_subtest_failure_and_skip_rejected(self):
        class Broken(unittest.TestCase):
            def test_body(self):
                with self.subTest(value=1):
                    self.fail("actual subtest failure")
        class Deferred(unittest.TestCase):
            def test_body(self):
                with self.subTest(value=1):
                    self.skipTest("runtime dependencies are not installed")
        for cls in (Broken, Deferred):
            row = record(cls)
            self.assertFalse(runner.strict_host_success(row))
            self.assertFalse(runner.strict_cpu_success(row, row["ids"]))
            self.assertFalse(results.closed_result(row, allow_skips=True, allow_failures=True))

    def test_a106_malformed_duplicate_mismatched_and_missing_record_rejected(self):
        row = record(Good)
        ids = row["ids"]
        for field, value in (
            ("passed_ids", [ids[0], ids[0]]),
            ("passed_ids", list(reversed(ids))),
            ("passed_ids", [ids[0], "replacement"]),
            ("stopped_ids", ids[:-1]),
            ("started_ids", list(reversed(ids))),
            ("executed_ids", list(reversed(ids))),
            ("discovered_ids", [ids[0], "replacement"]),
            ("tests_run", True),
            ("failures", False),
            ("failed_ids", [ids[0]]),
            ("skipped", [[ids[0], "duplicate terminal"]]),
            ("skipped", [[{}, "bad identity"]]),
            ("passed_ids", {"bad": "record"}),
        ):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(row)
                changed[field] = value
                self.assertFalse(runner.strict_host_success(changed))
                self.assertFalse(runner.strict_cpu_success(changed, ids))
        for field in ("passed_ids", "started_ids", "stopped_ids", "error_ids"):
            changed = copy.deepcopy(row)
            del changed[field]
            self.assertFalse(runner.strict_host_success(changed))
            self.assertFalse(runner.strict_cpu_success(changed, ids))

    def test_a107_real_staged_failures_close_but_never_pass(self):
        class Mixed(unittest.TestCase):
            def test_fail(self):
                self.fail("actual stage RED")
            def test_ok(self):
                self.assertTrue(True)
        suite = unittest.TestLoader().loadTestsFromTestCase(Mixed)
        result = results.run_recorded_suite(suite, stream=io.StringIO())
        row = runner.result_record(result, result.discovered_ids)
        self.assertEqual(len(row["failed_ids"]), 1)
        self.assertEqual(len(row["passed_ids"]), 1)
        self.assertTrue(results.exact_result(result, 2, allow_failures=True))
        self.assertFalse(results.exact_result(result, 2))
        self.assertFalse(runner.strict_host_success(row))
        self.assertFalse(runner.strict_cpu_success(row, row["ids"]))

    def test_a108_cross_process_pass_identity_control_and_replacement(self):
        discovered = frozen.FRONTEND_IDS + ["sr1_fixture.original.test_body"]
        self.assertTrue(frozen.host_correlation_gate(discovered, discovered))
        self.assertFalse(frozen.host_correlation_gate(
            discovered, frozen.FRONTEND_IDS + ["sr1_fixture.replacement.test_body"]))


def main():
    c1.require_no_torch()
    if frozen.yaml.__version__ != "6.0.2":
        raise RuntimeError("Require the original test-only YAML witness version")
    names = unittest.TestLoader().getTestCaseNames(ResultBehavior)
    if (len(names) != 8 or any(sum(n.startswith(f"test_a{i}_") for n in names) != 1
                               for i in range(101, 109))):
        raise RuntimeError("Invalid Fix A behavior identities")
    behavior = results.run_recorded_suite(unittest.TestSuite(ResultBehavior(n) for n in names))
    if not results.exact_result(behavior, 8):
        print("SR1-C1 BEHAVIOR HOLD", flush=True)
        return 2
    print("SR1-C1 BEHAVIOR: 8/8 actual PASS", flush=True)
    t_ok, t = frozen.execute(frozen.Targets, frozen.TARGET_NUMBERS)
    b_ok, b = frozen.execute(frozen.Baselines, frozen.BASE_NUMBERS)
    green = [i for i in t["expected_ids"] if i.rsplit(".", 1)[-1].startswith(
        ("test_d501_", "test_d502_", "test_d503_"))]
    red = set(t["expected_ids"]) - set(green)
    if (t_ok and b_ok and len(green) == 3 and len(red) == 16
            and t["passed_ids"] == green
            and set(t["failed_ids"]) == set(t["marked_failed_ids"]) == red
            and b["passed_ids"] == b["expected_ids"] and not b["failed_ids"]):
        print("E2-D-SR1-C1 SCOPED GREEN: 3/3 target PASS + 16/16 retained RED + "
              "12/12 baseline PASS + 8/8 result behavior PASS; "
              "0 ERROR/SKIP/xfail/xpass; integrated SR1 HOLD", flush=True)
        return 1
    print("E2-D-SR1-C1 INVALID STAGE SHAPE", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
