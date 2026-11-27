"""Shared unittest identity and terminal-outcome contract for E2 CI.

This is test infrastructure, not a model/runtime qualification authority.
Host skip ownership remains a separate policy. Results are observations of
unittest callbacks, not inference from testsRun minus failure/skip counts.
"""
from __future__ import annotations

import unittest


class StrictIdentityResult(unittest.TextTestResult):
    """Keep every start, stop and successful terminal callback in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_ids = []
        self.stopped_ids = []
        self.passed_ids = []
        self.discovered_ids = None

    def startTest(self, test):
        self.executed_ids.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        self.stopped_ids.append(test.id())
        super().stopTest(test)

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def suite_ids(suite):
    """Capture the identity order before unittest consumes its suite."""
    def walk(item):
        if isinstance(item, unittest.TestCase):
            yield item.id()
        elif isinstance(item, unittest.BaseTestSuite):
            for child in item:
                yield from walk(child)
        else:
            raise ValueError(f"Unsupported unittest discovery object: {type(item)!r}")

    ids = list(walk(suite))
    if (not _ids(ids) or any(i.startswith("unittest.loader._FailedTest.") for i in ids)):
        raise ValueError("Invalid, empty, duplicate or failed discovery identities")
    return ids


def run_recorded_suite(suite, *, verbosity=2, stream=None):
    ids = suite_ids(suite)
    result = unittest.TextTestRunner(
        stream=stream, verbosity=verbosity, resultclass=StrictIdentityResult).run(suite)
    result.discovered_ids = ids
    return result


def result_record(result, ids):
    """Record actual callbacks and actual unittest non-success outcomes.

    Keep the existing ids/executed_ids and aggregate fields for log consumers.
    Missing terminal callbacks must never be filled from aggregate counts.
    Class/module errors, skips and subtest nonpasses retain their actual
    unittest identities; the strict normal-suite contract rejects them if
    they cannot be closed against started test identities.
    """
    if not isinstance(result, StrictIdentityResult):
        raise TypeError("StrictIdentityResult required for execution evidence")
    expected = list(ids)
    if result.discovered_ids is not None and result.discovered_ids != expected:
        raise ValueError("Result discovery identity mismatch")
    return {
        "discovered": len(expected),
        "tests_run": result.testsRun,
        "ids": expected,
        "executed_ids": list(result.executed_ids),
        "discovered_ids": expected.copy(),
        "started_ids": list(result.executed_ids),
        "stopped_ids": list(result.stopped_ids),
        "passed_ids": list(result.passed_ids),
        "failures": len(result.failures),
        "failed_ids": [case.id() for case, _ in result.failures],
        "errors": len(result.errors),
        "error_ids": [case.id() for case, _ in result.errors],
        "skipped": [[case.id(), reason] for case, reason in result.skipped],
        "expected_failures": len(result.expectedFailures),
        "expected_failure_ids": [case.id() for case, _ in result.expectedFailures],
        "unexpected_successes": len(result.unexpectedSuccesses),
        "unexpected_success_ids": [case.id() for case in result.unexpectedSuccesses],
    }


def _ids(values, *, allow_empty=False):
    return (type(values) is list and (allow_empty or bool(values))
            and all(type(i) is str and bool(i) for i in values)
            and len(values) == len(set(values)))


def closed_result(record, *, allow_skips=False, allow_failures=False):
    """Validate a complete, ordered normal-suite result, fail closed.

    A strict PASS has no nonpass outcomes. A Host result can have explicit
    skips, and a staged RED result can have assertion failures. Neither is
    a successful test. Error/xfail/xpass always block. A skipped subtest or
    class-level skip without an individual start cannot count as a complete
    test outcome under this contract.
    """
    if type(record) is not dict:
        return False
    ids = record.get("discovered_ids")
    started = record.get("started_ids")
    stopped = record.get("stopped_ids")
    passed = record.get("passed_ids")
    if not (_ids(ids) and _ids(started) and _ids(stopped)
            and _ids(passed, allow_empty=True)):
        return False
    if (any(type(record.get(k)) is not int for k in ("discovered", "tests_run"))
            or record["discovered"] != len(ids) or record["tests_run"] != len(ids)
            or ids != started or ids != stopped
            or record.get("ids") != ids or record.get("executed_ids") != started):
        return False
    for count_key, ids_key in (
        ("failures", "failed_ids"), ("errors", "error_ids"),
        ("expected_failures", "expected_failure_ids"),
        ("unexpected_successes", "unexpected_success_ids"),
    ):
        outcomes = record.get(ids_key)
        count = record.get(count_key)
        if not (_ids(outcomes, allow_empty=True) and type(count) is int
                and count == len(outcomes)):
            return False
    if record["errors"] or record["expected_failures"] or record["unexpected_successes"]:
        return False
    failed = record["failed_ids"]
    if failed and not allow_failures:
        return False
    skips = record.get("skipped")
    if type(skips) is not list or any(
        type(row) is not list or len(row) != 2 or
        type(row[0]) is not str or not row[0] or type(row[1]) is not str
        for row in skips
    ):
        return False
    skipped = [row[0] for row in skips]
    if not _ids(skipped, allow_empty=True) or (skipped and not allow_skips):
        return False
    terminal = passed + failed + skipped
    if len(terminal) != len(ids) or len(set(terminal)) != len(terminal) or set(terminal) != set(ids):
        return False
    # Each outcome channel preserves its relative order within discovery.
    return all([i for i in ids if i in set(group)] == group
               for group in (passed, failed, skipped))


def exact_result(result, count, *, allow_failures=False):
    """For real in-process contract suites, including staged RED."""
    if not isinstance(result, StrictIdentityResult) or result.discovered_ids is None:
        return False
    if type(count) is not int or len(result.discovered_ids) != count:
        return False
    return closed_result(result_record(result, result.discovered_ids),
                         allow_failures=allow_failures)
