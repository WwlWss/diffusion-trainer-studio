"""QG-2-B4-I: exact common Host-only cumulative Admission GREEN assertion.

Read and run checked-in B301-B312 and B901-B904, no GPU, no FastAPI server.
All inherited CI users share this single fail-closed result contract.
"""
from __future__ import annotations

import unittest

from tests.test_parameter_policy_admission_red import (
    QG1B3QG2ExistingBaselines,
    QG1B3QG2TargetedRed,
)


def assert_cumulative_admission_green(stage: str) -> None:
    """Fail on a missing, failing, skipped or xfailed B312 / sibling case."""
    if type(stage) is not str or not stage.strip():
        raise AssertionError("B4-I requires a non-empty audited caller stage")

    targets = sorted(
        n for n in dir(QG1B3QG2TargetedRed) if n.startswith("test_b3")
    )
    baselines = sorted(
        n for n in dir(QG1B3QG2ExistingBaselines) if n.startswith("test_b9")
    )
    if len(targets) != 12 or len(baselines) != 4:
        raise AssertionError(
            f"{stage}: B4-I cumulative discovery mismatch "
            f"{len(targets)} target + {len(baselines)} baseline"
        )
    for number in range(301, 313):
        if sum(n.startswith(f"test_b{number}_") for n in targets) != 1:
            raise AssertionError(f"{stage}: B4-I missing/duplicate B{number}")
    for number in range(901, 905):
        if sum(n.startswith(f"test_b{number}_") for n in baselines) != 1:
            raise AssertionError(f"{stage}: B4-I missing/duplicate B{number}")

    cases = unittest.TestSuite(
        [QG1B3QG2TargetedRed(name) for name in targets]
        + [QG1B3QG2ExistingBaselines(name) for name in baselines]
    )
    result = unittest.TextTestRunner(verbosity=2).run(cases)
    if (
        result.testsRun != 16
        or result.failures
        or result.errors
        or result.skipped
        or result.expectedFailures
        or result.unexpectedSuccesses
    ):
        raise AssertionError(
            f"{stage}: B4-I cumulative contract requires 16/16 actual PASS; "
            f"ran={result.testsRun}, fail={len(result.failures)}, "
            f"error={len(result.errors)}, skip={len(result.skipped)}, "
            f"xfail={len(result.expectedFailures)}, "
            f"xpass={len(result.unexpectedSuccesses)}"
        )
    print(f"{stage}: B301-B312 + B901-B904 16/16 REAL PASS; "
          "B312 integrated GREEN; no ERROR/SKIP/xfail/xpass")
