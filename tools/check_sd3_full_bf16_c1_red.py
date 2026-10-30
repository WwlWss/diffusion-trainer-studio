"""Execute Fix C C1 RED as a strict expected-failure qualification.

C1 is intentionally RED: this executable runs the actual unittest suites and
asserts every failure is EXACTLY a known missing Worker/Observer interface.
Unexpected failures, skips, changes in test discovery or unexpectedly passing
missing-interface tests invalidate RED. This is NOT Fix C GREEN or CUDA PASS.
"""
from __future__ import annotations

import argparse
import os
import sys
import unittest


_HOST_WORKER_ERROR = "ModuleNotFoundError: No module named 'tools.sd3_full_bf16_evidence_worker'"
_HOST_TIMELINE_ERROR = (
    "AttributeError: module 'tools.sd3_full_bf16_evidence_support' "
    "has no attribute 'validate_worker_event_stream'"
)
_CPU_OBSERVER_ERROR = "ModuleNotFoundError: No module named 'tools.sd3_full_bf16_worker_observer'"

_HOST = {
    **{
        "WorkerEntryContractRED." + name: _HOST_WORKER_ERROR
        for name in (
            "test_entrypoint_has_no_torch_import_at_module_import",
            "test_worker_cli_accepts_one_absolute_ticket_only",
            "test_ticket_manifest_recomputed_digests_and_output",
        )
    },
    **{
        "TimelineContractRED." + name: _HOST_TIMELINE_ERROR
        for name in (
            "test_r1_minimal_complete_step_accepted_as_metadata_only",
            "test_duplicate_sequence_and_step_before_backward_rejected",
            "test_r2_first_microbatch_must_not_have_physical_step",
            "test_r2_reused_caption_identity_rejected",
        )
    },
}

_CPU = {
    "WorkerPhysicalObserverRED." + name: _CPU_OBSERVER_ERROR
    for name in (
        "test_exact_bf16_tensor_fingerprint_uses_original_bytes",
        "test_nested_optimizer_state_fingerprint_stable_across_object_instances",
        "test_actual_accelerate_optimizer_and_scheduler_identity",
        "test_resume_rng_snapshot_interface",
    )
}
_CPU_SUCCESS = {"WorkerPhysicalObserverRED.test_pinned_real_cpu_packages"}


def check_expected_red(stage: str) -> int:
    if stage == "host":
        module, expected_errors, expected_successes = (
            "tests.test_sd3_full_bf16_worker", _HOST, set()
        )
    elif stage == "cpu":
        module, expected_errors, expected_successes = (
            "tests.test_sd3_full_bf16_worker_cpu", _CPU, _CPU_SUCCESS
        )
    else:
        raise ValueError("Unknown RED stage: " + repr(stage))

    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    discovered = set()

    def walk(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                yield from walk(t)
            else:
                yield t

    for test in walk(suite):
        discovered.add(test.id().removeprefix(module + "."))
    if discovered != (set(expected_errors) | expected_successes):
        raise AssertionError(
            "C1 RED test discovery drift: "
            f"missing={sorted((set(expected_errors) | expected_successes) - discovered)} "
            f"extra={sorted(discovered - (set(expected_errors) | expected_successes))}"
        )

    # Unittest logs are retained as actual test-run output, including expected
    # tracebacks. A green marker here means expected RED was proven, not GREEN.
    print(f"Executing actual {stage.upper()} C1 RED tests", flush=True)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    errors = {
        test.id().removeprefix(module + "."): traceback
        for test, traceback in result.errors
    }
    failed = {
        test.id().removeprefix(module + "."): traceback
        for test, traceback in result.failures
    }
    successes = discovered - set(errors) - set(failed) - {
        test.id().removeprefix(module + ".") for test, _ in result.skipped
    }
    problems = []
    if result.testsRun != len(discovered):
        problems.append(f"unexpected executed count {result.testsRun}/{len(discovered)}")
    if failed:
        problems.append("unanticipated assertion failures: " + repr(sorted(failed)))
    if result.skipped:
        problems.append("tests unexpectedly skipped: " + repr(result.skipped))
    if successes != expected_successes:
        problems.append(
            f"unexpected success set={sorted(successes)} expected={sorted(expected_successes)}"
        )
    if set(errors) != set(expected_errors):
        problems.append(
            f"error set={sorted(errors)} expected={sorted(expected_errors)}"
        )
    for name, exact_tail in expected_errors.items():
        traceback = errors.get(name, "")
        if traceback and exact_tail not in traceback:
            problems.append(
                f"{name}: wrong RED failure (expected {exact_tail!r}): "
                + traceback[-900:]
            )
    if problems:
        print("C1 RED REJECTED: " + "; ".join(problems), flush=True)
        return 1

    print(
        f"C1 {stage.upper()} EXPECTED RED VERIFIED: "
        f"tests={result.testsRun}; intended_missing_interfaces={len(expected_errors)}; "
        f"baseline_success={len(expected_successes)}; assertions_failed=0; skipped=0. "
        "This does NOT claim Worker GREEN, CUDA or release qualification.",
        flush=True,
    )
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(
                f"### C1 {stage.upper()} expected RED verified\n\n"
                f"- Actual unittest executions: {result.testsRun}\n"
                f"- Intended missing-interface errors: {len(expected_errors)}\n"
                f"- Baseline passing tests: {len(expected_successes)}\n"
                "- Unexpected errors / skips / assertion failures: 0\n"
                "- **No real CUDA, no Worker GREEN, no release qualification.**\n\n"
            )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("host", "cpu"), required=True)
    args = parser.parse_args()
    raise SystemExit(check_expected_red(args.stage))
