"""Strict fixed-test discovery and exact outcome verifier for C1/C3 phases.

Historical C1 expected-RED cases use exact missing-interface errors.
After C3 implementation, all C1 Host and pinned CPU cases must ACTUALLY pass,
with no skip or unexpected failures; the verifier does not claim C4,
full-worker completion, GPU training, or Full-BF16 release qualification.
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

_HOST = {}

_CPU = {}
_HOST_SUCCESS = {
    "WorkerEntryContractRED." + name
    for name in (
        "test_entrypoint_has_no_torch_import_at_module_import",
        "test_worker_cli_accepts_one_absolute_ticket_only",
        "test_ticket_manifest_recomputed_digests_and_output",
    )
} | {
    "TimelineContractRED." + name
    for name in (
        "test_r1_minimal_complete_step_accepted_as_metadata_only",
        "test_duplicate_sequence_and_step_before_backward_rejected",
        "test_r2_first_microbatch_must_not_have_physical_step",
        "test_r2_reused_caption_identity_rejected",
    )
}
_CPU_SUCCESS = {
    "WorkerPhysicalObserverRED." + name
    for name in (
        "test_pinned_real_cpu_packages",
        "test_exact_bf16_tensor_fingerprint_uses_original_bytes",
        "test_nested_optimizer_state_fingerprint_stable_across_object_instances",
        "test_actual_accelerate_optimizer_and_scheduler_identity",
        "test_resume_rng_snapshot_interface",
    )
}


def check_expected_red(stage: str) -> int:
    if stage == "host":
        module, expected_errors, expected_successes = (
            "tests.test_sd3_full_bf16_worker", _HOST, _HOST_SUCCESS
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
    # unittest reports each failing subTest as a distinct _SubTest with an
    # id suffixed by its parameters. Normalize to the parent method, but keep
    # *every* individual error: a partly executing negative test is not RED.
    def parent_id(test):
        return getattr(test, "test_case", test).id().removeprefix(module + ".")

    errors = {}
    for test, traceback in result.errors:
        errors.setdefault(parent_id(test), []).append(traceback)
    failed = {}
    for test, traceback in result.failures:
        failed.setdefault(parent_id(test), []).append(traceback)
    successes = discovered - set(errors) - set(failed) - {
        parent_id(test) for test, _ in result.skipped
    }
    problems = []
    if result.testsRun != len(discovered):
        problems.append(f"unexpected executed count {result.testsRun}/{len(discovered)}")
    if failed:
        problems.append("unanticipated assertion failures: " + repr(sorted(failed)))
    # A subTest can contain several independent negative cases that are
    # expected to PASS on C3. They are counted as one parent unittest method.
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
        tracebacks = errors.get(name, [])
        count = 3 if (
            stage == "host"
            and name == "TimelineContractRED.test_duplicate_sequence_and_step_before_backward_rejected"
        ) else 1
        if len(tracebacks) != count:
            problems.append(f"{name}: {len(tracebacks)} error events; expected {count}")
        for traceback in tracebacks:
            if exact_tail not in traceback:
                problems.append(
                    f"{name}: wrong RED failure (expected {exact_tail!r}): "
                    + traceback[-900:]
                )
    if problems:
        print("C1 RED REJECTED: " + "; ".join(problems), flush=True)
        return 1

    outcome = (
        f"C1 {stage.upper()} EXPECTED RED VERIFIED"
        if expected_errors else f"C1/C3 {stage.upper()} REAL GREEN VERIFIED"
    )
    print(
        f"{outcome}: tests={result.testsRun}; "
        f"expected_missing_interfaces={len(expected_errors)}; "
        f"actual_pass={len(expected_successes)}; "
        "unexpected_failures=0; skipped=0. "
        "C4, real CUDA and release qualification remain unverified.",
        flush=True,
    )
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(
                f"### {outcome}\n\n"
                f"- Actual unittest executions: {result.testsRun}\n"
                f"- Expected missing-interface errors: {len(expected_errors)}\n"
                f"- Actual passing tests: {len(expected_successes)}\n"
                "- Unexpected errors, skipped tests or assertion failures: 0\n"
                "- **No C4 worker_complete, real CUDA or release qualification.**\n\n"
            )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("host", "cpu"), required=True)
    args = parser.parse_args()
    raise SystemExit(check_expected_red(args.stage))
