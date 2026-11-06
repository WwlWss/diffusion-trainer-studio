"""QG-1A3-SR1: six frozen F2/F3 negative-contract and safe-baseline tests.

Tests-only RED before F2/F3 fix: SR101-SR104 targeted FAIL, SR105-SR106
baseline PASS; zero ERROR/SKIP/xfail/xpass. The pure readiness reducer is
already present. Tests NEVER enable a training job, sign proof or run CUDA.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
import unittest

from mikazuki.parameter_policy_readiness import (
    ExecutionReadiness,
    ReadinessIssue,
    evaluate_execution_readiness,
)


_MARKER = "QG-1A3-SR1 CONTRACT NOT SATISFIED"
_SECRET = "WORKER_PRIVATE_KEY_MUST_NOT_BE_ECHOED"
_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "qg1a-readiness-host.yml"


class _ExplodingLengthSequence(Sequence):
    def __len__(self):
        raise RuntimeError(_SECRET)

    def __getitem__(self, index):
        raise IndexError(index)


class _ExplodingItemSequence(Sequence):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError(_SECRET)


class _ExplodingControlSequence(Sequence):
    def __init__(self, exception_type):
        self.exception_type = exception_type

    def __len__(self):
        raise self.exception_type("control exception")

    def __getitem__(self, index):
        raise IndexError(index)


def _event_paths_and_branches(raw: str, event: str) -> tuple[set[str], set[str]]:
    """Inspect the actual GitHub workflow event blocks, not random text.

    The frozen YAML structure uses on:<event>: branches/paths with literal
    list entries; no PyYAML/install or reliance on YAML 1.1 'on' coercion.
    """
    current_event = None
    current_section = None
    paths: set[str] = set()
    branches: set[str] = set()
    for line in raw.splitlines():
        if re.fullmatch(r"  (push|pull_request|workflow_dispatch):", line):
            current_event = line.strip()[:-1]
            current_section = None
            continue
        if line.startswith("  ") and not line.startswith("    "):
            current_event = None
            current_section = None
            continue
        if current_event != event:
            continue
        if line == "    paths:":
            current_section = "paths"
            continue
        if line == "    branches:":
            current_section = "branches"
            continue
        if line.startswith("    ") and not line.startswith("      "):
            current_section = None
            continue
        if line.startswith("      - ") and current_section:
            item = line.strip()[2:].strip().strip('"').strip("'")
            if current_section == "paths":
                paths.add(item)
            else:
                branches.add(item)
    return paths, branches


class QG1A3SR1Contract(unittest.TestCase):
    def _expect_no_leak_value_error(self, action):
        try:
            action()
        except ValueError as exc:
            self.assertNotIn(_SECRET, str(exc))
        except Exception as exc:
            self.fail(
                f"{_MARKER}: malformed Sequence raised "
                f"{type(exc).__name__} instead of ValueError"
            )
        else:
            self.fail(f"{_MARKER}: malformed Sequence was accepted")

    def test_sr101_custom_sequence_len_failure_has_stable_value_error(self):
        self._expect_no_leak_value_error(
            lambda: evaluate_execution_readiness(
                capability="implemented",
                hard_errors=_ExplodingLengthSequence(),
            )
        )

    def test_sr102_custom_sequence_getitem_failure_has_stable_value_error(self):
        self._expect_no_leak_value_error(
            lambda: evaluate_execution_readiness(
                capability="implemented",
                warnings=_ExplodingItemSequence(),
            )
        )

    def test_sr103_direct_readiness_construction_rejects_malformed_sequence(self):
        self._expect_no_leak_value_error(
            lambda: ExecutionReadiness(
                decision="allow",
                capability="implemented",
                hard_errors=_ExplodingLengthSequence(),
                warnings=(),
                evidence_tier="unverified",
            )
        )

    def test_sr104_push_and_pr_paths_watch_all_legacy_baseline_dependencies(self):
        source = _WORKFLOW.read_text(encoding="utf-8")
        required_paths = {
            ".github/workflows/qg1a-readiness-host.yml",
            "docs/qualification-gate-qg1a-readiness-contract.md",
            "tests/test_parameter_policy_qg1a_readiness.py",
            "tests/test_parameter_policy_qg1a_sr1.py",
            "mikazuki/parameter_policy_readiness.py",
            "mikazuki/parameter_policy_execution.py",
            "mikazuki/optimizer_profiles.py",
            "mikazuki/parameter_policy_runtime_bools.py",
            "mikazuki/parameter_policy.py",
            "mikazuki/training_request.py",
            "mikazuki/parameter_policy_trainer.py",
        }
        problems = []
        for event in ("push", "pull_request"):
            paths, branches = _event_paths_and_branches(source, event)
            missing = required_paths - paths
            if missing:
                problems.append(f"{event} paths missing {sorted(missing)}")
            if event == "push":
                missing_branches = {
                    "main", "feature/qualification-gate-qg1a-readiness-contract"
                } - branches
                if missing_branches:
                    problems.append(
                        f"push branches missing {sorted(missing_branches)}"
                    )
        if "  workflow_dispatch:" not in source:
            problems.append("workflow_dispatch missing")
        if problems:
            self.fail(f"{_MARKER}: " + "; ".join(problems))

    def test_sr105_normal_list_tuple_and_frozen_diagnostics_unchanged(self):
        warning = ReadinessIssue(
            code="qualification.backend.pending",
            message="Physical CUDA evidence pending",
            source="E02",
        )
        mutable = [warning]
        for value in (mutable, tuple(mutable)):
            with self.subTest(type=type(value).__name__):
                report = evaluate_execution_readiness(
                    capability="implemented",
                    warnings=value,
                    evidence_tier="unverified",
                )
                self.assertEqual(report.decision, "warn")
                self.assertTrue(report.can_start)
                self.assertEqual(report.hard_errors, ())
                self.assertEqual(report.warnings, (warning,))
                self.assertEqual(report.to_diagnostics()["evidence_tier"], "unverified")
                report.to_diagnostics()["warnings"][0]["message"] = "tampered"
                self.assertEqual(
                    report.to_diagnostics()["warnings"][0]["message"],
                    "Physical CUDA evidence pending",
                )
        mutable.clear()
        direct = ExecutionReadiness(
            decision="allow",
            capability="implemented",
            hard_errors=[],
            warnings=(),
            evidence_tier="host_cpu_checked",
        )
        self.assertEqual(direct.hard_errors, ())
        self.assertTrue(direct.can_start)

    def test_sr106_memory_and_system_interrupts_are_not_masked(self):
        for exception_type in (MemoryError, KeyboardInterrupt):
            with self.subTest(error=exception_type.__name__):
                with self.assertRaises(exception_type):
                    evaluate_execution_readiness(
                        capability="implemented",
                        hard_errors=_ExplodingControlSequence(exception_type),
                    )


if __name__ == "__main__":
    unittest.main()
