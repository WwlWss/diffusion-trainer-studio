"""QG-2-MR-FSR-PR69-SR2-B: exact source dependency CI trigger RED.

Six previously unobserved producer dependencies must trigger *both* the
push and pull_request forms of the two dedicated QG1B workflows. The test
reads the actual repository sources; it does not invoke training or CI.

This is intentionally RED at parent 806bd4dda0926677ad40df8413ede336f25249b3.
Only SR2-C may repair workflow paths after this RED evidence is accepted.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ORIGINAL = "qg1b-trusted-producer-host.yml"
SR1 = "qg1b-sr1-host.yml"
PRODUCER = ROOT / "mikazuki" / "parameter_policy_readiness_producer.py"
MARKER = "QG2-MR-SR2-B CI DEPENDENCY TRIGGER ABSENT"

# This is a source-anchored contract, not an unrestricted CI file glob.
DIRECT_MODULES = frozenset({
    "mikazuki/optimizer_profiles.py",
    "mikazuki/parameter_policy_evidence_coverage.py",
    "mikazuki/parameter_policy.py",
    "mikazuki/parameter_policy_compat.py",
    "mikazuki/parameter_policy_execution.py",
    "mikazuki/parameter_policy_matrix.py",
    "mikazuki/parameter_policy_readiness.py",
    "mikazuki/parameter_policy_runtime_bools.py",
})


def _producer_import_paths() -> set[str]:
    source = PRODUCER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PRODUCER))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module and node.module.startswith("mikazuki."):
                result.add(node.module.replace(".", "/") + ".py")
    return result


def _workflow_text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _event_lines(source: str, event: str) -> list[str]:
    """Parse the literal two-space YAML event subtree, without PyYAML.

    This deliberately refuses dynamic event aliases and requires the known
    simple GitHub Actions list shape. A malformed event cannot be GREEN.
    """
    lines = source.splitlines()
    if lines.count("on:") != 1:
        raise AssertionError("Expected one literal top-level on: block")
    start = lines.index("on:") + 1
    headers = [i for i in range(start, len(lines)) if lines[i] == f"  {event}:"]
    if len(headers) != 1:
        raise AssertionError(f"Expected exactly one event block: {event}")
    first = headers[0] + 1
    last = len(lines)
    for i in range(first, len(lines)):
        line = lines[i]
        if re.match(r"^  [a-z][a-z0-9_-]*:\s*(?:#.*)?$", line):
            last = i
            break
        if line and not line[0].isspace() and not line.startswith("#"):
            last = i
            break
    return lines[first:last]


def _event_list(source: str, event: str, field: str) -> tuple[str, ...]:
    lines = _event_lines(source, event)
    header = f"    {field}:"
    if lines.count(header) != 1:
        raise AssertionError(f"{event}.{field} must be an explicit list")
    at = lines.index(header)
    values: list[str] = []
    for line in lines[at + 1:]:
        if not line.startswith("      - "):
            break
        literal = line[len("      - "):].strip()
        try:
            parsed = ast.literal_eval(literal) if literal.startswith(('"', "'")) else literal
        except (SyntaxError, ValueError) as exc:
            raise AssertionError(f"Invalid YAML scalar in {event}.{field}") from exc
        if type(parsed) is not str or not parsed:
            raise AssertionError(f"Invalid nonstring path in {event}.{field}")
        values.append(parsed)
    if not values or len(set(values)) != len(values):
        raise AssertionError(f"{event}.{field} empty or contains duplicates")
    return tuple(values)


def _assert_dependency_covered(case: unittest.TestCase, workflow: str, path: str) -> None:
    imported = _producer_import_paths()
    case.assertIn(path, DIRECT_MODULES, f"not a frozen direct-import dependency: {path}")
    case.assertIn(path, imported, f"producer source no longer imports: {path}")
    source = _workflow_text(workflow)
    missing = [
        event for event in ("push", "pull_request")
        if path not in _event_list(source, event, "paths")
    ]
    if missing:
        case.fail(f"{MARKER}: {workflow} does not watch {path} for {','.join(missing)}")


class QG2MRFSRPR69SR2BTargetedRED(unittest.TestCase):
    def test_srb101_original_evidence_coverage(self):
        _assert_dependency_covered(self, ORIGINAL, "mikazuki/parameter_policy_evidence_coverage.py")

    def test_srb102_original_runtime_bools(self):
        _assert_dependency_covered(self, ORIGINAL, "mikazuki/parameter_policy_runtime_bools.py")

    def test_srb103_sr1_parameter_policy(self):
        _assert_dependency_covered(self, SR1, "mikazuki/parameter_policy.py")

    def test_srb104_sr1_policy_compat(self):
        _assert_dependency_covered(self, SR1, "mikazuki/parameter_policy_compat.py")

    def test_srb105_sr1_policy_matrix(self):
        _assert_dependency_covered(self, SR1, "mikazuki/parameter_policy_matrix.py")

    def test_srb106_sr1_runtime_bools(self):
        _assert_dependency_covered(self, SR1, "mikazuki/parameter_policy_runtime_bools.py")


class QG2MRFSRPR69SR2BBaselines(unittest.TestCase):
    def test_srb901_actual_producer_direct_imports_are_present(self):
        imported = _producer_import_paths()
        self.assertEqual(imported, set(DIRECT_MODULES))

    def test_srb902_original_keeps_existing_event_matrix(self):
        source = _workflow_text(ORIGINAL)
        self.assertIn("main", _event_list(source, "push", "branches"))
        self.assertIn("mikazuki/parameter_policy_execution.py", _event_list(source, "push", "paths"))
        self.assertIn("mikazuki/parameter_policy_execution.py", _event_list(source, "pull_request", "paths"))
        self.assertIn("os: [ubuntu-latest, windows-latest]", source)

    def test_srb903_original_green_is_strict_and_unskipped(self):
        source = _workflow_text(ORIGINAL)
        for literal in (
            '"tests.test_parameter_policy_qg1b_trusted_producer"',
            "found != 16", "result.errors", "result.skipped",
            "result.expectedFailures", "result.unexpectedSuccesses",
            "raise SystemExit(1)",
        ):
            self.assertIn(literal, source)

    def test_srb904_sr1_keeps_existing_event_matrix(self):
        source = _workflow_text(SR1)
        self.assertIn("main", _event_list(source, "push", "branches"))
        self.assertIn("mikazuki/parameter_policy_evidence_coverage.py", _event_list(source, "push", "paths"))
        self.assertIn("mikazuki/parameter_policy_evidence_coverage.py", _event_list(source, "pull_request", "paths"))
        self.assertIn("os: [ubuntu-latest, windows-latest]", source)

    def test_srb905_sr1_green_preserves_prior_and_d1_counts(self):
        source = _workflow_text(SR1)
        for literal in (
            '"tests.test_parameter_policy_qg1b_trusted_producer"',
            '"tests.test_parameter_policy_qg1b_sr1"',
            '"tests.test_parameter_policy_qg1b_evidence_coverage"',
            "result.testsRun != 16", "result.testsRun != 10",
            "result.skipped", "result.expectedFailures",
            "raise SystemExit(1)",
        ):
            self.assertIn(literal, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
