"""QG-1B3-B FSR: find runner-label escaping before any staged training tests."""
from __future__ import annotations
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".github/workflows/qg1b3b-typed-migration-host.yml"
MARKER = "QG-1B3-B RUNNER MATRIX CONTRACT ABSENT"
EXPRESSION = "$" + "{{ matrix.os }}"


class QG1B3BRunnerLabelTarget(unittest.TestCase):
    def test_r101_runs_on_is_actual_github_expression_not_literal_unicode(self):
        text = SOURCE.read_text(encoding="utf-8")
        rows = [line.strip() for line in text.splitlines()
                if line.strip().startswith("runs-on:")]
        if rows != ["runs-on: " + EXPRESSION]:
            self.fail(
                f"{MARKER}: runs-on must evaluate a valid OS matrix, "
                f"not a literal escaped runner label: {rows!r}"
            )


class QG1B3BRunnerLabelBaseline(unittest.TestCase):
    def test_r901_os_matrix_and_correct_job_name_are_still_both_present(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("os: [ubuntu-latest, windows-latest]", text)
        self.assertIn("    name: QG1B3-B typed certificate migration (" + EXPRESSION + ")", text)
        self.assertIn("      fail-fast: false", text)
