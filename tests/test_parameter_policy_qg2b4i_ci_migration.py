"""QG-2-B4-I: all 13 historical workflows must migrate B312 together.

These are real source-contract tests of the checked-in workflows.
Target RED identifies exactly the still-unmigrated CI consumers; after
migration the same tests require executable shared 16/16 GREEN assertions.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    "qg1b3-qg2-admission-host.yml",
    "qg1b3a-admission-adapter-host.yml",
    "qg1b3b-typed-migration-host.yml",
    "qg2a1-request-intent-host.yml",
    "qg2a1-sr1-host.yml",
    "qg2a2-api-host.yml",
    "qg2b2-cli-intent-host.yml",
    "qg2b3a-policy-snapshot-host.yml",
    "qg2b3b-early-bridge-host.yml",
    "qg2b3c-late-session-host.yml",
    "qg2b3c-sr1-cpu-runtime.yml",
    "qg2b4-0-b312-contract-red.yml",
    "qg2b4l1-launcher-transport-red.yml",
)
MARKER = "QG-2-B4-I INHERITED CI CONTRACT ABSENT"
HELPER_IMPORT = "from tests.qg2b4i_ci_contract import assert_cumulative_admission_green"
SHARED_FILE = "tests/qg2b4i_ci_contract.py"
BOUNDARIES = ("  push:", "  pull_request:", "  workflow_dispatch:", "permissions:", "jobs:")


def _paths_for_event(source: str, event: str) -> set[str]:
    lines = source.splitlines()
    heading = f"  {event}:"
    if heading not in lines:
        return set()
    first = lines.index(heading)
    last = next(
        (i for i in range(first + 1, len(lines))
         if lines[i] in BOUNDARIES),
        len(lines),
    )
    in_paths = False
    paths = set()
    for line in lines[first + 1:last]:
        if line.startswith("    paths:"):
            in_paths = True
            continue
        if in_paths and line.startswith("    ") and not line.startswith("      "):
            in_paths = False
        if in_paths and line.startswith("      - "):
            paths.add(line.strip()[2:].strip().strip('"').strip("'"))
    return paths


def _python_blocks(source: str):
    return [
        textwrap.dedent(m.group(1))
        for m in re.finditer(
            r"^\s*python - <<'PY'\s*\n(.*?)^\s*PY\s*$",
            source, re.M | re.S,
        )
    ]


class QG2B4IInheritedWorkflowMigration(unittest.TestCase):
    def _verify(self, filename: str):
        self.assertIn(filename, WORKFLOWS)
        source = (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
        if HELPER_IMPORT not in source or (
            not re.search(
                r"(?m)^\s*assert_cumulative_admission_green\([\s\S]*?\)\s*$",
                source,
            )
        ):
            self.fail(f"{MARKER}: {filename} does not execute shared B312 16/16 assertion")
        self.assertEqual(
            source.count(HELPER_IMPORT), 1,
            f"{filename}: shared B312 runner imported exactly once",
        )
        # The inherited B312 assertion is authoritative only if both event
        # triggers re-run it when producer code or the helper changes.
        for event in ("push", "pull_request"):
            paths = _paths_for_event(source, event)
            for needed in (
                f".github/workflows/{filename}",
                SHARED_FILE,
                "tests/test_parameter_policy_admission_red.py",
                "mikazuki/app/training_api.py",
                "mikazuki/training_launcher.py",
            ):
                self.assertIn(
                    needed, paths,
                    f"{filename}: {event} paths omit dependency {needed}",
                )
        self.assertIn("ubuntu-latest", source)
        self.assertIn("windows-latest", source)
        # The historical RED suites of B3-A/B3-B/B3-C remain independently
        # present; only stale B312-specific failure requirements are forbidden.
        self.assertNotIn("QG-1B3-QG2 ADMISSION CONTRACT ABSENT", source)
        self.assertNotIn("API must pass PreparedTrainingConfig intent to Launcher", source)
        self.assertNotIn("B312 EXACT ONE", source)
        self.assertNotIn("B312 API-only HOLD", source)
        blocks = _python_blocks(source)
        self.assertTrue(blocks, f"{filename}: missing executable Host suites")
        for block in blocks:
            ast.parse(block, filename=f"{filename}:embedded-python")

        # SR1: the actual cumulative check must be in one dedicated, enabled,
        # fail-closed execution step; source-string membership is insufficient.
        from tests.test_parameter_policy_qg2b4fsr_sr1_ci import require_executable_step
        require_executable_step(source, filename)

    def test_i101_qg1b3_qg2_admission_host(self):
        self._verify("qg1b3-qg2-admission-host.yml")

    def test_i102_qg1b3a_admission_adapter_host(self):
        self._verify("qg1b3a-admission-adapter-host.yml")

    def test_i103_qg1b3b_typed_migration_host(self):
        self._verify("qg1b3b-typed-migration-host.yml")

    def test_i104_qg2a1_request_intent_host(self):
        self._verify("qg2a1-request-intent-host.yml")

    def test_i105_qg2a1_sr1_host(self):
        self._verify("qg2a1-sr1-host.yml")

    def test_i106_qg2a2_api_host(self):
        self._verify("qg2a2-api-host.yml")

    def test_i107_qg2b2_cli_intent_host(self):
        self._verify("qg2b2-cli-intent-host.yml")

    def test_i108_qg2b3a_policy_snapshot_host(self):
        self._verify("qg2b3a-policy-snapshot-host.yml")

    def test_i109_qg2b3b_early_bridge_host(self):
        self._verify("qg2b3b-early-bridge-host.yml")

    def test_i110_qg2b3c_late_session_host(self):
        self._verify("qg2b3c-late-session-host.yml")

    def test_i111_qg2b3c_sr1_cpu_runtime(self):
        self._verify("qg2b3c-sr1-cpu-runtime.yml")

    def test_i112_qg2b4_0_b312_contract_red(self):
        self._verify("qg2b4-0-b312-contract-red.yml")

    def test_i113_qg2b4l1_launcher_transport_red(self):
        self._verify("qg2b4l1-launcher-transport-red.yml")


if __name__ == "__main__":
    unittest.main()
