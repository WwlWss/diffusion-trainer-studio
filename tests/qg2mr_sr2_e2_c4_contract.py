"""E2 permanent integrated qualification; no accepted RED execution path.

Exercise the actual embedded SR2 program, including negative unittest
outcomes. Keep all historical RED/stage sources unchanged for evidence.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import re
import sys
import textwrap
import unittest

from tests import qg2mr_sr2_e2_contract as frozen
from tests import qg2mr_sr2_e2_c1_contract as c1
from tests import qg2mr_sr2_e2_c2_contract as c2
from tests import qg2mr_sr2_e2_c3_contract as c3
from tests import test_qg2_mr_sr2_ci_dependencies_red as original

ROOT = frozen.ROOT
SR2_JOB = "qg1b-ci-dependency-red"
STRICT_STEP = "Exact six target and five baseline permanent strict GREEN"
INTEGRATED_COMMAND = "python -m tests.qg2mr_sr2_e2_c4_contract green"
SR2_CLASSES = (
    original.QG2MRFSRPR69SR2BTargetedRED,
    original.QG2MRFSRPR69SR2BBaselines,
)
TARGETS = (
    "test_srb101_original_evidence_coverage",
    "test_srb102_original_runtime_bools",
    "test_srb103_sr1_parameter_policy",
    "test_srb104_sr1_policy_compat",
    "test_srb105_sr1_policy_matrix",
    "test_srb106_sr1_runtime_bools",
)
BASELINES = (
    "test_srb901_actual_producer_direct_imports_are_present",
    "test_srb902_original_keeps_existing_event_matrix",
    "test_srb903_original_green_is_strict_and_unskipped",
    "test_srb904_sr1_keeps_existing_event_matrix",
    "test_srb905_sr1_green_preserves_prior_and_d1_counts",
)


def _enabled_job(text, job_name):
    job = frozen.job_text(text, job_name)
    if job is None or re.search(r"(?m)^    (if|continue-on-error):", job):
        raise AssertionError(f"Missing, conditional or tolerated job: {job_name}")
    if ("runs-on: $" + "{{ matrix.os }}" not in job
            or "os: [ubuntu-latest, windows-latest]" not in job
            or 'python-version: "3.11"' not in job):
        raise AssertionError(f"Missing dual-OS Python 3.11 job: {job_name}")
    return job


def require_sr2_events(text):
    events = frozen.events(text)
    if set(events) != {"push", "pull_request", "workflow_dispatch"}:
        raise AssertionError("SR2 event set changed")
    if (set(events["push"]) != {"branches", "paths"}
            or set(events["push"]["branches"]) != {"main", frozen.DEV, frozen.SR2_DEV}
            or set(events["pull_request"]) != {"paths"}
            or events["workflow_dispatch"] != {}):
        raise AssertionError("SR2 lifecycle is incomplete or narrowed")
    imported = original._producer_import_paths()
    if imported != original.DIRECT_MODULES:
        raise AssertionError("SR2 direct producer imports changed")
    required = frozen.SR2_INPUTS | imported
    for event in ("push", "pull_request"):
        if set(events[event]["paths"]) != required:
            raise AssertionError(f"SR2 exact input coverage changed: {event}")
    for path in required:
        if not (ROOT / path).is_file():
            raise AssertionError(f"SR2 path does not exist: {path}")


def sr2_program(text):
    """Extract the actual enabled single Python execution step, not comments."""
    require_sr2_events(text)
    job = _enabled_job(text, SR2_JOB)
    lines = job.splitlines()
    hits = [i for i, line in enumerate(lines) if line == "      - name: " + STRICT_STEP]
    if len(hits) != 1:
        raise AssertionError("SR2 requires a unique executable strict step")
    start = hits[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("      - ")), len(lines))
    step = lines[start:end]
    if (step[1:4] != ["        shell: bash", "        run: |", "          python - <<'PY'"]
            or step[-1] != "          PY"):
        raise AssertionError("SR2 strict command changed or is not unconditional")
    code = textwrap.dedent("\n".join(step[4:-1]))
    blocks = frozen._python_blocks(text)
    if len(blocks) != 1 or ast.dump(blocks[0]) != ast.dump(ast.parse(code)):
        raise AssertionError("SR2 strict step is not the unique actual Python run block")
    tree = ast.parse(code)
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and type(node.value) is str]
    if any("VALID TESTS-ONLY RED" in x or "GREEN CANDIDATE" in x for x in literals):
        raise AssertionError("SR2 historical acceptance branch restored")
    expected_tail = ast.parse('if __name__ == "__main__":\n    raise SystemExit(main())').body[0]
    if ast.dump(tree.body[-1]) != ast.dump(expected_tail):
        raise AssertionError("SR2 actual entry point not invoked")
    return code


def load_actual_program(text):
    # Code comes only from the reviewed repository workflow. Loading it as
    # a non-main module lets tests exercise the exact same callable with
    # real unittest fixtures without printing a false integrated success.
    scope = {"__name__": "_sr2_c4_behavior"}
    exec(compile(sr2_program(text), "SR2 actual embedded run", "exec"), scope)
    return scope


def require_integrated_workflow(text):
    ev = frozen.events(text)
    required = (frozen.R2_INPUTS | c2.EXTRA_INPUTS | c3.E2_INPUTS
                | {"tests/qg2mr_sr2_e2_c4_contract.py",
                   "docs/qualification-gate-qg2-mr-sr2-e2-c4-permanent-green.md"})
    if (set(ev["push"]["branches"]) != {"main", frozen.DEV, frozen.SR2_DEV}
            or set(ev["push"]["paths"]) != set(ev["pull_request"]["paths"])
            or set(ev["pull_request"]) != {"paths"}
            or ev.get("workflow_dispatch") != {}):
        raise AssertionError("Integrated lifecycle changed")
    for event in ("push", "pull_request"):
        if frozen.covers(ev[event], required):
            raise AssertionError(f"Integrated missing input: {event}")
    host = _enabled_job(text, "host-contract")
    if not frozen.actual_command(host, INTEGRATED_COMMAND):
        raise AssertionError("Integrated strict GREEN not actually wired")
    if not frozen.actual_command(_enabled_job(text, "host-partition"),
                                 "python -m tests.qg2mr_sr2_e2_c1_contract host"):
        raise AssertionError("Inherited Host execution lost")
    cpu = _enabled_job(text, "r2-cpu")
    for command in (
        frozen.CPU_COMMAND + "helper", frozen.CPU_COMMAND + "fsr",
        "python -m tests.qg2mr_sr2_e2_c2_contract partition-parity",
    ):
        if not frozen.actual_command(cpu, command):
            raise AssertionError(f"Inherited CPU execution lost: {command}")


class PermanentBehavior(unittest.TestCase):
    def test_c4101_exact_sr2_source_anchored_events_and_original_ids(self):
        scope = load_actual_program(frozen.source(frozen.SR2))
        self.assertEqual(scope["TARGETS"], TARGETS)
        self.assertEqual(scope["BASELINES"], BASELINES)
        for cls, names in zip(SR2_CLASSES, (TARGETS, BASELINES)):
            self.assertIs(scope[cls.__name__], cls)
            self.assertEqual(tuple(unittest.TestLoader().getTestCaseNames(cls)), names)

    def test_c4102_actual_program_green_and_exact_starts_successes(self):
        scope = load_actual_program(frozen.source(frozen.SR2))
        class Good(unittest.TestCase):
            def test_one(self):
                self.assertEqual(1 + 1, 2)
            def test_two(self):
                self.assertTrue(True)
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            self.assertEqual(scope["run_exact_suite"](Good, ("test_one", "test_two")), 2)
        import json
        records = [json.loads(l.split(" ", 1)[1]) for l in out.getvalue().splitlines()
                   if l.startswith("DTS_SR2_STRICT_RESULT_JSON ")]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["expected_ids"], record["started_ids"])
        self.assertEqual(record["expected_ids"], record["passed_ids"])
        self.assertEqual(record["tests_run"], 2)

    def test_c4103_actual_program_rejects_all_nonpass_unittest_outcomes(self):
        scope = load_actual_program(frozen.source(frozen.SR2))
        class Bad(unittest.TestCase):
            def test_error(self):
                raise RuntimeError("real error")
            def test_fail(self):
                self.fail("real failure")
            @unittest.skip("real skip")
            def test_skip(self):
                pass
            @unittest.expectedFailure
            def test_xfail(self):
                self.fail("real expected failure")
            @unittest.expectedFailure
            def test_xpass(self):
                pass
        import json
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "STRICT GREEN REGRESSION"):
                scope["run_exact_suite"](Bad, tuple(unittest.TestLoader().getTestCaseNames(Bad)))
        rows = [json.loads(l.split(" ", 1)[1]) for l in out.getvalue().splitlines()
                if l.startswith("DTS_SR2_STRICT_RESULT_JSON ")]
        self.assertEqual(len(rows), 1)
        for key in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes"):
            self.assertEqual(rows[0][key], 1, key)
        self.assertEqual(rows[0]["tests_run"], 5)
        self.assertEqual(rows[0]["passed_ids"], [])

    def test_c4104_actual_program_rejects_missing_duplicate_extra_and_wrong_ids(self):
        scope = load_actual_program(frozen.source(frozen.SR2))
        class Probe(unittest.TestCase):
            def test_a(self):
                pass
            def test_b(self):
                pass
        for names in (("test_a",), ("test_a", "test_a"),
                      ("test_a", "test_b", "test_c"), ("test_a", "test_c")):
            with self.subTest(names=names), self.assertRaisesRegex(SystemExit, "INVALID TEST IDENTITY"):
                scope["run_exact_suite"](Probe, names)

    def test_c4105_actual_sr2_main_never_accepts_an_incomplete_suite(self):
        from unittest import mock
        scope = load_actual_program(frozen.source(frozen.SR2))
        run = scope["run_exact_suite"]
        # Verify real main control flow; these are injected counter/failure
        # controls only. The dedicated SR2 job executes the real 6+5 suite.
        for side_effect in ([6, 4], [5, 5], [SystemExit("known RED")], [6, SystemExit("baseline failure")]):
            with self.subTest(side_effect=side_effect):
                with mock.patch.dict(scope, {"run_exact_suite": mock.Mock(side_effect=side_effect)}):
                    with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                        scope["main"]()
                    self.assertNotIn("PERMANENT STRICT GREEN: 11/11", out.getvalue())
        self.assertIs(scope["run_exact_suite"], run)

    def test_c4106_sr2_real_event_and_disabled_step_mutations_rejected(self):
        text = frozen.source(frozen.SR2)
        mutations = (
            text.replace("      - main\n", "", 1),
            text.replace('      - "mikazuki/parameter_policy_runtime_bools.py"\n', "", 1),
            text.replace("    runs-on:", "    if: false\n    runs-on:", 1),
            text.replace("      - name: " + STRICT_STEP + "\n",
                         "      - name: " + STRICT_STEP + "\n        continue-on-error: true\n", 1),
            text.replace("          python - <<'PY'\n", "          echo 11/11\n          python - <<'PY'\n", 1),
            text.replace("              raise SystemExit(main())", "              pass", 1),
        )
        for changed in mutations:
            with self.subTest(mutation=changed == text):
                self.assertNotEqual(text, changed)
                with self.assertRaises((AssertionError, ValueError)):
                    sr2_program(changed)

    def test_c4107_integrated_workflow_requires_all_three_enabled_jobs(self):
        text = frozen.source(frozen.E2)
        require_integrated_workflow(text)
        for old, new in (
            (INTEGRATED_COMMAND, "python -m tests.qg2mr_sr2_e2_c3_contract stage"),
            (frozen.CPU_COMMAND + "fsr", "echo 11/11"),
            ("      - name: Execute actual full Host suite and independently compare legacy outcomes\n",
             "      - name: Execute actual full Host suite and independently compare legacy outcomes\n        if: false\n"),
        ):
            changed = text.replace(old, new, 1)
            self.assertNotEqual(text, changed)
            with self.assertRaises(AssertionError):
                require_integrated_workflow(changed)


def run_behavior():
    names = unittest.defaultTestLoader.getTestCaseNames(PermanentBehavior)
    prefixes = [f"test_c4{i}_" for i in range(101, 108)]
    if len(names) != 7 or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
        raise RuntimeError(f"Wrong C4 behavior identities: {names}")
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(PermanentBehavior(n) for n in names))
    ok = c1.exact_result(result, 7) and not result.failures
    print(f"E2-C4 BEHAVIOR: {'7/7 PASS' if ok else 'HOLD'}", flush=True)
    return ok


def green():
    c1.require_no_torch()
    if not all(check() for check in (c1.run_behavior, c2.run_behavior,
                                     c3.run_behavior, run_behavior)):
        return 2
    result = frozen.run_exact("green")
    if result != 0:
        return result
    print("E2 INTEGRATED STRICT GREEN: 36/36 target + 8/8 baseline PASS; "
          "C1 10/10 + C2 6/6 + C3 7/7 + C4 7/7 behavior PASS; "
          "0 FAIL/ERROR/SKIP/xfail/xpass; independent review required", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["green"]:
        raise SystemExit("usage: python -m tests.qg2mr_sr2_e2_c4_contract green")
    raise SystemExit(green())
