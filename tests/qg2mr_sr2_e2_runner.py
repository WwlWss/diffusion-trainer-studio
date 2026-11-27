"""Parameter Policy test execution partition (QG-2 SR2 E2-C1).

Test infrastructure only. Keep the original unittest discovery module identity.
The two real Torch-only R2 modules are excluded before discovery/import.
E2-C2 owns their mandatory CPU execution; never use this as a runtime pass.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
PATTERN = "test_parameter_policy*.py"
R2_CPU_ONLY_FILES = frozenset({
    "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.py",
    "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.py",
})
DISCOVERY_MARKER = "DTS_E2_HOST_DISCOVERY_JSON "
RESULT_MARKER = "DTS_E2_HOST_RESULT_JSON "


class PartitionError(RuntimeError):
    """An unsupported or incomplete discovery is not a passing Host suite."""


def select_host_files(test_dir: Path) -> tuple[str, ...]:
    # The frozen tests directory is flat. unittest recurses into Python
    # packages even for an exact filename pattern. Reject a new package
    # instead of silently losing tests or discovering it repeatedly.
    if not test_dir.is_dir():
        raise PartitionError(f"Missing tests directory: {test_dir}")
    entries = tuple(test_dir.iterdir())
    packages = sorted(p.name for p in entries
                      if p.is_dir() and (p / "__init__.py").is_file())
    if packages:
        raise PartitionError(f"Discovery partition needs review for test packages: {packages}")
    files = {p.name for p in entries if p.is_file() and fnmatch.fnmatch(p.name, PATTERN)}
    missing = sorted(R2_CPU_ONLY_FILES - files)
    if missing:
        raise PartitionError(f"Frozen R2 CPU files missing: {missing}")
    invalid = sorted(name for name in files
                     if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", name))
    if invalid:
        raise PartitionError(f"Unsupported test module filename: {invalid}")
    selected = tuple(sorted(files - R2_CPU_ONLY_FILES))
    if not selected:
        raise PartitionError("Empty Host file selection")
    return selected


def loaded_cpu_modules() -> list[str]:
    names = {Path(name).stem for name in R2_CPU_ONLY_FILES}
    aliases = names | {"tests." + name for name in names}
    return sorted(name for name in aliases if name in sys.modules)


def require_no_cpu_modules() -> None:
    loaded = loaded_cpu_modules()
    if loaded:
        raise PartitionError(f"CPU-only modules loaded in Host partition: {loaded}")


def test_ids(suite: unittest.TestSuite) -> list[str]:
    def walk(item):
        if isinstance(item, unittest.TestCase):
            yield item.id()
        elif isinstance(item, unittest.BaseTestSuite):
            for child in item:
                yield from walk(child)
        else:
            raise PartitionError(f"Unexpected unittest discovery object: {type(item)!r}")

    ids = list(walk(suite))
    if not ids or any(type(i) is not str or not i for i in ids):
        raise PartitionError("Empty or invalid Host test identities")
    if any(i.startswith("unittest.loader._FailedTest.") for i in ids):
        raise PartitionError("Discovery contains _FailedTest")
    if len(ids) != len(set(ids)):
        raise PartitionError("Duplicate Host test identities")
    return ids


class _HostLoader(unittest.TestLoader):
    """Use exact discover patterns, but retain the original load_tests pattern."""

    def __init__(self, selected):
        super().__init__()
        self._selected = frozenset(selected)
        self._in_discover = False

    def discover(self, start_dir, pattern=PATTERN, top_level_dir=None):
        # A new load_tests hook must not recursively re-open the excluded
        # module set. Nested discovery needs an explicit partition review.
        if self._in_discover or pattern not in self._selected:
            raise PartitionError("Unreviewed recursive or broad Host discovery")
        self._in_discover = True
        try:
            return super().discover(start_dir, pattern=pattern,
                                    top_level_dir=top_level_dir)
        finally:
            self._in_discover = False

    def loadTestsFromModule(self, module, *, pattern=None):
        if self._in_discover and pattern in self._selected:
            pattern = PATTERN
        return super().loadTestsFromModule(module, pattern=pattern)


def discover_host(test_dir: Path = TESTS_DIR):
    """Partition BEFORE importing modules, via public unittest APIs."""
    require_no_cpu_modules()
    selected = select_host_files(test_dir)
    suite = unittest.TestSuite()
    loader = _HostLoader(selected)
    for filename in selected:
        suite.addTests(loader.discover(
            start_dir=str(test_dir),
            pattern=filename,
            top_level_dir=str(test_dir),
        ))
    if loader.errors:
        raise PartitionError("Host discovery errors:\n" + "\n".join(loader.errors))
    require_no_cpu_modules()
    ids = test_ids(suite)
    return suite, ids, selected


def result_record(result: unittest.TestResult, expected: int) -> dict:
    return {
        "discovered": expected,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": [[case.id(), reason] for case, reason in result.skipped],
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "loaded_cpu": loaded_cpu_modules(),
    }


def strict_host_success(record: dict) -> bool:
    # Preserve ordinary existing dependency-based unittest skips. Runtime
    # owners must run those tests independently; skip is not a PASS.
    return (
        record["tests_run"] == record["discovered"]
        and record["discovered"] > 0
        and not record["failures"]
        and not record["errors"]
        and not record["expected_failures"]
        and not record["unexpected_successes"]
        and not record["loaded_cpu"]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host",))
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise PartitionError("Run the test partition from the repository root")
    suite, ids, selected = discover_host()
    if args.discover_only:
        print(DISCOVERY_MARKER + json.dumps({
            "ids": ids,
            "errors": [],
            "loaded_cpu": loaded_cpu_modules(),
        }), flush=True)
        return 0
    print(f"E2 HOST PARTITION: {len(selected)} files; {len(ids)} test identities; "
          f"{len(R2_CPU_ONLY_FILES)} CPU-only files deferred", flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = result_record(result, len(ids))
    print(RESULT_MARKER + json.dumps(record), flush=True)
    return 0 if strict_host_success(record) else 1


if __name__ == "__main__":
    raise SystemExit(main())
