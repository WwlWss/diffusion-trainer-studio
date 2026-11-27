"""Parameter Policy test execution partition (QG-2 SR2 E2-C1).

Test infrastructure only. Keep the original unittest discovery module identity.
The two real Torch-only R2 modules are excluded before discovery/import.
R2 CPU execution is explicit and mandatory in its own pinned CPU job.
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


# Freeze original test identities, not merely suite lengths. The top-level
# module names intentionally match "unittest discover -s tests".
CPU_SUITE_FILES = {
    "helper": "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.py",
    "fsr": "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.py",
}
CPU_SUITE_IDS = {
    "helper": (
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G01_tiny_adamw_low_level_session",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G02_tiny_muon_low_level_session",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G03_tiny_muon_adamw_fallback_low_level_session",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G04_wrong_model_class_is_rejected_before_root_scan",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G05_inactive_precision_is_rejected_before_root_scan",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G06_production_full_bf16_flux_remains_blocked_pre_scan",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.IsolatedGpuEvidenceHelperCpuTests.test_R2G07_inactive_real_admission_still_constructs_legacy_session",
    ),
    "fsr": (
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR101FileOwnershipRed.test_SR101_B1_existing_git_guard_and_timeout_are_retained",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR101FileOwnershipRed.test_SR101_T1_probe_created_exclusively",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR101FileOwnershipRed.test_SR101_T2_runner_outputs_in_private_external_directory",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR101FileOwnershipRed.test_SR101_T3_never_unlinks_fixed_unowned_outputs",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR101FileOwnershipRed.test_SR101_T4_existing_external_files_survive_harness_rejection",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR102SyntheticScopeRed.test_SR102_T1_adamw_extra_argument",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR102SyntheticScopeRed.test_SR102_T2_muon_momentum_changed",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR102SyntheticScopeRed.test_SR102_T3_muon_ns_steps_changed",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR102SyntheticScopeRed.test_SR102_T4_muon_field_removed",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR102SyntheticScopeRed.test_SR102_T5_fallback_adamw_extra_argument",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.SR102SyntheticScopeRed.test_SR102_T6_same_name_and_shape_but_external_source",
    ),
}
CPU_PINS = {
    "accelerate": "1.6.0",
    "numpy": "1.26.4",
    "schedulefree": "1.4",
    "pytorch-optimizer": "3.10.0",
    "safetensors": "0.4.5",
}
CPU_DISCOVERY_MARKER = "DTS_E2_CPU_DISCOVERY_JSON "
CPU_RESULT_MARKER = "DTS_E2_CPU_RESULT_JSON "
CPU_ENV_MARKER = "DTS_E2_CPU_ENV_JSON "


def expected_cpu_ids(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return CPU_SUITE_IDS["helper"] + CPU_SUITE_IDS["fsr"]
    if selection not in CPU_SUITE_IDS:
        raise PartitionError(f"Unknown CPU suite: {selection}")
    return CPU_SUITE_IDS[selection]


def validate_cpu_ids(actual, expected) -> None:
    if (len(actual) != len(expected) or len(actual) != len(set(actual))
            or set(actual) != set(expected)):
        raise PartitionError(
            f"CPU identity mismatch: expected={len(expected)}, actual={len(actual)}, "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )


def verify_pinned_cpu() -> dict:
    # Import Torch only in the CPU execution/diagnostic entry, never Host.
    import importlib.metadata as metadata
    import os

    if sys.version_info[:2] != (3, 11):
        raise PartitionError("R2 CPU qualification requires Python 3.11")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PartitionError("R2 CPU qualification requires CUDA_VISIBLE_DEVICES to be empty")
    try:
        versions = {name: metadata.version(name) for name in ("torch", *CPU_PINS)}
    except metadata.PackageNotFoundError as exc:
        raise PartitionError(f"Missing pinned CPU dependency: {exc}") from exc
    if (versions["torch"] not in {"2.7.0", "2.7.0+cpu"}
            or any(versions[name] != pin for name, pin in CPU_PINS.items())):
        raise PartitionError(f"Unexpected pinned CPU dependencies: {versions}")
    import torch
    if (str(torch.__version__) not in {"2.7.0", "2.7.0+cpu"}
            or torch.version.cuda is not None
            or getattr(torch.version, "hip", None) is not None
            or torch.cuda.is_available()):
        raise PartitionError("R2 CPU qualification requires actual CPU-only Torch 2.7.0")
    if torch.zeros(1).device.type != "cpu":
        raise PartitionError("Default Torch tensor device is not CPU")
    record = {
        "python": sys.version.split()[0],
        "versions": versions,
        "torch_module_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "hip_build": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "tensor_device": "cpu",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    print(CPU_ENV_MARKER + json.dumps(record), flush=True)
    return record


def verify_cpu_sources(selection: str) -> None:
    # FSR imports the original helper by its top-level unittest identity.
    # Require exact checked-in module origins and reject duplicate aliases.
    expected = {"helper"} if selection == "helper" else {"helper", "fsr"}
    names = {Path(CPU_SUITE_FILES[k]).stem for k in expected}
    if set(loaded_cpu_modules()) != names:
        raise PartitionError(f"Unexpected CPU module aliases: {loaded_cpu_modules()}")
    for key in expected:
        filename = CPU_SUITE_FILES[key]
        module = sys.modules.get(Path(filename).stem)
        if module is None or Path(module.__file__).resolve() != (TESTS_DIR / filename).resolve():
            raise PartitionError(f"Unexpected CPU module source: {filename}")


def discover_cpu(selection: str):
    require_no_cpu_modules()
    select_host_files(TESTS_DIR)
    if frozenset(CPU_SUITE_FILES.values()) != R2_CPU_ONLY_FILES:
        raise PartitionError("CPU ownership does not match exact Host exclusions")
    expected = expected_cpu_ids(selection)
    keys = ("helper", "fsr") if selection == "all" else (selection,)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for key in keys:
        suite.addTests(loader.discover(
            start_dir=str(TESTS_DIR),
            pattern=CPU_SUITE_FILES[key],
            top_level_dir=str(TESTS_DIR),
        ))
    if loader.errors:
        raise PartitionError("CPU discovery errors:\n" + "\n".join(loader.errors))
    ids = test_ids(suite)
    validate_cpu_ids(ids, expected)
    verify_cpu_sources(selection)
    return suite, ids


class CpuIdentityResult(unittest.TextTestResult):
    """Record actual test starts; a matching count alone is insufficient."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_ids = []

    def startTest(self, test):
        self.executed_ids.append(test.id())
        super().startTest(test)


def cpu_result_record(result: CpuIdentityResult, ids: list[str]) -> dict:
    record = result_record(result, len(ids))
    record["ids"] = list(ids)
    record["executed_ids"] = list(result.executed_ids)
    return record


def strict_cpu_success(record: dict, expected) -> bool:
    try:
        validate_cpu_ids(record["ids"], expected)
        validate_cpu_ids(record["executed_ids"], expected)
    except PartitionError:
        return False
    return (
        record["discovered"] == record["tests_run"] == len(expected)
        and record["ids"] == record["executed_ids"]
        and not record["failures"]
        and not record["errors"]
        and not record["skipped"]
        and not record["expected_failures"]
        and not record["unexpected_successes"]
    )


def run_cpu(selection: str, *, discover_only: bool) -> int:
    verify_pinned_cpu()
    suite, ids = discover_cpu(selection)
    if discover_only:
        print(CPU_DISCOVERY_MARKER + json.dumps({
            "suite": selection, "ids": ids, "errors": [],
            "loaded_cpu": loaded_cpu_modules(),
        }), flush=True)
        return 0
    result = unittest.TextTestRunner(
        verbosity=2, resultclass=CpuIdentityResult).run(suite)
    verify_cpu_sources(selection)
    record = cpu_result_record(result, ids)
    record["suite"] = selection
    print(CPU_RESULT_MARKER + json.dumps(record), flush=True)
    if not strict_cpu_success(record, expected_cpu_ids(selection)):
        return 1
    print(f"E2-C2 R2 CPU {selection}: {len(ids)}/{len(ids)} actual PASS; "
          "0 FAIL/ERROR/SKIP/xfail/xpass", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host", "r2-cpu"))
    parser.add_argument("--suite", choices=("helper", "fsr", "all"))
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise PartitionError("Run the test partition from the repository root")
    if args.mode == "r2-cpu":
        return run_cpu(args.suite or "all", discover_only=args.discover_only)
    if args.suite is not None:
        parser.error("--suite is only valid with r2-cpu")
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
