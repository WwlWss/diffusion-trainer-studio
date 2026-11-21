"""QG-2-B4-FSR-SR1: genuine source-coverage/CI-step RED and mutation baselines."""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

from tests.test_parameter_policy_qg2b4i_ci_migration import WORKFLOWS

ROOT = Path(__file__).resolve().parents[1]
WF_DIR = ROOT / ".github" / "workflows"
INTEGRATION = "qg2b4i-integrated-green.yml"
STEP_TITLE = "B312 Integrated Admission 16/16 GREEN"
MARKER = "B4-FSR-SR1 SOURCE CONTRACT ABSENT"
SHARED_IMPORT = "from tests.qg2b4i_ci_contract import assert_cumulative_admission_green"
AUDITED_B312_INPUTS = (
    "mikazuki/parameter_policy.py",
    "mikazuki/parameter_policy_admission.py",
    "mikazuki/parameter_policy_trainer.py",
    "mikazuki/parameter_policy_readiness.py",
    "mikazuki/parameter_policy_readiness_producer.py",
    "mikazuki/parameter_policy_execution.py",
    "mikazuki/parameter_policy_matrix.py",
    "mikazuki/parameter_policy_runtime_bools.py",
    "mikazuki/parameter_policy_compat.py",
    "mikazuki/optimizer_profiles.py",
    "mikazuki/model_component_profiles.py",
    "mikazuki/direct_trainer_admission_intent.py",
    "mikazuki/training_request.py",
    "mikazuki/training_config.py",
    "mikazuki/training_validation.py",
    "mikazuki/app/training_api.py",
    "mikazuki/training_launcher.py",
    "scripts/stable/train_network.py",
    "scripts/dev/train_network.py",
    "scripts/stable/sdxl_train_network.py",
    "scripts/dev/sdxl_train_network.py",
    "scripts/dev/sd3_train_network.py",
    "scripts/dev/flux_train_network.py",
    "scripts/dev/lumina_train_network.py",
    "scripts/stable/library/dts_parameter_policy_bridge.py",
    "scripts/dev/library/dts_parameter_policy_bridge.py",
)
_BOUNDS = ("  push:", "  pull_request:", "  workflow_dispatch:", "permissions:", "jobs:")


def _event_section(source: str, event: str) -> list[str]:
    lines = source.splitlines()
    first = lines.index("  " + event + ":")
    stop = next((i for i in range(first + 1, len(lines))
                 if lines[i] in _BOUNDS), len(lines))
    return lines[first + 1:stop]


def _event_paths(source: str, event: str) -> set[str] | None:
    lines = _event_section(source, event)
    if "    paths:" not in lines:
        return None
    start = lines.index("    paths:")
    result: set[str] = set()
    for line in lines[start + 1:]:
        if not line.startswith("      - "):
            break
        result.add(line.strip()[2:].strip().strip('"').strip("'"))
    return result


def _covered(path: str, patterns: set[str]) -> bool:
    # Only exact paths and recursive path/** are eligible for this contract.
    # No ambiguous fnmatch('*') treatment of GitHub directory separators.
    return any(path == pat or (
        pat.endswith("/**") and path.startswith(pat[:-2])
    ) for pat in patterns)


def canonical_step(stage: str) -> str:
    return (
        "      - name: " + STEP_TITLE + "\n"
        "        shell: bash\n"
        "        run: |\n"
        "          python - <<'PY'\n"
        "          " + SHARED_IMPORT + "\n"
        '          assert_cumulative_admission_green("QG-2-B4-I:' + stage + '")\n'
        "          PY"
    )


def require_executable_step(source: str, filename: str) -> None:
    if type(filename) is not str or not filename.endswith(".yml"):
        raise AssertionError("invalid workflow name")
    stage = filename[:-4]
    fragments = source.split("      - name: " + STEP_TITLE + "\n")
    if len(fragments) != 2:
        raise AssertionError(f"{MARKER}: {filename} lacks unique dedicated B312 step")
    tail = fragments[1]
    next_step = tail.find("\n      - ")
    step = ("      - name: " + STEP_TITLE + "\n" +
            (tail if next_step < 0 else tail[:next_step])).strip("\n")
    if step != canonical_step(stage):
        raise AssertionError(f"{MARKER}: {filename} B312 block is disabled or not canonical")
    if source.count(SHARED_IMPORT) != 1 or source.count(
        "assert_cumulative_admission_green("
    ) != 1:
        raise AssertionError(f"{MARKER}: {filename} duplicate/indirect B312 invocation")
    if "continue-on-error:" in source or "    if: false" in source.lower():
        raise AssertionError(f"{MARKER}: {filename} masks a CI failure")
    parsed = ast.parse(
        SHARED_IMPORT + "\n" +
        'assert_cumulative_admission_green("QG-2-B4-I:' + stage + '")'
    )
    if (len(parsed.body) != 2 or
        not isinstance(parsed.body[0], ast.ImportFrom) or
        not isinstance(parsed.body[1], ast.Expr) or
        not isinstance(parsed.body[1].value, ast.Call)):
        raise AssertionError(f"{MARKER}: {filename} no top-level assertion")


class QG2B4FSRSR1Targeted(unittest.TestCase):
    def test_sr101_integration_push_covers_audited_sources_and_main(self):
        source = (WF_DIR / INTEGRATION).read_text(encoding="utf-8")
        push = _event_section(source, "push")
        paths = _event_paths(source, "push")
        if ("      - main" not in push and '      - "main"' not in push):
            self.fail(f"{MARKER}: main branch is not monitored by B4-I")
        if paths is None:
            self.fail(f"{MARKER}: B4-I push path contract is absent")
        required = AUDITED_B312_INPUTS + (
            "tests/qg2b4i_ci_contract.py",
            "tests/test_parameter_policy_admission_red.py",
            "tests/test_parameter_policy_qg2b4i_ci_migration.py",
            "tests/test_parameter_policy_qg2b4fsr_sr1_ci.py",
            ".github/workflows/" + INTEGRATION,
        )
        missed = [name for name in required if not _covered(name, paths)]
        if missed:
            self.fail(f"{MARKER}: B4-I omits source dependency paths: {missed}")

    def test_sr102_integration_pr_must_be_unfiltered_stable_required_check(self):
        source = (WF_DIR / INTEGRATION).read_text(encoding="utf-8")
        if _event_paths(source, "pull_request") is not None:
            self.fail(f"{MARKER}: B4-I PR event still has paths filter")
        self.assertIn("  all-13-source-contracts:\n", source)
        self.assertIn("os: [ubuntu-latest, windows-latest]", source)
        self.assertNotIn("    continue-on-error:", source)
        self.assertNotIn("    if: false", source.lower())

    def test_sr201_consumer_qg1b3_qg2_admission_host_yml(self):
        require_executable_step((WF_DIR / "qg1b3-qg2-admission-host.yml").read_text(encoding="utf-8"), "qg1b3-qg2-admission-host.yml")

    def test_sr202_consumer_qg1b3a_admission_adapter_host_yml(self):
        require_executable_step((WF_DIR / "qg1b3a-admission-adapter-host.yml").read_text(encoding="utf-8"), "qg1b3a-admission-adapter-host.yml")

    def test_sr203_consumer_qg1b3b_typed_migration_host_yml(self):
        require_executable_step((WF_DIR / "qg1b3b-typed-migration-host.yml").read_text(encoding="utf-8"), "qg1b3b-typed-migration-host.yml")

    def test_sr204_consumer_qg2a1_request_intent_host_yml(self):
        require_executable_step((WF_DIR / "qg2a1-request-intent-host.yml").read_text(encoding="utf-8"), "qg2a1-request-intent-host.yml")

    def test_sr205_consumer_qg2a1_sr1_host_yml(self):
        require_executable_step((WF_DIR / "qg2a1-sr1-host.yml").read_text(encoding="utf-8"), "qg2a1-sr1-host.yml")

    def test_sr206_consumer_qg2a2_api_host_yml(self):
        require_executable_step((WF_DIR / "qg2a2-api-host.yml").read_text(encoding="utf-8"), "qg2a2-api-host.yml")

    def test_sr207_consumer_qg2b2_cli_intent_host_yml(self):
        require_executable_step((WF_DIR / "qg2b2-cli-intent-host.yml").read_text(encoding="utf-8"), "qg2b2-cli-intent-host.yml")

    def test_sr208_consumer_qg2b3a_policy_snapshot_host_yml(self):
        require_executable_step((WF_DIR / "qg2b3a-policy-snapshot-host.yml").read_text(encoding="utf-8"), "qg2b3a-policy-snapshot-host.yml")

    def test_sr209_consumer_qg2b3b_early_bridge_host_yml(self):
        require_executable_step((WF_DIR / "qg2b3b-early-bridge-host.yml").read_text(encoding="utf-8"), "qg2b3b-early-bridge-host.yml")

    def test_sr210_consumer_qg2b3c_late_session_host_yml(self):
        require_executable_step((WF_DIR / "qg2b3c-late-session-host.yml").read_text(encoding="utf-8"), "qg2b3c-late-session-host.yml")

    def test_sr211_consumer_qg2b3c_sr1_cpu_runtime_yml(self):
        require_executable_step((WF_DIR / "qg2b3c-sr1-cpu-runtime.yml").read_text(encoding="utf-8"), "qg2b3c-sr1-cpu-runtime.yml")

    def test_sr212_consumer_qg2b4_0_b312_contract_red_yml(self):
        require_executable_step((WF_DIR / "qg2b4-0-b312-contract-red.yml").read_text(encoding="utf-8"), "qg2b4-0-b312-contract-red.yml")

    def test_sr213_consumer_qg2b4l1_launcher_transport_red_yml(self):
        require_executable_step((WF_DIR / "qg2b4l1-launcher-transport-red.yml").read_text(encoding="utf-8"), "qg2b4l1-launcher-transport-red.yml")


class QG2B4FSRSR1Baselines(unittest.TestCase):
    def test_sr901_canonical_step_is_valid_for_all_thirteen_consumers(self):
        self.assertEqual(len(WORKFLOWS), 13)
        self.assertEqual(len(set(WORKFLOWS)), 13)
        for filename in WORKFLOWS:
            with self.subTest(filename=filename):
                require_executable_step(canonical_step(filename[:-4]), filename)

    def test_sr902_negative_step_mutations_fail_closed(self):
        filename = WORKFLOWS[0]
        valid = canonical_step(filename[:-4])
        mutations = (
            valid.replace("          assert_cumulative", "          if False:\n            assert_cumulative"),
            valid.replace("          assert_cumulative", "          def dormant():\n            assert_cumulative"),
            valid.replace("          assert_cumulative", "          try:\n            assert_cumulative"),
            valid.replace("          assert_cumulative", "          # assert_cumulative"),
            valid.replace("          PY", "          PY || true"),
            valid.replace("        shell: bash", "        continue-on-error: true\n        shell: bash"),
            valid.replace("        shell: bash", "        if: false\n        shell: bash"),
            valid.replace("        shell: bash", "        shell: python"),
            valid.replace("          PY", "          PY\n          echo no-op"),
            valid.replace("QG-2-B4-I:", "QG-2-B4-OTHER:"),
            valid.replace("      - name: " + STEP_TITLE, "      - name: WRONG"),
        )
        for idx, bad in enumerate(mutations):
            with self.subTest(mutation=idx), self.assertRaises(AssertionError):
                require_executable_step(bad, filename)
