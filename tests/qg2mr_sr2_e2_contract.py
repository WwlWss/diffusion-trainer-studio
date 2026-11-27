"""QG-2-MR-FSR-PR69-SR2-E2-B: tests-only CI integration contract.

RED parent: 141a0d187fc32085dccf7b20147258c84ffe8fea.
This module deliberately is not discovered by test_parameter_policy*.py.
No Torch, CUDA, production training or workflow mutation in this module.
Run the exact RED contract with python -m tests.qg2mr_sr2_e2_contract red.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import unittest

from tests import qg2mr_sr2_e2_test_results as results
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
WF = ROOT / ".github/workflows"
MARKER = "QG2-MR-SR2-E2 CI INTEGRATION CONTRACT ABSENT"
DEV = "feature/qualification-gate-qg1b3-qg2-admission-red"
SR2_DEV = "feature/qg2-mr-fsr-pr69-sr2b-ci-deps-red"
ANIMA = "anima-qwen3-review.yml"
E2 = "qg2-mr-sr2-e2-ci-integration.yml"
SR2 = "qg2-mr-fsr-pr69-sr2b-ci-deps-red.yml"
RUNNER = "tests/qg2mr_sr2_e2_runner.py"
CONTRACT = "tests/qg2mr_sr2_e2_contract.py"
DOC = "docs/qualification-gate-qg2-mr-sr2-e2-ci-integration-contract.md"
R2_PREFIX = "test_parameter_policy_qg2_mr_r2_2_sr1"
HELPER = R2_PREFIX + "_gpu_helper.py"
FSR = R2_PREFIX + "_fsr_sr1_red.py"
R2_SOURCE = R2_PREFIX + "_red.py"
HARNESS = "test_parameter_policy_execution_gpu_harness.py"
CPU_FILES = frozenset((HELPER, FSR))
LEGACY_COMMAND = "python -m unittest discover -s tests -p 'test_parameter_policy*.py' -v"
HOST_COMMAND = "python -m tests.qg2mr_sr2_e2_runner host"
CPU_COMMAND = "python -m tests.qg2mr_sr2_e2_runner r2-cpu --suite "
DISCOVERY_MARKER = "DTS_E2_HOST_DISCOVERY_JSON "
REFERENCE_MARKER = "DTS_E2_REFERENCE_DISCOVERY_JSON "
FROZEN_FILES = frozenset(
    (
        "test_parameter_policy_adapter_metadata.py",
        "test_parameter_policy_admission_adapter_host.py",
        "test_parameter_policy_admission_red.py",
        "test_parameter_policy_anima_metadata.py",
        "test_parameter_policy_anima_runtime_smoke.py",
        "test_parameter_policy_anima_staging.py",
        "test_parameter_policy_backend_feature_gpu.py",
        "test_parameter_policy_bootstrap.py",
        "test_parameter_policy_compat.py",
        "test_parameter_policy_config.py",
        "test_parameter_policy_editor.py",
        "test_parameter_policy_editor_rehydrate.py",
        "test_parameter_policy_execution.py",
        "test_parameter_policy_execution_gpu_harness.py",
        "test_parameter_policy_execution_gpu_runtime.py",
        "test_parameter_policy_execution_gpu_support.py",
        "test_parameter_policy_frontend_behavior.py",
        "test_parameter_policy_full_runtime_smoke.py",
        "test_parameter_policy_gpu_qualification_environment.py",
        "test_parameter_policy_network_runtime_smoke.py",
        "test_parameter_policy_presets.py",
        "test_parameter_policy_qg1a_readiness.py",
        "test_parameter_policy_qg1a_sr1.py",
        "test_parameter_policy_qg1b3b_runner_label.py",
        "test_parameter_policy_qg1b3b_runtime_migration.py",
        "test_parameter_policy_qg1b_evidence_coverage.py",
        "test_parameter_policy_qg1b_sr1.py",
        "test_parameter_policy_qg1b_trusted_producer.py",
        "test_parameter_policy_qg2_mr_r2_2_sr1_fsr_sr1_red.py",
        "test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.py",
        "test_parameter_policy_qg2_mr_r2_2_sr1_red.py",
        "test_parameter_policy_qg2a1_request_red.py",
        "test_parameter_policy_qg2a1_sr1.py",
        "test_parameter_policy_qg2a1_sr1_lifecycle.py",
        "test_parameter_policy_qg2a2_api_red.py",
        "test_parameter_policy_qg2b2_cli_intent.py",
        "test_parameter_policy_qg2b3a_snapshot_red.py",
        "test_parameter_policy_qg2b3b_early_bridge_red.py",
        "test_parameter_policy_qg2b3b_sr1_dynamic_host_red.py",
        "test_parameter_policy_qg2b3c_late_dynamic_host.py",
        "test_parameter_policy_qg2b3c_late_session_red.py",
        "test_parameter_policy_qg2b3c_sr1_preapply_host.py",
        "test_parameter_policy_qg2b3c_sr1_real_late_cpu.py",
        "test_parameter_policy_qg2b3c_sr1_red.py",
        "test_parameter_policy_qg2b4_0_b312_rebase.py",
        "test_parameter_policy_qg2b4a1_api_handoff_red.py",
        "test_parameter_policy_qg2b4fsr_sr1_ci.py",
        "test_parameter_policy_qg2b4i_ci_migration.py",
        "test_parameter_policy_qg2b4l1_launcher_red.py",
        "test_parameter_policy_request_contract.py",
        "test_parameter_policy_resume.py",
        "test_parameter_policy_runtime.py",
        "test_parameter_policy_runtime_bools.py",
        "test_parameter_policy_schema.py",
        "test_parameter_policy_sdxl_weight_updates.py",
        "test_parameter_policy_step3_regression.py",
        "test_parameter_policy_step4_routing.py",
        "test_parameter_policy_step6a_contract.py",
        "test_parameter_policy_step6b_contract.py",
        "test_parameter_policy_step6c_contract.py",
        "test_parameter_policy_step6d_contract.py",
        "test_parameter_policy_step6e_contract.py",
        "test_parameter_policy_step6f_contract.py",
        "test_parameter_policy_step7_contract.py",
        "test_parameter_policy_torch_contract.py",
        "test_parameter_policy_torch_runtime_smoke.py",
        "test_parameter_policy_trainer_runtime_smoke.py",
    )
)
LIFECYCLE = (
    "qg1b3-qg2-admission-host.yml",
    "qg1b3a-admission-adapter-host.yml",
    "qg1b3b-fsr-runner.yml",
    "qg1b3b-typed-migration-host.yml",
    "qg2a1-request-intent-host.yml",
    "qg2a1-sr1-host.yml",
    "qg2a2-api-host.yml",
    "qg2b2-cli-intent-host.yml",
    "qg2b3a-policy-snapshot-host.yml",
    "qg2b3b-early-bridge-host.yml",
    "qg2b3b-sr1-dynamic-host.yml",
    "qg2b3c-late-session-host.yml",
    "qg2b3c-sr1-cpu-runtime.yml",
    "qg2b4-0-b312-contract-red.yml",
    "qg2b4a1-api-handoff-red.yml",
    "qg2b4fsr-sr1-ci.yml",
    "qg2b4l1-launcher-transport-red.yml",
    "qg2-mr-fsr-pr69-sr2b-ci-deps-red.yml",
)
ANIMA_INPUTS = frozenset((
    "tools/_parameter_policy_gpu_low_level_fixture.py",
    "tests/" + R2_SOURCE, "tests/" + HELPER, "tests/" + FSR,
    RUNNER, CONTRACT, ".github/workflows/" + E2,
))
R2_INPUTS = ANIMA_INPUTS | frozenset((
    DOC, "tests/" + HARNESS,
    "tools/run_parameter_policy_execution_gpu_matrix.py",
    "tools/run_parameter_policy_gpu_matrix.py",
    "tools/parameter_policy_execution_gpu_support.py",
    "tools/parameter_policy_backend_feature_gpu_support.py",
    "mikazuki/parameter_policy_admission.py",
    "mikazuki/direct_trainer_admission_intent.py",
    "mikazuki/parameter_policy_trainer.py",
    "mikazuki/parameter_policy.py",
    "mikazuki/parameter_policy_runtime.py",
    "mikazuki/parameter_policy_torch.py",
    "mikazuki/parameter_routing.py",
    "mikazuki/optimizer_profiles.py",
    "mikazuki/model_component_profiles.py",
    "mikazuki/parameter_policy_execution.py",
    "mikazuki/parameter_policy_compat.py",
    "mikazuki/parameter_policy_matrix.py",
    "mikazuki/parameter_policy_runtime_bools.py",
    "mikazuki/parameter_policy_readiness.py",
    "mikazuki/parameter_policy_readiness_producer.py",
    "mikazuki/parameter_policy_evidence_coverage.py",
))
SR2_INPUTS = frozenset((
    ".github/workflows/" + SR2,
    ".github/workflows/qg1b-trusted-producer-host.yml",
    ".github/workflows/qg1b-sr1-host.yml",
    "tests/test_qg2_mr_sr2_ci_dependencies_red.py",
    "docs/qualification-gate-qg2-mr-fsr-pr69-sr2b-ci-deps-tests-only-red.md",
    "mikazuki/parameter_policy_readiness_producer.py",
    "mikazuki/optimizer_profiles.py",
    "mikazuki/parameter_policy_evidence_coverage.py",
    "mikazuki/parameter_policy.py",
    "mikazuki/parameter_policy_compat.py",
    "mikazuki/parameter_policy_execution.py",
    "mikazuki/parameter_policy_matrix.py",
    "mikazuki/parameter_policy_readiness.py",
    "mikazuki/parameter_policy_runtime_bools.py",
))
PARITY = {
    "qg1b3a-admission-adapter-host.yml": (
        "mikazuki/parameter_policy.py",
        "tests/test_parameter_policy_qg1b3b_runtime_migration.py",
    ),
    "qg2b3b-early-bridge-host.yml": (
        "tests/test_parameter_policy_qg2b3a_snapshot_red.py",
        "mikazuki/parameter_policy_admission.py",
    ),
    "qg2b3c-late-session-host.yml": (
        "docs/qualification-gate-qg2b3c-late-session-tests-only-red.md",
        "tests/test_parameter_policy_qg2b3c_late_dynamic_host.py",
        "mikazuki/parameter_policy_admission.py",
        "mikazuki/parameter_policy.py",
    ),
    "qg2b3c-sr1-cpu-runtime.yml": (
        "docs/qualification-gate-qg2b3c-sr1-cpu-runtime-tests-only-red.md",
    ),
    "qg2b4a1-api-handoff-red.yml": (
        "docs/qualification-gate-qg2b4a1-api-handoff-tests-only-red.md",
    ),
    "qg2b4l1-launcher-transport-red.yml": (
        "docs/qualification-gate-qg2b4l1-launcher-transport-tests-only-red.md",
    ),
}
DYNAMIC = "qg2b3b-sr1-dynamic-host.yml"
DYNAMIC_INPUTS = frozenset((
    ".github/workflows/" + DYNAMIC,
    "tests/test_parameter_policy_qg2b3b_sr1_dynamic_host_red.py",
    "scripts/stable/train_network.py", "scripts/dev/train_network.py",
    "mikazuki/direct_trainer_admission_intent.py",
    "mikazuki/parameter_policy_admission.py",
    "scripts/stable/library/dts_parameter_policy_bridge.py",
    "scripts/dev/library/dts_parameter_policy_bridge.py",
))
FSR_INPUTS = frozenset((
    ".github/workflows/qg1b3b-fsr-runner.yml",
    ".github/workflows/qg1b3b-typed-migration-host.yml",
    "tests/test_parameter_policy_qg1b3b_runner_label.py",
    "tests/test_parameter_policy_qg1b3b_runtime_migration.py",
    "tests/test_parameter_policy_admission_red.py",
    "mikazuki/parameter_policy.py",
    "mikazuki/parameter_policy_admission.py",
    "mikazuki/parameter_policy_readiness_producer.py",
    "mikazuki/parameter_policy_execution.py",
    "mikazuki/parameter_policy_matrix.py",
))


def source(name: str) -> str:
    return (WF / name).read_text(encoding="utf-8")


def _section(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line == name + ":"]
    if len(hits) != 1:
        raise ValueError(f"Expected one literal top-level {name}: block")
    result = []
    for line in lines[hits[0]+1:]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        result.append(line)
    return result


def _scalar(value: str) -> str:
    value = value.strip()
    if value.startswith(('"', "'")):
        value = ast.literal_eval(value)
    if type(value) is not str or not value or not re.fullmatch(r"[A-Za-z0-9_./*?-]+", value):
        raise ValueError(f"Unsupported scalar in frozen event list: {value!r}")
    if value.startswith("!"):
        raise ValueError("Negative path filter not supported by explicit path contract")
    return value


def _list_values(lines: list[str], index: int, inline: str) -> tuple[str, ...]:
    if inline:
        if not inline.startswith("[") or not inline.endswith("]"):
            raise ValueError(f"Unsupported inline list: {inline}")
        values = [_scalar(part) for part in inline[1:-1].split(",")]
    else:
        values = []
        for line in lines[index+1:]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith("      - "):
                break
            values.append(_scalar(line[8:]))
    if not values or len(values) != len(set(values)):
        raise ValueError("Empty or duplicate event list")
    return tuple(values)


def events(text: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """Read the literal event syntax in the frozen workflows, not job strings."""
    lines = _section(text, "on")
    out = {}
    event = None
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        heading = re.fullmatch(r"  ([a-z][a-z0-9_]*):\s*", line)
        if heading:
            event = heading.group(1)
            if event in out:
                raise ValueError(f"Duplicate event: {event}")
            out[event] = {}
            continue
        field = re.fullmatch(r"    ([a-z][a-z0-9_-]*):\s*(.*?)\s*", line)
        if field:
            if event is None:
                raise ValueError("Event field before event")
            key, inline = field.groups()
            if key in out[event]:
                raise ValueError(f"Duplicate event field {event}.{key}")
            if key not in {"branches", "paths", "types", "branches-ignore", "paths-ignore"}:
                raise ValueError(f"Unsupported event field {event}.{key}")
            out[event][key] = _list_values(lines, i, inline)
            continue
        if line.startswith("      - "):
            continue
        raise ValueError(f"Unsupported literal event line: {line!r}")
    return out


def covers(fields: dict | None, inputs) -> list[str]:
    if fields is None:
        return sorted(inputs)
    if "paths-ignore" in fields or "branches-ignore" in fields:
        raise ValueError("Unexpected negative event filtering")
    paths = fields.get("paths")
    if paths is None:
        return []
    # Explicit paths are the contract for these narrow workflows.
    # Existing scoped directory/** filters are also valid coverage.
    return sorted(p for p in inputs if not any(
        p == q or (q.endswith("/**") and p.startswith(q[:-2]))
        for q in paths
    ))


def job_text(text: str, job: str) -> str | None:
    lines = _section(text, "jobs")
    headings = [(i, m.group(1)) for i, line in enumerate(lines)
                if (m := re.fullmatch(r"  ([A-Za-z0-9_-]+):", line))]
    hits = [i for i, name in headings if name == job]
    if not hits:
        return None
    if len(hits) != 1:
        raise ValueError(f"Duplicate job {job}")
    start = hits[0]
    end = next((i for i, _ in headings if i > start), len(lines))
    return "\n".join(lines[start:end])


def named_step_command(job: str, title: str) -> str:
    lines = job.splitlines()
    hits = [i for i, line in enumerate(lines)
            if line == "      - name: " + title]
    if len(hits) != 1:
        raise ValueError(f"Expected unique executable step: {title}")
    block = []
    for line in lines[hits[0]+1:]:
        if re.match(r"      - ", line):
            break
        block.append(line)
    if any(re.match(r"\s*(if|continue-on-error):", l) for l in block):
        raise ValueError(f"Conditional or tolerated step: {title}")
    runs = [l.strip()[5:] for l in block if l.startswith("        run: ")]
    if len(runs) != 1 or runs[0] == "|":
        raise ValueError(f"Expected one literal run command for {title}")
    return runs[0]


def actual_command(job: str | None, command: str) -> bool:
    if job is None or re.search(r"(?m)^    (if|continue-on-error):", job):
        return False
    lines = job.splitlines()
    for i, line in enumerate(lines):
        if line == "        run: " + command:
            start = next((j for j in range(i-1, -1, -1) if lines[j].startswith("      - ")), None)
            end = next((j for j in range(i+1, len(lines)) if lines[j].startswith("      - ")), len(lines))
            if start is not None and not any(
                re.match(r"\s*(if|continue-on-error):", l) for l in lines[start:end]
            ):
                return True
    return False


def _python_blocks(text: str) -> list[ast.Module]:
    blocks = re.findall(r"(?m)^([ ]+)python - <<'PY'\n([\s\S]*?)^\1PY\s*$", text)
    if not blocks:
        raise ValueError("No actual Python heredoc blocks")
    return [ast.parse(textwrap.dedent(body)) for _, body in blocks]


def _test_inputs(text: str) -> set[str]:
    result = set()
    for block in _python_blocks(text):
        for node in ast.walk(block):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests.test_"):
                result.add(node.module.replace(".", "/") + ".py")
            if isinstance(node, ast.Constant) and type(node.value) is str:
                if re.fullmatch(r"tests\.test_[A-Za-z0-9_]+", node.value):
                    result.add(node.value.replace(".", "/") + ".py")
    if not result:
        raise ValueError("No explicit test inputs found in executable blocks")
    return result


REFERENCE_CODE = r"""
import json, runpy, sys
from pathlib import Path
reference = runpy.run_path(sys.argv[1])["discover_reference"]
_, ids = reference(Path(sys.argv[2]), sys.argv[3:])
print("DTS_E2_REFERENCE_DISCOVERY_JSON " + json.dumps({
    "ids": ids, "errors": [], "loaded_cpu": [],
}))
"""


def _child_json(args: list[str], marker: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", *args], cwd=ROOT, text=True,
        capture_output=True, timeout=180, check=False,
        encoding="utf-8", errors="replace",
    )
    rows = [line[len(marker):] for line in proc.stdout.splitlines() if line.startswith(marker)]
    if proc.returncode or len(rows) != 1:
        raise RuntimeError(
            f"Invalid real discovery subprocess: rc={proc.returncode}, "
            f"records={len(rows)}\n{proc.stdout}\n{proc.stderr}"
        )
    result = json.loads(rows[0])
    if type(result) is not dict:
        raise ValueError("Discovery result must be an object")
    return result


def reference_discovery() -> list[str]:
    script = Path(__file__).resolve().with_name("qg2mr_sr2_e2_positive_reference.py")
    ref = _child_json(["-c", REFERENCE_CODE, str(script), str(TESTS),
                       *sorted(CPU_FILES)], REFERENCE_MARKER)
    ids = ref.get("ids")
    if (ref.get("errors") != [] or ref.get("loaded_cpu") != []
            or type(ids) is not list or not ids
            or any(type(i) is not str or not i for i in ids)
            or len(ids) != len(set(ids))):
        raise RuntimeError("Invalid positive reference Host identity census")
    print(f"E2 POSITIVE REFERENCE DISCOVERY: {len(ids)} Host identities; "
          "CPU ownership excluded before import", flush=True)
    return ids


class E2Targets(unittest.TestCase):
    def require(self, condition, reason):
        if not condition:
            self.fail(f"{MARKER}: {reason}")

    def both_paths(self, workflow, inputs):
        ev = events(source(workflow))
        missing = {event: covers(ev.get(event), inputs)
                   for event in ("push", "pull_request")}
        self.require(not any(missing.values()), f"{workflow} missing actual input paths: {missing}")

    def main_branch(self, workflow):
        ev = events(source(workflow))
        branches = ev.get("push", {}).get("branches", ())
        expected = (DEV, SR2_DEV) if workflow == SR2 else (DEV,)
        if not all(b in branches for b in expected) or "workflow_dispatch" not in ev:
            raise RuntimeError(f"Frozen development/dispatch contract lost: {workflow}")
        self.require("main" in branches, f"{workflow} lacks main push")

    def parity(self, workflow):
        ev = events(source(workflow))
        required = PARITY[workflow]
        if covers(ev.get("push"), required):
            raise RuntimeError(f"Frozen push reference paths lost: {workflow}")
        self.require(not covers(ev.get("pull_request"), required),
                     f"{workflow} PR lacks {covers(ev.get('pull_request'), required)}")

    def test_e2h101_real_no_torch_discovery_partition(self):
        validate = job_text(source(ANIMA), "validate")
        if validate is None:
            raise ValueError("Missing existing Anima validate job")
        command = named_step_command(validate, "Parameter Policy host contract")
        good = reference_discovery()
        if command == LEGACY_COMMAND:
            self.fail(MARKER + ": original broad Host discovery violates CPU ownership")
        self.require(command == HOST_COMMAND, "Anima is not wired to the reviewed Host partition")
        observed = _child_json(
            ["-m", "tests.qg2mr_sr2_e2_runner", "host", "--discover-only"],
            DISCOVERY_MARKER,
        )
        self.require(observed.get("errors") == [], "partition retained discovery errors")
        self.require(observed.get("loaded_cpu") == [], "partition imported a CPU-only module")
        ids = observed.get("ids")
        self.require(type(ids) is list and len(ids) == len(set(ids))
                     and ids == good, "partition lost, reordered or duplicated actual Host identities")

    def test_e2h102_helper_seven_cpu_execution_entry(self):
        job = job_text(source(E2), "r2-cpu")
        self.require(actual_command(job, CPU_COMMAND + "helper"),
                     "missing actual pinned CPU Helper suite command (R2G01-R2G07)")

    def test_e2h103_fsr_eleven_cpu_execution_entry(self):
        job = job_text(source(E2), "r2-cpu")
        self.require(actual_command(job, CPU_COMMAND + "fsr"),
                     "missing actual pinned CPU FSR suite command (SR101/SR102)")

    def test_e2h104_anima_r2_paths(self):
        ev = events(source(ANIMA))
        self.require(not covers(ev.get("pull_request"), ANIMA_INPUTS),
                     f"Anima PR lacks R2 inputs: {covers(ev.get('pull_request'), ANIMA_INPUTS)}")

    def test_e2h105_anima_main_lifecycle(self):
        ev = events(source(ANIMA))
        push = ev.get("push", {})
        pr_paths = ev.get("pull_request", {}).get("paths", ())
        self.require("main" in push.get("branches", ())
                     and not covers(push or None, set(pr_paths) | ANIMA_INPUTS),
                     "new merge contract: Anima main push must cover its PR inputs")

    def test_e2h106_cpu_matrix_runtime_and_trigger_contract(self):
        text = source(E2)
        job = job_text(text, "r2-cpu")
        ev = events(text)
        need = (
            "runs-on: ${{ matrix.os }}",
            "os: [ubuntu-latest, windows-latest]",
            'python-version: "3.11"',
            "https://download.pytorch.org/whl/cpu",
            "torch==2.7.0", "accelerate==1.6.0", "numpy==1.26.4",
            "schedulefree==1.4", "pytorch-optimizer==3.10.0", "safetensors==0.4.5",
            'CUDA_VISIBLE_DEVICES: ""',
        )
        self.require(job is not None and all(item in job for item in need)
                     and not re.search(r"(?m)^\s*continue-on-error:", job)
                     and "main" in ev.get("push", {}).get("branches", ())
                     and SR2_DEV in ev.get("push", {}).get("branches", ())
                     and "workflow_dispatch" in ev
                     and not covers(ev.get("push"), R2_INPUTS)
                     and not covers(ev.get("pull_request"), R2_INPUTS),
                     "missing permanent dual-OS pinned CPU matrix/input coverage")

    def test_e2l101_qg1b3_qg2_admission_host_main(self):
        self.main_branch("qg1b3-qg2-admission-host.yml")

    def test_e2l102_qg1b3a_admission_adapter_host_main(self):
        self.main_branch("qg1b3a-admission-adapter-host.yml")

    def test_e2l103_qg1b3b_fsr_runner_main(self):
        self.main_branch("qg1b3b-fsr-runner.yml")

    def test_e2l104_qg1b3b_typed_migration_host_main(self):
        self.main_branch("qg1b3b-typed-migration-host.yml")

    def test_e2l105_qg2a1_request_intent_host_main(self):
        self.main_branch("qg2a1-request-intent-host.yml")

    def test_e2l106_qg2a1_sr1_host_main(self):
        self.main_branch("qg2a1-sr1-host.yml")

    def test_e2l107_qg2a2_api_host_main(self):
        self.main_branch("qg2a2-api-host.yml")

    def test_e2l108_qg2b2_cli_intent_host_main(self):
        self.main_branch("qg2b2-cli-intent-host.yml")

    def test_e2l109_qg2b3a_policy_snapshot_host_main(self):
        self.main_branch("qg2b3a-policy-snapshot-host.yml")

    def test_e2l110_qg2b3b_early_bridge_host_main(self):
        self.main_branch("qg2b3b-early-bridge-host.yml")

    def test_e2l111_qg2b3b_sr1_dynamic_host_main(self):
        self.main_branch("qg2b3b-sr1-dynamic-host.yml")

    def test_e2l112_qg2b3c_late_session_host_main(self):
        self.main_branch("qg2b3c-late-session-host.yml")

    def test_e2l113_qg2b3c_sr1_cpu_runtime_main(self):
        self.main_branch("qg2b3c-sr1-cpu-runtime.yml")

    def test_e2l114_qg2b4_0_b312_contract_red_main(self):
        self.main_branch("qg2b4-0-b312-contract-red.yml")

    def test_e2l115_qg2b4a1_api_handoff_red_main(self):
        self.main_branch("qg2b4a1-api-handoff-red.yml")

    def test_e2l116_qg2b4fsr_sr1_ci_main(self):
        self.main_branch("qg2b4fsr-sr1-ci.yml")

    def test_e2l117_qg2b4l1_launcher_transport_red_main(self):
        self.main_branch("qg2b4l1-launcher-transport-red.yml")

    def test_e2l118_qg2_mr_fsr_pr69_sr2b_ci_deps_red_main(self):
        self.main_branch("qg2-mr-fsr-pr69-sr2b-ci-deps-red.yml")

    def test_e2l119_dynamic_host_pull_request(self):
        ev = events(source(DYNAMIC))
        self.require("pull_request" in ev, "Dynamic Host has no PR event")

    def test_e2l120_qg1b3a_admission_adapter_host_parity(self):
        self.parity("qg1b3a-admission-adapter-host.yml")

    def test_e2l121_qg2b3b_early_bridge_host_parity(self):
        self.parity("qg2b3b-early-bridge-host.yml")

    def test_e2l122_qg2b3c_late_session_host_parity(self):
        self.parity("qg2b3c-late-session-host.yml")

    def test_e2l123_qg2b3c_sr1_cpu_runtime_parity(self):
        self.parity("qg2b3c-sr1-cpu-runtime.yml")

    def test_e2l124_qg2b4a1_api_handoff_red_parity(self):
        self.parity("qg2b4a1-api-handoff-red.yml")

    def test_e2l125_qg2b4l1_launcher_transport_red_parity(self):
        self.parity("qg2b4l1-launcher-transport-red.yml")

    def test_e2l126_dynamic_real_inputs(self):
        # Push itself is missing four direct source inputs on the RED parent.
        self.both_paths(DYNAMIC, DYNAMIC_INPUTS)

    def test_e2l127_fsr_runner_real_inputs(self):
        self.both_paths("qg1b3b-fsr-runner.yml", FSR_INPUTS)

    def test_e2l128_b4a1_actual_inherited_test_inputs(self):
        name = "qg2b4a1-api-handoff-red.yml"
        inputs = _test_inputs(source(name))
        if "tests/test_parameter_policy_qg2b4l1_launcher_red.py" not in inputs or len(inputs) < 14:
            raise RuntimeError(f"Frozen inherited executable suite references not found: {inputs}")
        self.both_paths(name, inputs)

    def test_e2s101_permanent_sr2_strict_green(self):
        text = source(SR2)
        blocks = _python_blocks(text)
        literals = [node.value for block in blocks for node in ast.walk(block)
                    if isinstance(node, ast.Constant) and type(node.value) is str]
        self.require(not any("VALID TESTS-ONLY RED" in s or "GREEN CANDIDATE" in s
                             for s in literals),
                     "SR2 still uses historical RED/CANDIDATE branch")
        self.require("QG2MRFSRPR69SR2BTargetedRED" in text
                     and "QG2MRFSRPR69SR2BBaselines" in text
                     and all("." + s in text for s in (
                         "testsRun", "failures", "errors", "skipped",
                         "expectedFailures", "unexpectedSuccesses"))
                     and "11/11" in text,
                     "SR2 strict actual GREEN identities/results not retained")

    def test_e2s102_permanent_sr2_actual_dependency_paths(self):
        self.both_paths(SR2, SR2_INPUTS)


def strict_suite(name: str, count: int):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(name)
    if loader.errors or suite.countTestCases() != count:
        raise AssertionError(f"Invalid inherited suite discovery {name}: {loader.errors}")
    result = results.run_recorded_suite(suite, verbosity=1)
    if not results.exact_result(result, count):
        raise AssertionError(f"Inherited suite {name} did not execute {count}/{count} GREEN")
    print(f"E2 INHERITED GREEN: {name} {count}/{count} PASS", flush=True)


class E2Baselines(unittest.TestCase):
    def test_e2b901_original_census_and_cpu_paths(self):
        files = {p.name for p in TESTS.glob("test_parameter_policy*.py") if p.is_file()}
        self.assertEqual(len(FROZEN_FILES), 67)
        self.assertTrue(FROZEN_FILES <= files, sorted(FROZEN_FILES - files))
        self.assertEqual(CPU_FILES, frozenset((HELPER, FSR)))
        self.assertTrue(CPU_FILES <= files)
        # Ownership is the original method census and file path, not the
        # historical presence of an eager module-level Torch import.
        from tests import qg2mr_sr2_e2_runner as runner
        self.assertEqual(set(runner.CPU_SUITE_FILES.values()), CPU_FILES)
        for key, filename in runner.CPU_SUITE_FILES.items():
            tree = ast.parse((TESTS / filename).read_text(encoding="utf-8"))
            methods = [
                f"{Path(filename).stem}.{cls.name}.{method.name}"
                for cls in tree.body if isinstance(cls, ast.ClassDef)
                for method in cls.body if isinstance(method, ast.FunctionDef)
                and method.name.startswith("test_")
            ]
            expected = runner.expected_cpu_ids(key)
            self.assertEqual(len(methods), len(expected), filename)
            self.assertEqual(set(methods), set(expected), filename)
        fixture = (ROOT / "tools/_parameter_policy_gpu_low_level_fixture.py").read_text(encoding="utf-8")
        self.assertIn('"' + HELPER + '"', fixture)
        self.assertIn("_ALLOWED_MODEL_SOURCE_FILES", fixture)

    def test_e2b902_r2_source_thirteen(self):
        strict_suite("tests." + Path(R2_SOURCE).stem, 13)

    def test_e2b903_historical_harness_twenty_five(self):
        strict_suite("tests." + Path(HARNESS).stem, 25)

    def test_e2b904_sr2_dependency_eleven(self):
        strict_suite("tests.test_qg2_mr_sr2_ci_dependencies_red", 11)

    def test_e2b905_b312_sixteen(self):
        strict_suite("tests.test_parameter_policy_admission_red", 16)

    def test_e2b906_b4i_thirteen(self):
        strict_suite("tests.test_parameter_policy_qg2b4i_ci_migration", 13)

    def test_e2b907_b4fsr_seventeen(self):
        strict_suite("tests.test_parameter_policy_qg2b4fsr_sr1_ci", 17)

    def test_e2b908_preserve_unfiltered_pr_cpu_and_manual_gpu(self):
        b4i = events(source("qg2b4i-integrated-green.yml"))
        self.assertEqual(b4i["pull_request"], {})
        self.assertIn("main", b4i["push"]["branches"])
        for name in ("qg1b3b-fsr-runner.yml", "qg2b4fsr-sr1-ci.yml"):
            self.assertEqual(events(source(name))["pull_request"], {})
        cpu = job_text(source(ANIMA), "parameter-policy-runtime-smoke")
        self.assertIsNotNone(cpu)
        self.assertIn("needs: validate", cpu)
        for value in ("torch==2.7.0", "accelerate==1.6.0",
                      "pytorch-optimizer==3.10.0", 'CUDA_VISIBLE_DEVICES: ""',
                      "tests.test_parameter_policy_trainer_runtime_smoke"):
            self.assertIn(value, cpu)
        real = job_text(source("qg2b3c-sr1-cpu-runtime.yml"), "real-cpu-torch-red")
        self.assertIsNotNone(real)
        self.assertIn("os: [ubuntu-latest, windows-latest]", real)
        self.assertIn("torch==2.7.0", real)
        gpu = source("parameter-policy-gpu-matrix.yml")
        self.assertRegex(gpu, r"(?m)^  workflow_dispatch:$")
        self.assertNotRegex(gpu, r"(?m)^  (push|pull_request):")
        self.assertIn("self-hosted", gpu)


TARGET_PREFIXES = (
    tuple(f"test_e2h{i}_" for i in range(101, 107))
    + tuple(f"test_e2l{i}_" for i in range(101, 129))
    + tuple(f"test_e2s{i}_" for i in range(101, 103))
)
BASE_PREFIXES = tuple(f"test_e2b{i}_" for i in range(901, 909))


def run_exact(phase: str) -> int:
    if phase not in {"red", "green"}:
        raise SystemExit("Expected red or green")
    if importlib.util.find_spec("torch") is not None:
        raise SystemExit("E2 Host contract requires a real environment without Torch")
    suites = []
    names = []
    for cls, prefixes in ((E2Targets, TARGET_PREFIXES), (E2Baselines, BASE_PREFIXES)):
        found = unittest.defaultTestLoader.getTestCaseNames(cls)
        if len(found) != len(prefixes) or any(sum(n.startswith(p) for n in found) != 1 for p in prefixes):
            raise SystemExit(f"E2 INVALID IDENTITY/DISCOVERY: {cls.__name__} {found}")
        suites.append(unittest.TestSuite(cls(n) for n in found))
        names.append(set(found))
    targets, baselines = [
        results.run_recorded_suite(suite) for suite in suites
    ]
    if (not results.exact_result(targets, 36, allow_failures=True)
            or not results.exact_result(baselines, 8)):
        print("E2 INVALID EXECUTION: error/skip/xfail/xpass/baseline/count")
        return 2
    failed = {case.id().rsplit(".", 1)[-1]: trace for case, trace in targets.failures}
    if phase == "red":
        if (len(targets.failures) == 36 and set(failed) == names[0]
                and all(MARKER in trace for trace in failed.values())):
            print("E2-B VALID TESTS-ONLY RED: 36/36 exact target FAIL + 8/8 actual baseline PASS; "
                  "0 unexpected ERROR/SKIP/xfail/xpass", flush=True)
            return 1
        print(f"E2 INVALID RED SHAPE: {sorted(failed)}", flush=True)
        return 2
    if targets.failures:
        print(f"E2 GREEN HOLD: {sorted(failed)}")
        return 1
    print("E2 STRICT GREEN: 36/36 target + 8/8 baseline PASS; zero extra outcomes", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.qg2mr_sr2_e2_contract red|green")
    raise SystemExit(run_exact(sys.argv[1]))
