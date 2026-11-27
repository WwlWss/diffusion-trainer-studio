"""C3 scoped qualification of persistent workflow lifecycle.

The historical job digests are from the exact C2 parent. Header digests
freeze the reviewed C3 event migration, including unchanged names and
permissions. These are scoped audit snapshots, not a ban on future reviewed
workflow changes. Keep the E2-B and C1/C2 contracts unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
import unittest

from tests import qg2mr_sr2_e2_contract as frozen
from tests import qg2mr_sr2_e2_test_results as results
from tests import qg2mr_sr2_e2_c1_contract as c1
from tests import qg2mr_sr2_e2_c2_contract as c2

ROOT = frozen.ROOT
ANIMA = "anima-qwen3-review.yml"
E2 = "qg2-mr-sr2-e2-ci-integration.yml"
SR2 = "qg2-mr-fsr-pr69-sr2b-ci-deps-red.yml"
RETAINED_RED = ("test_e2l118_", "test_e2s101_", "test_e2s102_")
# name: (original jobs SHA256, reviewed migrated header SHA256)
SNAPSHOT = {
    "qg1b3-qg2-admission-host.yml": (
        "f9cd4fa8fd4150f056c2880cbe5e61c01efbee02276e46b9edff14c00f78fba0",
        "dae34ec8ce48267f88b5a178020306babbe35563362476091509af52e44c276e",
    ),
    "qg1b3a-admission-adapter-host.yml": (
        "e8e7a5fe7590fcb25b2b891b074718b431630e3b471d058f284c51023f07df8d",
        "f4acd80810d881b0436d18f05bcac303ad21785cb8e9e70b827b82607e1ae081",
    ),
    "qg1b3b-fsr-runner.yml": (
        "66954ed9cafee8514aeecedca17bb9e5e2ee4f11764e1c005aa7772685e2ce9f",
        "b999ea5137165c50c9fad6143d24dd9c50cfc47265560c647b7db50f95ebc6f6",
    ),
    "qg1b3b-typed-migration-host.yml": (
        "fd0114bf23f603c6b48d0ea310c75fec2218cb7d3e0599c64cc14ee74bffda95",
        "2b0226c040e4de3a5ec879cde05dc55106a95e9b097f5bd4fc966db3df839357",
    ),
    "qg2a1-request-intent-host.yml": (
        "f467eda2a2fada7bb0323e7a849a02ebea6e96bc905d201fb1440d3c3efdacd7",
        "a6f69ccf967f93e98f0e618258e0c66de89e42e2e4ac780f9a7911c81f2c76d2",
    ),
    "qg2a1-sr1-host.yml": (
        "c4cf022ba576be0e54fc60339e1200582542a61bc7961e0b2264fdd86f55f7bd",
        "a2131c712eb87ca059a91d6e2ad32f17c9f7333225af081a376606b7cef9fd7c",
    ),
    "qg2a2-api-host.yml": (
        "c4c4fca5c88e155bdf0d26d5b1a84f89ba85ff07c6c4989c93b427651a98f314",
        "36e3febe602fe893080ffed320ebbdbb4364c9eef0f1e32f5cdadbac0839651a",
    ),
    "qg2b2-cli-intent-host.yml": (
        "218c6838c1e3a5aa9a7798ad821e38ba62ee7a1423431f3bbf647ec7c4da9c27",
        "f35327dd517b53f622434cc5258666759d876b56ff1317dd620f70ce4a4335b7",
    ),
    "qg2b3a-policy-snapshot-host.yml": (
        "667ab3db424fa18fa77a09a4c4a5ea85575f589daecfa126b2e5cd92e6544b78",
        "91e7e2a680118c553d38f2ee277bb07693fd6f0772a5fb96610feba10dcb7359",
    ),
    "qg2b3b-early-bridge-host.yml": (
        "8cd19bdf32f40373846ad3f3aa5e7897887c650d86fa33a842278ddbe62d0581",
        "404c7c40205e3b009181f0b1fc8269119ee76521261623ac1a187c08bc7383b7",
    ),
    "qg2b3b-sr1-dynamic-host.yml": (
        "ad4335a185ad3c6c4d8d3a5a64893e259891f31eb2f1b3093c7c46c30a833c0e",
        "f91a433ba138fe112e229963825efec03029415051930e1b8864784ecd0fb419",
    ),
    "qg2b3c-late-session-host.yml": (
        "dfca9e55946eb11fc57bfdcf04b550af21bf228173bac95c401964882f62305a",
        "7f6b30ae18d2f6fbdaa32fc49d0d05b69f493b576ec0d5d05ecd8b41e3fc0009",
    ),
    "qg2b3c-sr1-cpu-runtime.yml": (
        "9c737048859a0003c6ecdfff8841d3eada1b411dd3ca513287e7d043d3e640a0",
        "91f721571244de4c01765f93fe5082ffc2ff46112e7465670a8a632e28dc6e10",
    ),
    "qg2b4-0-b312-contract-red.yml": (
        "f4a43dc43c4bc6fc816565a7e5eb4f9f9cee87da6c6058f1e2c945ef23b2946e",
        "62b8d85ab8886fffefd242951567757549aad02d77d0ddf4873258cc328b6734",
    ),
    "qg2b4a1-api-handoff-red.yml": (
        "5a6c6a60d4dd3ff789eb03b8f8a9cd9f256e574cf2fd5ab470aadbc8a083f65b",
        "017026fb89caa70b339e912c59e608207c5f3b5efbe7a84b4e52c6bfb14cdc7c",
    ),
    "qg2b4fsr-sr1-ci.yml": (
        "812b43d15d62f81ce6a2dc845dba492e573f09915d57c302c2b93c3e0d599428",
        "d1463a5f1fa0f2678e98f9283b80951624847e6b5ae3a63304949b9576c1e7fa",
    ),
    "qg2b4l1-launcher-transport-red.yml": (
        "9f6d521fdd6b054ebf65b0ad06b9e44f48d524b25f5a43c83096c0d221a21d52",
        "a926143450e3a893b7305d8df7c4f56450d3f5fd093930099bcab29516ed37ff",
    ),
    "anima-qwen3-review.yml": (
        "67a9a4bcd6653faf688e623cd5eef2cc5de672b914dc0ce6a41ae379e8d2a873",
        "739c0a335e19d89961a0407749eed8315d65b511ff2cd25a1b5fddb474ba5088",
    ),
}
UNFILTERED = frozenset((
    "qg1b3b-fsr-runner.yml", "qg2b4fsr-sr1-ci.yml", "qg2b4i-integrated-green.yml",
))
UNTOUCHED = {
    ".github/workflows/qg2b4i-integrated-green.yml": "7f87ab5d42c6ba9c00ba99cf2d06b9492a21a94e",
    ".github/workflows/parameter-policy-gpu-matrix.yml": "f09c22486f69b3b6fd98c4e6048eb12e41970467",
}
C3_SR2_BLOB = "ba4c12022eadb83dc80cec4aa25d7a16fc5cfe5a"
E2_INPUTS = frozenset((
    ".github/workflows/qg1a-readiness-host.yml",
    ".github/workflows/qg1a-sr1-host.yml",
    ".github/workflows/qg1b-sr1-host.yml",
    ".github/workflows/qg1b-trusted-producer-host.yml",
    ".github/workflows/qg1b3-qg2-admission-host.yml",
    ".github/workflows/qg1b3a-admission-adapter-host.yml",
    ".github/workflows/qg1b3b-fsr-runner.yml",
    ".github/workflows/qg1b3b-typed-migration-host.yml",
    ".github/workflows/qg2-mr-fsr-pr69-sr2b-ci-deps-red.yml",
    ".github/workflows/qg2a1-request-intent-host.yml",
    ".github/workflows/qg2a1-sr1-host.yml",
    ".github/workflows/qg2a2-api-host.yml",
    ".github/workflows/qg2b2-cli-intent-host.yml",
    ".github/workflows/qg2b3a-policy-snapshot-host.yml",
    ".github/workflows/qg2b3b-early-bridge-host.yml",
    ".github/workflows/qg2b3b-sr1-dynamic-host.yml",
    ".github/workflows/qg2b3c-late-session-host.yml",
    ".github/workflows/qg2b3c-sr1-cpu-runtime.yml",
    ".github/workflows/qg2b4-0-b312-contract-red.yml",
    ".github/workflows/qg2b4a1-api-handoff-red.yml",
    ".github/workflows/qg2b4fsr-sr1-ci.yml",
    ".github/workflows/qg2b4i-integrated-green.yml",
    ".github/workflows/qg2b4l1-launcher-transport-red.yml",
    ".github/workflows/qg2mr-sr1-tests-only-red.yml",
    "tests/qg2mr_sr2_e2_c3_contract.py",
    "docs/qualification-gate-qg2-mr-sr2-e2-c3-persistent-ci-lifecycle.md",
    "tests/qg2b4i_ci_contract.py",
    "tests/test_qg2_mr_sr2_ci_dependencies_red.py",
    "docs/qualification-gate-qg2-mr-fsr-pr69-sr2b-ci-deps-tests-only-red.md",
    "tests/test_parameter_policy_qg2b4i_ci_migration.py",
    "tests/test_parameter_policy_qg2b4fsr_sr1_ci.py",
    "tests/test_parameter_policy_admission_red.py",
))
FSR_EXTRA = frozenset((
    "mikazuki/parameter_policy_readiness.py",
    "mikazuki/parameter_policy_compat.py",
    "mikazuki/parameter_policy_runtime_bools.py",
    "mikazuki/parameter_policy_evidence_coverage.py",
    "mikazuki/model_component_profiles.py",
    "mikazuki/optimizer_profiles.py",
    "mikazuki/training_gui_args.py",
    "mikazuki/sd3_full_bf16_scope.py",
    "mikazuki/sd3_architecture_contract.py",
))
B4A_EXTRA = frozenset((
    "tests/test_parameter_policy_qg2a2_api_red.py",
    "mikazuki/training_config.py",
    "mikazuki/direct_trainer_admission_intent.py",
    "mikazuki/parameter_policy_admission.py",
    "scripts/stable/train_network.py",
))


def source(name):
    return frozen.source(name)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def blob_digest(text):
    content = text.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii")
                        + b"\0" + content).hexdigest()


def sections(text):
    if text.count("\njobs:\n") != 1:
        raise RuntimeError("Missing or duplicate jobs section")
    before, jobs = text.split("\njobs:\n")
    return before, "jobs:\n" + jobs


def direct_inputs(text):
    # Audited literal .py compile/run paths and test-module names in the
    # historical job sections. This does not claim complete transitive closure.
    _, jobs = sections(text)
    inputs = set(re.findall(
        r"\b(?:tests|mikazuki|scripts|tools)/[A-Za-z0-9_./-]+\.py\b", jobs))
    inputs.update("tests/" + name + ".py" for name in re.findall(
        r"\btests\.(test_[A-Za-z0-9_]+|qg2b4i_ci_contract|_parameter_policy_[A-Za-z0-9_]+)\b",
        jobs))
    return inputs


def require_events(name, text):
    ev = frozen.events(text)
    expected_branches = {"main"} if name == ANIMA else {"main", frozen.DEV}
    if set(ev.get("push", {}).get("branches", ())) != expected_branches:
        raise AssertionError(f"Persistent branch coverage lost: {name}")
    if name != ANIMA and ev.get("workflow_dispatch") != {}:
        raise AssertionError(f"Original dispatch lost: {name}")
    if "pull_request" not in ev:
        raise AssertionError(f"PR event missing: {name}")
    if name in UNFILTERED:
        if ev["pull_request"] != {}:
            raise AssertionError(f"Unfiltered PR narrowed: {name}")
    elif (set(ev["push"].get("paths", ())) != set(ev["pull_request"].get("paths", ()))
          or set(ev["pull_request"]) != {"paths"}):
        raise AssertionError(f"Push/PR path parity lost: {name}")
    if name != ANIMA:
        inputs = direct_inputs(text)
        for event in ("push", "pull_request"):
            missing = frozen.covers(ev[event], inputs)
            if missing:
                raise AssertionError(f"Explicit run input coverage lost: {name}:{event}:{missing}")
        for path in inputs:
            if not (ROOT / path).is_file():
                raise AssertionError(f"Explicit input does not exist: {name}:{path}")


def require_snapshot(name, text):
    require_events(name, text)
    header, body = sections(text)
    original_jobs, expected_header = SNAPSHOT[name]
    if digest(body) != original_jobs:
        raise AssertionError(f"Historical job body changed: {name}")
    if digest(header) != expected_header:
        raise AssertionError(f"Reviewed event/header snapshot changed: {name}")


class LifecycleBehavior(unittest.TestCase):
    def test_c3101_eighteen_historical_workflow_event_and_job_snapshots(self):
        self.assertEqual(set(SNAPSHOT), set(frozen.LIFECYCLE[:-1]) | {ANIMA})
        self.assertEqual(len(SNAPSHOT), 18)
        for name in SNAPSHOT:
            with self.subTest(workflow=name):
                require_snapshot(name, source(name))

    def test_c3102_all_explicit_job_inputs_and_old_pr_contracts(self):
        for name in SNAPSHOT:
            with self.subTest(workflow=name):
                require_events(name, source(name))
        for name in UNFILTERED:
            self.assertEqual(frozen.events(source(name))["pull_request"], {})

    def test_c3103_dynamic_fsr_and_frontend_actual_input_paths(self):
        inputs = (
            (frozen.DYNAMIC, frozen.DYNAMIC_INPUTS | {
                "tests/test_parameter_policy_qg2b3a_snapshot_red.py"}),
            ("qg1b3b-fsr-runner.yml", frozen.FSR_INPUTS | FSR_EXTRA),
            ("qg2a2-api-host.yml", {"frontend", ".gitmodules",
                                  "mikazuki/app/api_overlay.py"}),
        )
        for name, required in inputs:
            ev = frozen.events(source(name))
            for event in ("push", "pull_request"):
                self.assertFalse(frozen.covers(ev[event], required), (name, event))
            for path in required:
                self.assertTrue((ROOT / path).exists() or path == "frontend", path)

    def test_c3104_b4a1_actual_ast_and_imported_dependencies(self):
        name = "qg2b4a1-api-handoff-red.yml"
        text = source(name)
        inputs = frozen._test_inputs(text)
        self.assertEqual(len(inputs), 15)
        self.assertIn("tests/test_parameter_policy_qg2b4l1_launcher_red.py", inputs)
        for event in ("push", "pull_request"):
            self.assertFalse(frozen.covers(frozen.events(text)[event], inputs | B4A_EXTRA))
        for path in inputs | B4A_EXTRA:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_c3105_e2_watches_all_original_qg_workflow_contracts(self):
        ev = frozen.events(source(E2))
        required = E2_INPUTS | frozen.R2_INPUTS | c2.EXTRA_INPUTS
        self.assertEqual(set(ev["push"]["branches"]), {"main", frozen.DEV, frozen.SR2_DEV})
        self.assertEqual(set(ev["push"]["paths"]), set(ev["pull_request"]["paths"]))
        for event in ("push", "pull_request"):
            self.assertFalse(frozen.covers(ev[event], required), event)
        for path in E2_INPUTS:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_c3106_b4i_and_manual_gpu_workflows_unchanged(self):
        for path, expected in UNTOUCHED.items():
            self.assertEqual(blob_digest((ROOT / path).read_text(encoding="utf-8")),
                             expected, path)
        self.assertEqual(frozen.events(source("qg2b4i-integrated-green.yml"))["pull_request"], {})

    def test_c3107_negative_mutations_of_actual_migrated_workflows(self):
        name = "qg2b4a1-api-handoff-red.yml"
        text = source(name)
        require_snapshot(name, text)
        no_main = text.replace("      - main\n", "", 1)
        no_dep = text.replace('      - "tests/test_parameter_policy_qg2b4l1_launcher_red.py"\n', "", 1)
        disabled = text.replace("    runs-on:", "    if: false\n    runs-on:", 1)
        narrowed = text.replace("  pull_request:\n", "  pull_request:\n    branches: [main]\n", 1)
        for label, mutated in (("main", no_main), ("dependency", no_dep),
                               ("disabled", disabled), ("PR branches", narrowed)):
            with self.subTest(mutation=label):
                self.assertNotEqual(text, mutated)
                with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                    require_snapshot(name, mutated)
        name = "qg1b3b-fsr-runner.yml"
        text = source(name)
        narrowed = text.replace("  pull_request:\n",
            '  pull_request:\n    paths:\n      - "tests/test_parameter_policy_qg1b3b_runner_label.py"\n', 1)
        self.assertNotEqual(text, narrowed)
        with self.assertRaises(AssertionError):
            require_events(name, narrowed)


def run_behavior():
    names = unittest.defaultTestLoader.getTestCaseNames(LifecycleBehavior)
    prefixes = [f"test_c3{i}_" for i in range(101, 108)]
    if len(names) != 7 or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
        raise RuntimeError(f"Wrong C3 behavior identities: {names}")
    result = results.run_recorded_suite(
        unittest.TestSuite(LifecycleBehavior(n) for n in names))
    ok = c1.exact_result(result, 7) and not result.failures
    print(f"E2-C3 BEHAVIOR: {'7/7 PASS' if ok else 'HOLD'}", flush=True)
    return ok


def stage():
    c1.require_no_torch()
    # This guard is C3-only: C4 will intentionally migrate this workflow.
    if blob_digest(source(SR2)) != C3_SR2_BLOB:
        raise RuntimeError("C3 must preserve the exact SR2 workflow for C4")
    if not c1.run_behavior() or not c2.run_behavior() or not run_behavior():
        return 2
    targets = unittest.defaultTestLoader.getTestCaseNames(frozen.E2Targets)
    baselines = unittest.defaultTestLoader.getTestCaseNames(frozen.E2Baselines)
    for names, prefixes in ((targets, frozen.TARGET_PREFIXES), (baselines, frozen.BASE_PREFIXES)):
        if len(names) != len(prefixes) or any(sum(n.startswith(p) for n in names) != 1 for p in prefixes):
            raise RuntimeError(f"Frozen E2 identity mismatch: {names}")
    red = {n for n in targets if n.startswith(RETAINED_RED)}
    if len(red) != 3:
        raise RuntimeError(f"Wrong C3 retained RED identities: {red}")
    tr = results.run_recorded_suite(
        unittest.TestSuite(frozen.E2Targets(n) for n in targets))
    br = results.run_recorded_suite(
        unittest.TestSuite(frozen.E2Baselines(n) for n in baselines))
    failed = {case.id().rsplit(".", 1)[-1]: trace for case, trace in tr.failures}
    if (not c1.exact_result(tr, 36) or not c1.exact_result(br, 8)
            or br.failures or len(tr.failures) != 3 or set(failed) != red
            or not all(frozen.MARKER in trace for trace in failed.values())):
        print("E2-C3 INVALID STAGE SHAPE", flush=True)
        return 2
    print("E2-C3 SCOPED GREEN: 33/33 actual target PASS + 3/3 retained RED + "
          "8/8 baseline PASS + C1 10/10 + C2 6/6 + C3 7/7 behavior PASS; "
          "0 ERROR/SKIP/xfail/xpass; integrated E2 HOLD", flush=True)
    return 1


if __name__ == "__main__":
    if sys.argv[1:] != ["stage"]:
        raise SystemExit("usage: python -m tests.qg2mr_sr2_e2_c3_contract stage")
    raise SystemExit(stage())
