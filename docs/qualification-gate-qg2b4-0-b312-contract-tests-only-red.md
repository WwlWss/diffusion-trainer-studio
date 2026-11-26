# QG-2-B4-0 — B312 Contract Rebase and Tests-only RED

**Frozen parent:** PR #69, `aff397a6dd982286bc98b28f6541da63e3024fa1` (Draft / Open / Unmerged).
**Phase type:** ONLY tests, CI and documentation. Absolutely no production implementation or training permission changes.
**Goal:** Preserve the one intentional cumulative B312 RED, but make its message identify *only* the actual missing API → Launcher → Accelerate child CLI one-invocation intent transport.

## 1. Root cause: cumulative B312 became stale

QG-2-B3-A / B3-B / B3-C / B3-C-SR1 independently finished the Direct Trainer policy snapshot, strict Host and Late admission, and real CPU Torch / Pre-Apply source checks. B312 still looked for an obsolete `preflight_component_training_admission` name, while both base trainers and both bridges now correctly call `preflight_direct_trainer_admission`. Its bridge matcher checked `_called_names()`, which only sees bare `ast.Name` calls, not the actual `admission.prepare_direct_trainer_admission(...)` attribute call. Both were source-selector bugs, not missing functionality.

The canonical policy loader `mikazuki/parameter_policy.py:417+` is truly torch-free; the obsolete substring test `"import torch" in policy_source` matched the module's **documentation**, not a Python import. B4-0 checks AST `Import` / `ImportFrom` nodes and exercises the genuine loader. That must not be weakened into a general import whitelist for other modules.

The inherited B312 trace at frozen parent listed seven reasons; five were the above stale source matcher diagnoses. Two remain genuinely absent:

1. `typed strict-positive WebUI per-job Launcher intent and child CLI flag`
   - `mikazuki/training_launcher.py:20+` `run_prepared_train()` still lacks an `admission_intent` argument and does not append `--experimental_full_bf16_ack` to child argv.
2. `API must pass PreparedTrainingConfig intent to Launcher`
   - `mikazuki/app/training_api.py` `create_toml_file()` still invokes `run_prepared_train()` without the request-scoped `prepared.admission_intent`.

No code change here should manufacture an accepted experimental Start. **`experimental_host_candidate` still fails at Request Start, Direct Early and Late; Step 6F remains independent with its previous hard blockers.**

## 2. Frozen source selectors after rebase

- Canonical sidecar loader: existing `load_parameter_policy_sidecar()` + AST-based torch import inspection; no raw documentation substring heuristic.
- Stable/Dev base: real `preflight_direct_trainer_admission()` call order: first before dataset, second after model-specific `assert_extra_args` but before Accelerator. Pre-Apply and Late checks already have separate real Host/CPU tests.
- Both bridges: real pure `preflight_direct_trainer_admission()` lazily imports `mikazuki.parameter_policy_admission` and invokes the **attribute** `prepare_direct_trainer_admission()` factory. No trainer session/torch import on this path.
- Existing B2 CLI family / TOML / output_config, B3-A snapshots, B3-C root / optimizer / dtype / checkpoint guards unchanged.
- Real child CLI and API transport probes **remain RED**, not intentionally skipped or reclassified as passed.

## 3. Tests-only RED and independent baseline identities

`tests/test_parameter_policy_admission_red.py`: B301–B311 **11 genuine PASS**, B901–B904 **4 genuine PASS**, B312 **exactly one FAIL** with the *original* `QG-1B3-QG2 ADMISSION CONTRACT ABSENT` marker and exactly these **two** semicolon-separated reasons:

- `typed strict-positive WebUI per-job Launcher intent and child CLI flag`;
- `API must pass PreparedTrainingConfig intent to Launcher`.

Any other B312 reason, any missing one of these, empty reason, error, skip, xfail, xpass or unexpected failed test = **INVALID RED**. B312's historical own first-party status matrix was previously hidden behind `self.fail()`; an independent real Host baseline now runs it even while transport is absent.

New `tests/test_parameter_policy_qg2b4_0_b312_rebase.py`, exactly **six genuine PASS** baselines:
- B4001: canonical loader AST/actual strict JSON and SHA-256;
- B4002: both real Stable/Dev first/final/Pre-Apply call order;
- B4003: both real lazy pure first-party bridges and attribute factory call;
- B4004: B312 uses real source names, AST import nodes and no old function name;
- B4005: independently execute D1 legacy, pending no-ack, pending positive candidate, SD3 unsupported blocker;
- B4006: frozen typed snapshot, Late four-coordinate verification, candidate HOLD, require_assets and pre-scan ordering.

Run the existing inherited B3 and QG-1B tests on Ubuntu/Windows to detect regressions. Do not need to rerun GPU training in this phase. **This is a test correction, not Direct Trainer feature development.**

## 4. Exact dual-OS CI signoff rule

New workflow `.github/workflows/qg2b4-0-b312-contract-red.yml`, Ubuntu and Windows Python3.11. In each OS:
- Six B4001–B4006 real Host baselines PASS; discovery exactly 6.
- Inherited QG-1B/SR1 42, B2 8, B3-A 12, B3-B 12, B3-B-SR1 8, B3-C 10, B3-C Late Dynamic 8, B3-C-SR1 11 + Pre-Apply 5: all genuine PASS.
- Cumulative 16 cases: 15 real PASS, B312 one genuine FAIL **with exactly both quoted reasons and no other reason**. No unexpected ERROR/SKIP/xfail/xpass.
- On valid Tests-only RED intentionally `exit 1`. Future fully wired B312 16/16 can become GREEN only after Launcher/API behavior is independently tested and all inherited workflows expecting B312 RED are migrated in a single reviewed CI cutover. This B4-0 stage does not perform that cutover.

## 5. Stop line and next stage

Do not edit `mikazuki/training_launcher.py`, `mikazuki/app/training_api.py`, any Trainer/bridge, B3-A admission factory, readiness/qualification, frontend, optimizer, GPU/CUDA, Checkpoint/Resume or task metadata. Preserve Standard, qualified D1, Direct CLI, and **Candidate HOLD**.

**Next after exact valid dual-OS RED:** QG-2-B4-L1 Launcher strict typed transport Tests-only RED (real argv/task/env boundary, one-job no-replay) → B4-L2 Minimal GREEN → B4-A1 API handoff Tests-only RED/GREEN → B4-I integrated B312 GREEN and coordinated 11 inherited CI workflows → independent final source/behavior review. Never treat B312 GREEN as permission to remove experimental Full-BF16 HOLD.
