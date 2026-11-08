# QG-2-B1 — Direct Trainer / Launcher parity Tests-only RED

**Parent architecture review:** QG-2-B0, commit 2776d6680180f98c07f19a42c38935d41c4fa863.
**Code baseline before B1:** 598a81aee3a1b1181ee333055e6b3268fa24fc8d.
**Branch:** feature/qualification-gate-qg1b3-qg2-admission-red.
**PR:** #69, Draft and unmerged.

## 1. Stage objective and hard boundary

QG-2-B1 freezes the missing Direct Trainer / Launcher parity contract as a host-only Tests-only RED.

Allowed changes in this stage:
- tests/test_parameter_policy_admission_red.py: strengthen the existing B312 aggregate test and add passing assertions inside the four existing B901-B904 baselines.
- .github/workflows/qg1b3-qg2-admission-host.yml: scope the cumulative Ubuntu/Windows host workflow to the B1 source paths and exact B312 marker.
- This document.

Forbidden changes:
- No production Python, TypeScript, frontend, launcher, Trainer, API, Policy, optimizer or Checkpoint code.
- No qualification/evidence registry edits, CUDA, physical training, resume or optimizer execution.
- No change to the model-family allow-list, A2 candidate hold or PR merge status.
- No skip, xfail, expectedFailure, deleted negative assertion, unconditional failure or test-count manipulation.

## 2. Frozen test identity and expected count

The existing 16-case cumulative suite remains exactly 16 tests:
- B301-B312: twelve targeted contracts.
- B901-B904: four actual source/legacy baselines.

Expected result for this stage:
- B301-B311: 11 PASS.
- B312: exactly one intentional FAIL.
- B901-B904: 4 PASS.
- Total: 16 tests, 15 PASS + 1 targeted FAIL.
- 0 ERROR, 0 SKIP, 0 xfail, 0 xpass.

The single expected failure identity remains:
test_b312_direct_trainer_uses_same_admission_and_explicit_cli_intent.

Its only accepted failure marker remains:
QG-1B3-QG2 ADMISSION CONTRACT ABSENT.

The cumulative workflow also preserves the 42 previous QG-1B / SR1 host tests:
- 16 QG-1B trusted producer tests.
- 16 QG-1B SR1 tests.
- 10 D1 evidence coverage tests.

The exact same counts and outcome are required on ubuntu-latest and windows-latest, Python 3.11.

## 3. B312 aggregated contract

The B312 test aggregates the full Direct Trainer / Launcher control-plane contract into one test identity. It must not use independent failing subTests or split the missing seams into additional failing test methods.

### A. Typed one-invocation CLI intent

First-party source must provide:
- Frozen DirectTrainerAdmissionIntent with strict real-bool experimental_full_bf16_ack.
- A strict --experimental_full_bf16_ack CLI parser entry in both stable and dev sd-scripts parser paths.
- extract_direct_trainer_admission_intent(args) that creates the typed intent and removes the raw flag from the argparse Namespace.
- Stable and dev executable entrypoints extract the intent before read_config_from_file() and pass it separately to Trainer.train().

This is one invocation's user intent. It is not a qualification record, evidence token, environment variable or persisted training setting.

### B. TOML / output-config isolation

Both stable and dev read_config_from_file() paths must reject a TOML key named experimental_full_bf16_ack before building the merged argparse Namespace.

The CLI intent must be extracted and removed before config loading, so --output_config cannot serialize the confirmation. A parser store_true action by itself is not a TOML trust boundary.

### C. Two-stage pure Host Admission

Both stable and dev NetworkTrainer paths must:
1. Receive the typed admission_intent separately.
2. Run an initial Host Admission before tokenizer cache writes, dataset preparation, Accelerator creation and model loading.
3. Run a second Host Admission after model-family assert_extra_args() normalization and immediately before prepare_accelerator().
4. Use purpose="direct_trainer" and the same strict acknowledgement on both checks.

Both lazy dts_parameter_policy_bridge.py copies must expose this pure preflight without importing mikazuki.parameter_policy_trainer or torch.

### D. First-party classifier and immutable snapshot

The source-owned admission layer must provide:
- Frozen DirectTrainerAdmissionSnapshot containing the canonical Policy hash, effective-config fingerprint and typed admission result.
- A pure preparation helper that uses the existing assess_component_training_admission() with purpose="direct_trainer" and the strict experimental_ack value.
- A single canonical, torch-free Policy sidecar loading/hash path shared by early preflight and late Session.

The early result is Host admission only. It never replaces live CUDA / Tensor / optimizer / Checkpoint evidence.

### E. Late model-aware Session revalidation

create_parameter_policy_session() must accept the typed intent and immutable snapshot separately from args. Before scan_parameter_roots(), it must:
- Reload and validate the canonical Policy.
- Compare Policy hash and effective-config fingerprint with the final preflight snapshot.
- Re-run the first-party assess_component_training_admission() for Direct Trainer.
- Preserve semantic, model/target, optimizer and mandatory require_assets=True hard checks.

A positive acknowledgement may only satisfy the explicit acknowledgement condition for a source-implemented, otherwise eligible candidate. It cannot promote SD3 / Flux / Chroma or bypass real hard errors. The B1 test does not authorize training or remove the existing QG-2-B/final-review HOLD.

### F. Launcher / API transport

mikazuki/app/training_api.py:create_toml_file() must pass prepared.admission_intent to run_prepared_train().

run_prepared_train() must accept a typed per-job intent and append --experimental_full_bf16_ack to the child trainer arguments only for a strict positive WebUI Start intent. The child flag must appear after trainer_file so Accelerate forwards it to the trainer script.

Do not transport acknowledgement through:
- environment variables;
- task metadata;
- TOML or exported bundles;
- policy sidecars;
- Checkpoint manifest/progress;
- resume state.

The Direct CLI and API-launched child both construct a fresh child-process intent. The parent Request intent is not copied into durable trainer configuration.

### G. Existing live runtime and Checkpoint invariants

The existing late runtime checks remain mandatory:
- live Parameter Root identity and alias topology;
- requires_grad ownership;
- actual trainable BF16 Tensor dtype;
- actual Accelerator process count and distributed type;
- prepared optimizer parameter identity and device;
- Checkpoint manifest identity and progress;
- Resume manifest/progress validation and post-resume live-root audit.

The Checkpoint manifest must not contain experimental_full_bf16_ack. Resume never restores user intent.

## 4. Passing baselines B901-B904

The four existing baseline test identities are retained and strengthened without adding test cases:

- B901: existing SD1 Full-BF16 D1 source path remains legacy_approved for Direct Trainer without acknowledgement.
- B902: pending / unreviewed SD3 remains blocked even with experimental_ack=True.
- B903: inactive Full-BF16 / quoted pseudo-boolean legacy semantics remain unchanged; importing stable/dev bridge modules does not import the torch-heavy Trainer.
- B904: request/start and Direct Trainer checks remain before materialization/root scan; live runtime and Checkpoint/Resume invariant methods remain present; Checkpoint manifest has no acknowledgement field.

## 5. Workflow acceptance

The cumulative host workflow compiles the test harness, runs the 42 prior tests as strict GREEN, then runs the exact 16-case suite.

The workflow must fail with exit code 1 only after it has verified the one expected B312 failure and printed the B1 Tests-only RED ledger. It must fail on:
- changed test discovery count;
- any unrelated failure;
- missing B312 marker;
- error / skip / xfail / xpass;
- any prior baseline regression.

A workflow run ending failure is therefore expected only when both platform jobs report the same verified single B312 RED. A failed job without the exact ledger is invalid evidence.

## 6. Stop line and next stage

QG-2-B1 ends when the exact cumulative dual-OS RED is established and recorded. It does not implement any of the missing production interfaces.

After valid RED, proceed to a separately planned QG-2-B2 minimal GREEN for typed CLI intent and TOML isolation. Do not begin B2 in the B1 RED commit.

No physical training, CUDA qualification, Checkpoint migration, certificate unlock, PR merge or release authorization is implied by B1.
