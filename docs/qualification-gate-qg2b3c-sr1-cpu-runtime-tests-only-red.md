# QG-2-B3-C-SR1 — Historical CPU Torch migration, Full-BF16 isolation and Pre-Apply evidence — Tests-only RED

**Frozen reviewed parent:** `173353027a000dbb4e76e8e4348d20ac6a0c9bdb`.
**PR:** #69 on `feature/qualification-gate-qg1b3-qg2-admission-red`, Draft/open/unmerged.
**This phase:** tests, documentation and isolated CI **only**. Do not migrate existing smoke tests or edit any production source in RED.

## 1. Actual source-level inventory (13 direct historical calls)

All 13 are `create_parameter_policy_session(...)` without the required `admission_snapshot` and `admission_intent`.

| File | Source lines at frozen parent | Category |
| --- | --- | --- |
| `tests/test_parameter_policy_trainer_runtime_smoke.py` | 106, 874, 981, 1682 | Ordinary/Mixed-BF16 existing session tests (4) |
| same | 215, 261, 318 | Mock-qualified Flux Full-BF16 execution tests (3) |
| same | 552 | Genuine active Full-BF16 rejection-before-scan test (1) |
| `tests/test_parameter_policy_full_runtime_smoke.py` | 119, 161, 200 | Ordinary full-model routing/backprop smoke (3) |
| `tests/test_parameter_policy_network_runtime_smoke.py` | 456 | Ordinary Flux network/scheduler smoke (1) |
| `tests/test_parameter_policy_anima_runtime_smoke.py` | 94 | Ordinary Anima DiT/Qwen3 ownership smoke (1) |

The previous FSR inventory of 11 was incomplete. The complete audited inventory is 13, with 9 ordinary/Mixed, 3 low-level mock-qualified Full-BF16, and 1 genuine blocked path. The production signature change is verified at `mikazuki/parameter_policy_trainer.py:1619–1626`.

## 2. Causal and safety analysis

B3-C deliberately made a frozen `DirectTrainerAdmissionSnapshot` and an independently typed `DirectTrainerAdmissionIntent` mandatory at the **real** late session. This is correct, even for programmatic Component callers. Old tests invoke the now-invalid ABI; those tests were written before that safety contract existed. Do not change production defaults, permit `None` or accept caller-controlled statuses.

The three special Flux Full-BF16 tests legitimately test lower-level execution/optimizer/dtype/weak-reference/resume invariants by temporarily patching `FULL_BF16_*_QUALIFICATIONS`. Modern first-party readiness also guards `_AUDITED_BACKENDS={"sd-lora","sdxl-lora"}` and unconditionally blocks unsupported Full-BF16 backends. Changing release-table rows does NOT grant Full-BF16 Host Admission for `flux-finetune`, nor should it.

Separate these layers in future GREEN:
- **Real Admission:** genuine first-party B3-A factory, actual strict sidecar, 64-char policy/config SHA-256, exact typed one-shot intent. Every normal/Mixed smoke calls the production late function using both required args; do not fake `ComponentAdmission` statuses.
- **Real rejection:** Full-BF16 Flux with explicit typed snapshot and intent must reject **before root scan**, even if a release row is patched.
- **Isolated low-level implementation tests:** test-only helper(s) under `tests/` construct valid real routing/runtime/optimizer/ExecutionContract and Session internals without calling public `create_parameter_policy_session` to bypass a legitimate blocked Admission. Preserve original assertions. No test bypass flag in production; no mutated source capability/qualification table in production.

## 3. Source semantics requiring evidence before risky edits

The Stable/Dev `network.apply_to()` step changes original module `forward` and attaches child LoRA adapters before Late Session. A pure **Pre-Apply** identity check could catch a changed sidecar/config/intent before that mutation while leaving the real Late Session after LoRA attachment (necessary for root discovery). Its guard must invoke the existing B3-A pure loader/classifier, compare the entire four-coordinate snapshot and frozen first-party admission, and fail closed. The Late Session must independently rerun all checks; no TOCTOU guarantee across two moments and no attempt to undo third-party network hooks.

Stable/Dev also derive `args.max_train_steps` after Late Session using actual dataloader/process count; `save_every_n_epochs` is resolved later. The existing Scheduler Identity is calculated from final args in `_scheduler_identity_from_args`, after dataloader preparation. Thus the B3-A fingerprint binds a **pre-model configuration**. The downstream resolved scheduler/checkpoint identity must remain separate and intact. Do not retroactively remove any config fields, weaken SHA-256, or move model/dataloader/accelerator creation before the early gate.

For the public Python `train(self,args,admission_intent=None)`, preserve Standard legacy calls. Active Component `train(args)` must fail for lack of typed intent; active Component with exact `DirectTrainerAdmissionIntent(False)` must run full checks; ack=True is advisory only.

## 4. Frozen Tests-only RED identities

`tests/test_parameter_policy_qg2b3c_sr1_red.py` asserts six independent **actual missing obligations**, each with the exact marker `QG-2-B3-C-SR1 HISTORICAL CPU CONTRACT ABSENT`:

- SR101: Eight existing trainer-runtime direct Session calls migrated to typed two-keyword contract.
- SR102: Three existing full-model runtime calls migrated.
- SR103: Network and Anima one-call smokes migrated (including separately supplied config snapshots).
- SR104: Three mock-qualified Full-BF16 tests no longer enter production late Admission for low-level simulation; instead use an isolated **test-only** session construction layer and preserve all prior assertions.
- SR105: Genuine Full-BF16 Flux rejection-before-scan test supplies real typed snapshot/intent and verifies first-party blocked output, regardless of patched release rows.
- SR106: Stable and Dev perform pure four-coordinate Pre-Apply identity revalidation before `network.apply_to`; Late Session remains after attachment for parameter scan. The intended fix must not merely check presence of the string `preflight_direct_trainer_admission`; identity comparison and first-party status comparison must be evidenced.

Five live baselines SR901–905 prove typed first-party API and strict loader, Flux Full-BF16 hard block, existing B3-C Late guard, Scheduler Identity separation, and Standard/B312 boundaries. No manufactured `assert False`, swallowed exception, xfail or skip.

## 5. Real CPU Torch RED evidence

The SR1 workflow additionally runs the exact four **existing** CPU runtime smoke test methods through the *real imported torch module*, with `torch` and `accelerate` explicitly installed on Ubuntu/Windows:
- trainer runtime: `test_baseline_session_has_no_execution_contract_or_root_refs`;
- full runtime: `test_dreambooth_text_encoder_only_backpropagates_through_frozen_unet`;
- network runtime: `test_component_lr_logging_survives_accelerate_scheduler_wrapper`;
- Anima runtime: `test_anima_full_dit_and_qwen_ownership`.

A valid CPU RED requires four real missing-required-keyword `TypeError` results from these four methods, **not** import errors, missing dependencies, unexpected failures, xfail, or skipped tests. After fixture migration the same runner will require 4/4 PASS. This CPU evidence is separate from the 6 Host contract failures; do not silently count an absent Torch installation as RED. If dependency installation is unavailable, report CPU evidence as unverified rather than claiming signoff.

## 6. Cumulative inherited gates and stop line

Ubuntu + Windows Python 3.11: all inherited QG-1B/SR1 42, B2 8, B3-A 12, B3-B 12, B3-B-SR1 8, B3-C target/baseline 10 and CDYN 8 must stay PASS. B301–B311 and B901–B904 15 remain PASS, and B312 exactly one original expected RED. SR1 Host six distinct target failures plus five real baselines PASS, zero unexpected errors/skip/xfail/xpass.

**No production modifications**: no `mikazuki/parameter_policy_trainer.py`, Stable/Dev Trainer or bridge, derived model, qualification, launcher/API, optimizer, CUDA, checkpoint/resume production code. No existing smoke edits in Tests-only RED. Keep PR #69 Draft/unmerged, B312 HOLD, Full-BF16 candidate HOLD.

Next after exact accepted RED is **SR1 normal test migration → test-only Full-BF16 low-level construction → genuine CPU Torch green → conditional Pre-Apply guard → full dual-platform regression → independent Final Strict Code Review**.
