# DTS QG-1B — First-party trusted execution-readiness producer contract

**Baseline:** QG-1A3-SR1 reviewed HEAD `f6b0b757f58546922a2703608b662a3810476eff`, parent [PR #67](https://github.com/WwlWss/diffusion-trainer-studio/pull/67); QG-0 [114-gate inventory](https://github.com/WwlWss/diffusion-trainer-studio/blob/docs/qualification-gate-architecture-review/docs/qualification-gate-qg0-inventory-2026-11.md) in separate [PR #66](https://github.com/WwlWss/diffusion-trainer-studio/pull/66). This is a **stacked, independent QG-1B0 source contract + QG-1B1 Tests-only RED**, not a runtime-feature unlock.

## 1. Root cause and trust model

`mikazuki.parameter_policy_readiness.evaluate_execution_readiness()` (QG-1A) correctly reduces **already-classified** capability/issue facts. It cannot know whether a caller has demoted an actual `R03 runtime.dtype.mismatch` to `warnings` or promoted a mere `E02 qualification.backend.pending` to a hard error. Its `can_start` is diagnostic only.

Legacy `mikazuki/parameter_policy_execution.py:419–433,533–541,576–596` combines *formal CUDA evidence* with product eligibility, returning on backend `pending` before independent semantic/optimizer/source checks. Existing `parameter_policy_runtime_blockers` (`parameter_policy.py:543–661`), Preview/Start (`training_request.py:223–236`), and direct Trainer (`parameter_policy_trainer.py:1627–1653`) still run those legacy blockers. Current correctness restrictions (bad precision, optimizer ownership/device, missing assets, invalid policy, incompatible saved state, real BF16 dtype, SD3 mid-epoch Resume) **remain authoritative**.

**QG-1B first-party producer (new, not yet implemented)** determines capability and error severity *from concrete local code/registries and runtime facts*, never from Worker-provided issue lists, bool flags, claimed capabilities, requested decision, or certificate status. Evidence metadata may add a **warning** but cannot grant capability; runtime failures must always be hard even if presented to a remote caller as warnings.

**Critical scope:** A host-only classifier cannot inspect actual live Tensor dtype/grad/optimizer devices or real resumed state without a Trainer. Therefore its result is a **host preflight diagnosis**, NOT an unconditional launch permission. QG-1B1 must not pretend to prove a true R03 live dtype error from untrusted configuration text. Keep `ParameterPolicyTrainerSession.assert_execution_dtype_contract()`, `assert_live_root_identity_contract()`, optimizer/device and checkpoint assertions unchanged and fail-closed. QG-2 must recompute readiness plus run genuine Trainer audits; no HTTP/GUI consumer may trust a user-submitted `ExecutionReadiness` instance or `can_start`.

## 2. Frozen public ABI and producer inputs

New standalone host-side module:

`mikazuki/parameter_policy_readiness_producer.py`

Export one primary function, **not connected to any training entrypoint yet**:

```python
def classify_component_full_bf16_readiness(
    canonical_policy: Mapping[str, Any],
    *,
    train_type: str,
    effective_config: Mapping[str, Any],
    require_assets: bool = False,
) -> ExecutionReadiness | None:
    ...
```

- `None` only when `full_bf16` is **strictly inactive**. Standard/mixed BF16 stays a pure no-op. Invalid pseudo-bools must not silently yield `None`.
- Validate `canonical_policy` via existing `validate_parameter_policy()`; check trainer backend membership in `PARAMETER_POLICY_RUNTIME_TRAIN_TYPES`, resolve only **trainable referenced** optimizer primary/fallback profiles through `get_optimizer_capability()`, and retain `planned/restricted` implementation/ownership restrictions. Unknown policy/optimizer/backend and malformed core configuration must fail closed, with a stable source code or validation exception; the producer must not guess "implemented".
- Review real `full_bf16=true, mixed_precision=bf16` semantics, strict typed booleans (`full_bf16`, `v2`, `gradient_checkpointing`), existing semantic/target incompatibilities and hard runtime ownership boundary. The producer cannot skip or suppress independent noncertificate legacy blockers and must never silently bypass any checkpoint/trainer auditor.
- **Audited positive Full-BF16 production host cases initially only**: stock `sd-lora` with `v2=false`, stock `sdxl-lora` with `gradient_checkpointing=false`; source-trace compatible optimizer `AdamW` or `Muon` with valid explicit routing (fallback only where genuinely supported) and reviewed optimizer arguments. This is **host eligibility only**; real Torch/Accelerate, actual files and tensors still guarded later. Merely listing a train_type in `PARAMETER_POLICY_BACKEND_MATRIX`, setting `qualifications.status="qualified"`, or claiming a source SID must **not** upgrade any other backend to implemented.
- **Not yet audited Full-BF16** (especially `sd3-lora`, `sdxl-finetune`, Flux/Chroma/Anima, SD2 `v2=true`, extra optimizer/dtype variants) continues **BLOCK/HOLD for execution contract not established**, with an explicit *capability-audit-required* hard reason, **not** a misleading "GPU CUDA test failed" claim. QG-3 will source-review and qualify actual `experimental_candidate` configs; the missing release certificate alone is **not** the blocker.
- **Known incompatible:** SDXL Full-BF16 with gradient checkpointing and potentially trainable TE embeddings remains hard blocked even under a fake `qualified` row. Likewise real invalid mixed precision is hard blocked regardless of evidence.
- **Mutable test-only qualification-row injection:** Existing `FULL_BF16_BACKEND_QUALIFICATIONS` and `FULL_BF16_OPTIMIZER_QUALIFICATIONS` are read **only for evidence warnings/tiers**, after capability facts. Patch a trusted registry row from `qualified` to `pending` for the **same** already audited SD1 case: capability and execution identity do not change; decision changes from `allow` to `warn` only if a missing-certificate warning is generated (not `block`). Patch `qualified` without `evidence_case_id`: no valid formal tier. Never mutate either registry.
- Determine `evidence_tier` from trusted local release rows in this scoped host adapter: `formally_qualified` only when all selected *backend and referenced optimizer* rows have `status=qualified` **and nonblank source evidence IDs**, and the variant is in the audited evidence coverage. Otherwise `unverified` (do not fabricate `observed_real_gpu` or `host_cpu_checked` if no separate precise source). **This is a descriptive local registry reading, not external cryptographic proof.** A Worker cannot supply `evidence_tier`.
- The producer has **no** `capability`, `warnings`, `hard_errors`, `decision`, `can_start` or `evidence_tier` arguments. A forged `ReadinessIssue` or \`{"runtime.dtype.mismatch": "warning"}\` inside an otherwise valid GUI/effective config must not change true classifications. Unknown caller-provided kwargs cause `TypeError`. It may ignore **untrusted diagnostic-only** config extras but must not blindly ignore a real legacy feature flag.
- Stable code namespace examples: `runtime.bf16.precision_mismatch` (H), `runtime.bool.invalid` (H), `capability.backend.unknown` (H), `capability.backend.audit_required` (A), `capability.optimizer.not_implemented` (H), `capability.optimizer.audit_required` (A), `runtime.sdxl.gradient_checkpointing_unreviewed` (H/A), `qualification.backend.pending` (E), `qualification.backend.missing_evidence` (E), `qualification.optimizer.pending` (E). Codes are machine-owned; message wording is never authorization.
- Use existing `ReadinessIssue` and `evaluate_execution_readiness` to construct the immutable result. Avoid duplicating the pure reducer or forging source `R03` runtime findings without live tensors. Preserve stable `ParameterPolicyExecutionContract.execution_identity()` and `execution_signature()` exactly, regardless of certificate tier.

**Host-only and deployment limits:** No import of Torch, Accelerate, large models, CUDA, attestation Runner, OS signing service, filesystem mutation, network, time or user environment. `require_assets=False` never opens/checks model files; for `require_assets=True` preserve existing backend asset validation contract without claiming absence of real model is formally safe. This phase does NOT relocate actual runtime admission until later reviewed QG-2.

## 3. Frozen QG-1B1 Tests-only RED (12 target + 4 old baselines)

New `tests/test_parameter_policy_qg1b_trusted_producer.py` contains **16 unconditional cross-OS Host tests** with 12 new B101–B112 and four B901–B904 real baselines. Each B101–B112 must produce **one** exact tagged failure `QG-1B TRUSTED PRODUCER CONTRACT ABSENT` while `mikazuki.parameter_policy_readiness_producer` is missing (no `skip` or `expectedFailure`). No production source file is added in QG-1B1.

| Test | Frozen expected outcome *after future GREEN* |
| --- | --- |
| B101 | First-party producer module/function exists and signature excludes caller-supplied authority/severity/evidence claims; reducer outcome immutable |
| B102 | Source-audited SD1 + AdamW, patched backend `pending`: **implemented/warn/unverified**, stable E02 warning, never Block solely for certificate |
| B103 | Same audited SD1 with `qualified` but missing evidence ID: **warn/unverified**; does not mint formal tier |
| B104 | Invalid actual `mixed_precision=fp16` blocks, regardless of forged `runtime.dtype.mismatch` warning or `can_start=True` fields in config |
| B105 | Unknown backend/target blocked even if Worker claims `implemented` or backend qualification row is fake-qualified |
| B106 | Planned `pytorch_optimizer.CAME` is not executable, even if its release row is fake-qualified |
| B107 | Quoted `full_bf16="false"` is a hard runtime bool error; cannot become inactive Standard |
| B108 | SDXL full BF16 + real gradient_checkpointing=TRUE remains hard blocked regardless of formally qualified release row |
| B109 | SD3 Full-BF16 R1/R2 unreviewed implementation remains in **capability-audit-required HOLD**, even if Worker or evidence rows claim fully certified; no fabricated runtime R03 diagnosis |
| B110 | Only referenced optimizer primary/fallback impact evidence status; unused registered pending profiles must not downgrade or block audited SD1 |
| B111 | Exact execution identity/signature unchanged before/after trusted release row variations; source registries and original policy/config remain unchanged |
| B112 | Standard/full_bf16 inactive is a no-op; unsupported malformed strict config and untrusted extra kwargs fail closed; no signature/CUDA authority in producer |

**B901–B904 existing baselines:** prior QG-1A pure reducer decision semantics, legacy `sd3-lora full_bf16` still returns `pending` blocker, existing strict boolean/precision refusal, planned `pytorch_optimizer.CAME` remains blocked. Each baseline **PASS** before the new module exists. No previous legacy test assertion is altered.

**Exact Actions B1 RED:** Ubuntu + Windows Python 3.11; **16 discovered and executed, B101–B112 twelve tagged FAIL + B901–B904 four PASS**, zero ERROR/SKIP/xfail/xpass; both job failures are intentional. After B1 source-contract RED is accepted, QG-1B2 can implement a **new producer module only**; legacy `_qualification_blocker`/Preview/Start/direct Trainer still untouched until QG-1B3/QG-2 migration with its own tests.

## 4. F1 mandatory integration tests before changing Start

The independent final QG-1A3 review left F1 unresolved **for production integration**. This B1 Host source contract is necessary but not sufficient. Before any actual Start/Direct Trainer permission change, freeze and pass:

- **F1-01** actual live `ParameterPolicyTrainerSession.assert_execution_dtype_contract()` on a real BF16-required FP32 trainable Tensor blocks, no matter how the HTTP/Worker presents `warnings`, `capability` or `can_start`;
- **F1-02** backend/optimizer missing release case while code/ownership is proven produces advisory only, including both GUI and direct Trainer;
- **F1-03** unimplemented optimizer, invalid Component routing, untrusted capability claim, misbucketed hard E02, or independent runtime/Scheduler/Checkpoint issues cannot get a Start permission;
- **F1-04** no trained-model/weights-identity signature, original checkpoint manifest, attestation status or CUDA proof is generated from this host-only decision.

QG-1B1 is explicitly **no executable QG-1B migration or partial Full-BF16 acceptance**. QG-2 remains the explicit scope for Preview/Export/Start/direct parity, user-facing experimental acknowledgment and real runtime binding, after a separate source review.


## 5. QG-1B1 exact Tests-only RED ledger

**Source-contract + tests + CI freeze HEAD:** `e53caf4811ce8250f5a6ab9f4823b475ef0ae0fb`.
- [Push Windows+Ubuntu #42293546247](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42293546247): each **16 discovered/run = 12 B101–B112 tagged FAIL + 4 B901–B904 baseline PASS**, zero ERROR/SKIP/xfail/xpass. Every targeted failure included `QG-1B TRUSTED PRODUCER CONTRACT ABSENT`.
- [Stacked PR Windows+Ubuntu #42293600333](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42293600333): identical results on both OS, confirming RED on PR trigger as well.
- The red workflows intentionally **conclude FAILURE**; all failures are exact missing-new-module contract failures, not failed existing training functionality. The four baseline tests include prior pure readiness behavior, old SD3 certificate-only blocker, real invalid BF16/strict bool checks and planned CAME rejection.
- Files introduced in this stage: `docs/qualification-gate-qg1b-trusted-producer-contract.md`, `tests/test_parameter_policy_qg1b_trusted_producer.py`, and `.github/workflows/qg1b-trusted-producer-host.yml`. No `mikazuki/parameter_policy_readiness_producer.py` and no edit to any existing production module or legacy test.

**Next stage:** QG-1B2 implement **only the new host-side trusted producer** against these substantive contracts; require Ubuntu+Windows **16/16 PASS, 0 FAIL/ERROR/SKIP** on exact final HEAD, plus fresh independent Strict Review. The producer must never accept supplied capability/severity from Worker or grant Start. Actual live dtype/device/optimizer/Checkpoint audits remain in existing Trainer and will be regression-gated before the later production entrypoint migration.

**B1 HOLD:** No QG-1B production implementation, no Legacy `_qualification_blocker` migration, no GUI/headless experimental-override entrypoint, no Source-qualified SD3 true Full-BF16 R1/R2 or new CUDA GPU smoke, no external Runner attestation, and no merge. PR #68 remains Draft stacked on PR #67.


## 6. QG-1B2 minimum GREEN — source and actual CI

**Frozen source-contract RED** at `ec8f25f37686022d1a3c0390a1f2279e2a5c21af`: [push #42293734636](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42293734636) and [PR #42293743017](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42293743017), each OS 16 run = 12 targeted missing-producer FAIL +4 prior-source baseline PASS, 0 ERROR/SKIP/xfail/xpass. New classifier module absent at RED.

**QG-1B2 new module** at `171afa06b896a21fe309ac57232f4d8476fc2e3c`: `mikazuki/parameter_policy_readiness_producer.py` (351 added LOC), **the sole production file added**. Source-gated Host classification:
- Original `validate_parameter_policy` / strict boolean runtime / `build_parameter_policy_execution_contract` / `parameter_policy_v1_semantic_blockers` and existing non-Anima target/cache checks remain authoritative.
- An explicit, source-scoped audited Host capability for SD1.x stock LoRA and SDXL LoRA, AdamW or eligibility-routed Muon. Other backends including SD3 true Full-BF16, SD2.x, extra Optimizers and known SDXL gradient checkpointing ownership risks HOLD/BLOCK for **implementation/ownership audit**, independently of evidence metadata.
- Only referenced trainable primary/fallback optimizers affect evidence. Expected source-case IDs and local `qualified` statuses are read **only** for host diagnostic evidence tier and warning; no signing, physical CUDA observation, independent OS credential or Checkpoint original Load.
- User/Worker `can_start`, `decision`, `capability`, `qg_claim`, forged hard/warning buckets and asserted evidence tier in arbitrary effective-config extras do not grant support.
- This module is an **unwired Host diagnostic**; `require_assets=True` is an **ABI-reserved input that does not perform filesystem/model asset validation**, and MUST NOT be used as a launch asset check. `can_start` remains advisory, not proof of live dtype/device or a Start authorization. Launch asset validation remains the responsibility of existing production validators in later QG-2 integration.

**First B2 attempt identified an invalid test fixture**, not a new producer regression: B110 constructed `_policy(fallback="AdamW8bit")` with primary `AdamW`. Existing `parameter_policy.py:250–258` correctly rejects fallback on a primary without parameter eligibility. The first source run was 15 PASS +1 ERROR (B110), **not accepted GREEN**. Corrected **only the B110 test input** at `81390f62c143400e9731649b957be60fdfdd9d21` to `_policy(optimizer="Muon", fallback="AdamW8bit")`, a structurally valid eligibility-routed primary/fallback pair. No legacy production validator was relaxed and no other test behavior changed.

**Accepted actual scoped GREEN at `81390f62c143400e9731649b957be60fdfdd9d21`:**
- [Dedicated QG1B PR #42300460180](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42300460180) and [push #42300453070](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42300453070): **Ubuntu 16/16 PASS, Windows 16/16 PASS**, zero FAIL/ERROR/SKIP/xfail/xpass. This includes the four original-source baselines that retain the **legacy SD3 `pending` fatal gate** unchanged.
- [Anima Qwen3 PR #42300460183](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42300460183) completed **SUCCESS**, including Parameter Policy host contract (640 tests, 64 skipped due optional dependencies, **no fail/error**) and Parameter Policy runtime smoke (SUCCESS). These are Host/CPU tests, not a physical CUDA run.

**Qualification verdict:** QG-1B2 **SCOPED HOST GREEN**; **not** QG-1B3 trusted Start adapter integration, QG-2 Preview/Start/direct Trainer parity, an SD3 Full-BF16 usable release, real CUDA R1/R2, physical Checkpoint Resume, or external signed evidence. Legacy `_qualification_blocker`, UI, Trainer, formal qualification rows and Checkpoint code unchanged. Keep PR #68 **Draft and stacked on QG-1A PR #67**, unmerged until independent Strict Review and integration.

**Independent review watchpoints before QG-1B3/QG-2:** verify host evidence tier against true variant-specific coverage; never treat `require_assets=True` or pure diagnostic `can_start` as a signed/verified asset/trainer permission; ensure multiple optimizer audit failures are not silently obscured when stable issue codes are de-duplicated; preserve the distinction between unreviewed implementation and known broken functionality.


## 7. QG-1B2-SR1: evidence scope, asset API, diagnostics and capability correction

**Reviewed starting point:** `02a76e751310e8b9f97d823c4cf8429321e077c1`, [independent Final Strict Review #5811377478](https://github.com/WwlWss/diffusion-trainer-studio/pull/68#pullrequestreview-5811377478). Fixes are deliberately scoped to the **new Host producer, pure reference-coverage helper, new SR1 tests and their CI**. They do not modify QG-1A public enums, old training gates, model loader, actual optimizer/Checkpoint state, release rows or external proof service.

### F1 — D1 reference shape is NOT a physical certification of this run

Prior `parameter_policy_readiness_producer.py` emitted `formally_qualified` whenever broad backend/optimizer statuses and expected ID strings agreed, even for SDXL TE1-only and unqualified ranks. This overclaimed D1. D1 source `tools/parameter_policy_backend_feature_gpu_support.py:620–935` constrains the stock network, U-Net/CLIP trainable coverage, AdamW versus Muon+AdamW fallback, reference accumulation, Conv LoRA and strict fresh/resume model input identity. [SD1 D1 PR #57](https://github.com/WwlWss/diffusion-trainer-studio/pull/57) and [SDXL D1 PR #58](https://github.com/WwlWss/diffusion-trainer-studio/pull/58) document real physical CUDA evidence and its specific scope.

The new `mikazuki/parameter_policy_evidence_coverage.py` is a **pure reference-shape checker**, exposing immutable `EvidenceCoverage(covered,case_id,reasons)` and `match_full_bf16_evidence_coverage()`. It recognizes SD1 and SDXL **AdamW** or **Muon+explicit AdamW fallback** D1 topology only with explicit reference rank/alpha, 512×512, single-batch, accumulation 1/2, stock `networks.lora`, real Conv LoRA arguments, safetensors, required U-Net+Text Encoder trains, and no D1-outside precision/process/compile/cache/weights flags. All unconfirmed values fail **evidence-shape** coverage, **not** training capability.

**Important trust distinction:** `covered=True` compares *Host-visible shape* only. It cannot verify the actual pretrained model/dataset file SHA, clean exact head or original checkpoint identity. Therefore the QG-1B Host producer **never promotes this current user job to `formally_qualified`** without the missing trusted input binding, even when the known D1 shape matches. It reports `qualification.variant.identity_unverified` for matched shape and `qualification.variant.outside_scope` otherwise, with `evidence_tier=unverified`. This intentionally preserves runtime `implemented` and `decision=warn` for otherwise runnable source-audited configurations. Existing release statuses remain unchanged and still create independent warnings for `pending`, missing IDs and unqualified referenced optimizers. No user-provided `evidence_case_id` or self-declared model SHA grants proof.

Future QG-2/attestation may implement an authenticated *binding* from job identity to physical proof; only that independently reviewed producer may consider a higher evidence tier. A positive D1 shape match is **not** a physical CUDA pass for this job.

### F2 — fail closed when asset checking is requested but unavailable

`require_assets=False` remains pure Host diagnostic. `full_bf16=false` returns `None` as before. An active `full_bf16=true, require_assets=True` now raises a stable `ValueError` that actual asset preflight is **not implemented** and points back to the existing launch validator. No model path, Torch or file I/O was added. Never reinterpret that error as “model file is invalid”; it means “asset validation was not executed by this Host helper”.

### F3 — stable issue codes without lost Profiles

The old dictionaries kept only the first/last issue of each code. The producer now names every referenced optimizer **Profile ID + type** and groups all distinct `(source,message)` pairs deterministically under the original issue code; an aggregate has `source=QG1B` and preserves each original source inside its message. Both hard `capability.optimizer.audit_required` and qualification warning `qualification.optimizer.pending` preserve independent affected Profile details. Unused profiles remain excluded. QG-1A's strict unique-code and immutable snapshot contracts stay unchanged.

### F4 — audit_required ≠ implementation absent

Known `planned` Optimizers remain `not_implemented`, genuine runtime incompatibility stays `known_incompatible`, and registered but unreviewed SD3/SD2/extra Optimizer true Full-BF16 is now `experimental_candidate` **with an unconditional hard `audit_required` error and `decision=block`** until separate QG-3 source/ownership work. A deterministic severity order resolves mixed findings (`known_incompatible > not_implemented > experimental_candidate > implemented`), retaining all hard issues. No actual SD3 Full-BF16 execution was enabled, and no change to QG-1A enum was required.

### Frozen RED and scoped GREEN evidence

- **Tests-only RED HEAD `fb6205a9432bbdcc76d83ada9b906cde58604a06`:** [push #42361711291](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42361711291) and [PR #42361714910](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42361714910). Each Ubuntu/Windows ran **16 SR1 tests: 12 exact tagged targeted FAIL +4 baseline PASS, 0 ERROR/SKIP/xfail/xpass**; prior QG-1B 16/16 PASS. Dedicated jobs intentionally FAILURE on valid RED. No production code had yet been modified.
- **First minimum correction HEAD `310a782b474b9de8f4fb9de4c398ac6ba0c7c955`:** [push #42361882638](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42361882638) and [PR #42361885590](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42361885590), each OS **QG-1B 16/16 + SR1 16/16 PASS**, 0 FAIL/ERROR/SKIP/xfail/xpass.
- The separate `tests/test_parameter_policy_qg1b_evidence_coverage.py` adds **10 actual Host shape positive/negative tests** covering all four D1 reference families, default/missing scope, non-reference target and geometry, and the distinction between matched shape and actual per-job CUDA proof. [PR #42361942114](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42361942114) and [push #42361939187](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42361939187): each OS **16+16+10 = 42/42 PASS**, no skip/failure.
- **Additional strict negative check:** SR1 coverage test injected `full_bf16=false`, `mixed_precision=fp16`, `num_processes=2`, multi-GPU distributed mode, `torch_compile` and fused groups into otherwise D1-shaped inputs. The old matcher incorrectly accepted them as reference matches. [Tests-only CI #42362024765](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42362024765) demonstrated precisely this additional negative failure on both OS, while the original 16+16 retained GREEN. `mikazuki/parameter_policy_evidence_coverage.py` now excludes those inputs from shape coverage without introducing new runtime hard errors.
- **Corrected source HEAD `ff71a6ba2888eec8f8fb9ff78b4ba6463198ae7c`:** [push #42362074019](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42362074019) and [PR #42362080270](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42362080270): each OS **42/42 PASS** (original 16 + SR1 16 + D1 helper 10), zero FAIL/ERROR/SKIP/xfail/xpass. [Anima Qwen3 PR #42362080267](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42362080267) remains a separate whole-suite and runtime-smoke check. This audit entry is documentation-only and must be followed by exact-final-head CI confirmation.

**HOLD:** New Host output is advisory only; it is not wired to Preview/Start/Direct Trainer. Actual live BF16 Tensor/device/optimizer group/Checkpoint checks remain independently enforced in legacy code. No physical CUDA smoke was run here, no release-row mutation or credential issuance, no independent P1-C original Checkpoint Load, no user-visible Full-BF16 unlock. PR #68 stays Draft stacked on PR #67 until exact-final-head independent Strict Review.
