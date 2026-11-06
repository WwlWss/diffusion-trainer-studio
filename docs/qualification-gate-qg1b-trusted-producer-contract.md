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
