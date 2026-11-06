# DTS Qualification Gate Architecture Review — execution eligibility vs evidence certification

**Review date:** 2026-11-06  
**Review baseline:** `main@1dc4f56fb862aade574add48c4f1bea2c54f3e96`; optional evidence harness [Draft PR #65](https://github.com/WwlWss/diffusion-trainer-studio/pull/65) at `1033105789c8cfaac910a6490d521dfd46a41568` (unmerged).  
**Decision:** **redirect development toward making reachable training modes usable, without fabricating official evidence**. Defer protected-OS Runner B3/P1-C as mandatory normal-training release blockers. **This document does not change any runtime source, certificates or existing gates.**

## 1. What is actually complete: distinct physical evidence scopes

| Capability | Physical evidence already obtained | Scope boundary |
| --- | --- | --- |
| Shared Full-BF16 AdamW/Muon and explicit AdamW fallback | [PR #56](https://github.com/WwlWss/diffusion-trainer-studio/pull/56): actual RTX 4090 Laptop C1–C4 CUDA 8/8 PASS | Approved shared optimizer cases; not all models/optimizers |
| SD1.5 stock LoRA Full-BF16 | [PR #57](https://github.com/WwlWss/diffusion-trainer-studio/pull/57): real AdamW and Muon+fallback Fresh Step-1, save, new-process resume Step-2 PASS | SD1.x reviewed baseline, not SD2 or arbitrary variants |
| SDXL Base stock LoRA Full-BF16 | [PR #58](https://github.com/WwlWss/diffusion-trainer-studio/pull/58): final user-run simpler-console rerun of **both AdamW and Muon+fallback Fresh/Resume PASS**; initial Rich logging `MemoryError` in the older PR description was subsequently addressed by a successful rerun | Scoped SDXL Base baseline, not arbitrary geometry/TE gradient checkpointing |
| SD3.5 Medium normal/mixed-BF16 LoRA | [PR #60](https://github.com/WwlWss/diffusion-trainer-studio/pull/60): actual production Trainer real CUDA Fresh/Step-1 save/new-process resume Step-2 PASS; [PR #61](https://github.com/WwlWss/diffusion-trainer-studio/pull/61): WebUI API Preview→Start→Fresh→Resume GPU PASS | Proven normal training and scoped epoch-boundary single-process Resume; not Full-BF16 R1/R2 or general mid-epoch/DDP |
| SD3.5 Medium Component mixed-BF16 | [PR #62](https://github.com/WwlWss/diffusion-trainer-studio/pull/62): real CUDA R0 526 trainable /789 total LoRA tensors, optimizer state and Step-1→Step-2 PASS | Component routing in mixed BF16; not physical true Full-BF16 trainable parameter/gradient R1/R2 |
| SD3.5 Medium Component true Full-BF16 | [PR #63](https://github.com/WwlWss/diffusion-trainer-studio/pull/63) scoped R1/R2 Host/CPU contract and [PR #65](https://github.com/WwlWss/diffusion-trainer-studio/pull/65) independent evidence-worker Host/CPU | **No completed direct physical R1/R2 candidate qualification**; draft Worker original `Accelerator.load_state` deliberately rejects |

**Important:** The existing production SD3 LoRA resume in PR #60/#61 is a different flow from the deliberately denied **evidence-worker original Load** under PR #65. Training feature completion is scoped: real training works in the listed modes; other untested variants must not be called proven, but the absence of a certificate does not automatically make them impossible.

## 2. Source-level root causes

### F1: one overloaded blocker list binds release certificates to user execution

- `mikazuki/parameter_policy_execution.py:169–275`: `FULL_BF16_BACKEND_QUALIFICATIONS` and optimizer statuses include `pending` rows (e.g. `sd3-lora`, many `bitsandbytes`/Lion/ScheduleFree variants).
- `parameter_policy_execution.py:419–433,479–598`: an absent or pending certification becomes a fatal string blocker; backend early-return at ~533–541 prevents later scope/asset reporting. **A failed certification is not the same as a known runtime failure**.
- `mikazuki/parameter_policy.py:543–610` aggregates this with legitimate semantic blockers. `mikazuki/training_request.py:223–236` makes Start raise the first blocker; `mikazuki/parameter_policy_trainer.py:1640–1653` makes direct Trainer raise before parameter scan. `parameter_policy_trainer.py:981–1007` also treats the lack of distributed-mode qualification as a runtime error.
- Standard mode and inactive `full_bf16` do not consult these certificate tables; **this specific fault is concentrated on active Component execution-feature admission**, not all DTS training.

### F2: R1/R2 test-vector geometry has been incorrectly promoted into product eligibility

`mikazuki/sd3_full_bf16_scope.py:25–100,146–214,282–364` freezes benchmark-specific constraints: exact trainable components, optimizer profile argument dictionaries, `network_dim=4`, `network_alpha=4`, learning rate `1e-4`, `train_batch_size=1`, 512×512, accumulation 1/2, `max_data_loader_n_workers=0`, bucket disabled, and many optional runtime flags. These are appropriate as **qualification matrix selectors**, not automatically as universal proof that every other configuration cannot train.

On the current path the backend `pending` gate returns before this later R1/R2 validation even executes, obscuring which configuration failures are real.

### F3: some existing blocks genuinely represent missing/unsafe runtime semantics

- `mikazuki/parameter_policy_matrix.py:69–129`: fused/DeepSpeed/multi-GPU ownership and unsupported semantic features.
- `mikazuki/parameter_policy_compat.py:203–340`: quoted boolean truthiness hazards, FP8/full-FP16, fused optimizer ownership, offload/block swap and compiler/optimizer identity.
- `mikazuki/optimizer_profiles.py:39–208`: `supported`, `restricted`, `planned` describe **Component implementation capability**, not Full-BF16 physical evidence; adaptive DAdaptation/Prodigy and Custom/CAME cannot be blindly unlocked.
- `mikazuki/parameter_policy_trainer.py:950–1027,1310–1333`: real tensor BF16 dtype, live-root identity, optimizer topology/device and resume-state invariants must remain strict.
- Model/asset validation, text-encoder caching conflicts, pre-load forward-semantics version checks and corrupt/incompatible Checkpoint must remain hard blockers.

### F4: trust attestation scope is overextended into local-workflow priorities

PR #65's `tools/sd3_full_bf16_worker_step_adapter.py:755–796` rejects original evidence-worker Resume until independent Fresh-parent authority is proven. This is reasonable for a **cryptographically attested third-party/evidence-worker certificate**, but unrelated to a user restoring their own local Checkpoint via the existing production trainer. Requiring distinct Windows/Linux protected service principal, signer private key and durable nonce ledger before normal local training would be a product-level category mistake.

## 3. Accepted architecture: separate three dimensions

**Execution capability:** `implemented`, `experimental_candidate`, `not_implemented`, `known_incompatible`. Assigned from actual source support, dependency/optimizer ownership and runtime invariants, never from the presence of a certificate alone.

**Runtime decision:** `ALLOW`, `WARN`, `BLOCK`. BLOCK when the implementation cannot execute with the selected semantics or would violate hard data/runtime/Checkpoint invariants. WARN when a reachable, runtime-checked feature has insufficient physical coverage; user may intentionally start an experimental job. `unverified` is never, by itself, a sufficient reason for BLOCK. Lack of implementation or unassessed high-risk ownership changes is **not** automatically a reachable experimental candidate.

**Evidence tier:** `formally_qualified`, `observed_real_gpu`, `host_cpu_checked`, `unverified`, `failed`. These describe *what may be claimed*, not whether a viable user job may be attempted. There must be **no** hidden `status='qualified'` mutation or artificial CUDA result when the user overrides an advisory warning.

Suggested immutable host-side result (illustrative, not yet an ABI):

```python
ExecutionReadiness(
    decision="warn",                       # allow | warn | block
    capability="experimental_candidate",   # not a certification
    hard_errors=(),                         # typed reason codes
    warnings=("full_bf16.sd3_r1_r2_not_physically_qualified",),
    evidence_tier="unverified",
)
```

The **stable execution identity/checkpoint compatibility schema must not depend on evidence tier, UI warning acknowledgment or current qualification metadata**. Record user acknowledgment and test coverage in separate run/audit metadata, not a fabricated trusted attestation or a persistent release-row mutation.

## 4. Preliminary classification matrix — subject to traced code and tests

| Gate | Proposed behavior | Why |
| --- | --- | --- |
| Invalid quoted booleans; contradictory BF16/FP16; missing assets; unsupported model/module; wrong Component/optimizer profile ownership; corrupted or incompatible checkpoint; trainable tensor dtype/device/root identity mismatch; known cached-TE/trained-TE conflict | **BLOCK** | Runtime/semantic or state-integrity error, not merely missing certification |
| Optimizer/profile actually unimplemented (`planned` Custom/CAME), missing adaptive/schedule-free routing support, fused optimizer operation with no equivalent Component implementation | **BLOCK** | There is no validated executable semantic contract |
| Multi-GPU/DDP, DeepSpeed, FP8 base, Full FP16, compiled/wrapped model when ownership changes, active offload/block swap | **HOLD as BLOCK until targeted implementation audit** | Some can break parameter ownership, optimizer state or checkpoint. Unverified alone is insufficient, but neither is it evidence they safely work |
| Backend `pending` solely because no exact-head GPU certificate, when actual production execution/dtype invariants are implemented | **WARN + allow experimental candidate** | Certification status should not dictate user execution permission |
| R1/R2 example-specific learning rate, rank/alpha, 512 resolution, bucket, batch size, worker count, VAE dtype and other actually supported training values | **WARN if test scope exceeded**, runtime/resource errors remain real errors | Normal trainer may support the configuration; benchmark coverage is narrower |
| `AdamW8bit`, Paged variants, Lion and ScheduleFree Full-BF16 | **Per-optimizer capability audit** | `component_support=supported` does not prove BF16 optimizer state/step/resume, yet missing certificate alone is not a failure |
| SD3 ordinary mixed-BF16 Fresh→Resume | **ALLOW** when existing PR #60/#61 conditions hold | Actual production GPU path already passed; unrelated evidence-worker signature must not gate it |

## 5. New roadmap: independent from Draft PR #65

| Step | Work | Required acceptance |
| --- | --- | --- |
| **QG-0** | Freeze complete blocker inventory, source/feature/affected mode, rationale and classification; distinguish confirmed failures from mere untested cases | No production change; identify unsafe and uncertain gates explicitly |
| **QG-1** | Tests-only RED → minimal GREEN for pure `ExecutionReadiness` decision contract | No change to old execution identity or official qualification records; deterministic typed hard errors vs warnings vs evidence tier |
| **QG-2** | Preview/Export/Start and direct Trainer parity; GUI/headless per-job experimental warning acknowledgment | Ordinary baseline unchanged; experimental *reachable* path runs without cheating existing hard guards; no hidden environment-variable bypass |
| **QG-3** | SD3 Component true Full-BF16 candidate audit and controlled experimental enablement on **ordinary production trainer**, starting with AdamW and Muon+explicit fallback | Real BF16 params/grads, optimizer ownership, memory/dtype/model assets; preserve existing Resume/forward semantics guards |
| **QG-4** | Actual GPU R1/R2 short Fresh→Save→new-process Resume→Step-2 smoke on user's 16 GB-class GPU, and WebUI API path | Explicit `--full_bf16 --mixed_precision bf16`, genuine optimizer/scheduler state and LoRA changes; no substitution of mixed-BF16 R0 or CPU-only synthetic data |
| **QG-5** | Independent strict review, batch/rank/resolution variant release based on actual support, UX docs and regressions | Known invalid still blocked; unverified-but-runnable warns; qualification not spoofed; SD1/SDXL and SD3 mixed path regressions stay GREEN |
| **QG-6** | Optional externally attested evidence Runner / protected OS services (B3/P1-C) | Deprioritized, separate from normal local training release, only if users actually need independent cross-principal proof |

### Implementation/merge rules

1. Begin a **new source branch from `main`**, not from PR #65. PR #65 has >80 changed files and numerous deliberately RED external-attestation tests; do not merge as one release dependency.
2. Preserve existing test evidence and already-merged training/resume code. Do not delete safety tests or globally allow unsupported optimizers/FP8/DeepSpeed/multi-GPU just to obtain GREEN.
3. Advisory warning authorization must never grant `backend_qualification_eligible`, `physical_cuda_qualified`, signed evidence, protected Runner status or `worker_complete`. Formal certification keeps its own exact-head provenance gate and physical scope.
4. Standard and mixed-precision modes must remain unaffected. Direct Trainer must not become an accidental bypass for actual invariants while Preview/Start changes.
5. The normal production Resume path must require compatible actual loaded state/forward metadata and avoid asserting true RNG restoration solely from the return of `Accelerator.load_state`.

## 6. Acceptance and next immediate action

**QG-0 → QG-1** is the next development milestone. First author *tests-only* that distinguish identical runnable semantics under different certificate statuses (same execution identity, different evidence tier); maintain the old failing corpus for genuinely unsupported/invalid semantics; then implement the pure decision engine and both host entrypoint integrations in small reviewed steps.

**Roadmap change:** Do not proceed to mandatory protected OS Runner B3, P1-C independent signature issuer, C4-C2 evidence-worker original Load, C4-D or full CUDA certification as prerequisites for users running supported SD3.5 Medium Component training. Those are optional formal evidence goals and may continue later as a separate project track.

**Status of this document:** Architecture review and revised roadmap only. No production execution behavior is changed; true SD3 Full-BF16 experiment availability requires subsequent QG source work and real GPU verification.
