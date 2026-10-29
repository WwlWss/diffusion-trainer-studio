# D1-SD3C2-B0 — SD3.5 Medium Component Full-BF16 real-CUDA evidence contract

> **Status:** DESIGN / HOST REVIEW PENDING; implementation and CUDA execution have not started.
> **Contract baseline:** `main@1dc4f56fb862aade574add48c4f1bea2c54f3e96` (merged PR #63, with PR #64 ancestor).
> **Release status:** `FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"].status == "pending"`; B0/B1/B2/B3 must not silently promote it.
> **Purpose:** define the bounded, production-trainer-based evidence infrastructure for two **candidate** SD3.5 Medium Component Full-BF16 cases. A host or synthetic PASS is not physical CUDA qualification.

## 1. Authority, history, and immutable boundaries

| Stage | Evidence authority | Current state |
| --- | --- | --- |
| D1-SD3A, PR #59 | `docs/parameter-policy-sd3-architecture-audit.md`; strict SD3.5 Medium MMDiT header and target audit | merged |
| D1-SD3B, PR #60/#61 | `docs/parameter-policy-sd3-runtime-correctness.md`; real mixed-BF16 Forward / fresh-save / epoch-boundary Resume | merged |
| D1-SD3C1, PR #62 | `docs/parameter-policy-sd3-component-baseline.md`; R0 production Component mixed-BF16 AdamW CUDA | merged |
| D1-SD3C2-A, PR #63 | `docs/parameter-policy-sd3-full-bf16-scope.md`; pure R1/R2 host admission and asset gate | merged; backend pending |
| Shared safety, PR #64 | `mikazuki/parameter_policy_runtime_bools.py` and early direct-trainer preflight | merged |
| **D1-SD3C2-B0** | This plan, generator, runner, worker, tests | planning; no CUDA claim |

The following are **sources of truth**, rather than a second runtime implementation:

- `mikazuki/sd3_full_bf16_scope.py`: exact R1/R2 policy, configuration and physical asset admission.
- `mikazuki/parameter_policy_execution.py`: shared Full-BF16 backend/optimizer qualification gate, Execution Identity v1.
- `mikazuki/parameter_policy_trainer.py`, `mikazuki/parameter_policy_torch.py`: live routing, CompositeOptimizer/CompositeLRScheduler, Manifest v2, Checkpoint Progress v1, state restoration.
- `scripts/dev/sd3_train_network.py` and `scripts/dev/train_network.py`: **only** production Forward/Backward/Optimizer/Save/Resume training loop.
- `scripts/dev/library/sd3_resume_cursor.py` and SD3 forward-semantics hooks: real single-process epoch-boundary Resume.
- `tools/run_sd3_runtime_gpu_evidence.py`, `tools/run_sd3_component_gpu_evidence.py`: reusable **generic** source/Checkpoint/LoRA comparisons; R0's forbidden Full-BF16 semantics stay unchanged.
- `tools/parameter_policy_execution_gpu_support.py::temporary_execution_qualification` and `tools/parameter_policy_execution_gpu_runtime.py`: local-process temporary qualification and Torch fingerprint helpers.
- `tools/parameter_policy_backend_feature_gpu_support.py`: generic Full-BF16 Manifest v2 / Progress v1 validation, **not** the SD1/SDXL-only stock LoRA case registry.

**No production code is modified in B0:** no changes to `FULL_BF16_BACKEND_QUALIFICATIONS`, optimizer eligibility, scheduler, model forward, actual training loop, SD3 Resume hooks, UI, or R0 runner. No permissive global environment flag, `--qualification_override`, synthetic CUDA evidence, or stock SD1/SDXL `D1_STOCK_LORA_SPECS` shortcut.

## 2. B0 deliverables and phase gates

| Artifact | Responsibility | B0 exit criterion |
| --- | --- | --- |
| `docs/d1-sd3c2b-plan.md` | normative implementation/evidence/negative-test contract | reviewed |
| `tools/prepare_sd3_full_bf16_cases.py` | deterministic R1/R2 dataset+policy+manifest generator, entirely outside repo | Host fixtures pass |
| `tools/run_sd3_full_bf16_gpu_evidence.py` | pure preflight, exact-head/file gates, orchestrate two fresh Python processes, validate post-run proof | Host simulated fixtures pass |
| `tools/sd3_full_bf16_evidence_worker.py` | **sole** scoped in-process qualification lease and actual production Sd3NetworkTrainer execution, read-only instrumentation | import/dispatch/denial Host tests pass |
| `tests/test_sd3_full_bf16_gpu_evidence.py` | contract, generator, parser, fake event, tamper, isolation, resume and child-state negatives | focused Host CI pass |
| `.github/workflows/anima-qwen3-review.yml` | explicit path triggers and focused test invocation | exact-head CI pass |

The B0 implementation PR may initially add helpers/tests alongside this plan, but must be independently strict-reviewed before running CUDA. B0 acceptance **does not** require or claim `R1 CUDA PASS` or `R2 CUDA PASS`.

Sequence: **B0 Contract Review → B0 RED tests → B0 implementation → B0 independent Strict Host Review and CI → B1 R1 CUDA → B2 R2 CUDA → B3 same-HEAD regression and conditional release review.**

## 3. Exact candidate cases

| Contract | R1 — AdamW baseline | R2 — Muon primary + independent AdamW fallback |
| --- | --- | --- |
| Model | loader-compatible complete `sd35-medium` MMDiT | same |
| Trainable | `mmdit.attention.adapter`, `mmdit.mlp.adapter`, `mmdit.modulation_norm.adapter` | same |
| Frozen | `mmdit.other.adapter`, `clip_l.adapter`, `clip_g.adapter`, `t5xxl.adapter` | same |
| Optimizer profiles | exactly one AdamW `args={"weight_decay":0.01}` | exactly Muon `args={}` and AdamW `args={}` |
| Routes | all three primary AdamW, no fallback | all three primary Muon; **explicit** AdamW fallback on every trained route |
| Learning rates | all primary `0.0001` | all primary and fallback `0.0001` |
| Grad accumulation | `1` | `2` |
| Effective DataLoader batches / epoch | exactly `1` | exactly `2` |
| Update / epoch | `1` | `1` |
| Physical optimizer children | one nonempty AdamW | nonempty, disjoint Muon and AdamW |
| Training target | absolute total `max_train_steps=2` in **both** processes | same |

Shared execution constraints: `networks.lora_sd3`; rank/alpha 4/4; resolution exactly 512x512; batch 1; one CUDA device and one process; `mixed_precision=bf16` **and** `full_bf16=true`; constant scheduler; `save_model_as=safetensors`; `save_state`, save every optimizer step; `cache_latents` and `cache_text_encoder_outputs` true; zero DataLoader workers; `enable_bucket=false`; no trainable encoder adapter, network/base preload, arbitrary network arguments, alternate datasets, FP8, Full-FP16, gradient checkpointing, offload/lowram/swap, DeepSpeed, compile, fused paths, non-reference VAE precision, or other unreviewed mode. The **full** additional flag/field constraints are owned by C2-A and should be invoked, not reimplemented loosely.

The R2 dataset needs **two actual** batches for each one-step epoch. `gradient_accumulation_steps=2` with a one-batch DataLoader is not valid proof.

**Scheduler identity caveat:** `_scheduler_identity_from_args` binds `max_train_steps`. Fresh and resumed commands must both request **total 2** steps. As in the accepted R0 lifecycle, the Fresh process may finish Step 2; the authoritative handoff is **its Step-1 state**, resumed in a newly launched process to produce a separate Step-2 state. Do not change Fresh to `max_train_steps=1` solely to make it stop early: that changes Scheduler Identity.

## 4. Evidence worker trust boundary

### 4.1 Process layout

`Host generator (no trainer) → Host runner (strict preflight) → fresh Worker Python process → production Sd3NetworkTrainer.train(args) → Step-1 state + observed proof → Host checkpoint verification → **new** resume Worker Python process → production Sd3NetworkTrainer.train(args) → Step-2 state + restored/advanced proof → Host final verifier`.

The runner does **not** apply the qualification lease to itself and then spawn `scripts/dev/sd3_train_network.py`: qualification tables are in-memory, and this would leave the real child process at `pending`. Instead, each **worker**, inside its own process, establishes the lease before invoking the actual trainer.

### 4.2 Restricted entry and temporary lease

1. Host runner verifies exact expected 40-hex Git HEAD, clean tree including relevant submodules, fixed worker module path, manifest schema/case, immutable input hashes, and fresh output absence. It creates an **one-shot per-phase ticket** under the external work directory with case, phase, expected SHA, manifest/input-contract digest and dedicated evidence identifier. Ticket files are never loaded from or written into repo.
2. Worker rejects direct invocation without a valid runner-created ticket, unknown case/phase, reused ticket, source/manifest drift, and any mismatch with the strict generated trainer command. The worker has no arbitrary script/entrypoint, arbitrary qualification list, or free-form forwarded option mechanism. Tickets are accidental-bypass prevention, **not** a cryptographic sandbox against a local user able to rewrite code/files.
3. Worker builds real args using the **production SD3 parser and config normalization** (no `--config_file` indirection). It runs C2-A's pure config and strict physical asset checks on the resolved effective Namespace. It rejects unreviewed defaults even if not present in raw CLI.
4. Immediately before production `Sd3NetworkTrainer.train(args)`, worker checks backend row is exactly `pending` and applies `temporary_execution_qualification(backend="sd3-lora", optimizers=("AdamW",)` or `("Muon","AdamW")`, `evidence_case_id=<fixed B candidate evidence ID>)`. Only the backend pending record receives a temporary in-memory lease; globally already-qualified AdamW/Muon records retain their identities.
5. The worker invokes **only** the real `Sd3NetworkTrainer` (or a subclass that exclusively installs observation callbacks and delegates the inherited real training method). No duplicated Forward/Backward/step/save/Resume code is allowed. `try/finally` restores original wrapped methods/observations and temporary registry; the final event reports original and restored qualification snapshots.
6. Fresh and Resume each use **new `sys.executable` processes**. Child exit status and state proof must be checked by the host; worker `status="pass"` alone is insufficient.

Neither WebUI nor ordinary CLI imports the evidence Worker or sees the lease. `sd3-lora` continues to return its production pending blocker when launched normally. Any new production-facing environment toggle, import-time monkeypatch, or persistent registry edit is a **P0 rejection**.

### 4.3 Observation seam constraints

Prefer a Worker-only observation subclass of `Sd3NetworkTrainer`:

- `configure_parameter_policy_training`: call `super`, retain the actual `ParameterPolicyTrainerSession` and physical CompositeOptimizer instance, audit resolved routing/physical parameter identities. Never change train/freeze or optimizer construction.
- `register_additional_state_hooks`: call `super`; register strictly observational callbacks or wrap the **instance** `accelerator.backward` only while running to capture **post-backward, pre-step/pre-zero** gradients, and record actual train and optimizer/scheduler states at the Step-1 save boundary. Do not replace or skip the existing SD3/Parameter Policy Accelerate hooks.
- `on_step_start`: call `super`, count real training DataLoader microbatches; bind a stable dataset/caption sample identity where available. A call with `is_train=false` is excluded. Do not rely on synthetic iterations.
- `plan_backend_resume_cursor`: call `super` after real `accelerator.load_state` and post-resume audit; before Step-2 training, snapshot **actually restored** child optimizer, scheduler and adapter parameters. Assert saved Step-1 handoff matches restored state.
- A Worker-only observation around the real `CompositeOptimizer.step`/Accelerate step path can record pre/post physical child state, but must call the original exactly once and preserve the Accelerator's accumulation/no-op behavior. If a proposed interception changes optimizer/scheduler step counts or causes an extra step, reject it.
- Keep large tensor payloads out of JSON. Use audited, hash-based fingerprints/counters and bounded failure diagnostics. Atomic write to an external evidence directory; no trainer parameter mutation by observation.

The source's physical loop order is `accelerator.backward(loss) → optional grad clipping → optimizer.step() → scheduler.step() → optimizer.zero_grad()`. The gradient evidence must be captured after **that real** `accelerator.backward`, not after zero-grad or merely from a mocked graph.

The exact hook implementation/order is a **blocking B0 source-review item**: demonstrate that the worker observes the optimizer that Accelerate actually owns (not a stale pre-`prepare` object), and does not suppress/duplicate an inherited production hook.

## 5. Generator, external inputs and content identity

The generator must use the C2-A canonical contract, rather than handcrafting a visually similar but semantically different policy. Reuse `validate_parameter_policy` and `serialize_parameter_policy`; R1 can derive the reviewed `prepare_sd3_component_baseline.make_policy()` and R2 can derive exactly the host-verified shape in `tests/test_sd3_full_bf16_scope.py::r2_policy`, with production-owned definitions preferred over importing test helpers at runtime.

External layout, **never inside the source checkout**, for example:

`<new-work>/r1/{data,policy.json,manifest.json,fresh/,resume/,events/,result.json}`
`<new-work>/r2/{data,policy.json,manifest.json,fresh/,resume/,events/,result.json}`

R1: one valid 512x512 sample with its corresponding caption. R2: **two distinguishable** 512x512 real samples plus captions, exactly one repeat each in one ordinary DreamBooth subset (e.g. `1_dts_sd3c2b`). Produce isolated copies/hardlink-free materialization rather than modifying R0 data; reject duplicate physical file/caption identities when distinct cases are required. Restrict to recognized image extensions and the one-caption-per-image format the production Dataset parser really consumes; reject extra images/subfolders/`.npz` or cached-output files that could modify the effective dataset. No alternate `dataset_config`, training/validation split or regularization images.

The generator should **not infer effective batches from image count**. Its inventory is an input precondition only. The Worker and checkpoint `dts_sd3_resume_progress.json` must physically prove `dataloader_batches == 1` (R1), `== 2` (R2), and `num_processes == 1`.

### 5.1 Hash contract

Compute streaming SHA256 of the **whole bytes**, not header/size/mtime alone, for:

- base SD3.5 Medium checkpoint;
- whichever external CLIP-L, CLIP-G, T5XXL and VAE files are needed, or record explicit embedded-source identity tied to the base file digest;
- canonical serialized Policy bytes **and** canonical Policy hash;
- every final input image and UTF-8 caption, plus sorted path→digest inventory and dataset-level signature;
- manifest body excluding any self-referential digest field; command/normalized effective options;
- exact source HEAD and all in-use Git submodule gitlink/checkout identities.

Source/case/input hashes are bound **before Fresh**, rechecked **after Fresh/before Resume**, and **after Resume**. Reject changed bytes, missing files, symlinks/realpath escape, duplicate logical paths, model/path swaps, or overwritten output evidence. Preflight may be I/O intensive for large checkpoints; it must not silently replace whole-file hashing with the prior SD3A shape index (which proves topology only).

The source audit validates `sd35-medium` full DTS-loader-compatible MMDiT topology (24 blocks, hidden 1536, 24 heads/head-dim 64, RMS QK-norm, 384 position grid, extra-attention on blocks 0–12). This does **not** prove official release provenance; the external full-file hashes establish only identity of the **tested local files**. **Important release boundary:** C2-A production admission is architecture/asset-based, not an exact official-weight SHA allowlist. B3 must not claim that tested digests automatically restrict future production checkpoints; if release is intended to cover *only* a particular model weight release, a separately reviewed production digest gate is needed.

## 6. Manifest and proof schema v1

Recommended independent schema constants (B0 final review may freeze spelling):

- Input: `schema="dts.sd3c2b.full-bf16-case-manifest"`, `version=1`.
- Worker phases: `schema="dts.sd3c2b.full-bf16-worker-phase"`, `version=1`.
- Aggregate: `schema="dts.sd3c2b.full-bf16-gpu-evidence"`, `version=1`.
- Candidate cases: `r1-adamw-accum1` and `r2-muon-adamw-fallback-accum2`. Never infer cases from optimizer names alone.
- Fixed test-only identifier: `phase-d1:sd3c2b:sd35-medium-component-full-bf16-candidates:v1`; this is **not** an already released backend evidence ID.

Minimum manifest contract:

~~~json
{
  "schema": "dts.sd3c2b.full-bf16-case-manifest",
  "version": 1,
  "case_id": "r2-muon-adamw-fallback-accum2",
  "source_commit": "<exact candidate SHA>",
  "input_identity": {
    "model_sha256": "<whole-file SHA256>",
    "assets": {},
    "dataset_files": {},
    "dataset_signature": "<SHA256>",
    "policy_bytes_sha256": "<SHA256>",
    "canonical_policy_hash": "<SHA256>"
  },
  "reference": {
    "train_type": "sd3-lora",
    "model_variant": "sd35-medium",
    "resolution": [512, 512],
    "train_batch_size": 1,
    "effective_batches_per_epoch": 2,
    "gradient_accumulation_steps": 2,
    "max_train_steps": 2,
    "optimizer_profiles": ["Muon", "AdamW"]
  },
  "fresh_command": ["scripts/dev/sd3_train_network.py", "..."],
  "resume_command": ["scripts/dev/sd3_train_network.py", "..."],
  "fresh_checkpoint_dir": "<absolute fresh Step-1 state>",
  "resume_checkpoint_dir": "<absolute resume Step-2 state>",
  "fresh_adapter_path": "<absolute fresh Step-1 safetensors>",
  "resume_adapter_path": "<absolute resumed Step-2 safetensors>"
}
~~~

The actual `...` placeholders above are **documentation-only**; a generated manifest must have fully concrete, explicit, deterministic paths and options. Canonicalization must reject duplicate switches, `--x=value` ambiguity, boolean-string pseudo-values, unexpected positional tokens, `--config_file`, wrong launcher, case/dataset mismatches, stale output files, and all unreviewed effective defaults. Fresh and Resume training options differ **only** in `--resume` and `--output_dir`; both carry `--full_bf16` and `max_train_steps=2`. Any phase-specific evidence metadata lives outside the production trainer args.

Aggregate evidence must contain:

`source_identity` (HEAD, clean-tree/submodules), `input_identity`, `cuda_environment` (GPU name/index, device capabilities, BF16 support, Python/Torch/Accelerate/pytorch-optimizer versions, process count), `architecture`, `case_id`, `policy_hash`, `parameter_ownership`, `dtype_and_gradient_evidence`, `microbatch_timeline`, `child_optimizer_state`, `scheduler_state`, `checkpoint_manifest_v2`, `checkpoint_progress_v1`, `sd3_forward_semantics`, `sd3_resume_progress`, `adapter_updates`, `qualification_lease` and phase exit statuses/log paths.

Success is a conjunction of independently verified predicates, not an uncontrolled Worker-provided `"pass"`. Default record `status="incomplete"`; any failure is `"fail"` with the original error preserved. A preflight-only result uses `"preflight_pass"`, **never** `"pass"` or `"qualified"`.

All B0/B1/B2 evidence must explicitly report `production_qualification_mutated=false` and `backend_qualification_eligible=false` while the actual registry remains pending. B3 may compute a **separate reviewed candidate eligibility summary**, never mark production released automatically.

## 7. Physical assertions — what is a real PASS?

### 7.1 Device, dtype, ownership

1. Require actual one-process CUDA with supported BF16; fail on CPU-only, fake CUDA claims, distributed execution, nonfinite losses.
2. On **prepared** training roots, every optimizer-owned, trainable parameter is genuinely `torch.bfloat16`, CUDA resident, and present in exactly one physical Child param-group. A mere `mixed_precision="bf16"` / autocast string is insufficient.
3. Audit by physical `id(parameter)` matched to actual routing descriptors and component metadata, rejecting duplicates, unknown root targets and leakage of frozen components.
4. For every trained Component, at least one parameter shows a real, finite, nonzero weight change over the saved/resumed lifecycle. The frozen four Components must have zero optimizer ownership and no observed change; verify a complete adapter key set, not a substring guess alone.
5. After actual Backward and before clearing gradients, capture every owned gradient with dtype/device/finite/nonzero markers. At minimum each Child and each trained Component must have a nonempty finite BF16 gradient during the appropriate synchronized logical step; report missing/zero rows, never silently skip them. Preserve the distinction between valid zero gradients on an individual tensor and a completely inactive component.

### 7.2 R1 AdamW

- Exactly one physical `CompositeOptimizer.entries` child, `optimizer_type="AdamW"`, with nonempty owned param set.
- Fresh Step-1 state nonempty and advanced; Step-1 checkpoint Manifest v2 and Progress v1 valid.
- A new Python process constructs a **new** network/optimizer/scheduler and loads exact Step-1 state. Its **immediate restored** optimizer, scheduler and adapter hashes must equal the saved Step-1 snapshots *before* any resumed Backward/Step.
- Then the resumed physical Step-2 state/weights advance; Progress v1 profiles report `[1] → [2]`, Scheduler advances exactly once; SD3 Cursor epoch boundary 1→2. It is insufficient to compare only whole `optimizer*.bin` hashes.

### 7.3 R2 Muon + explicit AdamW fallback

- **Two** physical nonempty children: one pinned `pytorch_optimizer.Muon` and one independent `torch.optim.AdamW`; disjoint physical IDs, union equal exactly all trainable IDs.
- Verify routing plan `route_kind="primary"` for genuinely Muon-eligible Attention/MLP 2-D hidden weights, `route_kind="fallback"` for incompatible AdaLN/Modulation (and any additional *legitimately ineligible* matrices/weights). Do not invent fixed cardinalities; derive exact expected counts from live audited SD3 targets and assert **both** classes populated.
- Every Muon group has `use_muon is True` and no internally used AdamW-fallback group/state; AdamW fallback owns ineligible targets by **different Profile**, never by a Muon internal option.
- Exactly **two physically observed training DataLoader batches per epoch**; microbatch 1 must not advance either Child, Scheduler, optimizer-step counter or adapter weights; microbatch 2 is the accumulation boundary and must advance both children and Scheduler once.
- Capture actual BF16 gradients after each Backward; no false claim from an empty gradient after zero-grad.
- After Fresh Step-1, both children possess nonempty appropriate state. Before resumed Step-2, the newly constructed children are initially fresh; after actual checkpoint load their state fingerprints **equal** saved Step-1. After resumed optimizer Step-2, **both** child fingerprints, step counters and relevant adapter weights advance.
- Verify Muon provider version/class and explicit `use_muon` topology; fallback AdamW state keys/step values; child Scheduler state per Profile; `runtime_topology_fingerprint`, `execution_identity`/signature, `scheduler_identity`/signature identical across Fresh and Resume.

### 7.4 Saved state and SD3 Resume lineage

Use the actual checkpoint directory files:

- `dts_parameter_policy_manifest.json` v2, including correct canonical Policy SHA, seven-component train/freeze set, physical child profile topology, Full-BF16 `Execution Identity v1`, and signed Scheduler Identity.
- `dts_parameter_policy_progress.json` v1: valid `checkpoint_id`, `resume_source_checkpoint_id` lineage, every child `step_values=[1]` then `[2]`, correct Scheduler `step_count`/`last_epoch`.
- `dts_sd3_forward_semantics.json`: corrected extra-attention head-dim contract; SD3.5 Medium architecture.
- `dts_sd3_resume_progress.json` plus `train_state.json`: one-process, expected effective-batch/accumulation geometry, Step/Epoch 1→2, epoch-boundary-only Resume.
- Actual Accelerate optimizer/scheduler/network/RNG state load; require exact saved→restored fingerprints before resumed optimization and separate fresh/resumed checkpoint directories.

A changed overall optimizer file digest is only **supporting** evidence; it cannot replace child-specific saved/restored/advanced assertions. Exact stochastic replay and mid-epoch sampler identity remain **out of scope**.

## 8. Host negative tests and independent review gates

B0 tests are CPU/Host; real CUDA is reserved for B1/B2. Use synthetic case fixture and fake subprocess/event streams for failure injection, but never label them CUDA.

| Group | Must reject |
| --- | --- |
| Schema/command | wrong schema/version/case; wrong entrypoint; duplicate or unknown option; `--config_file`; missing `--full_bf16`; wrong accum/batch/resolution/total steps; raw flag vs effective Namespace mismatch |
| C2-A admission | SD3 Medium/Large; incomplete MMDiT; trainable CLIP/T5/Other; wrong optimizer profile args/LR; missing explicit fallback; bucket enabled; FP8, no_half_vae, gradient checkpointing, lowram/offload/compile; quoted boolean pseudo-values |
| Source/data identity | dirty/wrong HEAD; submodule drift; missing encoder/VAE; altered whole-model/caption/image/policy content; symlink escape; duplicate R2 sample; alternative Dataset Blueprint; stale or preexisting output |
| Worker authority | direct/no-ticket call; reused ticket; forged phase/case/source; attempt to lease other backend/optimizer; lease absent before actual trainer; leaked lease after error; WebUI/ordinary CLI still blocked |
| Grad/ownership | BF16 autocast with FP32 trainable params; FP32/missing/NaN gradient; gradient sampled after zero; CPU-owned parameter; duplicate or frozen Child ownership; missing updated Component |
| Muon & accumulation | Muon Child empty; internal fallback `use_muon=false`; absent independent AdamW; only one actual R2 batch; early microbatch update; only one Child Step advances |
| Resume/checkpoints | absent/forged Manifest v2, Execution Identity, Policy Hash, child state or `Progress v1`; wrong parent checkpoint ID; divergent saved/restored fingerprint; mismatched Scheduler Identity; mid-epoch or multiprocess SD3 cursor; missing SD3 forward sidecar |
| Evidence result | partial evidence marked PASS; failed subprocess ignored; sha mismatch mid-run; false production promotion claim |

Strict Host Review checklist:

1. Review **all new code paths** for fixed worker/trainer entrypoint, ticket replay protection, import-time side effects and PYTHONPATH/code-shadowing risks; fail closed without accessing real model/CUDA in pure preflight.
2. Prove the exact observational seam is non-invasive: no extra Backward, optimizer/scheduler Step, or lost save/load hook; observation survives `Accelerator.prepare` and Resume.
3. Check each claimed physical predicate is sourced from a real runtime object or checkpoint, not solely policy metadata or hardcoded/duplicated results.
4. Run focused tests, `test_sd3_full_bf16_scope.py`, `test_sd3_component_gpu_evidence.py`, SD3 forward/resume tests and existing Parameter Policy runtime/boolean tests on the B0 commit.
5. Wire changed-file triggers **and** test steps in `.github/workflows/anima-qwen3-review.yml`; confirm full exact-head `validate` and `Parameter Policy runtime smoke` pass.
6. No GPU PASS, no release-status change, no file written inside repository by the generator/runner, no quiet lowering of requirements to accommodate 16-GB memory.

B0 is **ready for B1**, not `qualified`, only after independent Strict Host Review, complete negative cases and green exact-head CI.

## 9. Subsequent CUDA stages and release boundary

**B1 / R1:** on one clean exact candidate HEAD and pinned environment, generate **new** external evidence, preflight strict input hashes, then real RTX 4090 Laptop CUDA Fresh Step 1 → Save → new-process Resume Step 2. Require one Child AdamW, actual BF16 parameters/grads, physical save/load/Step lineage and finite nonzero changes. If OOM or a CPU-only/FP32 path occurs, record FAIL; do not secretly enable gradient checkpointing, swapping, smaller rank/resolution or fallback.

**B2 / R2:** separate new dataset and output; same HEAD, two real batches/epoch, two disjoint physical optimizer children, explicit fallback, full two-microbatch timeline, both child states restored and updated. A single batch with `accum2` is FAIL.

**B3:** on the **same final candidate SHA**, rerun D0 shared C1–C4 8/8 and SD3C1 R0, plus R1 and R2; verify tool/environment/input identities, zero pending qualifications mutated, exact-head CI, independent Final Strict Review and a signed, evidence-ID-bound summary. Historical GPU proof on an older SHA cannot be relabeled as proof for a changed candidate commit.

**Release is a distinct PR:** after approved B3 evidence, narrowly update the `sd3-lora` backend qualification record with a dedicated SD3 evidence ID while keeping C2-A's restrictive second gate. If that promotion changes HEAD, repeat the required exact-head checks and state clearly what the new commit has actually been tested against. The grant is restricted to recognized SD3.5 Medium topology + three MMDiT LoRA Components + R1/R2 geometry and optimizer shapes; it is **not** SD3/3.5-wide, encoder-training-wide or optimizer-wide Full-BF16.

### Open implementation decisions resolved at B0 Host Review

- Pick and test the **actual** interception point for prepared Accelerate optimizer state and post-backward gradient; do not ship a speculative callback that observes an unused object.
- Define unambiguous Worker/Runner phase ticket path, allowed effective argv schema, atomic result format, retry behavior and filename normalization.
- Verify the DreamBooth input directory convention for the true SD3 production Dataset Blueprint and enforce the *physical* effective batch count after loading.
- Resolve whether a future production release claims **topology-based SD3.5 Medium** (current C2-A) or exact official weight identity (would need a separately reviewed digest allowlist); never conflate whole-file evidence hashes with a production provenance gate.

**B0 stop condition:** if these decisions cannot be proven in Host tests/source review, do not advance to B1 GPU training. The safety/identity contracts take precedence over making the evidence runner green.
