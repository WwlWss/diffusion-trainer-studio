# Parameter Policy Development Status and Roadmap

> Status baseline: `main@9d9446ad1fd8c5bce45e6322c763043441702add`  
> This document is the engineering ledger for the Parameter Policy / Component-wise Optimization project.  
> It complements the user-facing guide in [parameter-policy.md](./parameter-policy.md) and the per-step design documents under `docs/parameter-policy-step*.md`.

## 1. Why this document exists

The Parameter Policy project has grown from a small optimizer experiment into a cross-layer training subsystem spanning:

- optimizer capability metadata;
- canonical policy configuration;
- model component registries;
- parameter identity scanning and routing;
- LoRA original-target metadata;
- multiple child optimizers behind one Accelerate-facing optimizer;
- scheduler ownership and ScheduleFree lifecycle;
- trainer integration across ten backends;
- checkpoint/resume identity;
- GPU qualification;
- GUI editing, bootstrap, presets, import/export and runtime-readiness UX;
- follow-up compatibility and lifecycle hardening.

The original step documents are intentionally implementation-focused and immutable historical records. They are not a concise answer to:

1. What was the original goal?
2. How many engineering steps were actually completed?
3. What concrete code was added in each step?
4. What does the finished subsystem do today?
5. Which claims are implementation claims, which are qualification claims, and which are still blocked?
6. What should be developed next?

This document answers those questions and should be updated whenever the release capability surface changes.

---

## 2. Project goal

### 2.1 Original requirement

DTS historically owns optimization through one global Standard-mode optimizer/LR configuration. That works for ordinary training, but it cannot directly express recipes such as:

- transformer blocks using Muon while biases/norms use AdamW;
- U-Net, Text Encoder and adapter families using different optimizers or learning rates;
- freezing individual architectural components while leaving other components trainable;
- routing optimizer-ineligible parameters to an explicit fallback optimizer;
- checkpointing and resuming a multi-optimizer topology without losing ownership identity.

The Parameter Policy project therefore introduced a second, opt-in optimization authority:

~~~text
Standard mode
    historical trainer optimizer/LR behavior

Component mode
    Parameter Policy owns:
    - Train / Freeze
    - component -> optimizer profile
    - component -> LR
    - primary/fallback routing
    - optimizer topology
    - optimizer/scheduler checkpoint identity
~~~

### 2.2 Non-negotiable safety rule

Component support must never silently change Standard behavior.

The implementation is therefore intentionally split into two paths:

- **Standard** remains the historical trainer path.
- **Component** is explicit and fail-closed when DTS cannot represent or qualify the requested runtime exactly.

A successful one-off run is not sufficient to remove a blocker. Ownership, device placement, checkpoint/resume, scheduler lifecycle and parameter routing must all remain well-defined.

---

## 3. Development chronology

The project consists of **7 main Steps**.

Because Step 6 and Step 7 were deliberately decomposed, the actual implementation contains **17 formal engineering units**:

~~~text
Step 1
Step 2
Step 3
Step 4
Step 5
Step 6A
Step 6B
Step 6C
Step 6D
Step 6E
Step 6F
Step 7A
Step 7B
Step 7C
Step 7D
Step 7E
Step 7F
~~~

After Step 7F, **9 post-release hardening PRs (#38-#46)** closed defects found by real GUI/runtime use and stricter source-level review.

### 3.1 Step / PR map

| Unit | PR | Main result |
| --- | ---: | --- |
| Step 1 | #21 | Optimizer capability registry foundation |
| Step 2 | #22 | Canonical Parameter Policy host/config contract |
| Step 3 | #23 | Model Component Profiles + pure parameter routing |
| Step 4 | #24 | LoRA original-target metadata |
| Step 5 | #25 | CompositeOptimizer + CompositeLRScheduler runtime |
| Step 6A | #26 | Shared trainer integration boundary |
| Step 6B | #27 | LoRA NetworkTrainer integration |
| Step 6C | #28 | Non-Anima full-trainer integration |
| Step 6D | #29 | Staged Anima integration |
| Step 6E | #30 | Runtime hardening + checkpoint manifest v2 |
| Step 6F | #31 | GPU matrix + baseline Component Start |
| Step 7A | #32 | Host-side editor contract |
| Step 7B | #33 | Generated Schemastery editor |
| Step 7C | #34 | Bootstrap + runtime-readiness frontend state machine |
| Step 7D | #35 | Curated presets + user guide |
| Step 7E | #36 | Ten-backend release regression closure |
| Step 7F | #37 | `lora-basic` acceptance closure |
| Follow-up | #38-#46 | Round-trip, GUI, device, cache and non-Anima hardening |

---

## 4. Step-by-step engineering record

## 4.1 Step 1 — Optimizer capability foundation

**Goal:** create a host-side source of truth for optimizer behavior before any trainer integration.

### Code delivered

Primary module:

- `mikazuki/optimizer_profiles.py`

Implemented:

- canonical optimizer names;
- Component support state:
  - `supported`
  - `restricted`
  - `planned`
- group-LR capability;
- external-scheduler vs optimizer-managed scheduler semantics;
- dependency metadata;
- optimizer implementation metadata;
- parameter eligibility policy metadata;
- dedicated Muon argument validation;
- pinned Muon provider via `pytorch-optimizer==3.10.0`;
- fail-closed runtime resolution.

### Result

DTS can reason about optimizer capabilities without constructing the optimizer.

This became the source of truth used later by routing, GUI option generation and runtime construction.

### Important design decision

Muon's internal AdamW fallback is deliberately **not** used. Parameter Policy owns fallback routing explicitly as a separate Optimizer Profile.

---

## 4.2 Step 2 — Canonical Parameter Policy host contract

**Goal:** define a versioned configuration format before routing or runtime code exists.

### Code delivered

Primary module:

- `mikazuki/parameter_policy.py`

Implemented:

- `optimization_mode = standard | component`;
- versioned canonical Parameter Policy JSON;
- named Optimizer Profiles;
- per-component Train/Freeze;
- primary optimizer profile + LR;
- optional fallback profile + fallback LR;
- structural Muon fallback validation;
- deterministic serialization;
- content-addressed sidecar paths;
- policy validation separated from runtime readiness.

### Result

Component configuration became a stable data contract rather than GUI-only state.

This also established the future import/export/checkpoint identity seam.

---

## 4.3 Step 3 — Model Component Profiles and pure routing

**Goal:** determine exactly which physical parameter belongs to which Component and optimizer route, without constructing an optimizer.

### Code delivered

Primary modules:

- `mikazuki/model_component_profiles.py`
- `mikazuki/parameter_routing.py`
- extensions to `mikazuki/optimizer_profiles.py`
- `mikazuki/parameter_policy_bootstrap.py`
- `mikazuki/parameter_policy_compat.py`

Implemented:

- backend-specific Component registries;
- higher-level Training Target Profiles;
- real parameter identity scanning with `id(parameter)`;
- alias preservation and alias-consensus checks;
- architectural component classification;
- structural parameter classification;
- Muon eligibility:
  - model-profile-approved
  - hidden-layer
  - 2D matrix weight
- primary / fallback / frozen / unavailable routing;
- exact ownership audit;
- Standard -> Component bootstrap;
- semantic compatibility blockers for Standard configurations that cannot be represented exactly.

### Result

Before optimizer construction, DTS can answer:

~~~text
parameter
 -> architectural component
 -> target availability
 -> primary eligibility
 -> fallback route if required
 -> final ownership
~~~

The router fails closed on unassigned, conflicting or unavailable ownership.

---

## 4.4 Step 4 — LoRA original-target metadata

**Goal:** make adapter parameters routable by the architecture they modify instead of guessing from generated LoRA names.

### Code delivered

Implemented metadata transport for:

- SD / SDXL LoRA;
- Flux LoRA;
- Chroma LoRA;
- SD3 LoRA;
- staged Anima LoRA.

The scanner discovers original-target metadata from the adapter/module ancestry.

### Result

A LoRA parameter can be classified by its original target module, allowing Component routes such as attention vs MLP/FFN without relying on fragile string parsing.

### Standard-mode impact

None. Metadata is additive and does not change legacy optimizer semantics.

---

## 4.5 Step 5 — Composite optimizer and scheduler runtime

**Goal:** convert a pure RoutingPlan into a real torch runtime that Accelerate can treat as one optimizer.

### Code delivered

Primary modules:

- `mikazuki/parameter_policy_runtime.py`
- `mikazuki/parameter_policy_torch.py`

Implemented:

- deterministic Runtime Spec compiler;
- one child optimizer per Optimizer Profile;
- immutable optimizer ownership topology;
- `CompositeOptimizer` facade;
- flattened live child param groups for Accelerate/GradScaler visibility;
- `CompositeLRScheduler`;
- external scheduler children;
- optimizer-managed ScheduleFree children;
- `train()/eval()` lifecycle forwarding;
- optimizer and scheduler state dictionaries;
- topology fingerprints;
- strict resume compatibility checks.

### Result

Multiple optimizers can participate in one trainer while exposing a single real torch optimizer/scheduler surface to Accelerate.

---

## 4.6 Step 6A — Shared trainer integration boundary

**Goal:** add one reusable Parameter Policy runtime seam before touching individual trainers.

### Code delivered

Primary module:

- `mikazuki/parameter_policy_trainer.py`

Bridges:

- `scripts/stable/library/dts_parameter_policy_bridge.py`
- `scripts/dev/library/dts_parameter_policy_bridge.py`

Implemented:

- lazy runtime import;
- policy-sidecar loading;
- runtime blocker evaluation;
- model root scanning;
- RoutingPlan -> Runtime Spec -> CompositeOptimizer orchestration;
- `requires_grad` ownership contract;
- scheduler construction seam;
- runtime diagnostics;
- Standard-path isolation.

### Result

Trainer-specific code only needs to provide model roots and lifecycle hooks; it does not reimplement policy validation or optimizer construction.

---

## 4.7 Step 6B — LoRA NetworkTrainer integration

**Goal:** integrate Component ownership into the shared NetworkTrainer lifecycle.

### Backends

- `sd-lora`
- `sdxl-lora`
- `flux-lora`
- `chroma-lora`
- `sd3-lora`

### Code delivered

Implemented:

- bypass of legacy optimizer construction in Component mode;
- policy-owned U-Net/transformer/TE train flags;
- per-Text-Encoder train state;
- CompositeOptimizer / scheduler preparation;
- LR logging;
- checkpoint integration;
- Text Encoder cache safety.

### Result

LoRA backends can run real Component-owned training rather than only previewing routes.

---

## 4.8 Step 6C — Non-Anima full-trainer integration

**Goal:** bring full-training backends under the same ownership contract.

### Backends

- `sd-dreambooth`
- `sdxl-finetune`
- `flux-finetune`

### Code delivered

Implemented:

- DreamBooth ownership integration;
- disabling legacy dynamic TE-stop ownership in Component mode;
- SDXL structural TE freezes before Runtime Spec construction;
- Flux full-trainer ownership;
- shared optimizer/scheduler lifecycle;
- policy-owned clipping/logging.

### Result

The shared Component runtime is no longer LoRA-specific.

---

## 4.9 Step 6D — Staged Anima integration

**Goal:** integrate Anima without mutating the pinned sd-scripts submodule.

### Code delivered

Implemented:

- content-addressed `parameter_policy_runtime` staging feature;
- composition with:
  - Qwen3 joint-training patch;
  - Multi-Caption patch;
  - LoRA target metadata patch;
- staged DTS runtime bridge;
- Anima LoRA integration;
- Anima full-finetune integration;
- DiT/Qwen ownership;
- policy-aware Text Encoder cache behavior;
- checkpoint/runtime contracts.

### Result

Anima participates in the same Component ownership model while preserving isolated upstream staging.

---

## 4.10 Step 6E — Runtime hardening

**Goal:** make the runtime auditable and resume-safe rather than merely runnable.

### Code delivered

Implemented:

- unified `finalize_after_prepare()` lifecycle;
- post-resume and epoch-start runtime assertions;
- optimizer ownership audit;
- device audit;
- `requires_grad` audit;
- checkpoint manifest v2;
- deterministic policy/runtime topology identity;
- component/profile/tensor diagnostics;
- model metadata;
- save/load hook assertions.

### Result

The runtime can prove that the live optimizer owns exactly the routed parameters and that checkpoint topology matches the requested policy.

---

## 4.11 Step 6F — GPU matrix and baseline Component Start

**Goal:** move from internal runtime implementation to an explicitly releasable backend matrix.

### Code delivered

Primary modules/tools:

- `mikazuki/parameter_policy_matrix.py`
- `tools/run_parameter_policy_gpu_matrix.py`
- `tools/run_parameter_policy_backend_gpu_matrix.py`
- `.github/workflows/parameter-policy-gpu-matrix.yml`

Implemented:

- exact ten-backend runtime matrix;
- Component Start gate;
- qualification blocker registry;
- semantic blocker registry;
- explicit single-GPU restriction;
- synthetic real-CUDA optimizer/runtime matrix;
- real-backend fresh/resume harness;
- machine-readable evidence JSON;
- read-only validation before launch side effects.

### Result

Baseline Component Start is opened only for the ten integrated backends and only when no runtime/semantic blocker is active.

---

## 4.12 Step 7A — Host-side editor contract

**Goal:** make the policy editable through stable host APIs instead of hand-editing JSON.

### Code delivered

Primary module:

- `mikazuki/parameter_policy_editor.py`

Implemented:

- registry-derived editor metadata;
- supported optimizer choices;
- backend Component metadata;
- exact Standard -> Component bootstrap;
- preservation of complete imported Component policy;
- model-free Preview helper;
- literal normalization for profile args.

### Result

The GUI has a stable host contract independent of handwritten per-backend policy forms.

---

## 4.13 Step 7B — Generated Schemastery editor

**Goal:** expose the host contract in the existing DTS GUI.

### Code delivered

Primary module:

- `mikazuki/parameter_policy_schema.py`

Implemented:

- generated Optimization Mode UI;
- dynamic Optimizer Profiles;
- backend-specific Component rows;
- Train/Freeze;
- primary/fallback profile and LR;
- supported/restricted/planned optimizer presentation;
- dedicated typed Muon controls;
- idempotent schema wrapping.

### Result

Component policy became a first-class GUI workflow.

---

## 4.14 Step 7C — Bootstrap and runtime-readiness UX

**Goal:** make mode switching safe under asynchronous frontend state.

### Code delivered

Primary frontend overlay:

- `mikazuki/frontend_training_patch.py`

Implemented:

- Standard -> Component bootstrap request;
- bootstrap generation guards;
- Preview debounce/generation guards;
- runtime-ready state;
- Start disabled until current Preview is green;
- stale Preview invalidation after policy edits;
- import/history/preset mode synchronization;
- backend Start validation remains authoritative.

### Result

The frontend no longer treats a syntactically populated Component form as equivalent to a runtime-ready policy.

---

## 4.15 Step 7D — Curated presets and user documentation

**Goal:** make the feature discoverable without turning every Standard preset into a Component duplicate.

### Code delivered

Four curated Component examples:

1. Flux Full — Muon + AdamW fallback;
2. Anima Full — DiT Muon + AdamW fallback;
3. SDXL Full — split U-Net/Text Encoder LR;
4. SDXL LoRA — selective U-Net components.

Also delivered:

- exact Component preset merge semantics;
- preservation of unrelated model/data/output state;
- user guide: `docs/parameter-policy.md`.

### Result

The subsystem gained runnable examples and an end-user description.

---

## 4.16 Step 7E — Full release regression closure

**Goal:** prove all layers still agree after the editor and release workflow were added.

### Code delivered

Primary release-level matrix:

- `tests/test_parameter_policy_step7_contract.py`

Implemented:

- canonical ten-backend fixture table;
- Standard missing/explicit/stale-policy equivalence;
- Standard -> Component bootstrap;
- policy sidecar generation;
- Component Preview;
- rehydrate;
- second Preview;
- exact semantic round-trip;
- backend/profile/editor registry closure;
- blocker taxonomy freeze.

### Result

The project gained a release-level cross-layer contract rather than a collection of disconnected unit tests.

---

## 4.17 Step 7F — `lora-basic` acceptance closure

**Goal:** cover the visible basic-LoRA page alias in addition to the canonical ten-backend release matrix.

### Code delivered

Test-only acceptance coverage for:

- Standard equivalence;
- Standard -> Component bootstrap;
- basic U-Net LR mapping;
- rehydrate behavior.

### Result

The user-facing basic LoRA route is explicitly covered without pretending it is an eleventh backend.

---

## 5. Post-release hardening (#38-#46)

The post-Step-7 work was not a new feature generation phase. It was a hardening phase driven by real GUI usage, source-level review and actual runtime behavior.

## 5.1 #38 — Basic LoRA rehydrate round-trip repair

Fixed `lora-basic` memory-mode projection so rehydrate does not synthesize false `lowram/highvram` state.

## 5.2 #39 — Component editor polish and Multi-Caption default repair

Parameter Policy work included:

- improved Component editor presentation;
- localized help/descriptions;
- selectable profile references;
- typed Muon editor.

## 5.3 #40 — Legacy Schemastery discriminated-union stability

Fixed branch-selection failures where temporarily invalid child fields could collapse the active Component optimizer/editor branch.

## 5.4 #41 / #42 — Profile name / optimizer-type lifecycle

Separated:

~~~text
Profile name
Optimizer type
~~~

into independently editable concepts and restored dict entry actions.

This is required because profile names such as `fallback` may intentionally map to optimizer type `AdamW`.

## 5.5 #43 — Generated frontend asset MIME repair

Fixed generated Component editor CSS/JS asset serving. No optimizer semantics changed.

## 5.6 #44 — GUI lifecycle closure

Implemented:

- unnamed draft profile handling;
- duplicate/empty profile-name validation;
- rename retargeting;
- referenced-profile delete blocking;
- capability-driven fallback cleanup;
- dynamic profile selectors;
- page-global lifecycle hook cleanup.

A key change is that fallback behavior is derived from optimizer capability metadata, not hard-coded only to the string `Muon`.

## 5.7 #45 — Implicit CUDA device audit repair

Fixed false runtime failures caused by:

~~~text
accelerator.device = cuda
parameter.device   = cuda:0
~~~

while preserving strict explicit ordinal mismatches.

Also added a real-CUDA device-audit matrix case.

## 5.8 #46 — Non-Anima semantic/runtime hardening

This was the largest follow-up and closed multiple host/runtime mismatches.

Implemented or repaired:

- bootstrap representability blockers separated from runtime qualification blockers;
- non-Anima Component host preflight;
- exact all-frozen/target checks;
- SD3 target semantic normalization;
- Standard Flux partial-cache preservation;
- Flux/SD3 policy-aware Text Encoder cache routing;
- SD3 CLIP-L/CLIP-G co-residency;
- exact upstream `train_t5xxl` wire semantics;
- raw key/value/whitespace compatibility;
- validated pre-routing Component Train flags;
- removal of unnecessary partial tokenization when all TEs are frozen;
- early T5+cache rejection;
- real `Accelerator.prepare()` SD3 CLIP-pair qualification;
- real-CUDA SD3 CLIP pair qualification case.

### Final backend hardening state after #46

The review chain closed all findings raised in that hardening cycle:

~~~text
P0 = 0
P1 = 0
P2 = 0
~~~

This statement is scoped to the reviewed backend-hardening findings, not to every future capability listed below.

---

## 6. Current product/runtime capability

## 6.1 Qualified baseline backends

Component Start is opened for exactly ten runtime backends:

| Runtime backend | Trainer family |
| --- | --- |
| `sd-lora` | stable NetworkTrainer |
| `sdxl-lora` | stable NetworkTrainer |
| `sd-dreambooth` | stable full trainer |
| `sdxl-finetune` | stable full trainer |
| `sd3-lora` | dev NetworkTrainer |
| `flux-lora` | dev NetworkTrainer |
| `chroma-lora` | dev NetworkTrainer |
| `flux-finetune` | dev full trainer |
| `anima-lora` | staged Anima NetworkTrainer |
| `anima-finetune` | staged Anima full trainer |

Baseline currently means:

- one process;
- ordinary supported trainer execution path;
- Component-owned optimizer/LR routing;
- no active qualification blocker;
- no active semantic blocker.

---

## 6.2 Optimizer capability status

The current registry contains three distinct statuses. These must not be described as one undifferentiated “optimizer support” claim.

### Component `supported`

Currently selectable in the Component editor:

- AdamW
- AdamW8bit
- PagedAdamW8bit
- PagedAdamW
- PagedAdamW32bit
- Lion
- Lion8bit
- PagedLion8bit
- SGDNesterov
- SGDNesterov8bit
- RAdamScheduleFree
- AdamWScheduleFree
- SGDScheduleFree
- Muon

Runtime construction exists for all of the above.

The shared CUDA matrix is designed to exercise every registry entry marked `supported`, subject to its installed dependency.

### Component `restricted`

Known to DTS but not exposed as ordinary Component-v1 choices until their optimizer-specific semantics are implemented and qualified:

- DAdaptation
- DAdaptAdamPreprint
- DAdaptAdam
- DAdaptAdaGrad
- DAdaptAdan
- DAdaptAdanIP
- DAdaptLion
- DAdaptSGD
- Prodigy
- AdaFactor
- `prodigyplus.ProdigyPlusScheduleFree`

These are not all blocked for the same reason.

Examples:

- DAdapt/Prodigy need an explicit adaptive-LR ownership model.
- AdaFactor requires a reviewed `relative_step=False` contract.
- ProdigyPlusScheduleFree combines adaptive and optimizer-managed-scheduler semantics.

### Component `planned`

Registered, but intentionally unavailable:

- `pytorch_optimizer.CAME`
- Custom optimizer

CAME currently requires dedicated runtime/GPU qualification. Arbitrary Custom optimizers require an explicit capability contract rather than “try importing a class and hope”.

---

## 6.3 Muon and fallback status

### Component mode

Implemented:

- Muon as a primary optimizer profile;
- explicit eligibility policy;
- model-aware hidden-layer 2D matrix routing;
- explicit fallback Optimizer Profile;
- independent fallback LR;
- deterministic ownership/checkpoint identity;
- no use of Muon's internal AdamW fallback.

Canonical example:

~~~text
primary profile:   Muon
fallback profile:  AdamW
~~~

### Standard mode

Standard mode does **not** currently have the Component concept of:

~~~text
primary optimizer profile
+
per-parameter eligibility router
+
fallback optimizer profile
~~~

Therefore “Muon + fallback in Standard mode” is **not currently equivalent to the Component feature**.

This is a real future compatibility/design task, not merely a missing GUI option.

Possible future designs must decide whether Standard should:

1. gain an explicit Muon+fallback mini-policy; or
2. remain a single-optimizer legacy path and require users to switch to Component for explicit fallback routing.

Until that decision is made, Standard must not silently emulate Component fallback semantics.

---

## 6.4 Checkpoint and resume guarantees

Implemented:

- CompositeOptimizer state;
- CompositeLRScheduler state;
- child optimizer identity;
- topology fingerprints;
- checkpoint manifest v2;
- policy hash;
- trainable/frozen Component diagnostics;
- parameter counts;
- fail-closed topology mismatch;
- post-prepare and post-resume runtime audits.

A Component checkpoint is therefore not treated as merely “weights plus any optimizer that happens to load”.

---

## 6.5 Text Encoder caching status

Current non-Anima hardening provides:

- Standard Flux CLIP partial caching preserved;
- Component Flux CLIP-live + T5-cached path;
- Component SD3 CLIP-L/G live + T5-cached path;
- policy-aware pre-routing cache hints;
- full cache when all relevant TEs are frozen;
- early T5-train + cache rejection;
- SD3 CLIP pair co-residency protection;
- real Accelerate prepare qualification for one-live/one-frozen CLIP routes.

Preloaded non-zero adapter + cached TE combinations remain semantic blockers where cache timing would omit adapter effects.

---

## 7. Current blocker taxonomy

A major lesson from the project is that “unsupported” has multiple meanings.

## 7.1 Qualification blockers

These features are conceptually compatible with Component ownership, but DTS has not yet established a sufficient runtime/physical-GPU contract.

Current registry:

- `torch_compile`
- Anima `compile`
- DeepSpeed
- `fused_backward_pass`
- `fused_optimizer_groups`
- `blockwise_fused_optimizers`
- `cpu_offload_checkpointing`
- `unsloth_offload_checkpointing`
- `blocks_to_swap`
- `double_blocks_to_swap`
- `single_blocks_to_swap`
- `full_fp16`
- `full_bf16`
- `fp8_base`
- `fp8_base_unet`
- explicit multi-GPU/DDP selection

These may be unblocked incrementally after dedicated implementation/audit/evidence.

## 7.2 Semantic blockers

These are not safely unblockable by “run one GPU test”. Parameter Policy v1 cannot currently represent their semantics exactly, or their state mutation conflicts with static ownership.

Current feature registry:

- LoRA+
- regex-specific LR
- SD/SDXL LoRA block LR
- SDXL Full block LR
- `scale_weight_norms`
- DreamBooth dynamic Text Encoder stop
- preloaded adapter + Text Encoder cache hazards
- Anima Qwen-only
- unreviewed custom `network_module`

Removing these blockers requires schema/routing semantics work first.

---

## 8. What is still missing

This section is the forward development backlog. Items are ordered by engineering dependency, not by user-visible importance.

## 8.1 Full BF16 / Full FP16 Component qualification

### Current state

Ordinary mixed precision is baseline-supported.

`full_bf16` and `full_fp16` remain qualification blockers.

### Why this is not equivalent to ordinary mixed precision

Full precision-mode flags change the actual model/gradient dtype ownership path rather than only entering autocast around selected operations.

Component runtime must prove:

- Runtime Spec parameter identity survives dtype conversion;
- optimizer children receive the intended dtype/state;
- FP16 GradScaler behavior remains correct;
- BF16 path does not incorrectly introduce scaling;
- Accelerate wrapping preserves ownership;
- checkpoint optimizer state restores with the same dtype/topology;
- Anima-specific model dtype behavior remains intentional.

### Required closure

A future qualification PR should include:

1. source-level dtype lifecycle review per trainer family;
2. synthetic CompositeOptimizer full-BF16/full-FP16 cases;
3. real CUDA fresh step;
4. checkpoint save;
5. resume;
6. post-resume ownership/device audit;
7. backend representative matrix;
8. only then remove `full_bf16` / `full_fp16` blockers.

---

## 8.2 FP8 Component qualification

Still blocked:

- `fp8_base`
- `fp8_base_unet`

Required work includes:

- base-model dtype/residency audit;
- optimizer-owned parameter dtype audit;
- module conversion ordering;
- checkpoint identity;
- backend-specific Flux/SD3/Anima behavior;
- real CUDA evidence.

Do not combine this mechanically with full BF16. FP8 changes a different part of the model/optimizer contract.

---

## 8.3 Optimizer compatibility beyond the strongest-qualified paths

The current implementation already supports more than AdamW and Muon, but future work should increase evidence quality and close restricted optimizers.

### Supported registry entries

Keep a real-CUDA smoke for every `component_support == "supported"` optimizer.

Evidence should include at minimum:

- construction;
- one or more real optimizer steps;
- group LR propagation;
- state_dict save/load;
- CompositeOptimizer wrapping;
- scheduler behavior;
- mixed-child topology where relevant.

### Restricted optimizers

#### DAdapt family / Prodigy

Need a Component-owned adaptive-LR semantic contract.

Open questions:

- What does a component LR mean for an optimizer whose effective step size adapts internally?
- Can different Components safely use independent adaptive child optimizers?
- Which legacy Standard LR rewrites must be bypassed?
- What should Preview display as the authoritative LR?

#### AdaFactor

Need an explicit `relative_step=False` policy and tests preventing incompatible relative-step behavior.

#### ProdigyPlusScheduleFree

Needs both:

- adaptive optimizer semantics;
- optimizer-managed scheduler/lifecycle semantics.

### Planned CAME

Needs:

- child optimizer implementation in `parameter_policy_torch.py`;
- capability validation;
- CPU runtime smoke;
- CUDA runtime smoke;
- state/resume qualification.

### Custom optimizer

Should remain blocked until DTS defines a declarative capability contract. Arbitrary import strings are not enough to guarantee:

- group-LR behavior;
- scheduler ownership;
- state serialization;
- Accelerate compatibility;
- fallback eligibility.

---

## 8.4 Muon + fallback compatibility in Standard mode

### Current state

Component mode owns this feature completely.

Standard mode retains historical single-optimizer semantics.

### Missing decision

The project must explicitly decide whether “Standard Muon” means:

- Muon alone with provider-native internal fallback controls;
- Muon + DTS explicit fallback;
- or “not a Standard feature; use Component mode”.

Using provider-native Muon fallback would conflict with the existing Component design principle that fallback ownership must be explicit and auditable.

### Recommended requirement before implementation

Write a dedicated Standard compatibility design that specifies:

- GUI representation;
- effective TOML representation;
- optimizer construction ownership;
- parameter eligibility;
- checkpoint format;
- Standard -> Component bootstrap mapping;
- Component -> Standard rehydrate/export expectations;
- behavior when fallback cannot be represented losslessly.

Do not add a Muon option to the legacy optimizer dropdown without answering these questions.

---

## 8.5 Legacy optimizer/LR ownership UX in Component mode

Backend compilation already treats Parameter Policy as authoritative in Component mode, but the public GUI can still expose historical optimizer/LR fields from the underlying trainer schema.

Future frontend work should make ownership visually unambiguous.

Desired behavior:

- Standard:
  - legacy optimizer/LR controls editable;
  - Parameter Policy editor inactive/hidden as designed.
- Component:
  - legacy optimizer/LR controls that no longer own runtime optimization should be disabled or clearly marked as non-authoritative;
  - scheduler controls that are still legitimately shared should remain editable;
  - no user value should be silently deleted when switching modes;
  - switching back to Standard should restore historical configuration semantics.

This is a UX/ownership clarification task, not a request to rewrite Standard compilation.

---

## 8.6 Compile qualification

Blocked:

- `torch_compile`
- Anima block-level `compile`

Required evidence:

- parameter identity before/after wrapping;
- optimizer references remain attached to the real trainable Parameters;
- save/resume topology;
- adapter metadata survival;
- no duplicate/wrapped ownership ambiguity.

---

## 8.7 Swap/offload qualification

Blocked:

- CPU checkpoint offload;
- Unsloth checkpoint offload;
- Flux/Anima block swap fields.

The current device audit assumes stable ownership of optimizer parameters. Dynamic residency requires a phase-aware contract describing when CPU/GPU movement is legal and what exactly is audited.

---

## 8.8 Fused optimizer/backward paths

Blocked:

- fused backward;
- fused optimizer groups;
- blockwise fused optimizers.

These are not a simple performance toggle under Component mode because the historical trainer path may create and step optimizers independently.

Component support requires one coherent ownership model that does not bypass CompositeOptimizer topology.

---

## 8.9 DeepSpeed and multi-GPU/DDP

Current Component baseline is single-process.

### Missing work

Need a distributed ownership contract for:

- CompositeOptimizer wrapping;
- sharded/replicated optimizer state;
- rank-local vs global topology identity;
- parameter aliases after wrapping;
- checkpoint save/load;
- resume across the same topology;
- explicit GPU selection;
- backend matrix evidence.

This should be treated as a separate development phase rather than removing the current blocker opportunistically.

---

## 8.10 Semantic-feature extensions

The following require new schema/routing semantics before runtime work:

### LoRA+

Parameter-level LR multipliers do not map directly to the current static component LR schema.

### Regex LR

Flux/Chroma/Anima regex-specific LR can split parameters below the current Component ownership layer.

### SD/SDXL block LR and SDXL Full block LR

Need a representable subcomponent/group model or an explicit nested-LR extension.

### Dynamic DreamBooth TE stop

Current policy is static ownership. Temporal Train -> Freeze transitions require a time-dependent policy/state-machine contract.

### `scale_weight_norms`

Can mutate parameters that Parameter Policy considers frozen; static optimizer ownership alone is insufficient.

---

## 9. Standard vs Component compatibility policy

Future work must preserve the following rule:

> Component may add capability, but enabling Component support must not redefine Standard semantics.

When a Standard configuration is migrated:

- exact representable behavior may bootstrap into Component;
- unsupported semantics must return a blocker;
- DTS must not invent fallback routes or approximate optimizer ownership.

When Component state is imported:

- the policy is authoritative;
- incomplete/incompatible state must fail closed;
- rehydrate must not silently fall back to Standard.

When switching back to Standard:

- hidden Component policy state must not affect effective Standard config.

---

## 10. Qualification levels

Future status reports should use the following vocabulary.

### Level 0 — Registered

The optimizer/backend/feature exists in metadata.

No runtime claim.

### Level 1 — Host representable

The host can validate and serialize it.

No torch runtime claim.

### Level 2 — Runtime implemented

Torch/trainer code exists and dependency-light/runtime tests cover construction.

### Level 3 — Synthetic physical-GPU qualified

Real CUDA evidence covers the core runtime seam with small synthetic modules/tensors.

### Level 4 — Real backend qualified

A real trainer/model/dataset fresh run and resume pass the ownership/checkpoint contract.

### Level 5 — Release baseline

The blocker is removed and the configuration is allowed through Component Start.

This avoids statements such as “optimizer X is supported” when only one layer is actually implemented.

---

## 11. Current qualification infrastructure

### Dependency-light host CI

Covers:

- policy validation;
- bootstrap;
- compatibility;
- routing contracts;
- editor/schema contracts;
- release matrices;
- Standard regression.

### CPU torch runtime smoke

Pinned runtime currently exercises:

- PyTorch 2.7;
- Accelerate 1.6;
- ScheduleFree 1.4;
- pytorch-optimizer 3.10.0;
- CompositeOptimizer/Scheduler;
- state/resume;
- real `Accelerator.prepare()` seams.

### Real CUDA optimizer/runtime matrix

`tools/run_parameter_policy_gpu_matrix.py` covers:

- supported optimizer implementations;
- FP16 autocast + GradScaler;
- BF16 autocast;
- mixed child optimizers;
- optimizer state reload;
- implicit `cuda` device audit;
- SD3 CLIP-L/G prepare co-residency.

### Real backend matrix

`tools/run_parameter_policy_backend_gpu_matrix.py` accepts machine-local commands for exactly the ten release backends and requires:

- fresh run success;
- checkpoint manifest v2;
- correct backend identity;
- resume success;
- unchanged manifest identity.

---

## 12. Recommended next development sequence

The following order minimizes repeated runtime churn.

### Phase A — Frontend ownership clarity

1. disable/annotate legacy optimizer/LR controls in Component mode;
2. preserve scheduler controls where still authoritative;
3. preserve Standard round-trip;
4. make fallback selectors capability-aware everywhere;
5. clarify target availability vs Component Train state.

This changes UX, not runtime topology.

### Phase B — Optimizer evidence and restricted optimizer semantics

1. audit current supported optimizer GPU evidence;
2. fill missing real-CUDA evidence;
3. implement CAME;
4. design AdaFactor contract;
5. design adaptive optimizer contract for DAdapt/Prodigy;
6. separately design ProdigyPlusScheduleFree.

### Phase C — Full BF16 / FP16 qualification

Treat this as the next precision qualification project.

Do not remove blockers until fresh/resume GPU evidence is complete.

### Phase D — FP8 qualification

Separate from full BF16/FP16.

### Phase E — Standard Muon/fallback decision

Write the compatibility spec first; then either implement it or explicitly document Component mode as the only audited Muon+fallback path.

### Phase F — compile / swap / offload / fused runtime

Each feature should get an independent runtime contract and evidence.

### Phase G — distributed Component runtime

Design DDP/DeepSpeed ownership and checkpoint topology before implementation.

### Phase H — semantic-policy v2 candidates

Only after baseline runtime is mature:

- nested/block LR;
- regex LR;
- temporal Train/Freeze;
- LoRA+;
- other parameter-level routing semantics.

---

## 13. Definition of done for future capability removal

A qualification blocker should not be removed merely because a local training process reaches step 1.

Minimum acceptance:

1. host Preview correctly identifies the configuration;
2. Standard behavior remains unchanged;
3. Component policy has deterministic ownership;
4. actual optimizer parameters match Runtime Spec;
5. device/residency audit is valid for that execution mode;
6. at least one optimizer step succeeds;
7. checkpoint saves;
8. optimizer/scheduler state saves;
9. resume succeeds;
10. topology identity remains stable;
11. post-resume runtime audit succeeds;
12. relevant real CUDA evidence is recorded;
13. backend-specific hazards are covered;
14. blocker registry + tests are updated in the same change.

For semantic blockers, add a prior requirement:

0. the policy schema can represent the requested semantics without approximation.

---

## 14. Source-of-truth map

| Concern | Source of truth |
| --- | --- |
| Optimizer capabilities | `mikazuki/optimizer_profiles.py` |
| Canonical policy schema | `mikazuki/parameter_policy.py` |
| Component registry / target availability | `mikazuki/model_component_profiles.py` |
| Physical parameter routing | `mikazuki/parameter_routing.py` |
| Standard -> Component migration | `mikazuki/parameter_policy_bootstrap.py` |
| Semantic/runtime blockers | `mikazuki/parameter_policy_compat.py` |
| Runtime backend matrix | `mikazuki/parameter_policy_matrix.py` |
| Runtime Spec | `mikazuki/parameter_policy_runtime.py` |
| Composite torch runtime | `mikazuki/parameter_policy_torch.py` |
| Trainer lifecycle | `mikazuki/parameter_policy_trainer.py` |
| Editor metadata | `mikazuki/parameter_policy_editor.py` |
| GUI schema | `mikazuki/parameter_policy_schema.py` |
| Frontend state machine | `mikazuki/frontend_training_patch.py` |
| Shared CUDA matrix | `tools/run_parameter_policy_gpu_matrix.py` |
| Real backend matrix | `tools/run_parameter_policy_backend_gpu_matrix.py` |
| User documentation | `docs/parameter-policy.md` |
| Historical implementation plans | `docs/parameter-policy-step*.md` |

---

## 15. Maintenance rule

Whenever a future PR changes Parameter Policy capability:

1. update the machine-readable registry first;
2. update or add tests;
3. update the relevant detailed plan if it is still an active implementation phase;
4. update this development-status document if the completed/missing capability surface changes;
5. update the user guide only when the user-visible workflow changes.

Do not use documentation to claim support that the blocker registry or runtime evidence still rejects.
