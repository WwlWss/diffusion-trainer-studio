# Parameter Training Policy — Step 3 Implementation Plan

Baseline: `main@a065c2983db90d0d9f5fc2e5324788b4a17307f2` (PR #21 and #22 merged)

## 1. Goal

Step 3 implements **Model Component Profiles + pure parameter routing**.

At the end of this step DTS must be able to answer, deterministically and without constructing any optimizer:

1. What real parameter objects exist under the trainer-owned model roots?
2. Which aliases refer to the same parameter object?
3. Which model component owns each parameter?
4. Is that parameter eligible for the primary optimizer's parameter-eligibility policy?
5. If not, does it have a valid fallback route?
6. Is any parameter unassigned, multiply owned, unavailable under the higher-level training target, or otherwise inconsistent?
7. What exact primary / fallback / frozen routing plan would a later runtime consume?

This step does **not** make Component-wise training runnable yet.

## 2. Hard scope boundary

### Step 3 includes

- explicit optimizer eligibility-policy metadata;
- model-family Component Profile registry;
- higher-level Training Target Profile resolution;
- parameter identity scanning and alias consolidation;
- structural parameter classification;
- pure routing to primary / fallback / frozen / unavailable;
- ownership and fail-closed audits;
- Standard -> Component bootstrap for model components;
- Component compatibility blockers for legacy optimizer/group semantics that are not yet representable;
- dependency-light unit tests and CI wiring.

### Step 3 deliberately excludes

- optimizer construction;
- `optimizer.step()` / `zero_grad()`;
- LR scheduler construction;
- CompositeOptimizer;
- Accelerate wrapping;
- optimizer/scheduler save and resume;
- any `requires_grad_` mutation;
- model loading;
- tensor cloning, `.cpu()`, value statistics, norms, or device movement;
- trainer CLI/runtime activation;
- removing the existing Component Start blocker;
- modifying LoRA implementations to attach original-target metadata (Step 4);
- per-model trainer integration (Step 6);
- GUI editor work (Step 7).

Standard mode must remain a true legacy path.

## 3. CI / dependency constraint

The current host-side review workflow intentionally runs without installing PyTorch.

Therefore the new routing modules must:

- not import `torch` at module import time;
- use duck typing for module/parameter inspection;
- inspect metadata only (`shape`, `ndim`, `dtype`, `requires_grad`, identity);
- keep all tests runnable with small fake Module/Parameter objects.

Actual PyTorch integration is validated later when trainer runtime integration begins.

## 4. File layout

### 4.1 `mikazuki/optimizer_profiles.py`

Extend optimizer capability metadata from a boolean-only eligibility flag to an explicit policy identifier.

Planned contract:

```python
@dataclass(frozen=True)
class OptimizerCapability:
    ...
    eligibility_policy: str | None = None

    @property
    def requires_parameter_eligibility(self) -> bool:
        return self.eligibility_policy is not None
```

Muon becomes:

```python
eligibility_policy="model_hidden_2d_weight"
```

The compatibility property keeps PR #22 callers stable while the router consumes the explicit policy name.

No optimizer implementation is constructed in this step.

### 4.2 `mikazuki/model_component_profiles.py`

Own all model-specific knowledge.

This file must not construct models and must not import trainer model classes. Classification uses stable root IDs, module paths, module class names, parameter roles, structural classes, and (later) explicit adapter-target metadata.

Core types:

```python
ParameterClass = Literal[
    "matrix_weight",
    "bias",
    "norm_weight",
    "embedding_weight",
    "conv_weight",
    "other",
]

@dataclass(frozen=True)
class ComponentDefinition:
    component_id: str
    display_name: str
    description: str
    roots: frozenset[str]

@dataclass(frozen=True)
class TrainingTargetProfile:
    train_type: str
    available_components: frozenset[str]
    unavailable_reasons: Mapping[str, str]

@dataclass(frozen=True)
class ModelComponentProfile:
    train_type: str
    components: Mapping[str, ComponentDefinition]
    classify_alias: Callable[..., str | None]
    eligibility_check: Callable[..., tuple[bool, str]]
```

Public entry points:

```python
get_model_component_profile(train_type)
resolve_training_target_profile(train_type, effective_config)
```

The separation is intentional:

- **ModelComponentProfile** answers “what architectural component is this parameter?”
- **TrainingTargetProfile** answers “is that component allowed to train for this trainer/config?”

A Component policy is never allowed to expand a higher-level trainer target.

### 4.3 `mikazuki/parameter_routing.py`

Own all model-independent identity scanning and routing.

Planned data model:

```python
@dataclass(frozen=True)
class AdapterTargetMetadata:
    root: str
    module_path: str
    module_type: str

@dataclass(frozen=True)
class ParameterAlias:
    root: str
    full_name: str
    module_path: str
    module_type: str
    module_class: str
    parameter_role: str
    parameter_class: ParameterClass
    adapter_target: AdapterTargetMetadata | None = None

@dataclass(frozen=True)
class ParameterDescriptor:
    parameter: Any
    parameter_id: int
    aliases: tuple[ParameterAlias, ...]
    shape: tuple[int, ...]
    ndim: int
    dtype: str
    requires_grad: bool

@dataclass(frozen=True)
class RoutingAssignment:
    parameter_id: int
    component_id: str
    route_kind: Literal["primary", "fallback", "frozen", "unavailable"]
    optimizer_profile: str | None
    learning_rate: float | None
    reason: str

@dataclass(frozen=True)
class RoutingIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    parameter_id: int | None = None
    component_id: str | None = None

@dataclass(frozen=True)
class RoutingStats:
    total_tensors: int
    total_numel: int
    by_component: Mapping[str, int]
    by_route: Mapping[str, int]
    by_parameter_class: Mapping[str, int]

@dataclass(frozen=True)
class RoutingPlan:
    assignments: tuple[RoutingAssignment, ...]
    issues: tuple[RoutingIssue, ...]
    stats: RoutingStats

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)
```

Public entry points:

```python
scan_parameter_roots(
    roots: Mapping[str, Any],
    *,
    adapter_targets: Mapping[int, AdapterTargetMetadata] | None = None,
)
build_parameter_routing_plan(
    policy,
    *,
    train_type,
    effective_config,
    descriptors,
)
```

## 5. Parameter scan algorithm

For each trainer-provided root:

1. accept an explicit `Mapping[root_id, module]`; root IDs are trainer-owned semantic roots such as `unet`, `transformer`, `text_encoder_1`, `qwen3`;
2. walk `named_modules(remove_duplicate=False)`;
3. for each module, walk only `named_parameters(recurse=False, remove_duplicate=False)`;
4. form an alias record from root + module path + local parameter role;
5. capture both a qualified module type (`type(module).__module__ + "." + __qualname__`) and the short class name;
6. if `adapter_targets` contains `id(module)`, attach that metadata to **this alias**;
7. aggregate by `id(parameter)`, never by name;
8. preserve every alias;
9. sort roots/aliases deterministically before producing descriptors.

Adapter metadata is intentionally keyed by adapter **module identity**, not parameter identity or string name. Step 4 may provide this registry without requiring Step 3 to guess from `lora_name`.

The scanner must never:

- mutate `requires_grad`;
- call `.cpu()`, `.cuda()`, `.to()`, `.clone()`, `.detach()`;
- inspect tensor values;
- perform routing repeatedly inside the training loop.

Routing is a startup operation.

### Alias conflicts

A shared parameter is legal only if all aliases agree on the same effective component, target state, and optimizer-eligibility result.

Adapter target metadata is alias-scoped for the same reason: a shared parameter must not silently inherit one arbitrary alias's original-target metadata.

If aliases imply different components, roots, target metadata, availability, or eligibility, the plan records a hard conflict and fails closed.

## 6. Structural parameter classification

Structural class is derived from the owning alias, not from parameter names alone.

Order:

1. role `bias` -> `bias`;
2. normalization module -> `norm_weight`;
3. embedding module -> `embedding_weight`;
4. convolution module -> `conv_weight`;
5. role `weight` and `ndim == 2` -> `matrix_weight`;
6. otherwise -> `other`.

Normalization detection covers the repository's real names such as LayerNorm, GroupNorm, RMSNorm and QKNorm-style modules.

This classification alone does **not** make a matrix Muon-eligible.

Component classification and structural parameter classification are separate axes. A profile should classify the architectural owner first; `ParameterClass` must not become an optimizer-shaped substitute for Component IDs. For example, a bias owned by an attention projection remains in that attention/transformer Component and is later rejected by Muon eligibility, allowing fallback routing to do real work.

## 7. Muon eligibility

Muon uses:

```text
eligibility_policy = model_hidden_2d_weight
```

Eligibility requires both:

1. structural class is a 2D matrix weight; and
2. the active Model Component Profile explicitly authorizes that architectural location as a hidden-layer matrix.

Examples that remain ineligible even if 2D:

- embeddings;
- normalization affine parameters;
- input/output/final projections that the model profile marks non-hidden;
- modulation/conditioning projections when the profile excludes them;
- convolution weights;
- biases;
- arbitrary unknown 2D parameters.

For LoRA, eligibility must eventually depend on the **original target module metadata**, not merely the shape/name of `lora_down` / `lora_up`. Step 3 defines the metadata slot and tests synthetic descriptors; Step 4 populates it in real LoRA modules.

## 8. Routing algorithm

Routing has a policy-level pre-pass and a descriptor-level pass.

Policy pre-pass:

1. every Component ID present in the sidecar must exist in the active Model Component Profile;
2. resolve higher-level Training Target availability for every declared row;
3. reject `Train=true` on a target-unavailable Component;
4. do not yet assume that a profile-declared Component exists in the actual loaded model.

Descriptor pass:

1. classify every alias with the Model Component Profile;
2. require alias consensus for component, target state, adapter metadata, and eligibility;
3. derive the set of **actually present** Components from classified descriptors;
4. if a present + target-available Component has no policy row => `unassigned` hard error;
5. if `Train=false` => `frozen`;
6. resolve primary Optimizer Profile;
7. if primary has no eligibility policy => `primary`;
8. otherwise evaluate the model-profile eligibility policy:
    - eligible => `primary`;
    - ineligible + fallback exists => `fallback`;
    - ineligible + no fallback => hard error;
9. fallback LR resolves to explicit `fallback_learning_rate` when present, otherwise the Component `learning_rate`;
10. audit global ownership.

Presence audit:

- a valid profile Component that is absent from the actual descriptors and has `Train=false` is allowed;
- an absent Component with `Train=true` is a hard `component_absent` error, preventing optional/mismatched model structures from silently training nothing;
- target-unavailable + `Train=false` remains representable and does not require model presence.

The router does not mutate parameters.

### Required final audit

A plan is routing-valid only when:

- Unassigned = 0;
- Duplicate/conflicting ownership = 0;
- Unknown component = 0;
- Train=true on unavailable component = 0;
- Missing required fallback for an actually ineligible parameter = 0;
- Train=true Component with zero matched parameters = 0.

A declared fallback that is never used is allowed and is represented as a warning, not an error.

## 9. Initial Component Profile IDs

These IDs become the stable internal contract for later UI/runtime work.

### SD DreamBooth

- `unet.transformer`
- `unet.conv_resnet`
- `unet.norm_bias_other`
- `unet.base_other`
- `text_encoder`

### SDXL Full

- `unet.transformer`
- `unet.conv_resnet`
- `unet.norm_bias_other`
- `unet.base_other`
- `text_encoder_1`
- `text_encoder_2`

The profile is aligned with the repository's `BasicTransformerBlock`, `CrossAttention`, `FeedForward`, `ResnetBlock2D`, convolution, normalization and outer U-Net structures.

### Flux Full

- `transformer.double_stream`
- `transformer.single_stream`
- `transformer.modulation_norm_other`
- `transformer.input_conditioning`
- `transformer.final`
- `transformer.other`

Important repository-specific rule: `SingleStreamBlock.linear1` and `linear2` are fused attention/MLP projections. The profile must classify those real modules directly; it must not assume q_proj/k_proj/v_proj names. Their biases remain in the same architectural Component; `ParameterClass` then makes those biases Muon-ineligible so fallback routing is exercised correctly.

Flux Full never gains CLIP-L/T5 training merely because Component mode exists.

### Anima Full

- `dit.self_attention`
- `dit.cross_attention`
- `dit.mlp`
- `dit.modulation`
- `dit.llm_adapter`
- `dit.base_other`
- `qwen3`

These preserve the pinned trainer's existing six DiT LR groups while making Qwen3 an explicit higher-level component.

Qwen3 availability remains controlled by the existing `train_qwen3_text_encoder` contract.

### SD / SDXL LoRA (metadata-driven; real wiring is Step 4)

SD:

- `unet.attention.adapter`
- `unet.feed_forward.adapter`
- `unet.conv.adapter`
- `unet.other.adapter`
- `text_encoder.adapter`

SDXL:

- same U-Net adapter groups;
- `text_encoder_1.adapter`;
- `text_encoder_2.adapter`.

### Flux LoRA (metadata-driven)

- `transformer.double_stream.adapter`
- `transformer.single_stream.adapter`
- `transformer.input_conditioning.adapter`
- `clip_l.adapter`
- `t5xxl.adapter`

### Chroma LoRA (metadata-driven)

- `transformer.double_stream.adapter`
- `transformer.single_stream.adapter`
- `transformer.input_conditioning.adapter`
- `t5xxl.adapter`

There is deliberately no CLIP-L Component for Chroma.

### SD3 LoRA (metadata-driven)

- `mmdit.attention.adapter`
- `mmdit.mlp.adapter`
- `mmdit.modulation_norm.adapter`
- `mmdit.other.adapter`
- `clip_l.adapter`
- `clip_g.adapter`
- `t5xxl.adapter`

### Anima LoRA (metadata-driven)

- `dit.self_attention.adapter`
- `dit.cross_attention.adapter`
- `dit.mlp.adapter`
- `dit.modulation.adapter`
- `dit.other.adapter`
- `llm_adapter.adapter`
- `qwen3.adapter`

## 10. Higher-level Training Target rules

The resolver consumes the already normalized effective config and must preserve current trainer semantics.

Examples:

- SD/SDXL LoRA: `network_train_unet_only` / `network_train_text_encoder_only` remain authoritative.
- Flux LoRA: normalized Flux target remains authoritative.
- Chroma LoRA: CLIP-L is never available.
- Flux Full: only the transformer is trainable in the current trainer; CLIP-L/T5 stay unavailable.
- SDXL Full: text encoders are available only under the existing `train_text_encoder` contract.
- SD DreamBooth: the existing text-encoder training switch remains authoritative.
- Anima Full: Qwen3 is available only when the existing Qwen3 joint-training target enables it. Existing legacy optimizer/LR restrictions in `anima_qwen_config.py` are not reinterpreted as Component optimizer rules; runtime integration of those constraints is Step 6.
- Anima LoRA: `network_train_unet_only` / `network_train_text_encoder_only` preserve the DiT/Qwen3 target; `network_args train_llm_adapter=True` controls whether LLM Adapter LoRA can exist.
- SD3 LoRA: `network_train_unet_only` controls MMDiT vs CLIP-L/CLIP-G availability, while `network_args train_t5xxl=True` independently controls T5XXL availability.
- cache/target conflicts remain validated by the existing semantic compiler before routing.

## 11. Standard -> Component bootstrap

Extend `mikazuki/parameter_policy_bootstrap.py` with a full model-component bootstrap.

Planned entry point:

```python
bootstrap_parameter_policy_from_standard(
    config,
    page_type,
    *,
    resolve_backend,
) -> dict
```

Flow:

1. compile the exact existing Standard config;
2. build `legacy_main` with the existing optimizer bootstrap;
3. resolve the model component profile;
4. map current legacy LRs/targets into Component rows;
5. convert every legacy LR=0 freeze into explicit `{"train": false}`;
6. fail closed instead of approximating unsupported legacy semantics;
7. canonicalize through the existing sidecar validator.

### Anima Full mapping

- `learning_rate` -> `dit.base_other`;
- `self_attn_lr` -> `dit.self_attention`;
- `cross_attn_lr` -> `dit.cross_attention`;
- `mlp_lr` -> `dit.mlp`;
- `mod_lr` -> `dit.modulation`;
- `llm_adapter_lr` -> `dit.llm_adapter`;
- `qwen3_lr` -> `qwen3` only when Qwen3 joint training is enabled.

Missing optional Anima sub-LRs inherit the base LR, matching the pinned trainer.

### SDXL Full mapping

Without `block_lr`:

- all U-Net components inherit `learning_rate`;
- TE1 uses `learning_rate_te1` or base LR;
- TE2 uses `learning_rate_te2` or base LR;
- legacy zero becomes explicit Train Off.

Active `block_lr` is not approximated in Step 3; it is a compatibility blocker.

### SD DreamBooth mapping

- all U-Net components inherit the legacy U-Net LR;
- text encoder uses its existing separate LR when present, otherwise the legacy base LR;
- disabled text-encoder training becomes Train Off.

### Flux Full

All transformer components inherit the legacy base LR.

### LoRA families

Component rows inherit the already normalized legacy U-Net/DiT/text-encoder LR semantics.

Block-weight LR and LoRA+ features are compatibility blockers rather than being silently flattened.

## 12. Compatibility gate

Add `mikazuki/parameter_policy_compat.py`.

Public API:

```python
parameter_policy_compatibility_blockers(
    effective_config,
    train_type,
) -> list[str]
```

Step 3 must fail closed for active semantics that Component v1 does not yet reproduce exactly.

Initial blocker set:

- SDXL Full `block_lr`;
- SD/SDXL LoRA block LR weighting in `network_args`:
  - `down_lr_weight`;
  - `mid_lr_weight`;
  - `up_lr_weight`;
  - `block_lr_zero_threshold`;
- LoRA+ ratios in `network_args`:
  - `loraplus_lr_ratio`;
  - `loraplus_unet_lr_ratio`;
  - `loraplus_text_encoder_lr_ratio`;
- regex-specific LoRA learning rates where implemented (Flux/Anima), especially `network_reg_lrs`;
- `fused_backward_pass`;
- `fused_optimizer_groups`;
- `blockwise_fused_optimizers`;
- DeepSpeed until the CompositeOptimizer/runtime ownership contract is proven;
- any other optimizer-specific path whose current semantics create or step multiple optimizers outside the future CompositeOptimizer contract.

The parser for `network_args` must inspect normalized key/value entries; it must not use substring guessing.

This compatibility gate supplements the existing optimizer capability blockers.

## 13. Effective-config ownership is deferred to Step 6

Step 3 does **not** strip legacy optimizer/LR fields from the effective trainer config.

Reason:

- Component Start is still blocked in Step 3;
- current trainer semantic validators (especially Anima Qwen3) still use legacy LR/optimizer fields while compiling existing trainer configuration;
- stripping or bypassing those fields now would partially integrate Component runtime and create two competing semantic compilers.

Step 3 may expose pure compatibility/bootstrap helpers, but request/trainer integration keeps the Step 2 behavior unchanged.

Step 6, when per-model runtime integration is implemented, becomes responsible for establishing the single trainer-side optimizer/LR authority and removing superseded legacy controls at the correct boundary.

Standard mode remains unchanged.

## 14. LoRA Step 3 / Step 4 boundary

Step 3 defines and consumes:

```python
AdapterTargetMetadata(
    root=...,
    module_path=...,
    module_type=...,
)
```

Step 3 tests routing with synthetic adapter metadata.

Step 3 must **not** modify:

- `scripts/stable/networks/lora.py`;
- `scripts/dev/networks/lora_flux.py`;
- `scripts/dev/networks/lora_sd3.py`;
- pinned `sd-scripts/networks/lora_anima.py`.

Step 4 will attach reliable original-target metadata when adapters are created.

If a real LoRA descriptor lacks required target metadata, routing fails closed instead of guessing from `lora_name`, adapter parameter names, or A/B shapes.

## 15. Tests

### 15.1 Optimizer capability tests

Extend `tests/test_optimizer_profile_capabilities.py`:

- Muon exposes `eligibility_policy == "model_hidden_2d_weight"`;
- compatibility property still reports `requires_parameter_eligibility=True`;
- ordinary optimizers expose no eligibility policy.

### 15.2 Model profile tests

New `tests/test_model_component_profiles.py`:

- all supported backend IDs resolve;
- unknown backend fails closed;
- SD/SDXL transformer matrix vs conv/resnet/norm/other classification;
- Flux double/single fused projection classification;
- Flux input/final/modulation exclusions from Muon eligibility;
- Anima self/cross/MLP/mod/adapter/base classification;
- Chroma has no CLIP-L component;
- SD3 profile recognizes MMDiT target families;
- target availability cannot be expanded by policy.

### 15.3 Scanner/routing tests

New `tests/test_parameter_routing.py` using fake modules/parameters:

- scan preserves aliases;
- identity is `id(parameter)`;
- shared same-component alias is accepted;
- adapter metadata is alias-scoped;
- conflicting alias ownership / adapter-target / eligibility is rejected;
- no parameter mutation;
- all-eligible Muon component works without fallback;
- mixed Muon component requires fallback only for actual ineligible parameters;
- fallback LR inheritance;
- missing fallback produces a hard issue;
- non-eligibility optimizer stays on primary;
- Train Off routes frozen;
- missing component row produces unassigned;
- unknown parameter produces unassigned;
- unavailable Train=true fails;
- unavailable Train=false remains unavailable;
- Train=true on a profile-valid but actually absent Component fails closed;
- warning/error severity is preserved;
- explicit routing stats (tensor count, numel, component/route/class counts) are correct.

### 15.4 Bootstrap/compatibility tests

Extend/create host tests for:

- legacy LR=0 -> Train Off;
- Anima optional LR inheritance;
- Qwen3 enabled/disabled mapping;
- SDXL TE1/TE2 mapping;
- SDXL `block_lr` blocker;
- LoRA block-weight blocker from normalized `network_args`;
- LoRA+ blocker;
- `network_reg_lrs` blocker for Flux/Anima LoRA;
- fused/DeepSpeed blockers;
- Standard config remains unchanged;
- Step 3 does not strip or reinterpret trainer optimizer/LR fields.

### 15.5 CI

Update `.github/workflows/anima-qwen3-review.yml` to:

- compile the new host modules;
- include them in path triggers;
- run `test_model_component_profiles.py`;
- run `test_parameter_routing.py`;
- keep all existing Step 1/2 tests.

No PyTorch installation is added.

## 16. Planned commit sequence

### Commit 1 — eligibility contract

- explicit optimizer eligibility policy;
- preserve old boolean property;
- tests.

### Commit 2 — model component profiles

- new profile/target registry;
- full-trainer classifiers;
- metadata-driven LoRA profile definitions;
- profile tests.

### Commit 3 — parameter scanner and pure router

- descriptor/alias data model;
- identity scan;
- primary/fallback/frozen/unavailable routing;
- ownership audit;
- router tests.

### Commit 4 — bootstrap and compatibility gate

- full Standard -> Component bootstrap;
- legacy LR=0 migration;
- compatibility blockers;
- pure bootstrap/compatibility tests;
- no request/runtime ownership rewrite.

### Commit 5 — CI and final contract review

- workflow coverage;
- compile/test paths;
- final regression sweep;
- documentation updates if code contracts changed during implementation.

## 17. Step 3 acceptance criteria

Step 3 is complete only when all of the following are true:

- Standard mode has no new runtime call path.
- Component Start is still blocked before trainer runtime staging.
- The host can build deterministic descriptors from supplied model roots.
- Every descriptor is keyed by real parameter identity, not only a string name.
- Alias conflicts (including adapter-target or eligibility disagreement) fail closed.
- Train=true on a profile-valid but actually absent Component fails closed.
- Every known full-model family has an explicit Component Profile.
- LoRA profiles consume explicit original-target metadata and never guess when metadata is absent.
- Muon eligibility is model-profile-approved, not `ndim == 2`.
- Pure-Muon components do not need a fallback.
- Mixed eligible/ineligible Muon components require fallback only when the real scan proves it is needed.
- Frozen parameters are represented explicitly by Train Off; Step 3 itself does not mutate `requires_grad`.
- Unassigned and duplicate/conflicting ownership counts are zero for a valid plan.
- Higher-level trainer targets remain authoritative.
- Legacy LR=0 bootstrap becomes Train Off.
- Unsupported legacy multi-group/fused/LoRA+/regex-LR semantics fail closed.
- Step 3 does not strip or reinterpret trainer-side optimizer/LR fields; ownership transfer is deferred to Step 6.
- No optimizer, scheduler, Accelerate, save/resume, device movement, or tensor-value inspection is introduced.
- All host tests pass without installing PyTorch.

## 18. What Step 4 receives

Step 4 can assume the following stable contract exists:

- Component IDs;
- `AdapterTargetMetadata`;
- descriptor/routing APIs;
- eligibility-policy API;
- fail-closed behavior when adapter metadata is missing.

Step 4's job is then narrowly scoped to making real LoRA modules provide the metadata required by this router.
