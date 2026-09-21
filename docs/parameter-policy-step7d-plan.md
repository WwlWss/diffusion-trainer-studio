# Parameter Policy Step 7D — Curated Presets and User Documentation

Step 7D packages the merged Step 7A/7B/7C editor/runtime work into a usable
release surface. It adds a small set of deliberately non-trivial Component
presets, makes the legacy preset application path safe for Component state, and
adds a user guide.

This stage does **not** add trainer/runtime semantics, change Component IDs,
relax Step 6F blockers, or change Parameter Policy sidecar v1.

> Step 7D was planned on the Step 7C head. PR #35 targets `main`, which
> contains the merged Step 7C implementation.

## Goals

1. Ship a small set of useful Component examples rather than ten redundant
   "same as Standard" presets.
2. Ensure Component presets preserve explicit `false` / `0` values instead
   of losing them to the legacy preset-diff helper.
3. Keep presets partial: they may configure optimizer/policy semantics and a few
   compatibility controls, but must not overwrite user model/dataset/output
   paths.
4. Validate every preset against the same backend registries and runtime
   blocker logic used by Preview/Start.
5. Add one authoritative user guide for Standard vs Component, Muon fallback,
   blockers, export/import, checkpoint identity, and troubleshooting.
6. Add README entry points without rewriting unrelated documentation.

## Non-goals

Step 7D does not:

- create a second preset loader;
- create one preset per backend merely to mirror Standard;
- make example learning rates into recommended universal defaults;
- hide Standard optimizer/LR controls;
- dynamically populate profile-reference dropdowns;
- remove Step 6F qualification blockers;
- enable explicit multi-GPU/DDP;
- enable DeepSpeed, compile, fused paths, swap/offload, full FP16/BF16, or FP8;
- add CUDA CI;
- change checkpoint/resume semantics.

# 7D1 — Component preset application semantics

## Existing problem

The pinned frontend applies a preset through:

~~~javascript
$=_=>{
    let m=findChangedDataBySchema(_,n.value);
    a.value==null
        ? a.value=clone(m)
        : a.value=Object.assign({},a.value,m)
}
~~~

`findChangedDataBySchema()` deliberately removes values equal to schema
defaults. That is fine for historical Standard presets but unsafe for Component
examples.

Example:

~~~toml
train_qwen3_text_encoder = false
blocks_to_swap = 0
deepspeed = false
~~~

Those explicit values may be filtered as defaults. If the current browser form
contains `true`, a nonzero swap value, or another stale advanced setting, the
Component preset cannot reset it.

The same problem exists for policy editor values when a preset intentionally
contains a complete Component state.

## Frontend change

Modify `mikazuki/frontend_training_patch.py`.

Add a pure helper adjacent to the 7C helpers:

~~~javascript
__isComponentPreset=_=>{
    let P=_&&_.parameter_policy_profiles,
        C=_&&_.parameter_policy_components;
    return String(_&&_.optimization_mode||"").toLowerCase()=="component"
        && P&&typeof P=="object"&&Object.keys(P).length
        && C&&typeof C=="object"&&Object.keys(C).length
}
~~~

Patch the existing preset-data merge function `$=_=>{...}` only.

Recommended final behavior:

~~~javascript
$=_=>{
    const Pc=__isComponentPreset(_);

    if(Pc){
        ++__policyBootstrapGeneration.value;
        __policyBootstrapPending.value=false;
    }

    let m=Pc
        ? clone(_)
        : findChangedDataBySchema(_,n.value);

    a.value==null
        ? a.value=clone(m)
        : a.value=Object.assign({},a.value,m);

    if(Pc){
        __runtimeReady.value=false;
        __runtimeBlockers.value=[];
        __refreshPreview();
    }

    console.log(a.value)
}
~~~

### Why exact merge instead of full state replacement

A preset must not overwrite:

- model checkpoint paths;
- dataset paths;
- output paths/names;
- sampling prompts;
- unrelated user settings not listed in the preset.

Therefore Component presets still **merge only their listed keys** into the
current form. The difference is that their listed values are treated literally;
`false`, `0`, empty arrays, and nested policy dictionaries are not reduced
to "differences from schema defaults".

### Why cancel an in-flight bootstrap

A user can open/apply a Component preset while an earlier Standard -> Component
bootstrap request is still pending. Component preset application is an explicit
user action and becomes authoritative.

Incrementing `__policyBootstrapGeneration` ensures the old bootstrap response
cannot overwrite the preset. Clearing `__policyBootstrapPending` allows the
preset's Preview to run immediately.

### Standard preset compatibility

Historical Standard presets continue to use
`findChangedDataBySchema(_, n.value)` unchanged.

Do not key this behavior on a new metadata flag. A complete Component preset is
identified from the same public state contract used everywhere else:

- `optimization_mode=component`;
- non-empty `parameter_policy_profiles`;
- non-empty `parameter_policy_components`.

## 7D1 tests

Extend `tests/test_frontend_effective_config_patch.py`.

Lock:

1. `__isComponentPreset` requires Component mode plus non-empty profile and
   component maps.
2. Component preset data uses `clone(_)`, not
   `findChangedDataBySchema(_, n.value)`.
3. Historical Standard preset data still uses
   `findChangedDataBySchema(_, n.value)`.
4. Component preset application increments
   `__policyBootstrapGeneration`.
5. It clears `__policyBootstrapPending`.
6. It invalidates `__runtimeReady` and triggers Preview.
7. It still merges into the current form instead of replacing `a.value`.

# 7D2 — Four curated Component presets

Do not create ten "same as Standard" presets. Standard -> Component bootstrap is
already the migration path for that use case.

Add exactly four curated examples.

All optimizer-profile `args` values must use **editor strings**, because the
7B public schema is `Schema.dict(Schema.string())`.

All learning-rate fields inside
`parameter_policy_components` must also be strings.

All component IDs must be quoted TOML keys so dots stay literal.

## Preset A — Flux Full: Muon + AdamW fallback

File:

`config/presets/component-flux-finetune-muon.toml`

Metadata:

~~~toml
[metadata]
name = "Component：Flux Full — Muon + AdamW fallback"
version = "1.0"
author = "Diffusion Trainer Studio"
train_type = "flux-finetune"
description = "示例：Flux 全参按组件使用 Muon，非 Muon eligible 参数回退到 AdamW。学习率仅用于演示，请按数据集调整。"
~~~

Baseline data should explicitly keep the preset inside Step 6F:

~~~toml
[data]
optimization_mode = "component"
mixed_precision = "bf16"
full_fp16 = false
full_bf16 = false
fp8_base = false
fp8_base_unet = false
cpu_offload_checkpointing = false
fused_backward_pass = false
blockwise_fused_optimizers = false
blocks_to_swap = 0
deepspeed = false
torch_compile = false
~~~

Profiles:

- `muon`: `Muon`, empty args;
- `fallback`: `AdamW`, empty args.

Components must cover exactly:

- `transformer.double_stream`
- `transformer.single_stream`
- `transformer.modulation_norm_other`
- `transformer.input_conditioning`
- `transformer.final`
- `transformer.other`

Example routing:

- double/single stream: `1e-4`
- remaining components: `5e-5`
- primary `muon`
- fallback `fallback`
- fallback LR equal to the route LR

This demonstrates eligibility routing without pretending every parameter can use
Muon.

## Preset B — Anima Full: DiT Muon + AdamW fallback, Qwen3 frozen

File:

`config/presets/component-anima-finetune-muon.toml`

Metadata `train_type = "anima-finetune"`.

Explicit baseline:

~~~toml
[data]
optimization_mode = "component"
anima_precision_mode = "mixed_bf16"
anima_checkpoint_mode = "standard"
train_qwen3_text_encoder = false
fused_backward_pass = false
blocks_to_swap = 0
deepspeed = false
torch_compile = false
~~~

Do **not** copy the historical Standard preset's
`anima_precision_mode = "full_bf16"`; Step 6F intentionally blocks that in
Component mode.

Profiles:

- `muon`
- `fallback = AdamW`

Components must cover exactly:

- `dit.self_attention`
- `dit.cross_attention`
- `dit.mlp`
- `dit.modulation`
- `dit.llm_adapter`
- `dit.base_other`
- `qwen3`

Suggested example:

- self/cross/MLP: trained with Muon + fallback at `1e-4`;
- modulation/base_other: trained with Muon + fallback at `5e-5`;
- optional `dit.llm_adapter`: frozen;
- `qwen3`: frozen.

This must remain consistent with `train_qwen3_text_encoder = false`.

## Preset C — SDXL Full: split U-Net / Text Encoder LR

File:

`config/presets/component-sdxl-finetune-split-lr.toml`

Important frontend metadata:

~~~toml
train_type = "sdxl-full"
~~~

The visible page/schema key is `sdxl-full`; `PAGE_BACKEND_MAP` resolves it to
the runtime backend `sdxl-finetune`.

Explicit baseline:

~~~toml
[data]
optimization_mode = "component"
train_text_encoder = true
cache_text_encoder_outputs = false
cache_text_encoder_outputs_to_disk = false
mixed_precision = "bf16"
full_fp16 = false
full_bf16 = false
fused_backward_pass = false
deepspeed = false
torch_compile = false
~~~

Use one `AdamW` profile.

Components must cover exactly:

- `unet.transformer`
- `unet.conv_resnet`
- `unet.norm_bias_other`
- `unet.base_other`
- `text_encoder_1`
- `text_encoder_2`

Illustrative LRs:

- U-Net transformer: `1e-6`
- U-Net conv/resnet: `8e-7`
- U-Net norm/bias + other: `5e-7`
- TE1: `5e-7`
- TE2: `2.5e-7`

The point is to demonstrate independent component LR, not to claim an optimal
SDXL recipe.

## Preset D — SDXL LoRA: selective U-Net components

File:

`config/presets/component-sdxl-lora-selective.toml`

Metadata `train_type = "sdxl-lora"`.

Explicit target state:

~~~toml
[data]
optimization_mode = "component"
lora_target = "unet"
mixed_precision = "bf16"
~~~

Use one `AdamW` profile.

Components must cover exactly:

- `unet.attention.adapter`
- `unet.feed_forward.adapter`
- `unet.conv.adapter`
- `unet.other.adapter`
- `text_encoder_1.adapter`
- `text_encoder_2.adapter`

Example:

- train U-Net attention at `1e-4`;
- train U-Net FFN at `7.5e-5`;
- freeze U-Net conv/other;
- freeze both text-encoder adapters.

This is the compact LoRA example for Train/Freeze plus independent LR.

## Preset dependency policy

Use `AdamW`, not `AdamW8bit`, as the fallback/default example optimizer so
the presets do not add a bitsandbytes dependency.

Muon examples rely only on DTS's already-pinned
`pytorch-optimizer==3.10.0`.

No preset should use restricted/planned optimizer types.

# 7D3 — Preset runtime contract

Add:

`tests/test_parameter_policy_presets.py`

This is not a string-only test.

## Static structure

Define the exact four curated filenames and expected visible page type/backend.

For every file:

1. parse TOML;
2. require `metadata.train_type`;
3. resolve backend via `PAGE_BACKEND_MAP`;
4. require `data.optimization_mode == "component"`;
5. require non-empty profile/component objects;
6. call `normalize_parameter_policy_editor_state()`;
7. canonicalize/validate the policy;
8. assert component IDs exactly equal
   `get_model_component_profile(backend).components`;
9. assert every profile's optimizer is registered;
10. assert every newly selectable profile has
    `component_support == "supported"`.

## Muon contract

For every route whose primary profile is Muon:

- a fallback profile must be present;
- the fallback optimizer must not require eligibility;
- fallback cannot self-reference;
- fallback LR must be a positive editor value.

The test should rely on the canonical policy validator rather than duplicate
Muon routing rules.

## Runtime blocker contract

The dependency-light host contract should still run the same semantic layers
that matter before trainer launch:

~~~python
normalize_parameter_policy_editor_state(data)
policy_path, sidecars, policy = build_parameter_policy_sidecar(data, page_type)
prepared = prepare_training_config(
    data,
    page_train_type=page_type,
    resolve_backend=fake_resolve_backend,
    launch=False,
)
blockers = parameter_policy_runtime_blockers(
    policy,
    train_type=prepared.train_type,
    effective_config=prepared.config,
    integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
)
target = resolve_training_target_profile(
    prepared.train_type,
    prepared.config,
)
~~~

Require zero request-level blockers and require every `Train=true` policy
component to be available in the resolved target profile.

This intentionally validates:

- visible page alias -> backend resolution;
- Component sidecar compilation;
- effective-config semantic normalization;
- public target controls such as `lora_target`;
- target-profile availability;
- Step 6F compatibility;
- optimizer capability;
- Qwen3/TE target consistency.

The test must remain dependency-light; it should not import the FastAPI request
layer merely to validate presets.

## Curated-only contract

Require exactly these four `component-*.toml` presets in Step 7D. This prevents
accidental growth into ten redundant Standard mirrors.

# 7D4 — User guide

Add:

`docs/parameter-policy.md`

Use this as the authoritative technical user guide.

Required sections:

1. **Standard vs Component**
   - Standard is the historical DTS optimizer path.
   - Component is opt-in and owns optimizer/LR routing when active.
   - hidden/stale Component state does not affect Standard.

2. **Switching Standard -> Component**
   - 7C resolves the current Standard schema first, including GUI defaults;
   - backend bootstrap is exact-or-fail for representable semantics;
   - existing complete Component state is never overwritten.

3. **Optimizer Profiles**
   - profile name, optimizer type, optimizer args;
   - supported choices vs disabled restricted/planned choices;
   - editor args are string literals and are normalized by the host.

4. **Components**
   - Train/Freeze;
   - primary LR;
   - optimizer profile reference;
   - fallback profile/LR;
   - backend Component IDs are registry-owned.

5. **Muon eligibility and fallback**
   - only model-profile-approved hidden-layer 2D weights are Muon-eligible;
   - other parameters route to the explicit fallback profile;
   - DTS does not use Muon's internal AdamW fallback controls;
   - `pytorch-optimizer==3.10.0`.

6. **Preview and runtime readiness**
   - Component edits invalidate stale green readiness immediately;
   - Preview exposes runtime blockers;
   - Start remains disabled until current Preview is ready;
   - backend Start validation remains authoritative.

7. **Curated presets**
   - explain the four examples;
   - learning rates are illustrative;
   - Component presets merge listed keys only;
   - unrelated user model/dataset/output fields are preserved;
   - Preview remains authoritative if unrelated stale advanced settings exist.

8. **Export / Import**
   - ordinary Standard/no-sidecar export may remain TOML;
   - Component export carries the content-addressed policy sidecar in
     `.dts.json`;
   - import validates hashes/owned paths and rehydrates editor-safe values.

9. **Checkpoint / resume identity**
   - document only guarantees currently enforced by manifest/runtime tests;
   - explain that incompatible Component ownership/optimizer topology is
     rejected rather than silently resumed.

10. **Supported backend matrix**
    exactly ten runtime backends:
    - sd-lora
    - sdxl-lora
    - sd-dreambooth
    - sdxl-finetune
    - sd3-lora
    - flux-lora
    - chroma-lora
    - flux-finetune
    - anima-lora
    - anima-finetune

    Also show visible aliases where useful, especially
    `sdxl-full -> sdxl-finetune`.

11. **Qualification blockers vs semantic blockers**
    Explain the distinction.

    Current qualification examples:
    - full FP16/BF16;
    - FP8 base;
    - compile;
    - DeepSpeed;
    - fused paths;
    - swap/offload;
    - explicit multi-GPU.

    Current semantic examples should be taken from the actual compatibility
    registry/tests at implementation time, not copied from memory.

12. **Troubleshooting / return to Standard**
    - switch back to Standard at any time;
    - hidden Component state is ignored by Standard;
    - Reset All can be used before a fresh migration;
    - Preview blocker text is the first diagnostic;
    - imported restricted optimizer policies remain readable but Start-blocked.

## README entry points

Modify:

- `README.md`
- `README-zh.md`

Add a concise highlight/link only:

- Component-wise Parameter Policy is available on the ten qualified backends;
- link to `docs/parameter-policy.md`.

Do not duplicate the full blocker matrix into both READMEs.

## Stale comments / planning drift

Update:

- `mikazuki/parameter_policy_editor.py`
  - change "future GUI surface" to the now-public GUI surface.
- `docs/parameter-policy-step7c-plan.md`
  - update the bootstrap snapshot example to show
    `return __resolveGuiState(R)`;
  - add `__policyBootstrapPending` to bootstrap entry guards.

Historical Step 3-6 plan documents should remain historical unless they contain
an actively misleading user-facing statement.

# 7D5 — CI/path wiring

Modify `.github/workflows/anima-qwen3-review.yml` path filters so preset/doc-only
follow-up PRs still run relevant host checks:

- `config/presets/component-*.toml`
- `docs/parameter-policy*.md`

The existing:

~~~text
python -m unittest discover -s tests -p 'test_parameter_policy*.py' -v
~~~

will automatically execute `test_parameter_policy_presets.py`.

No new workflow/job is needed.

# Planned commit sequence

## 7D1 — Component preset application contract

Files:

- `mikazuki/frontend_training_patch.py`
- `tests/test_frontend_effective_config_patch.py`

Review boundary:

- Standard preset behavior unchanged;
- Component preset exact listed-value merge;
- stale bootstrap cancellation.

## 7D2 — Curated preset set

Files:

- four new `config/presets/component-*.toml`
- new `tests/test_parameter_policy_presets.py`
- workflow path filter update if needed in the same commit

Review boundary:

- exact backend component sets;
- supported optimizers only;
- no runtime blockers on clean Preview preparation.

## 7D3 — User guide and release entry points

Files:

- `docs/parameter-policy.md`
- `README.md`
- `README-zh.md`
- `mikazuki/parameter_policy_editor.py` stale docstring
- `docs/parameter-policy-step7c-plan.md` implementation drift cleanup

No runtime behavior changes.

## 7D4 — Final preset/document regression review

Only fix issues found by:

- final patched frontend syntax;
- Parameter Policy host contract;
- preset runtime blocker test;
- runtime smoke;
- source-level review.

Do not add new feature scope here.

# Merge criteria

Step 7D is mergeable when:

- all four Component presets appear on their intended visible training pages;
- applying a Component preset preserves explicit `false` / `0` values;
- ordinary Standard preset behavior is unchanged;
- applying a Component preset cannot be overwritten by an older bootstrap
  response;
- every preset contains the exact backend Component set;
- every preset uses only supported optimizer capabilities;
- every Muon route has an explicit valid fallback;
- every preset prepares with zero Parameter Policy runtime blockers on a clean
  host Preview path;
- no preset enables a Step 6F advanced blocker;
- README links resolve to the user guide;
- the guide accurately describes the current ten-backend matrix and blocker
  model;
- source/CPU/runtime-smoke CI remains green.
