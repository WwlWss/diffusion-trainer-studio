# Parameter Training Policy — Step 3 / Commit 4 Detailed Plan

Baseline: PR #23 head `0af424f3`.

Commit 4 adds the **Standard -> Component bootstrap** and a **fail-closed
compatibility gate**. It remains host-only and pure: it does not load a model,
scan parameters, construct optimizers/schedulers, alter request/launch
ownership, or make Component Start runnable.

> Naming note: this is **Step 3 / Commit 4** inside PR #23. It is not the later
> project Step 4 that wires real LoRA original-target metadata.

## 1. Commit 4 output

Add:

- `mikazuki/parameter_policy_compat.py`
- `tests/test_parameter_policy_bootstrap.py`
- `tests/test_parameter_policy_compat.py`

Extend:

- `mikazuki/parameter_policy_bootstrap.py`

Small documentation/test updates are allowed where required.

Do **not** modify:

- `mikazuki/training_request.py`;
- request Preview/Export/Start behavior;
- trainer runtime files;
- LoRA network implementations;
- `parameter_policy_runtime_blockers()` ownership;
- Component Start blocking;
- scheduler/Accelerate/save-resume behavior.

Standard mode remains an untouched legacy runtime path.

## 2. Public APIs

### 2.1 Compatibility gate

```python
parameter_policy_compatibility_blockers(
    effective_config: Mapping[str, Any],
    train_type: str,
) -> list[str]
```

Properties:

- input is an already compiled **Standard effective config**;
- does not mutate input;
- deterministic ordering;
- duplicate messages removed;
- exact semantic detection, never substring guessing;
- returns every independent blocker it can prove in one pass;
- does not include optimizer capability support/planned blockers; those remain
  the job of `parameter_policy_runtime_blockers()`.

### 2.2 Full bootstrap

```python
bootstrap_parameter_policy_from_standard(
    config: Mapping[str, Any],
    page_type: str | None,
    *,
    resolve_backend: Callable,
) -> dict[str, Any]
```

Returns a strict canonical v1 Parameter Policy:

```json
{
  "version": 1,
  "optimizer_profiles": {
    "legacy_main": { "...": "..." }
  },
  "components": {
    "...": { "...": "..." }
  }
}
```

If the Standard configuration uses semantics that cannot be reproduced exactly,
the bootstrap raises a deterministic `ValueError` listing the compatibility
blockers. It never returns an approximate policy.

The existing:

```python
bootstrap_parameter_policy_optimizer_profile(...)
```

remains public and behavior-compatible.

## 3. Compile Standard exactly once

Refactor the existing bootstrap helper around an internal function:

```python
_prepare_standard_snapshot(
    config,
    page_type,
    *,
    resolve_backend,
) -> PreparedTrainingConfig
```

Algorithm:

1. copy caller config;
2. remove all Parameter Policy GUI keys;
3. remove any stale/derived `parameter_policy_config`;
4. run `train_utils.fix_config_types()`;
5. call the existing `prepare_training_config(..., launch=False)`;
6. return the exact resulting `PreparedTrainingConfig`.

Both bootstrap entry points use this helper.

Important invariants:

- caller input is never mutated;
- `ui_custom_params` still applies with the exact current Standard
  last-write-wins semantics;
- backend normalization occurs before bootstrap;
- bootstrap never reimplements the raw GUI compiler;
- no argparse defaults are invented after the effective config is produced.

## 4. Bootstrap order

`bootstrap_parameter_policy_from_standard()` executes in this order:

1. compile Standard snapshot;
2. run `parameter_policy_compatibility_blockers(prepared.config, prepared.train_type)`;
3. if blockers exist, fail closed before producing policy;
4. create `legacy_main` using existing
   `bootstrap_legacy_optimizer_profile(prepared.config)`;
5. resolve `ModelComponentProfile`;
6. resolve `TrainingTargetProfile`;
7. calculate the exact legacy LR for every profile Component;
8. convert target-unavailable or legacy LR=0 components to
   `{"train": false}`;
9. convert positive LR components to
   `{"train": true, "optimizer_profile": "legacy_main", "learning_rate": ...}`;
10. do **not** invent a fallback optimizer;
11. strict-validate the completed policy through
    `validate_parameter_policy()`;
12. return the canonical result.

The returned Component key set must equal the active
`ModelComponentProfile.components` key set. This makes bootstrap output
predictable and keeps the UI from receiving an arbitrary partial Component
schema.

## 5. Numeric LR contract

Bootstrap uses explicit helpers instead of Python truthiness such as
`value or base_lr`.

### 5.1 Base LR

A backend that needs legacy `learning_rate` must find it in the compiled
effective config.

Bootstrap does not guess sd-scripts' argparse default if the effective config
did not materialize the value.

Accepted LR values:

- finite number > 0 -> Train=true;
- exactly 0 -> Train=false;
- negative / NaN / infinity / boolean / malformed -> bootstrap error.

### 5.2 Verified fallback only

`None` / empty may inherit another LR only where the real trainer source has a
verified fallback rule.

Examples:

- stable LoRA `unet_lr is None -> learning_rate`;
- stable LoRA `text_encoder_lr is None -> learning_rate`;
- Anima Full optional sub-LR `None -> base_lr`;
- SDXL Full TE LR `None -> learning_rate`.

No other missing field is guessed.

### 5.3 Legacy zero means explicit freeze

No bootstrapped trained row may contain `learning_rate=0`.

All verified legacy zero-LR semantics become exactly:

```json
{"train": false}
```

This is the migration bridge from historical trainer freeze-by-zero to the new
explicit Train flag.

## 6. Multiple Text Encoder LR helper

Dev NetworkTrainer implementations have real list semantics, so Commit 4 must
not collapse them to one scalar.

Internal helper:

```python
_expand_text_encoder_lrs(
    raw_value,
    *,
    count: int,
    base_lr: float,
) -> tuple[float, ...]
```

Exact reviewed semantics:

### Flux / Chroma, count=2

- missing / empty list -> `[base, base]`;
- scalar -> `[value, value]`;
- one item -> `[item0, item0]`;
- two-or-more -> first two values.

Meaning:

- index 0 = CLIP-L;
- index 1 = T5XXL.

Chroma has no CLIP-L Component, but its T5 Component still uses index 1 because
the shared Flux LoRA network's optimizer grouping uses that slot.

### SD3, count=3

- missing / empty -> `[base, base, base]`;
- scalar -> repeat all three;
- one item -> repeat item0;
- two items -> `[item0, item1, item1]`;
- three-or-more -> first three.

Meaning:

- index 0 = CLIP-L;
- index 1 = CLIP-G;
- index 2 = T5XXL.

Every expanded value goes through finite/nonnegative LR validation.

Stable SD/SDXL LoRA does not use this helper: its reviewed LoRA implementation
has one Text Encoder LR for all text encoders.

## 7. Per-backend bootstrap mapping

Higher-level `TrainingTargetProfile` is authoritative in every mapping. A
target-unavailable Component is always emitted as `Train=false`, regardless
of stale LR fields.

### 7.1 SD LoRA

U-Net adapter Components:

- `unet.attention.adapter`
- `unet.feed_forward.adapter`
- `unet.conv.adapter`
- `unet.other.adapter`

LR:

```text
unet_lr if explicitly present else learning_rate
```

Text encoder:

```text
text_encoder.adapter
-> text_encoder_lr if explicitly present else learning_rate
```

Target flags may disable either side.

### 7.2 SDXL LoRA

Same four U-Net adapter Components and U-Net LR rule as SD LoRA.

Both:

- `text_encoder_1.adapter`
- `text_encoder_2.adapter`

inherit the single stable-LoRA:

```text
text_encoder_lr if present else learning_rate
```

This intentionally does not invent separate TE1/TE2 LRs that the existing
stable LoRA path did not have.

### 7.3 Flux LoRA

Transformer Components:

- `transformer.double_stream.adapter`
- `transformer.single_stream.adapter`
- `transformer.input_conditioning.adapter`

use:

```text
unet_lr if present else learning_rate
```

Text encoders use the reviewed 2-slot expansion:

- `clip_l.adapter` -> TE slot 0;
- `t5xxl.adapter` -> TE slot 1.

Target availability still controls whether CLIP-L/T5 are Train Off.

### 7.4 Chroma LoRA

Transformer Components use `unet_lr/base`.

There is no CLIP-L Component.

`t5xxl.adapter` uses Flux TE slot 1.

### 7.5 SD3 LoRA

MMDiT Components:

- `mmdit.attention.adapter`
- `mmdit.mlp.adapter`
- `mmdit.modulation_norm.adapter`
- `mmdit.other.adapter`

use `unet_lr/base`.

Text encoders use the reviewed 3-slot expansion:

- CLIP-L -> slot 0;
- CLIP-G -> slot 1;
- T5XXL -> slot 2.

### 7.6 Anima LoRA

DiT adapter Components:

- `dit.self_attention.adapter`
- `dit.cross_attention.adapter`
- `dit.mlp.adapter`
- `dit.modulation.adapter`
- `dit.other.adapter`
- `llm_adapter.adapter`

use `unet_lr/base`.

`qwen3.adapter` uses `text_encoder_lr/base`.

The target profile still owns DiT-only, Qwen-only, joint, and
`train_llm_adapter` availability.

### 7.7 SD DreamBooth

All U-Net Components use `learning_rate`.

Text encoder:

```text
learning_rate_te if present else learning_rate
```

Only two static Standard states are exactly bootstrappable:

- `stop_text_encoder_training is None` -> TE target available;
- `stop_text_encoder_training < 0` -> TE Train=false.

A nonnegative explicit stop value is a compatibility blocker because it means
"train the text encoder until global step N, then freeze it". Static Component
Train=true/false cannot reproduce that temporal behavior.

Commit 4 deliberately blocks even values that might happen to exceed the
current planned run length rather than trying to infer training-duration
equivalence from unrelated schedule fields.

### 7.8 SDXL Full

All four U-Net Components use `learning_rate`.

If `train_text_encoder=false`:

- TE1/TE2 -> Train=false.

Otherwise:

- TE1 -> `learning_rate_te1 if not None else learning_rate`;
- TE2 -> `learning_rate_te2 if not None else learning_rate`.

Explicit zero freezes each independently.

`block_lr` is blocked before mapping.

### 7.9 Flux Full

Every Transformer Component uses `learning_rate`.

No text encoder Components exist in the profile and Component mode must not
create them.

### 7.10 Anima Full

- `dit.base_other` -> `learning_rate`;
- `dit.self_attention` -> `self_attn_lr or base`;
- `dit.cross_attention` -> `cross_attn_lr or base`;
- `dit.mlp` -> `mlp_lr or base`;
- `dit.modulation` -> `mod_lr or base`;
- `dit.llm_adapter` -> `llm_adapter_lr or base`;
- `qwen3` -> `qwen3_lr` only when Qwen joint training is target-available.

The optional LLM Adapter deserves explicit treatment:

- Standard trains it at `llm_adapter_lr/base` **if it exists**;
- host bootstrap does not load the checkpoint, so it cannot prove presence;
- bootstrap therefore preserves Standard intent with the corresponding
  Train=true row when LR > 0;
- the real parameter scan later remains authoritative and may return
  `component_absent` if the selected checkpoint has no LLM Adapter.

This is fail-closed and preserves intent. Silently changing it to Train=false
would alter Standard behavior for checkpoints that do contain the adapter.

## 8. Potential vs actual LoRA Components

Model Component Profiles define the potential architectural Component schema.
A particular LoRA network may not instantiate every potential adapter family.

Bootstrap preserves the legacy target/LR **intent** for the full potential
schema. It does not pretend to know actual adapter presence without a model
scan.

Therefore a bootstrap policy is guaranteed to be structurally canonical, but
not guaranteed routing-valid for a concrete network until
`build_parameter_routing_plan()` sees the real descriptors.

This is intentional:

- never silently Train Off a component that Standard would train if present;
- never invent adapter presence;
- later `component_absent` is a safe, actionable failure.

## 9. Compatibility parser for network_args

Add a dependency-light exact parser for normalized `network_args`.

Input forms accepted:

- missing / empty -> empty mapping;
- string -> one normalized item (or line-separated items);
- list / tuple -> items in order.

Each non-empty item must be a `key=value` entry.

Rules:

- split only on the first `=`;
- trim key;
- casefold key for comparison;
- preserve value as text; compatibility only needs semantic presence here;
- duplicate key -> later item wins, matching current host normalization;
- malformed item / empty key -> `ValueError`.

Blockers compare exact keys. For example, `my_loraplus_lr_ratio_backup` does
not match `loraplus_lr_ratio`.

## 10. Compatibility blockers

### 10.1 Global optimizer-runtime blockers

For every Component bootstrap backend, block active:

- `fused_backward_pass=true`;
- `deepspeed=true`;
- positive `fused_optimizer_groups`;
- `blockwise_fused_optimizers=true`.

These change optimizer ownership/stepping semantics and belong to later
CompositeOptimizer/runtime integration.

Do not block unrelated memory/offload/compile switches merely because they are
advanced options.

### 10.2 SDXL Full block LR

For `sdxl-finetune`:

- non-empty `block_lr` -> blocker.

Commit 4 does not flatten 23 per-block LRs into four U-Net Components.

### 10.3 SD DreamBooth temporal TE freeze

For `sd-dreambooth`:

- explicit `stop_text_encoder_training >= 0` -> blocker.

This is a newly identified compatibility requirement and is mandatory.

### 10.4 SD / SDXL LoRA block LR weighting

For `sd-lora` / `sdxl-lora`:

if any of these exact keys is present:

- `down_lr_weight`;
- `mid_lr_weight`;
- `up_lr_weight`;

emit one block-weight LR blocker.

`block_lr_zero_threshold` by itself is inert in the reviewed LoRA
implementation because block weighting returns disabled when all three weight
keys are absent. It only participates in the same blocker when block weighting
is actually active.

### 10.5 LoRA+

For reviewed LoRA families, block presence of any exact key:

- `loraplus_lr_ratio`;
- `loraplus_unet_lr_ratio`;
- `loraplus_text_encoder_lr_ratio`.

This is blocked even when the numeric ratio happens to be 1 because the legacy
implementation creates separate `lora_up` optimizer groups and Component v1
does not yet own that grouping contract.

Applies to:

- SD / SDXL LoRA;
- Flux / Chroma LoRA;
- SD3 LoRA;
- Anima LoRA where the pinned network exposes the same semantics.

### 10.6 Regex-specific LR

Block exact `network_reg_lrs` where the reviewed implementation uses regex
matches to create LR groups:

- Flux LoRA;
- Chroma LoRA;
- Anima LoRA.

Do not block `network_reg_dims`; rank/dimension selection is not itself an LR
ownership semantic.

Do not invent an SD3 regex-LR blocker when the reviewed SD3 implementation does
not expose that feature.

### 10.7 Unreviewed/custom network optimizer grouping

A Standard LoRA config may select a different `network_module` whose
`prepare_optimizer_params()` owns custom grouping semantics.

Commit 4 must not claim exact migration for an implementation it has not
reviewed.

Canonical reviewed modules:

- SD / SDXL LoRA: `networks.lora`;
- Flux / Chroma LoRA: `networks.lora_flux`;
- SD3 LoRA: `networks.lora_sd3`;
- Anima LoRA: `networks.lora_anima`.

Missing `network_module` is allowed when the backend compiler intentionally
uses its canonical trainer default.

An explicitly different network module is a compatibility blocker until its
grouping contract is reviewed. This covers arbitrary Custom network modules,
LyCORIS/DyLoRA/OFT-style implementations, and prevents bootstrap from silently
assuming regular-LoRA optimizer semantics.

## 11. Compatibility gate vs capability gate

Keep these separate.

`parameter_policy_compatibility_blockers()` asks:

> Can Component v1 represent this Standard optimizer/LR/grouping behavior
> exactly?

`parameter_policy_runtime_blockers(policy)` asks:

> Is the selected Component optimizer implementation currently runnable?

Examples:

- legacy LoRA+ -> compatibility blocker before bootstrap;
- a structurally representable but not-yet-supported optimizer -> may bootstrap,
  then remain blocked by optimizer capability/runtime gating.

Do not merge these responsibilities into one function.

## 12. No fallback invented during migration

Bootstrap produces only `legacy_main`.

It never creates a second optimizer profile merely because the current
optimizer has parameter eligibility semantics.

If a legacy optimizer is structurally representable as an eligibility-gated
profile and later routing proves that some real parameters need a fallback,
routing fails closed as designed.

Inventing AdamW or another fallback would not be an exact Standard migration.

## 13. No request/runtime integration

Commit 4 does not call the new bootstrap or compatibility helpers from normal
Preview/Export/Start.

It does not:

- add compatibility blockers to normal Standard requests;
- change the Component sidecar compiler;
- strip legacy optimizer/LR fields;
- materialize a routing plan;
- load a checkpoint/model;
- enable Component Start.

The helper is an explicit migration/bootstrap operation for later UI/API work.
Standard training behavior therefore cannot regress from Commit 4.

## 14. Purity and performance

Commit 4 adds no new model loading, tensor work, device work, or parameter scanning.
The bootstrap deliberately reuses the existing Standard effective-config compiler;
therefore it may inherit lightweight checks that Standard already performs, such
as Anima safetensors header inspection for model-variant validation. Commit 4
must not add any additional checkpoint/model inspection beyond that existing
compiler behavior.

No:

- torch import;
- trainer source parsing at runtime;
- tensor work;
- device work;
- model parameter scanning.

Expected additional bootstrap cost after Standard compilation is
O(number of config fields + network_args + Component IDs).

The implementation must not call the model parameter scanner.

## 15. Tests — compatibility

Create `tests/test_parameter_policy_compat.py`.

Required tests:

1. no blockers for ordinary reviewed configs;
2. input config is not mutated;
3. blocker order and dedup are deterministic;
4. network_args exact-key matching, not substring matching;
5. duplicate network_args keys use last occurrence;
6. malformed network_args fails closed;
7. SDXL Full `block_lr`;
8. DreamBooth nonnegative `stop_text_encoder_training`;
9. DreamBooth None / negative stop are not blocked;
10. SD/SDXL block-weight LR;
11. lone `block_lr_zero_threshold` is inert;
12. all LoRA+ keys across applicable families;
13. Flux/Chroma/Anima `network_reg_lrs`;
14. SD3 does not falsely claim regex-LR support;
15. fused backward;
16. fused optimizer groups;
17. blockwise fused optimizers;
18. DeepSpeed;
19. noncanonical explicit network module;
20. unrelated network_args remain allowed.

## 16. Tests — bootstrap

Create `tests/test_parameter_policy_bootstrap.py`.

Required tests:

1. caller raw config remains unchanged;
2. existing profile-only bootstrap behavior remains unchanged;
3. `ui_custom_params` is honored because bootstrap consumes the final Standard
   effective config;
4. every supported backend returns exactly its Model Component Profile key set;
5. output passes strict `validate_parameter_policy()`;
6. output is deterministic for equivalent input;
7. no trained row contains LR=0;
8. legacy zero becomes Train=false;
9. target-unavailable overrides stale positive LR;
10. missing/negative/NaN/infinite required LR fails closed;
11. SD LoRA U-Net/TE fallback rules;
12. SDXL LoRA uses one TE LR for both encoders;
13. Flux scalar/list TE LR expansion;
14. Chroma T5 uses Flux slot 1;
15. SD3 1/2/3-item TE LR expansion, especially 2 -> [0,1,1];
16. Anima LoRA DiT/Qwen mapping;
17. Anima Full optional sub-LR inheritance;
18. Anima Full explicit zero sub-LR -> Train=false;
19. Anima Qwen disabled -> Train=false;
20. Anima Qwen enabled -> qwen3_lr;
21. SDXL Full TE1/TE2 fallback and independent zero freeze;
22. Flux Full all Transformer Components use base LR;
23. DreamBooth TE disabled-from-start mapping;
24. DreamBooth temporal stop rejects bootstrap through compatibility gate;
25. compatibility blockers prevent policy creation instead of returning an
   approximation;
26. bootstrap does not invent fallback optimizer profiles.

## 17. Implementation order

### 4A — compatibility parser and blocker registry

Implement `parameter_policy_compat.py` first.

No bootstrap mapping yet.

Run compatibility tests.

### 4B — Standard snapshot and LR primitives

Refactor existing profile bootstrap to share one Standard compilation helper.

Add:

- finite nonnegative LR parser;
- explicit row builder;
- multiple-TE LR expander.

Run existing bootstrap tests plus primitive tests.

### 4C — backend mapping table/functions

Implement backend-specific mapping functions one family at a time:

1. SD / SDXL LoRA;
2. Flux / Chroma / SD3 LoRA;
3. DreamBooth / SDXL Full / Flux Full;
4. Anima LoRA / Full.

Every mapper consumes the already resolved Model/Target profiles.

### 4D — public full bootstrap and strict round-trip

Add `bootstrap_parameter_policy_from_standard()`.

Final output goes through `validate_parameter_policy()`.

Run all Step 1-3 and Commit 4 host tests.

## 18. Commit 4 acceptance gate

Commit 4 is complete only when:

- ordinary Standard behavior has no new call path;
- caller config is never mutated by migration helpers;
- all supported backend mappings have explicit tests;
- every legacy zero freeze becomes Train=false;
- no missing LR is guessed except a verified trainer fallback;
- multi-TE LR precedence matches the reviewed trainer sources;
- DreamBooth temporal TE freeze is blocked;
- SDXL block LR is blocked;
- LoRA block-weight, LoRA+, regex-LR, fused/multi-optimizer, DeepSpeed, and
  unreviewed network optimizer grouping semantics fail closed;
- compatibility and optimizer capability blockers remain separate;
- bootstrapped output is strict sidecar v1;
- no fallback optimizer is invented;
- no torch/model/runtime dependency is introduced;
- Component Start remains blocked;
- current full CI remains green.
