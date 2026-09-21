# Parameter Policy Step 7E — Full Regression Closure

Step 7E freezes feature scope and closes the release regression matrix for the
merged Step 7A–7D implementation.

This stage is intentionally test-first. By default it should not add new product
behavior, new backends, new optimizer semantics, or remove any Step 6F blocker.
Production code should change only when the new cross-layer regression matrix
finds a real defect.

## Starting point

Step 7D is merged on `main`.

The public workflow now exists end-to-end:

- generated Schemastery Component editor;
- Standard -> Component exact-or-fail bootstrap;
- runtime-readiness Preview and Start gating;
- portable Component sidecars / import-rehydrate;
- curated Component presets;
- user documentation;
- ten qualified Component backends.

Existing unit/contract coverage is already strong. Step 7E must therefore avoid
duplicating lower-level assertions that already exist in:

- `test_parameter_policy_editor.py`
- `test_parameter_policy_schema.py`
- `test_training_schema_overrides.py`
- `test_frontend_effective_config_patch.py`
- `test_parameter_policy_config.py`
- `test_parameter_policy_bootstrap.py`
- `test_parameter_policy_compat.py`
- Step 3/4/6 routing/runtime suites.

The missing value is a release-level matrix that proves those layers still agree
with each other.

# 7E1 — One canonical ten-backend release fixture table

Add:

`tests/test_parameter_policy_step7_contract.py`

Keep it dependency-light:

- no FastAPI import;
- no `mikazuki.app.training_api`;
- no torch runtime import;
- no model load;
- no CUDA;
- no filesystem materialization.

Use one canonical visible page per release backend:

| Visible page | Runtime backend |
| --- | --- |
| `lora-master` | `sd-lora` |
| `sdxl-lora` | `sdxl-lora` |
| `dreambooth` | `sd-dreambooth` |
| `sdxl-full` | `sdxl-finetune` |
| `sd3-lora` | `sd3-lora` |
| `flux-lora` | `flux-lora` |
| `chroma-lora` | `chroma-lora` |
| `flux-finetune` | `flux-finetune` |
| `anima-lora` | `anima-lora` |
| `anima-finetune` | `anima-finetune` |

Represent the table as an immutable tuple/dataclass rather than ten separate
test methods.

Suggested fixture type:

~~~python
@dataclass(frozen=True)
class ReleaseCase:
    page_type: str
    backend: str
    standard_raw: Mapping[str, object]
~~~

The fixture set must map bijectively onto
`PARAMETER_POLICY_RUNTIME_TRAIN_TYPES`.

## Minimal Standard fixtures

Fixtures should use ordinary supported `AdamW` and stay inside currently
representable Standard semantics.

Suggested baselines:

### SD LoRA

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-4",
    "lora_target": "unet_text_encoder",
}
~~~

### SDXL LoRA

Same semantic target, using the `sdxl-lora` page.

### SD DreamBooth

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-6",
}
~~~

Do not set a positive `stop_text_encoder_training`.

### SDXL Full

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-6",
    "train_text_encoder": True,
    "mixed_precision": "bf16",
}
~~~

### SD3 LoRA

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-4",
}
~~~

Missing optional T5 target state is allowed; unavailable components should be
frozen by bootstrap.

### Flux LoRA

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-4",
    "flux_lora_target": "dit",
}
~~~

### Chroma LoRA

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-4",
    "flux_lora_target": "dit",
}
~~~

### Flux Full

~~~python
{
    "optimizer_type": "AdamW",
    "learning_rate": "1e-6",
    "mixed_precision": "bf16",
    "blocks_to_swap": 0,
}
~~~

### Anima LoRA

Use explicit public Anima semantics rather than legacy derived flags:

~~~python
{
    "anima_model_variant": "base",
    "network_module": "networks.lora_anima",
    "optimizer_type": "AdamW",
    "learning_rate": "1e-4",
    "anima_lora_target": "dit",
    "mixed_precision": "bf16",
    "blocks_to_swap": 0,
}
~~~

Do not enable TE output cache in the regression baseline.

### Anima Full

~~~python
{
    "anima_model_variant": "base",
    "optimizer_type": "AdamW",
    "anima_finetune_learning_rate": "1e-5",
    "lr_scheduler": "constant",
    "anima_precision_mode": "mixed_bf16",
    "anima_checkpoint_mode": "standard",
    "blocks_to_swap": 0,
    "train_qwen3_text_encoder": False,
}
~~~

If an existing schema/effective-config default is required for one backend,
add it explicitly to that fixture instead of weakening the matrix.

# 7E2 — Dependency-light host request seam

Inside the new test file, add one helper that mirrors the Parameter Policy
portion of `prepare_request_config()` without importing the FastAPI legacy
module:

~~~python
def _resolve_backend(config, requested):
    return requested, f"./{requested}.py"
~~~

~~~python
def _prepare_policy_request(raw, page_type):
    config = deepcopy(raw)

    train_utils.fix_config_types(config)
    normalize_parameter_policy_editor_state(config)

    policy_path, sidecars, policy = build_parameter_policy_sidecar(
        config,
        page_type,
    )

    prepared = prepare_training_config(
        config,
        page_train_type=page_type,
        resolve_backend=_resolve_backend,
        launch=False,
    )

    prepared.sidecars.update(sidecars)

    if policy is not None:
        prepared.runtime_blockers.extend(
            parameter_policy_runtime_blockers(
                policy,
                train_type=prepared.train_type,
                effective_config=prepared.config,
                integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
            )
        )
        prepared.runtime_blockers.extend(
            parameter_policy_gpu_selection_blockers(prepared.gpu_ids)
        )

    validate_prepared_config(prepared, False)
    return prepared, policy
~~~

The Step 7E fixture intentionally excludes non-Standard Multi-Caption and
prompt-generation controls. The dependency-light helper still calls
`build_multi_caption_sidecar()` so rehydrate's `caption_mode="standard"`
sentinel is normalized exactly like the real request pipeline. Non-Standard
Multi-Caption and prompt sidecars retain their own existing tests; 7E is
closing the Parameter Policy regression surface, not replacing all request
tests.

This helper must preserve the real ordering:

~~~text
fix GUI types
 -> normalize editor literals
 -> extract/build policy sidecar
 -> compile effective trainer config
 -> runtime blocker evaluation
 -> preview validation
~~~

# 7E3 — Standard three-state equivalence matrix

For every `ReleaseCase`, compile three inputs:

1. no `optimization_mode`;
2. explicit `optimization_mode="standard"`;
3. Standard plus stale hidden policy GUI values.

The stale state should be intentionally invalid as Component data, for example:

~~~python
{
    "optimization_mode": "standard",
    "parameter_policy_profiles": {
        "stale": {
            "type": "NotARealOptimizer",
            "args": {"broken": "["},
        }
    },
    "parameter_policy_components": {
        "stale.component": {
            "train": True,
            "optimizer_profile": "missing",
            "learning_rate": "not-a-number",
        }
    },
}
~~~

Because mode is Standard, those hidden editor values must be discarded before
Component validation.

For all three variants require:

- same runtime backend;
- same trainer file;
- same effective trainer config;
- same warnings;
- same GPU selection;
- `policy is None`;
- `prepared.sidecars == {}`;
- `prepared.runtime_blockers == []`;
- no `parameter_policy_config` in effective config.

This is the release-level proof that the public GUI did not change historical
Standard behavior.

Do not duplicate the existing source-level torch-import assertion. The existing
`test_step5_runtime_is_not_wired_into_request_launch_path` remains the
authority for that boundary.

# 7E4 — Ten-backend Component semantic round-trip

For each release fixture perform:

~~~text
Standard raw GUI
 -> bootstrap_parameter_policy_editor()
 -> Component GUI state
 -> host Preview preparation
 -> canonical sidecar snapshot
 -> rehydrate_trainer_config()
 -> host Preview preparation again
~~~

This is the dependency-light semantic equivalent of:

~~~text
Standard
 -> Bootstrap
 -> Preview
 -> Export bundle
 -> Rehydrate
 -> Preview
~~~

The actual API JSON/TOML bundle envelope remains covered by
`test_training_api_overlay_contract.py` and frontend export/import contract
tests. Step 7E should not import FastAPI or the third-party TOML writer merely
to duplicate that wiring.

## First Component pass

~~~python
gui = deepcopy(case.standard_raw)
gui.update(
    bootstrap_parameter_policy_editor(
        case.standard_raw,
        case.page_type,
        resolve_backend=_resolve_backend,
    )
)
prepared_a, policy_a = _prepare_policy_request(gui, case.page_type)
~~~

Require:

- `policy_a` is not `None`;
- one content-addressed policy sidecar exists;
- `parameter_policy_config` points to that staged sidecar;
- `runtime_blockers == []`;
- every `Train=true` route is available in
  `resolve_training_target_profile(prepared_a.train_type, prepared_a.config)`;
- `parameter_policy_editor_preview(...)` returns
  `runtime_ready=True`.

## Rehydrate

Pass the already-staged sidecars directly:

~~~python
rehydrated = rehydrate_trainer_config(
    deepcopy(prepared_a.config),
    case.page_type,
    sidecars=deepcopy(prepared_a.sidecars),
)
~~~

Require:

- `rehydrated["optimization_mode"] == "component"`;
- policy editor profiles/components exist;
- no fallback to Standard.

## Second Component pass

~~~python
prepared_b, policy_b = _prepare_policy_request(
    rehydrated,
    case.page_type,
)
~~~

Require semantic identity:

- `validate_parameter_policy(policy_a) == validate_parameter_policy(policy_b)`;
- same content-addressed policy path;
- same policy sidecar text;
- `prepared_a.config == prepared_b.config`;
- same warnings;
- same runtime blockers;
- same target availability for every `Train=true` route;
- editor Preview summaries are equal.

Exact effective-config comparison permits one already-documented inverse
mapping normalization only: rehydrate emits `memory_mode="auto"` when
`lowram/highvram` are absent, and re-preparation materializes those two flags
as explicit `False`. Normalize only absent-vs-False for `lowram` and
`highvram`; all other effective fields remain exact.

# 7E5 — Cross-layer matrix closure

The new contract should also ensure the release matrix itself cannot drift.

## Release backend bijection

Require:

~~~python
{case.backend for case in RELEASE_CASES}
    == PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
~~~

Require every page resolves through `PAGE_BACKEND_MAP` to its declared
backend.

## Editor / Model Component Profile closure

For every case:

- `parameter_policy_editor_metadata(page_type)["train_type"] == backend`;
- editor component IDs equal
  `get_model_component_profile(backend).components`;
- editor newly selectable optimizer types equal the registry entries whose
  `component_support == "supported"`.

This is intentionally one integrated check. Detailed schema formatting remains
owned by existing schema tests.

## Qualification blocker freeze

Step 7E must prove the Step 6F blocker surface did not shrink accidentally.

Lock the exact current field set:

~~~text
torch_compile
compile
deepspeed
fused_backward_pass
fused_optimizer_groups
blockwise_fused_optimizers
cpu_offload_checkpointing
unsloth_offload_checkpointing
blocks_to_swap
double_blocks_to_swap
single_blocks_to_swap
full_fp16
full_bf16
fp8_base
fp8_base_unet
~~~

Use representative active values:

- booleans -> `True`;
- fused optimizer groups -> `2`;
- block swap counts -> `1`.

Use `anima-finetune` for the Anima-only `compile` field and
`flux-finetune` for the remaining generic qualification fields.

For each field require
`parameter_policy_v1_semantic_blockers(...)` to return a blocker.

Also lock the current semantic-feature registry:

~~~text
lora_plus
regex_lr
sd_lora_block_lr
sdxl_full_block_lr
scale_weight_norms
dreambooth_dynamic_text_encoder_stop
preloaded_adapter_text_encoder_cache
anima_qwen_only
unreviewed_network_module
~~~

Do not reimplement each semantic-blocker parser here. Their detailed behavior
remains owned by `test_parameter_policy_compat.py`; Step 7E only freezes the
release taxonomy.

# 7E6 — Existing suite extensions

Do not add redundant tests merely because the original Step 7 document listed
them.

Existing coverage already proves:

- one editor per release schema;
- component IDs derive from Model Component Profiles;
- supported/restricted optimizer schema choices;
- Standard default;
- frontend bootstrap guards;
- import preserves complete Component state;
- blocked Preview disables Start;
- preset exact-merge behavior;
- API portable-bundle wiring;
- runtime smoke.

Only extend existing suites if the new matrix exposes a concrete uncovered bug.

Likely touched files for bug fixes, if needed:

- `tests/test_parameter_policy_editor.py`
- `tests/test_parameter_policy_config.py`
- `tests/test_parameter_policy_request_contract.py`
- `tests/test_frontend_effective_config_patch.py`
- production code only at the exact failing seam.

# 7E7 — CI wiring

No new workflow should be required.

The existing workflow already has:

~~~yaml
- "tests/test_parameter_policy*.py"
~~~

and already runs:

~~~text
python -m unittest discover -s tests -p 'test_parameter_policy*.py' -v
~~~

Therefore `test_parameter_policy_step7_contract.py` will automatically run in
the dependency-light Parameter Policy host contract.

It will also run before the real runtime smoke job, which is the desired order.

Do not add CUDA or Windows self-hosted requirements in 7E.

# Planned commit sequence

## 7E1 — Release matrix + Standard equivalence

Files:

- new `tests/test_parameter_policy_step7_contract.py`

Implement:

- canonical ten-backend fixture table;
- dependency-light policy request helper;
- backend bijection;
- Standard missing/explicit/stale-policy equivalence.

No production code changes.

## 7E2 — Component bootstrap / rehydrate round-trip

Same test file.

Implement:

- ten-backend Standard -> Component bootstrap;
- first effective Preview preparation;
- canonical sidecar capture;
- target availability;
- rehydrate;
- second Preview preparation;
- exact policy/effective-config semantic identity.

No API materialization.

## 7E3 — Blocker and registry closure

Same test file, plus existing tests only if a real gap appears.

Implement:

- editor/model-profile/optimizer registry closure;
- exact qualification-blocker field freeze;
- semantic-blocker taxonomy freeze.

## 7E4 — Regression fixes only

If CI exposes a real bug:

1. identify the owning layer;
2. add the smallest focused regression;
3. fix only that seam;
4. rerun the full Parameter Policy host suite and runtime smoke.

Do not use this commit to add new UX, presets, backends, optimizers, or unblock
advanced execution modes.

# Source-review checklist

Before merge, review specifically for:

1. Standard three-state equality on all ten backends.
2. No hidden policy sidecar in Standard.
3. Bootstrap never mutates the Standard fixture.
4. Component round-trip preserves canonical sidecar identity.
5. Rehydrate never silently falls back to Standard.
6. Train=true components remain target-available after both preparations.
7. Request-level runtime blockers stay empty for the baseline matrix.
8. Step 6F qualification blockers remain intact.
9. Semantic blocker taxonomy remains intact.
10. New host tests remain torch/FastAPI/CUDA independent.

# Merge criteria

Step 7E is mergeable when:

- the canonical fixture table covers exactly the ten qualified runtime backends;
- all ten Standard missing/explicit/stale-policy variants are semantically
  identical;
- all ten Standard -> Component bootstrap paths succeed for the representable
  baselines;
- all ten Component states Preview without request-level blockers;
- all ten Component policies survive rehydrate/reprepare with identical
  canonical sidecar semantics;
- all Train=true routes remain available after effective-config normalization;
- Step 6F qualification blockers and semantic taxonomy are unchanged;
- no new product behavior is added merely to satisfy the tests;
- dependency-light Parameter Policy host CI is green;
- real CPU runtime smoke remains green.

After 7E merges, Step 7 enters 7F code freeze / final source review and physical
CUDA evidence. Any qualification-blocker removal belongs to a later focused PR,
not to 7E.
