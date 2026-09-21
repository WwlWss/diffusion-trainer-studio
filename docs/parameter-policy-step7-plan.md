# Parameter Policy Step 7 — GUI Editor, Presets, Regression, and Release Closure

Step 7 turns the completed Parameter Policy runtime into a user-facing,
release-ready feature. It does not add new trainer families or relax Step 6F
qualification blockers by default.

The implementation is deliberately split into independently reviewable stages.
The hard rule is that Standard mode remains the historical DTS path and
Component-wise mode continues to fail closed whenever runtime semantics are
not qualified.

## Starting point

Step 6F opened baseline single-process Component Start for exactly ten backends:

- `sd-lora`
- `sdxl-lora`
- `sd-dreambooth`
- `sdxl-finetune`
- `sd3-lora`
- `flux-lora`
- `chroma-lora`
- `flux-finetune`
- `anima-lora`
- `anima-finetune`

The backend contract, routing engine, CompositeOptimizer/Scheduler runtime,
trainer integration, checkpoint manifest v2, Preview/Export/Rehydrate sidecars,
and Start gate already exist.

What is intentionally still missing is the public GUI editor. The current
training schemas do not expose `optimization_mode`,
`parameter_policy_profiles`, or `parameter_policy_components`. Therefore
Step 7 must finish the actual user workflow instead of being documentation-only.

## Non-goals

Step 7 does **not**:

- add another backend;
- change Component IDs or Parameter Policy sidecar v1;
- load models in the host GUI just to preview parameter-level routing;
- silently approximate unsupported Standard semantics;
- enable explicit multi-GPU/DDP;
- enable DeepSpeed, compile, fused optimizer paths, block swap/offload,
  full FP16/BF16, or FP8 merely because the GUI can represent them;
- make Standard depend on Parameter Policy torch runtime;
- remove the final trainer-side ownership/checkpoint assertions.

## 7A — Host-side editor contract

### New module: `mikazuki/parameter_policy_editor.py`

Keep canonical policy semantics in `parameter_policy.py`. The new module owns
only GUI/editor concerns.

Planned public helpers:

~~~python
def parameter_policy_editor_metadata(train_type: str) -> dict:
    ...

def bootstrap_parameter_policy_editor(
    raw_config: Mapping[str, Any],
    page_train_type: str,
) -> dict:
    ...

def normalize_parameter_policy_editor_state(config: dict) -> None:
    ...

def parameter_policy_editor_preview(
    policy: Mapping[str, Any],
    train_type: str,
    runtime_blockers: Sequence[str],
) -> dict:
    ...
~~~

### Metadata source of truth

`parameter_policy_editor_metadata()` must derive data from existing registries,
never duplicate it:

- Components come from `get_model_component_profile(train_type)`.
- Optimizer choices come from `list_optimizer_capabilities()`.
- The editor dropdown exposes only capabilities whose
  `component_support == "supported"`.
- Restricted/planned optimizer types remain valid sidecar data for old imports,
  but are not offered as new runnable choices.
- Component labels/descriptions/groups come from the model profile definitions.
- The response records Parameter Policy version and backend train type.

Suggested payload:

~~~json
{
  "version": 1,
  "train_type": "flux-finetune",
  "optimizer_types": [
    {"type": "AdamW", "requires_parameter_eligibility": false},
    {"type": "Muon", "requires_parameter_eligibility": true}
  ],
  "components": [
    {
      "id": "transformer.double_stream",
      "label": "Double Stream",
      "description": "...",
      "groups": ["transformer", "flux"]
    }
  ]
}
~~~

### Ergonomic optimizer args

The sidecar remains:

~~~json
{
  "optimizer_profiles": {
    "main": {
      "type": "Muon",
      "args": {"momentum": 0.95}
    }
  }
}
~~~

The GUI may represent dictionary values as text because Schemastery's legacy
dictionary editor is string-oriented. `normalize_parameter_policy_editor_state`
must therefore convert editor string values to JSON/Python-literal values before
calling the existing canonical validator.

Rules:

1. native bool/int/float/string values are preserved;
2. textual literals are parsed deterministically;
3. malformed values fail closed with profile/key context;
4. reserved ownership keys such as LR/fallback routing remain forbidden;
5. canonical sidecar serialization stays unchanged.

This conversion must happen before
`build_parameter_policy_sidecar()`; `parameter_policy.py` remains the strict
canonical-policy layer.

### Standard -> Component bootstrap

`bootstrap_parameter_policy_editor()` must:

1. deep-copy the raw GUI state;
2. force the snapshot to Standard mode;
3. remove stale policy editor fields from that snapshot;
4. call the existing `bootstrap_parameter_policy_from_standard()`;
5. return `rehydrate_parameter_policy(policy)`.

It must never invent a fallback optimizer. Existing Step 3 compatibility
blockers stay authoritative.

## 7B — Public Schemastery editor

### New module: `mikazuki/parameter_policy_schema.py`

Generate the editor schema from the same backend registries used by runtime.

Planned helpers:

~~~python
def parameter_policy_schema_fragment(train_type: str) -> str:
    ...

def wrap_parameter_policy_editor(schema: str, train_type: str) -> str:
    ...
~~~

The wrapper approach is preferred over editing six source schema files by hand:

~~~text
existing backend schema
        +
generated Parameter Policy editor fragment
        =
Schema.intersect([...])
~~~

This keeps the ten backend pages synchronized and prevents Component IDs or
optimizer choices from drifting from Python registries.

### Visible fields

Every integrated backend gets:

- `Optimization Mode`
  - `standard` (default)
  - `component`
- `Optimizer Profiles`
  - dictionary key = user profile name
  - optimizer type = supported registry choice
  - optimizer args = dictionary of literal values
- `Components`
  - fixed rows generated from that backend's Model Component Profile
  - `Train`
  - `Learning Rate`
  - `Optimizer Profile`
  - optional `Fallback Optimizer Profile`
  - optional `Fallback Learning Rate`

Frozen rows must not emit stale hidden LR/profile values. The existing canonical
normalizer remains the final enforcement layer.

Fallback fields remain visible/optional instead of trying to make the static
schema inspect a dynamically referenced optimizer profile. Backend validation
continues to enforce:

- fallback is illegal for unrestricted optimizers;
- fallback is required only when real parameter routing discovers ineligible
  parameters for an eligibility-gated primary optimizer;
- fallback cannot point to another eligibility-gated optimizer.

### Injection points

- `fixed_sd_schema()`
- `fixed_flux_family_schema()`
- directly loaded fixed schemas such as `sd3-lora`, `sdxl-full`, and
  `flux-finetune`

Use the existing `PAGE_BACKEND_MAP` when the page schema name differs from the
runtime backend name.

Do not hand-maintain a second ten-backend mapping if an existing mapping can be
reused.

## 7C — Bootstrap UX and runtime-readiness preview

### API additions in `mikazuki/app/training_api.py`

Add:

~~~text
GET  /training/parameter-policy/metadata?train_type=<page>
POST /training/parameter-policy/bootstrap
~~~

The bootstrap endpoint receives the same request envelope as Preview:

~~~json
{
  "train_type": "flux-finetune",
  "config": {...current raw GUI state...}
}
~~~

Success response:

~~~json
{
  "gui_state": {
    "optimization_mode": "component",
    "parameter_policy_profiles": {...},
    "parameter_policy_components": {...}
  }
}
~~~

No file is written by either endpoint.

### Extend existing Preview payload

When Component mode is active, `_prepared_payload()` should additionally return
a read-only editor summary derived from the canonical sidecar:

~~~json
{
  "runtime_ready": false,
  "runtime_blockers": ["..."],
  "parameter_policy_preview": {
    "profiles": [...],
    "components": [...]
  }
}
~~~

This is a component-level plan only. Step 7 must **not** host-load a model to
claim exact tensor counts or Muon eligibility counts. Parameter-level routing
remains trainer-owned and is audited by the Step 6E startup diagnostics and
manifest.

### Frontend patch

Extend `mikazuki/frontend_training_patch.py` with three pieces of state:

~~~text
__runtimeReady
__policyBootstrapPending
__policyModeGuard
~~~

Behavior:

1. default Standard page behaves exactly as today;
2. when `optimization_mode` changes Standard -> Component:
   - if policy profiles/components already exist, preserve them;
   - otherwise call `/api/training/parameter-policy/bootstrap`;
   - merge only the returned policy GUI keys into the current form;
3. Component -> Standard hides the editor but does not destructively erase the
   editor state in the browser; backend Standard compilation still strips it;
4. imported Component bundles already contain policy fields and must not be
   auto-bootstrap-overwritten;
5. Preview success with `runtime_ready=false` must surface
   `runtime_blockers` in the existing error area instead of silently showing
   an apparently valid form;
6. Start is disabled while bootstrap is pending or the latest Component preview
   is runtime-blocked;
7. backend Start remains authoritative even if frontend state is stale.

The mode watcher must be guarded against deep-watch feedback loops. A bootstrap
response may trigger another preview, but must never trigger another bootstrap.

### Important UX decision

Do not hide the historical global optimizer/LR controls in Step 7 v1. They
remain useful as the Standard configuration and as the migration seed.
Component mode clearly labels the Component editor as authoritative, and the
effective TOML preview demonstrates that policy-owned global optimizer/LR keys
are stripped.

This avoids a fragile rewrite of every legacy optimizer section in the pinned
frontend.

## 7D — Presets and documentation

### Presets

Do not create ten redundant "same as Standard" policies; the bootstrap button
already handles that migration.

Add a small set of intentionally different example presets, for example:

- Flux Full — Muon primary + AdamW fallback;
- Anima Full — DiT Muon + AdamW fallback, Qwen3 frozen;
- SDXL Full — separate U-Net/Text Encoder component learning rates;
- one LoRA example demonstrating component freeze/independent LR.

Every Component preset must stay inside the Step 6F baseline:

- ordinary mixed FP16/BF16 only;
- no `full_fp16` / `full_bf16`;
- no FP8 base;
- no compile;
- no DeepSpeed;
- no explicit multi-GPU;
- no swap/offload/fused paths.

In particular, do not copy the current Standard Anima Full
`anima_precision_mode = "full_bf16"` into a Component preset because Step 6F
correctly blocks that path.

### User documentation

Add `docs/parameter-policy.md` covering:

1. Standard vs Component;
2. Standard -> Component migration;
3. Optimizer Profiles;
4. Component Train/Freeze and LR;
5. Muon eligibility + fallback;
6. Preview/runtime blockers;
7. Export bundle + import/rehydrate;
8. checkpoint/resume identity guarantees;
9. supported ten-backend matrix;
10. qualification blockers vs semantic blockers;
11. troubleshooting and how to return to Standard.

Update stale source comments that still say there is no public GUI.

## 7E — Regression closure

### New tests

`tests/test_parameter_policy_editor.py`

- metadata comes from registries;
- exactly ten release backends are accepted;
- unsupported backend fails closed;
- only `component_support == "supported"` optimizers are offered;
- editor optimizer args normalize deterministically;
- Standard -> Component bootstrap preserves canonical semantics;
- bootstrap rejects every existing compatibility blocker;
- existing Component editor state is never overwritten by bootstrap helper.

`tests/test_parameter_policy_step7_contract.py`

- every release backend schema contains one Optimization Mode editor;
- every schema's component IDs equal its Model Component Profile exactly;
- schema optimizer choices equal supported registry choices;
- Standard remains the default;
- Component preview exposes runtime blockers;
- frontend Start is disabled on blocked Component preview;
- frontend bootstrap only fires for missing Component policy state;
- import/rehydrate does not trigger destructive bootstrap;
- Step 6F qualification blocker fields remain present.

Extend existing suites:

- `test_frontend_effective_config_patch.py`
- `test_training_schema_factory.py`
- `test_training_schema_overrides.py`
- `test_parameter_policy_config.py`
- `test_parameter_policy_request_contract.py`

### Standard regression matrix

For all ten backends compare:

1. no `optimization_mode`;
2. explicit `optimization_mode=standard`;
3. Standard with stale hidden policy GUI values.

All three must compile to the same effective Standard trainer configuration
apart from already-documented normalization.

No Standard request may import Parameter Policy torch runtime or write a policy
sidecar.

### Component round trip

For each backend:

~~~text
Standard raw GUI
 -> bootstrap
 -> Component GUI
 -> Preview
 -> Export bundle
 -> Rehydrate
 -> Preview again
~~~

Canonical Parameter Policy sidecar and effective trainer config must remain
semantically identical.

## 7F — Final release acceptance and CUDA evidence

Only after 7A–7E are source-reviewed and CI-green do physical GPU tests run.

The Windows self-hosted GitHub runner is not a Step 7 development dependency.
The failed Step 6F attempts were environment/bootstrap failures before the CUDA
matrix itself, so Step 7 development must not keep spending commits on runner
setup.

Preferred final acceptance is on the user's normal DTS Python environment after
pulling the completed Step 7 branch.

### Tier 1: shared CUDA runtime

~~~powershell
python tools/run_parameter_policy_gpu_matrix.py --output parameter-policy-gpu-matrix.json
~~~

This qualifies only what the script actually tests:

- supported optimizer construction/step;
- bitsandbytes paths;
- ScheduleFree;
- Muon construction/step;
- FP16 autocast + GradScaler;
- BF16 autocast;
- mixed optimizer children;
- optimizer state_dict reload.

### Tier 2: real trainer fresh/resume matrix

When the required model/dataset assets are available:

~~~powershell
python tools/run_parameter_policy_backend_gpu_matrix.py --manifest <local-manifest.json> --output parameter-policy-backend-gpu-matrix.json
~~~

The manifest remains machine-local. The evidence runner requires exactly the
ten release backends and validates Step 6E manifest v2 identity across fresh and
resume runs.

If all ten assets are not available, do not fabricate coverage. Record the
available backend evidence explicitly and keep unqualified execution features
blocked.

### Blocker-removal policy

Step 7 release closure itself does not remove:

- `full_fp16`
- `full_bf16`
- `fp8_base`
- `fp8_base_unet`
- compile
- DeepSpeed
- fused paths
- swap/offload
- explicit multi-GPU

A later focused PR may remove a qualification blocker only after the exact
runtime combination has dedicated evidence and checkpoint/resume coverage.

## Planned commit sequence

### 7A — Editor host contract
Files:
- new `mikazuki/parameter_policy_editor.py`
- `mikazuki/app/training_api.py`
- `mikazuki/training_request.py`
- new `tests/test_parameter_policy_editor.py`

### 7B — Generated Schemastery editor
Files:
- new `mikazuki/parameter_policy_schema.py`
- `mikazuki/training_schema_factory.py`
- `mikazuki/training_schema_overrides.py`
- schema factory/override tests

### 7C — Frontend bootstrap/readiness UX
Files:
- `mikazuki/frontend_training_patch.py`
- `tests/test_frontend_effective_config_patch.py`
- `tests/test_training_api_overlay_contract.py`

### 7D — Presets and user guide
Files:
- curated `config/presets/*.toml`
- new `docs/parameter-policy.md`
- stale source comments / README links as appropriate

### 7E — Full regression closure
Files:
- new `tests/test_parameter_policy_step7_contract.py`
- existing Parameter Policy request/config/schema tests
- CI wiring only where explicit coverage is not already reached by discovery

### 7F — Final source review + local CUDA evidence
No runtime semantics should change here unless the final review finds a bug.

## Merge criteria

Step 7 is mergeable when:

- the GUI can create/edit/import/export Component policies on all ten backends;
- Standard -> Component bootstrap is lossless for representable Standard
  semantics and fail-closed otherwise;
- Component runtime blockers are visible before Start;
- Standard mode remains semantically unchanged;
- all ten schemas derive Component IDs from the same model profile registry;
- all newly offered optimizer types come from the supported capability registry;
- Component bundle rehydrate round-trips;
- Step 6F qualification/semantic blockers remain enforced;
- source/CPU CI is green;
- final physical CUDA evidence is run after code freeze, with evidence scope
  described accurately.
