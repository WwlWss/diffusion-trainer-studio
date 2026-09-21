# Parameter Policy Step 7C — Frontend Bootstrap and Runtime-Readiness UX

Step 7C connects the merged Step 7A host editor contract and Step 7B generated
Schemastery editor to the pinned legacy frontend.

It does not change trainer/runtime semantics, optimizer routing, checkpoint
identity, or any Step 6F qualification blocker.

## Starting point

Step 7A already provides:

- `GET /training/parameter-policy/metadata`
- `POST /training/parameter-policy/bootstrap`
- Standard -> Component exact-or-fail migration
- editor-safe rehydrate/encoding
- model-free `parameter_policy_editor_preview()`

Step 7B already provides:

- public `optimization_mode`
- Optimizer Profiles
- backend-fixed Component rows
- supported optimizer choices
- disabled restricted/planned optimizer values for old bundle compatibility
- one final schema injection path through `override_raw_schema()`

The missing product behavior is now entirely frontend orchestration:

1. switching Standard -> Component must populate a real policy;
2. imported Component bundles must not be overwritten;
3. runtime blockers must be visible before Start;
4. Start must be disabled while bootstrap/preview state is not current;
5. asynchronous Preview/bootstrap responses must never overwrite newer form state.

## Non-goals

Step 7C does not:

- add new trainer backends;
- remove any Step 6F blocker;
- change Parameter Policy sidecar v1;
- host-load models to compute parameter/tensor routing;
- add dynamic optimizer-profile dropdowns;
- hide or delete Standard optimizer/LR controls;
- destructively erase Component state when switching back to Standard;
- move runtime authority from backend Start validation into JavaScript.

## 7C1 — Backend runtime-readiness preview payload

### Modify `mikazuki/app/training_api.py`

Import:

~~~python
from mikazuki.parameter_policy_editor import (
    bootstrap_parameter_policy_editor,
    parameter_policy_editor_metadata,
    parameter_policy_editor_preview,
)
~~~

Add one private helper that reads the already-staged content-addressed Parameter
Policy sidecar from `prepared.sidecars`:

~~~python
def _prepared_parameter_policy_preview(prepared) -> dict | None:
    path = prepared.config.get("parameter_policy_config")
    if not path:
        return None

    content = prepared.sidecars.get(str(path))
    if content is None:
        raise RuntimeError(
            "Prepared Component request is missing its host-owned Parameter Policy sidecar."
        )

    try:
        policy = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Prepared Component request contains an invalid Parameter Policy sidecar."
        ) from exc

    return parameter_policy_editor_preview(
        policy,
        prepared.train_type,
        prepared.runtime_blockers,
    )
~~~

Do not add a new field to `PreparedTrainingConfig`. The canonical policy is
already present as the staged sidecar, so a second ownership channel would
create drift risk.

Extend `_prepared_payload()`:

~~~python
preview = _prepared_parameter_policy_preview(prepared)
if preview is not None:
    payload["runtime_ready"] = preview["runtime_ready"]
    payload["runtime_blockers"] = preview["runtime_blockers"]
    payload["parameter_policy_preview"] = {
        "version": preview["version"],
        "train_type": preview["train_type"],
        "profiles": preview["profiles"],
        "components": preview["components"],
    }
~~~

Standard payloads must omit all three Parameter Policy readiness fields.

The preview remains model-free. It must not claim:

- exact parameter counts;
- Muon eligibility counts;
- optimizer ownership tensor counts;
- device placement.

Those remain trainer-owned Step 6E/6F diagnostics.

## 7C2 — Frontend Parameter Policy state machine

### Modify `mikazuki/frontend_training_patch.py`

The existing patch already owns:

- backend-only Preview;
- debounced Preview generations;
- Start pending state;
- import/rehydrate;
- export;
- deep form watching.

Step 7C should extend this existing patch rather than create a second patch.

### New refs

Extend the current ref declaration with:

~~~text
__runtimeReady
__runtimeBlockers
__policyBootstrapPending
__policyModeGuard
__policyBootstrapGeneration
~~~

Recommended initial values:

~~~javascript
__runtimeReady=ref(!0)
__runtimeBlockers=ref([])
__policyBootstrapPending=ref(!1)
__policyModeGuard=ref(!1)
__policyBootstrapGeneration=ref(0)
~~~

A separate bootstrap generation counter is required. Preview generation cannot
safely substitute for it because the two request streams have different
lifetimes and cancellation rules.

### Pure helpers

Add:

~~~javascript
__isComponentMode=()=>String(
    a.value&&a.value.optimization_mode||"standard"
).toLowerCase()=="component"
~~~

A policy is considered real only when both containers are non-empty:

~~~javascript
__hasPolicyState=()=>{
    let P=a.value&&a.value.parameter_policy_profiles,
        C=a.value&&a.value.parameter_policy_components;
    return !!(
        P&&typeof P=="object"&&Object.keys(P).length &&
        C&&typeof C=="object"&&Object.keys(C).length
    )
}
~~~

Do not test only for property existence. The legacy Schemastery renderer may
materialize empty placeholder objects when a conditional branch becomes active.

### Bootstrap snapshot

Bootstrap must **not** call `T()`.

Immediately after the user selects Component, the Component schema may not yet
contain a complete policy. Running the Component resolver first can therefore
fail before migration has a chance to run.

Instead:

~~~javascript
__policyBootstrapSnapshot=()=>{
    let R=clone(a.value||{});
    delete R.parameter_policy_profiles;
    delete R.parameter_policy_components;
    R.optimization_mode="standard";
    return __resolveGuiState(R);
}
~~~

This sends the current raw Standard GUI controls to the existing exact-or-fail
backend migration path.

### Bootstrap request

Add:

~~~javascript
__bootstrapPolicy=async()=>{ ... }
~~~

Required behavior:

1. return immediately when:
   - `__policyBootstrapPending` is active;
   - `__policyModeGuard` is active;
   - mode is no longer Component;
   - a non-empty policy already exists;
2. increment `__policyBootstrapGeneration`;
3. set:
   - `__policyBootstrapPending=true`;
   - `__runtimeReady=false`;
   - clear runtime blockers;
4. POST:
   - endpoint: `/api/training/parameter-policy/bootstrap`
   - config: `__policyBootstrapSnapshot()`
5. reject a stale response when:
   - generation changed;
   - current mode is no longer Component;
6. validate returned `gui_state` contains both:
   - `parameter_policy_profiles`
   - `parameter_policy_components`
7. set `__policyModeGuard=true`;
8. merge **only** those two policy keys into the current form:

~~~javascript
Object.assign(a.value,{
    parameter_policy_profiles:clone(B.parameter_policy_profiles),
    parameter_policy_components:clone(B.parameter_policy_components)
})
~~~

Do not replace `a.value`. This preserves any unrelated edits the user made
while bootstrap was in flight.

9. after `nextTick()`, clear the guard/pending state;
10. trigger one new Preview.

Bootstrap failure behavior:

- keep the user's explicit Component mode;
- do not silently revert to Standard;
- set `__runtimeReady=false`;
- surface the bootstrap error through the existing `d.value` error area;
- allow the user to switch back to Standard or manually build a policy.

### Mode synchronization

Add one function:

~~~javascript
__syncPolicyMode=async()=>{ ... }
~~~

Behavior:

#### Standard

- increment bootstrap generation to invalidate any in-flight bootstrap;
- set `__policyBootstrapPending=false`;
- set `__runtimeReady=true`;
- clear runtime blockers;
- do not delete policy state;
- schedule Standard Preview.

#### Component with existing policy

- set `__runtimeReady=false` immediately;
- preserve policy state;
- schedule Preview;
- never call bootstrap.

This is the import/rehydrate path.

#### Component without a real policy

- set `__runtimeReady=false`;
- call `__bootstrapPolicy()`.

Register:

~~~javascript
watch(
    ()=>a.value&&a.value.optimization_mode,
    __syncPolicyMode
)
~~~

Do not use a deep watcher for mode/bootstrap orchestration.

### Initial mount

Replace the current:

~~~javascript
onMounted(async()=>{I(),y(),await nextTick(),__refreshPreview()})
~~~

with:

~~~javascript
onMounted(async()=>{
    I(),
    y(),
    await nextTick(),
    await __syncPolicyMode()
})
~~~

This handles:

- ordinary Standard page startup;
- future Component presets;
- restored Component state;
- old state where Component mode exists but policy fields are missing.

### Imported bundles

The existing import path already does:

~~~javascript
a.value=clone(B)
await nextTick()
__refreshPreview()
~~~

Keep whole-state replacement for explicit user import.

The mode watcher sees:

- Component + populated policy -> preserve and Preview;
- Standard -> ordinary Standard Preview.

Do not call bootstrap unconditionally after import.

### Preview readiness

Update `__refreshPreview()`.

At the beginning of any Component refresh:

~~~javascript
__runtimeReady.value=false
~~~

This is important. A previously green Preview must not leave Start enabled
during the 300 ms debounce after the user has edited the Component policy.

If bootstrap is pending:

- cancel/debounce any previous Preview;
- increment Preview generation;
- do not issue a request yet.

On successful Standard Preview:

~~~javascript
__runtimeReady.value=true
__runtimeBlockers.value=[]
d.value=[]
~~~

On successful Component Preview:

~~~javascript
__runtimeReady.value=R.runtime_ready===true
__runtimeBlockers.value=Array.isArray(R.runtime_blockers)
    ? R.runtime_blockers
    : []
d.value=__runtimeReady.value
    ? []
    : (
        __runtimeBlockers.value.length
            ? __runtimeBlockers.value
            : ["Component runtime is not ready."]
      )
~~~

Keep warnings in `C.value`.

On Component Preview failure:

- `__runtimeReady=false`;
- put the request error into both the existing error area and runtime blocker
  state.

Preview generation checks remain authoritative for stale async responses.

### Start gate

Change both the button disabled state and the Start handler's first line.

Button:

~~~javascript
disabled:
    __startPending.value ||
    __policyBootstrapPending.value ||
    (__isComponentMode()&&!__runtimeReady.value)
~~~

Start handler:

~~~javascript
if(
    __startPending.value ||
    __policyBootstrapPending.value ||
    (__isComponentMode()&&!__runtimeReady.value)
)return;
~~~

The JavaScript gate is UX only. `POST /api/run` continues to perform the real
Step 6F runtime-blocker checks.

## 7C3 — Regression closure

### Backend/API tests

Extend `tests/test_training_api_overlay_contract.py`:

- `parameter_policy_editor_preview` is imported;
- `_prepared_payload()` exposes:
  - `runtime_ready`
  - `runtime_blockers`
  - `parameter_policy_preview`;
- Standard path does not fabricate readiness fields;
- sidecar content is read from `prepared.sidecars`, not the filesystem.

Add a small direct unit test if importing `training_api.py` can remain
dependency-light; otherwise keep this as source contract plus the existing
`parameter_policy_editor_preview()` behavioral tests.

### Frontend patch tests

Extend `tests/test_frontend_effective_config_patch.py`.

Lock all of the following:

1. state refs exist:
   - `__runtimeReady`
   - `__runtimeBlockers`
   - `__policyBootstrapPending`
   - `__policyModeGuard`
   - `__policyBootstrapGeneration`;
2. bootstrap uses:
   - `/api/training/parameter-policy/bootstrap`;
   - `clone(a.value)` snapshot semantics;
   - deleted policy fields;
   - forced Standard snapshot;
3. bootstrap does **not** use `T()`;
4. existing non-empty Component policy skips bootstrap;
5. bootstrap merges only profiles/components and does not replace `a.value`;
6. stale bootstrap responses check generation and current mode;
7. initial mount calls `__syncPolicyMode()`;
8. mode watcher watches only `optimization_mode`;
9. switching Standard never deletes Component editor state;
10. Component refresh sets readiness false before debounce;
11. Preview success consumes top-level backend readiness;
12. blockers are copied into the existing error area;
13. imported Component state is not destructively bootstrapped;
14. Start button is disabled for:
    - start pending;
    - bootstrap pending;
    - blocked/stale Component Preview;
15. Start handler has the same defensive guard;
16. backend `/api/run` remains the launch authority.

### Syntax/regression suites

Existing CI must continue to pass:

- final patched frontend bundle syntax check;
- frontend effective-config contract;
- runtime schema specialization;
- Parameter Policy host contract;
- Parameter Policy runtime smoke;
- Multi-Caption contract.

Step 7C must not require CUDA.

## Race scenarios that must be handled

### A. Standard -> Component -> Standard before bootstrap returns

The Standard transition increments `__policyBootstrapGeneration`. The stale
bootstrap response is ignored and cannot write Component fields back.

### B. Standard -> Component -> Standard -> Component

The second Component transition receives a new bootstrap generation. The first
request cannot overwrite the second.

### C. Import Component bundle

The imported state already has non-empty profiles/components. Mode sync skips
bootstrap and performs only Preview.

### D. Component edit while previous Preview is green

The deep watcher immediately marks Component readiness false before debounce,
so Start cannot use stale green readiness.

### E. Bootstrap mutates policy fields

`__policyModeGuard` prevents orchestration feedback. The deep watcher may
schedule Preview, but Preview exits while bootstrap remains pending. One final
Preview runs after bootstrap completion.

### F. Component -> Standard

Policy objects remain in browser state. The schema hides them, and backend
Standard compilation strips them. Returning to Component can reuse the existing
policy without another bootstrap.

## Planned commit sequence

### 7C1 — Backend readiness payload

Files:

- `mikazuki/app/training_api.py`
- `tests/test_training_api_overlay_contract.py`

No frontend changes.

### 7C2 — Frontend bootstrap state machine

Files:

- `mikazuki/frontend_training_patch.py`
- `tests/test_frontend_effective_config_patch.py`

No trainer/runtime changes.

### 7C3 — Cross-feature regression hardening

Files only as required by discovered regressions.

Expected work:

- final patched bundle syntax;
- import/rehydrate preservation;
- Standard no-op;
- Multi-Caption coexistence;
- source-level final review.

## Merge criteria

7C is mergeable when:

- Standard startup remains behaviorally unchanged;
- Standard -> Component with no policy runs exactly one effective bootstrap;
- a complete existing/imported Component policy is never overwritten;
- rapid mode changes cannot apply stale bootstrap results;
- Component policy edits immediately invalidate stale green readiness;
- runtime blockers are visible before Start;
- Start is disabled while Component readiness is false or bootstrap is pending;
- backend Start remains authoritative and fail-closed;
- Component -> Standard preserves browser-side policy state;
- Preview remains side-effect free;
- all CPU/source/frontend CI is green.
