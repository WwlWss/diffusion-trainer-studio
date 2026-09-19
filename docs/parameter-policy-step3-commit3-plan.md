# Parameter Policy Step 3 — Commit 3 Detailed Design

Baseline: PR #23 head `fea7c93d`.

Commit 3 adds the dependency-light parameter scanner, pure router, and ownership audit. It does not integrate them into request/preview/launch yet.

## Scope

Add:

- `mikazuki/parameter_routing.py`
- `tests/test_parameter_routing.py`

Allowed defensive change:

- make unknown future eligibility policies in `model_component_profiles.py` fail closed instead of returning ordinary "ineligible".

Explicitly excluded:

- optimizer/scheduler construction;
- param_groups;
- Accelerate;
- save/resume;
- trainer launch;
- `requires_grad_` mutation;
- real LoRA metadata injection;
- request/preview/export integration;
- Component Start activation.

## Failure model

Scanner failures raise `ParameterScanError` because a partial parameter inventory is unsafe.

Examples:

- no alias-preserving `named_modules(remove_duplicate=False)`;
- no local `named_parameters(recurse=False, remove_duplicate=False)`;
- inconsistent shape/ndim;
- conflicting nested adapter metadata.

Routing/model-policy failures return an invalid `RoutingPlan` containing aggregated `RoutingIssue` objects so Preview can eventually show all independent problems together.

Malformed sidecars still raise through `validate_parameter_policy()`.

## Data model

```python
class ParameterScanError(ValueError):
    pass


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
    ancestor_module_types: tuple[str, ...]
    ancestor_module_classes: tuple[str, ...]
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
    numel: int
    dtype: str
    requires_grad: bool

    @property
    def canonical_alias(self) -> ParameterAlias: ...

    @property
    def canonical_name(self) -> str: ...


@dataclass(frozen=True)
class RoutingAssignment:
    parameter: Any
    parameter_id: int
    canonical_name: str
    parameter_class: ParameterClass
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
    component_id: str | None = None
    parameter_id: int | None = None
    count: int = 1
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterStat:
    tensors: int
    numel: int


@dataclass(frozen=True)
class RoutingStats:
    total: ParameterStat
    assigned: ParameterStat
    unassigned: ParameterStat
    conflicts: ParameterStat
    unroutable: ParameterStat
    by_component: Mapping[str, ParameterStat]
    by_route: Mapping[str, ParameterStat]
    by_parameter_class: Mapping[str, ParameterStat]


@dataclass(frozen=True)
class RoutingPlan:
    assignments: tuple[RoutingAssignment, ...]
    issues: tuple[RoutingIssue, ...]
    stats: RoutingStats

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)
```

Assignments carry the actual parameter object so Step 5 can construct optimizer groups without rescanning or recovering objects from names. No parameter object is serialized.

## Determinism

`id(parameter)` is used only for identity/deduplication, never for ordering.

- sort aliases by semantic name;
- canonical alias = first sorted alias;
- sort descriptors by canonical alias;
- preserve descriptor order in assignments;
- sort issues and stats maps by stable string keys.

Output must therefore remain stable across processes even though object IDs change.

## Scanner input

```python
scan_parameter_roots(
    roots: Mapping[str, Any],
    *,
    adapter_targets: Mapping[int, AdapterTargetMetadata] | None = None,
) -> tuple[ParameterDescriptor, ...]
```

Root IDs are semantic trainer roots such as `unet`, `transformer`, `dit`, `qwen3`, `text_encoder_1`.

Empty roots and normalized duplicate root IDs are scanner errors.

## Alias-preserving traversal

For each root:

1. call exactly `named_modules(remove_duplicate=False)`;
2. materialize path -> module aliases;
3. derive path-specific ancestry;
4. call each module's `named_parameters(recurse=False, remove_duplicate=False)`;
5. never recursively enumerate all parameters from every module;
6. aggregate final parameters by `id(parameter)`.

Do not retry with duplicate removal enabled if a custom object rejects these keyword contracts. Losing aliases would make ownership auditing unsound.

## Adapter metadata propagation

Step 4 registers metadata by adapter module identity:

```python
adapter_targets[id(adapter)] = AdapterTargetMetadata(...)
```

Real LoRA trainable parameters live in descendant modules such as `adapter.lora_down.weight` and `adapter.lora_up.weight`.

Therefore, for each parameter alias, scanner searches the current module plus its ancestry for registered adapter identities.

- no registration -> metadata is None;
- one or more identical registrations -> inherit nearest ancestor metadata;
- conflicting nested registrations -> `ParameterScanError`.

Never infer the original target from `lora_name`, adapter names, or A/B shapes.

## Parameter metadata

Capture metadata only:

- `shape`
- `ndim`
- derived `numel`
- `dtype` string
- `requires_grad` snapshot

No `.cpu()`, `.cuda()`, `.to()`, `.clone()`, `.detach()`, reductions, norms, or value access.

`numel` is derived from shape.

`requires_grad` is observational only. Router decisions never filter on current `requires_grad`; higher-level target + policy remain authoritative. Commit 3 never mutates it.

## Structural ParameterClass

Classification order per alias:

1. bias role -> `bias`;
2. normalization module -> `norm_weight`;
3. embedding module -> `embedding_weight`;
4. convolution module -> `conv_weight`;
5. role `weight` + ndim 2 -> `matrix_weight`;
6. otherwise -> `other`.

Bias recognition covers `bias`, `*_bias`, `bias_*`.

Norm recognition covers LayerNorm, GroupNorm, RMSNorm, QKNorm, LLMAdapterRMSNorm and BatchNorm*/InstanceNorm* names.

Embedding recognition is conservative: true embedding modules only; wrapper names such as TimestepEmbedding are not automatically embedding matrices.

## Effective architectural root

Normal parameter:

```text
effective_root = alias.root
```

LoRA parameter:

```text
effective_root = alias.adapter_target.root
```

This effective root is checked against `ComponentDefinition.roots`.

The scanner root for a LoRA network is not the original architectural root.

## Alias consensus

One physical parameter is routable only when aliases agree on:

- Component ID;
- valid effective root for that Component;
- ParameterClass;
- AdapterTargetMetadata presence/value;
- eligibility boolean when applicable.

Outcomes:

- all aliases classify None -> unassigned;
- some None/some component, or multiple components -> `alias_component_conflict`;
- different structural classes -> `alias_parameter_class_conflict`;
- different adapter metadata -> `alias_adapter_target_conflict`;
- different eligibility booleans -> `alias_eligibility_conflict`.

Same-object/same-semantics aliases are legal and produce exactly one assignment.

This deliberately fails closed for tied weights exposed through semantically different roles.

## Router entry

```python
build_parameter_routing_plan(
    policy: Mapping[str, Any],
    *,
    train_type: str,
    effective_config: Mapping[str, Any],
    descriptors: Sequence[ParameterDescriptor],
) -> RoutingPlan
```

Sequence:

1. strict sidecar validation;
2. resolve Model Component Profile;
3. resolve Training Target Profile;
4. reject duplicate descriptor identities;
5. audit policy Component rows;
6. resolve descriptor ownership;
7. route valid descriptors;
8. audit component presence/fallback use;
9. build deterministic stats/issues.

No optimizer is constructed.

## Policy pre-pass

For each policy Component:

- unknown Component ID -> `unknown_policy_component`;
- target unavailable + Train=true -> `target_unavailable_train_enabled`;
- target unavailable + Train=false -> legal;
- presence is checked only after real descriptors are classified.

Optimizer support/restriction blockers remain owned by existing `parameter_policy_runtime_blockers()`; router does not duplicate them.

## Target-unavailable descriptors

Higher-level target wins before optimizer policy.

- emit `unavailable` assignment;
- do not evaluate optimizer eligibility;
- missing policy row is allowed because trainer target already forbids training;
- illegal Train=true row is still reported by the component pre-pass.

## Target-available descriptors

A real target-available Component requires a policy row.

Missing row:

- no assignment;
- count in `stats.unassigned`;
- aggregate one `missing_component_policy` issue per Component.

Train=false:

- emit `frozen`;
- do not evaluate eligibility.

Train=true:

- continue to optimizer routing.

## Primary/fallback routing

Primary optimizer without eligibility policy -> `primary`.

Primary optimizer with eligibility policy:

1. evaluate every alias through `profile.eligibility_check(...)`;
2. alias eligibility disagreement -> conflict, no assignment;
3. all eligible -> `primary`;
4. all ineligible + fallback -> `fallback`;
5. all ineligible + no fallback -> unroutable, aggregate `missing_fallback`.

Fallback LR:

- explicit `fallback_learning_rate` if supplied;
- otherwise inherit Component `learning_rate`.

Unknown eligibility policy support must raise/fail closed in the profile layer, never silently route to fallback.

## Presence audit

Target-available Train=true Component with zero real resolved descriptors -> `component_absent`.

Train=false absent Component -> allowed.

Target-unavailable absent Component -> allowed unless policy illegally says Train=true.

Observed candidates from a conflicting parameter suppress redundant `component_absent` noise; ownership conflict is the real diagnosis.

## Unused fallback

If a present, otherwise-valid Train=true Component declares fallback but no parameter actually uses it, emit one warning `unused_fallback`.

Do not warn for absent/invalid Components.

## Issue aggregation

Avoid O(parameters) duplicate errors.

Aggregate component-wide errors:

- `missing_component_policy`
- `missing_fallback`
- bulk unassigned parameters where practical

Each issue carries:

- affected tensor count;
- up to 5 deterministic canonical-name examples.

Parameter-specific alias conflicts may carry `parameter_id`.

## Stats

`ParameterStat` = tensor count + summed numel.

`RoutingStats.total` covers every unique descriptor.

`assigned` contains deterministic ownership outcomes:

- primary
- fallback
- frozen
- unavailable

Separate error buckets:

- unassigned
- conflicts
- unroutable

`by_route` contains only the four assignment kinds.

`by_component` and `by_parameter_class` count resolved ownership only.

Valid-plan invariant:

```text
total.tensors == assigned.tensors
unassigned.tensors == 0
conflicts.tensors == 0
unroutable.tensors == 0
no error issues
```

## Final ownership invariant

Before returning the plan:

- assignment parameter IDs are unique;
- `id(assignment.parameter) == assignment.parameter_id`;
- no physical parameter appears in two route kinds.

Violation makes the plan invalid. This is the last guard before Step 5 optimizer-group construction.

## Performance

One module-tree scan only.

Complexity is proportional to module aliases + parameter aliases + unique parameters, never tensor element count.

No repeated scan inside routing and never inside the training loop.

## Test fakes

`FakeParameter` exposes metadata only. Its device/copy methods raise if called.

`FakeModule` supports only the required alias-preserving/local traversal contracts and can reject incorrect duplicate/recurse flags.

Dynamic fake subclasses reproduce class names needed by model profiles without importing torch.

## Scanner test gate

Must cover:

1. unique scan;
2. shared identity deduplication;
3. alias preservation;
4. deterministic ordering independent of object creation order;
5. path-specific ancestry;
6. all ParameterClass categories;
7. adapter metadata inherited from adapter ancestor;
8. conflicting nested adapter metadata fails;
9. incompatible traversal API fails instead of degrading;
10. no tensor/device/mutation calls;
11. requires_grad snapshot only.

## Router test gate

Must cover:

1. ordinary optimizer primary;
2. pure eligible Muon without fallback;
3. mixed Muon primary/fallback;
4. fallback required only when real ineligible parameters exist;
5. fallback LR inherit/override;
6. Train=false frozen without eligibility evaluation;
7. target-unavailable unavailable;
8. target-unavailable + Train=true error;
9. missing row on available real Component unassigned;
10. missing row on unavailable Component allowed;
11. unknown policy Component error;
12. Train=true absent Component error;
13. Train=false absent allowed;
14. shared same-semantics aliases one assignment;
15. component/class/adapter/eligibility alias conflicts;
16. duplicate descriptor identity invalid;
17. LoRA missing metadata is never guessed;
18. unused fallback warning;
19. requires_grad does not affect routing;
20. correct tensor/numel stats;
21. deterministic issues and assignments.

## Implementation order

### 3A Scanner primitives

Dataclasses, metadata readers, traversal, ancestry, structural class, adapter inheritance.

### 3B Identity aggregation

Dedup by object identity, aliases, canonical names, deterministic descriptor ordering.

### 3C Router ownership/target pass

Strict sidecar validation, model/target profiles, policy row audit, alias consensus, unavailable/frozen handling.

### 3D Eligibility/fallback

Optimizer capability lookup, Muon eligibility, fallback semantics.

### 3E Presence/stats/final audit

Component absence, unused fallback warning, aggregated issues, stats, unique assignment invariant.

Only after every stage's tests pass should the implementation be committed.

## Commit 4 gate

Do not begin Commit 4 unless all existing Step 1/2/Commit 2 tests and the complete scanner/router suite pass without PyTorch installed, Component Start remains blocked, and Standard mode has no new call path.
