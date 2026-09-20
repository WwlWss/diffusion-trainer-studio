# Parameter Policy Step 6A — Trainer Runtime Integration Boundary

## Scope

Step 6A installs the reusable trainer-side boundary required by later model-family
integration. It does **not** enable Component-wise Start for any backend.

The request-level runtime allow-list remains empty until the complete Step 6
backend matrix has been integrated and reviewed.

## Ownership

When a later trainer opts into Component-wise mode it must call
`create_parameter_policy_session()` after the model/network exists and before any
legacy optimizer-group helper takes ownership.

The session owns:

1. sidecar loading and canonical policy identity;
2. fail-closed v1 semantic compatibility checks;
3. one metadata-only parameter scan;
4. audited routing and Runtime Spec compilation;
5. CompositeOptimizer construction;
6. final per-parameter requires-grad state;
7. CompositeLRScheduler construction through a child-aware legacy scheduler
   adapter;
8. post-`Accelerator.prepare()` parameter/device ownership audit;
9. checkpoint policy/topology manifest validation;
10. scheduler effective-configuration identity for safe resume.

## Standard-mode boundary

`scripts/*/library/dts_parameter_policy_bridge.py` is deliberately lazy.
Standard trainers must not import the bridge, so the new torch runtime remains
outside the historical optimizer path when `parameter_policy_config=None`.

Trainer-side parsing is also fail-closed. A non-empty
`parameter_policy_config` is rejected by `verify_training_args()` unless that
trainer explicitly opts into the Parameter Policy runtime. The legacy
`get_optimizer()` path always rejects a managed policy, providing a second
sentinel against accidental fallback during later model-family integration.

## Compatibility gate

The v1 semantic blocker is shared by Standard -> Component bootstrap checks and
trainer runtime preflight. Unsupported ownership-changing modes fail closed,
including DeepSpeed, fused optimizer modes, block swap, and checkpoint/offload
paths that have not completed Component-wise validation.

## Host semantics

Legacy D-Adaptation LR rewriting is skipped when a managed
`parameter_policy_config` is present. Anima Full likewise no longer requires
the legacy global `learning_rate` merely to preview/export a Component-wise
configuration.

## Deferred to 6B+

Step 6A does not patch training loops, model-family root maps, clipping sources,
or Anima staged trainers. Those integrations consume this boundary in 6B–6D.
The global Start gate is removed only after the full matrix and runtime smoke
suite are complete.

## Resume identity

Component-wise state resume is bound to three independent identities:

- canonical Parameter Policy hash;
- optimizer/runtime topology fingerprint;
- scheduler effective-configuration signature.

The scheduler signature is derived from the same frozen argument snapshot used
to construct sd-scripts external child schedulers, including effective training,
warmup, decay, cycle, power, timescale, minimum-LR and custom scheduler
arguments. This prevents a checkpoint created under one scheduler configuration
from silently loading under another scheduler of the same Python class.

When every child optimizer manages its own schedule (for example a pure
ScheduleFree policy), unused global scheduler controls do not participate in the
identity and cannot cause false resume mismatches.

The Accelerate load pre-hook validates the manifest before optimizer/scheduler
state is restored. Runtime smoke coverage exercises a real save/load round trip
and verifies that scheduler mismatch rejection leaves fresh optimizer and
scheduler state unchanged.
