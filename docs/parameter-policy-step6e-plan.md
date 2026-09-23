# Parameter Policy Step 6E — Runtime hardening

Step 6E does not add new backends. It hardens the runtime contract shared by all
Component-wise trainer integrations completed in Steps 6B–6D.

## Unified runtime lifecycle

Every integrated trainer now uses the same lifecycle:

1. build the immutable Parameter Policy session;
2. prepare model/optimizer/scheduler with Accelerate;
3. call `finalize_after_prepare()`;
4. resume state;
5. assert the `post_resume` runtime contract;
6. assert the `epoch_start` runtime contract at every epoch boundary.

`finalize_after_prepare()` performs optimizer ownership, trainable-device, and
requires-grad auditing; registers checkpoint hooks; and emits one startup
diagnostic summary.

Standard mode continues to bypass this runtime physically.

## Checkpoint manifest v2

`dts_parameter_policy_manifest.json` is now version 2. In addition to policy,
runtime topology, optimizer, and scheduler identity, it records deterministic
diagnostics:

- train type;
- trainable and frozen component IDs;
- trainable parameter tensor/element counts;
- optimizer profile-to-type mapping.

Save/load pre-hooks assert the current runtime contract before state mutation.
A mutated requires-grad contract or optimizer ownership therefore fails before a
new manifest is committed or resumed state is applied.

## Diagnostics and metadata

The session owns a single `model_metadata()` representation for metadata-aware
NetworkTrainer outputs. Stable/dev NetworkTrainer and staged Anima LoRA consume
this API instead of maintaining separate DTS metadata implementations.

Startup diagnostics print the policy hash, topology fingerprint, component
ownership, optimizer profiles, trainable parameter counts, and scheduler
signature once per Component run. Per-step logging remains limited to
component-oriented LR values.

## Device and requires-grad safety

Device auditing intentionally covers optimizer-owned trainable parameters only.
Frozen roots may remain on CPU when trainers use caching/offload-safe frozen
paths. Requires-grad auditing covers all scanned parameters, including
structural freezes.

Single-process Accelerate may expose the CUDA target as an implicit `cuda` device
while prepared tensors report an explicit logical ordinal such as `cuda:0`.
The runtime audit resolves only an implicit CUDA ordinal against
`torch.cuda.current_device()` before comparison. Explicit CUDA ordinals remain
strict: `cuda:0` and `cuda:1` are never treated as equivalent. Device
resolution is read-only and does not call `torch.cuda.set_device()`, move
tensors, or alter optimizer ownership.

The runtime smoke suite deliberately mutates frozen/trainable flags, optimizer
ownership, and expected devices to verify fail-closed behavior.

## Safety boundary

Step 6E does not open previously blocked execution modes such as DeepSpeed,
fused backward, block swapping/offload, Dynamo torch.compile, or Anima block
compile. The global Component Start allow-list remains empty until Step 6F.
