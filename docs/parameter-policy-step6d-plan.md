# Parameter Policy Step 6D — Anima staged integration

Step 6D integrates Parameter Policy with the pinned Anima sd-scripts runtime
without mutating the submodule.  The global GUI Start allow-list remains closed.

## Runtime staging

A new content-addressed feature, `parameter_policy_runtime`, is materialized
after Qwen3 joint-finetune, Multi-Caption, and LoRA target-metadata patches.
The staged tree receives the reviewed DTS Parameter Policy bridge and patched
Anima LoRA/full trainer seams.  The cache key includes the runtime patch and
bridge source.

## Anima LoRA

The staged NetworkTrainer uses the same Component ownership contract as Step 6B:
legacy optimizer construction is bypassed, scheduler construction occurs after
final max_train_steps, clipping uses session-owned parameters, LR logging is
component-aware, and the checkpoint manifest is registered before trainer
resume hooks.  Anima's subclass maps actual policy routes to DiT and Qwen3
adapter train flags.  Preloaded LoRA + Text Encoder cache and Anima per-block
compile remain fail-closed.

## Anima full finetune

The staged full trainer scans `dit` and, when target permission is enabled,
`qwen3`.  `train_qwen3_text_encoder` is target permission; Parameter Policy
owns final Qwen3 train/freeze and all optimizer/LR semantics.  Qwen3-only is not
supported in Component v1.  Qwen3 output caching is allowed only when the policy
freezes Qwen3.

Parameter Policy owns `requires_grad` and optimizer membership.  The trainer
continues to own model mode, dtype/device placement, Qwen3 gradient
checkpointing, paired Qwen3 sidecars, and sampling/save behavior.

The Parameter Policy checkpoint manifest is registered before the existing
Qwen3 ON/OFF resume-mode hook so topology/scheduler mismatches fail before state
mutation.

## Safety boundary

Component mode still blocks DeepSpeed, fused-backward, block swap/offload,
Accelerate Dynamo `torch_compile`, and Anima's separate per-block `compile`
path.  `PARAMETER_POLICY_RUNTIME_TRAIN_TYPES` remains empty until Step 6F.
