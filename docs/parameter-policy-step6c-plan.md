# Parameter Policy Step 6C — Full trainer integration

Step 6C integrates the Step 6A runtime boundary into the non-Anima full trainers
used by DTS while keeping the global host Start allow-list closed.

## Backends

- stable DreamBooth: `sd-dreambooth`
- stable SDXL full fine-tune: `sdxl-finetune`
- dev Flux full fine-tune: `flux-finetune`

## DreamBooth

Component mode creates one session over `unet` and `text_encoder`. The
legacy dynamic `stop_text_encoder_training` state machine is not allowed to
participate in Component semantics; non-negative stop steps remain fail-closed.
The session can therefore represent U-Net-only, Text-Encoder-only, or mixed
partial-U-Net ownership without legacy `requires_grad_(True)` reopening frozen
parameters.

## SDXL full

The session owns `unet`, `text_encoder_1`, and `text_encoder_2`. TE1's
last encoder layer and final layer norm are trainer-intrinsic structural freezes
and are excluded before RuntimeSpec construction. Text-encoder output caching is
validated after routing: it is allowed when both text encoders are policy-frozen
and rejected when either encoder is actually trained.

## Flux full

The session owns only the Flux transformer root. Legacy blockwise/fused
optimizers, block swapping, CPU-offload checkpointing and DeepSpeed remain
fail-closed. Existing Flux ScheduleFree sampling/save lifecycle is reused with
CompositeOptimizer.train/eval.

## Common ownership

Component mode bypasses legacy optimizer construction, builds the composite
scheduler only after final `max_train_steps` is known, audits prepared
optimizer/device ownership, registers the checkpoint manifest before resume,
clips only session-owned parameters, and logs LR by component.

`torch_compile` is fail-closed for all Component v1 paths until Dynamo-wrapped
parameter identity and save/resume semantics are covered by the Step 6F GPU
matrix.

The global `PARAMETER_POLICY_RUNTIME_TRAIN_TYPES` allow-list remains empty in
Step 6C.
