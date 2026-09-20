# Parameter Policy Step 6F — GPU Matrix and Component Start

Step 6F closes the trainer-integration phase. Baseline Component-wise Start is
opened only for the ten backends integrated in Steps 6B–6D, while execution
modes that change optimizer ownership, parameter residency, distributed state,
or model wrapping remain fail-closed until they have a dedicated runtime
contract and qualification evidence.

## Baseline backend matrix

The machine-readable source of truth is
`mikazuki/parameter_policy_matrix.py`.

Baseline Start backends:

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

"Baseline" means the ordinary single-process trainer path with no blocked
execution feature enabled. Standard mode remains unchanged.

## Qualification matrix

The GPU matrix has three orthogonal axes rather than one Cartesian product.

1. Backend matrix: every integrated backend must preserve the Step 6E lifecycle,
   policy-owned optimizer membership, requires-grad contract, metadata, and
   checkpoint identity.
2. Optimizer/runtime matrix: CUDA covers currently-supported optimizer
   implementations, ScheduleFree, bitsandbytes, Muon construction, FP16
   autocast + GradScaler, BF16 autocast, mixed optimizer children, and
   CompositeOptimizer state_dict reload.
3. Trainer/backend matrix: real trainer commands cover Step 6E lifecycle,
   Accelerator integration, checkpoint manifest v2, and resume per backend.
4. Execution-feature matrix: compile, distributed/sharded ownership, fused
   optimizer paths, swap/offload, full-precision training, and FP8 remain
   independently qualified.

`tools/run_parameter_policy_gpu_matrix.py` is the shared optimizer/runtime
CUDA probe. It emits JSON containing repository/runtime versions, GPU identity,
case name, and pass/fail details.

`tools/run_parameter_policy_backend_gpu_matrix.py` is the real trainer matrix
runner. Its machine-local JSON manifest must contain exactly the ten release
backends. Every case supplies a fresh trainer argv, a resume trainer argv, and
the checkpoint directory. The harness requires the fresh run to create Step 6E
manifest v2 with the matching train type, requires the resume command to
succeed, and requires checkpoint identity to remain unchanged across resume.
This keeps model/dataset paths outside the repository while making physical GPU
evidence reproducible and machine-readable.

## Blocker policy

Step 6F distinguishes two blocker classes.

Qualification blockers may eventually be removed after a dedicated runtime
contract and GPU evidence:

- Accelerate Dynamo `torch_compile`;
- Anima per-block `compile`;
- DeepSpeed/distributed optimizer ownership;
- fused backward / fused optimizer groups / blockwise fused optimizers;
- CPU/Unsloth checkpoint offload;
- Flux/Anima block swap;
- `full_fp16` / `full_bf16`;
- `fp8_base` / `fp8_base_unet`.

Semantic blockers remain blocked in v1 even if a one-off training run succeeds,
because the current Component schema cannot represent their semantics exactly:

- LoRA+;
- regex-specific LR;
- SD/SDXL LoRA block LR;
- SDXL Full block LR;
- `scale_weight_norms`;
- DreamBooth dynamic `stop_text_encoder_training`;
- preloaded adapter + Text Encoder cache hazards;
- Anima Qwen-only;
- unreviewed custom `network_module`.

Explicit multi-GPU selection also remains fail-closed until the DDP matrix is
qualified. A machine with multiple visible GPUs still uses the existing
single-process Accelerate config unless the user explicitly selects multiple
GPU ids. Likewise, ordinary `mixed_precision=fp16/bf16` remains baseline,
while full-precision and FP8 base modes stay blocked until physical GPU
qualification.

## Start pipeline

Component Start keeps launch side effects behind the runtime gate:

1. compile effective config without Component launch-only side effects;
2. evaluate backend/runtime/optimizer/GPU blockers;
3. perform read-only trainer/model/dataset validation;
4. materialize any reviewed staged Anima runtime;
5. perform Component launch-only finalization;
6. materialize content-addressed sidecars and final TOML;
7. spawn Accelerate.

A blocked Component request therefore cannot create a staged trainer, policy
sidecar, or training TOML.

## Release invariants

- `PARAMETER_POLICY_RUNTIME_TRAIN_TYPES` comes only from the release matrix.
- All ten entries must agree with the host trainer mapping.
- Unknown/new backends remain blocked by default.
- Standard mode never imports or constructs Parameter Policy torch runtime.
- Specific compatibility blockers remain authoritative after the global backend
  gate opens.
- Component multi-GPU remains explicitly blocked until DDP qualification.
- The Step 6E manifest v2/runtime assertions remain mandatory on every opened
  backend.
