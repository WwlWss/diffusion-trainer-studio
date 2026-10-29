# D1-SD3C2-A — SD3.5 Medium Component Full-BF16 scoped admission foundation

**Status: scope contract implementation / qualification still PENDING.**
Original base: `main@bfdac9cb040515b4ff3edb422ea414215df4e825`
(merged PR #62). **Shared Component Boolean Safety PR #64 is now merged
to `main@8b00a3b16b0f71d98cab2f63294656da5ed025a2`; PR #63 has
incorporated its exact final Anima safety preflight fix and targets `main`.
No Full-BF16 SD3 release qualification is implied.**

## Why this is separate from C2-B

D1-SD3C1 proved real **Component mixed-BF16** SD3.5 Medium LoRA
Fresh Step 1 -> checkpoint -> new-process Resume Step 2 on CUDA with
AdamW. This is not true Full-BF16 and does not qualify Muon fallback.

The Full-BF16 release registry in
`mikazuki/parameter_policy_execution.py` is currently keyed by backend.
Granting `sd3-lora` without another scope gate would permit untested model
families, trainable CLIP/T5 adapters, optimizer layouts and execution flags.

C2-A adds the *second* fail-closed boundary, while preserving the pending
release record. Only a later exact-head CUDA evidence PR may promote it.

## Authority and entrypoints

`parameter_policy_execution_blockers(...)` is the shared host-side
execution gate for Preview, backend Start, and direct production
`create_parameter_policy_session(...)` calls.

1. Inactive `full_bf16`: return without consulting qualification metadata or
   reading model files; Standard/mixed-BF16 semantics are unchanged.
2. Active `full_bf16`: require `mixed_precision=bf16`.
3. Require released evidence-backed backend qualification. **During C2-A,
   `sd3-lora` remains pending: all such jobs are blocked.**
4. Only if that backend is subsequently released: run
   `sd3_full_bf16_config_blockers(...)` in the shared request and direct
   execution gate. This pure stage never reads model files or imports Torch.
5. **Preview/Export** call the shared gate with `require_assets=False`;
   the Preview API remains model-file-independent and reports configuration
   readiness only. **Start** calls with `require_assets=True` using the
   original `launch` intent (not `effective_launch`, which is deliberately
   false for Component requests); it checks model headers and external asset
   paths before creating launch side effects.
6. **Direct trainer** independently calls the shared gate with
   `require_assets=True` before parameter routing/optimizer allocation,
   regardless of any Preview/Start state. Direct trainer code may already
   have loaded the model before this Session-level check: it is *not* claimed
   to prevent earlier CUDA model loading.
7. The combined helper `sd3_full_bf16_scope_blockers(...,
   require_assets=True)` defaults to strict asset checking for any caller
   that has not explicitly opted into pure Preview behavior.
8. Enforce existing per-optimizer Full-BF16 gates and live-root/dtype/device
   audits in the production `ParameterPolicyTrainerSession`.

There is no separate WebUI exception, second optimizer implementation,
temporary qualification lease in production, or execution-identity schema
change. Host tests patch the backend qualification row only **inside** the
test process to exercise the future-qualified path.

## Model and target scope

- Explicit local, absolute `.safetensors` base checkpoint; read the
  *header only*, using `safetensors.safe_open.get_slice().get_shape()`,
  not model tensor values or GPU memory.
- Reuse `validate_sd3_mmdit_checkpoint(...,
  require_loader_compatible_root=True)`; the variant must be exactly
  `sd35-medium` with complete DTS MMDiT names and expected shapes.
- External CLIP-L, CLIP-G, T5XXL and VAE source files are required when
  their corresponding weights are not embedded in the base checkpoint.
- Exactly seven registered `sd3-lora` Components. Train
  `mmdit.attention.adapter`, `mmdit.mlp.adapter`,
  `mmdit.modulation_norm.adapter`. Freeze `mmdit.other.adapter`,
  `clip_l.adapter`, `clip_g.adapter`, `t5xxl.adapter`.
- Only `networks.lora_sd3`, `network_train_unet_only=true`, rank=4,
  alpha=4, batch=1, **fixed 512x512** resolution, zero workers, constant
  scheduler, safetensors output, cached latents/text-encoder outputs.
- **Ordinary DreamBooth must set `enable_bucket=false`** for the initial
  reference qualification. Even without an external Dataset Config,
  `enable_bucket=true` would let the standard Dataset Blueprint generate
  variable aspect-ratio/geometry buckets. Ignored `min_bucket_reso`,
  `max_bucket_reso` and `bucket_no_upscale` values need not be overwritten
  when buckets are off. Other SD3 precision modes retain their bucket support.
- Fixed resolution is a *semantic check*, not a rewrite: the GUI sends
  `"512,512"`, CLI may send `"512"`, and the production
  `train_util.prepare_dataset_args` changes both into a
  `(512, 512)` integer tuple before the direct trainer Session check.
  All three forms describe the same reference geometry. Non-square,
  wrong-sized and unsupported input shapes remain blocked. The scope leaves
  each incoming representation untouched.
- **Merged Boolean safety prerequisite (PR #64):** only real Boolean, None/off,
  or numeric 0/1 values are permitted for Component runtime switches.
  Quoted TOML pseudo-booleans such as `fp8_base = "false"` are rejected,
  because `bool("false")` is true inside the legacy Trainer. The R1/R2
  scope reuses the shared `component_runtime_boolean_type_blockers` and
  `strict_component_runtime_bool`, not a second permissive string parser.
  Both forbidden and required-true switches are covered. Preview still
  avoids model-header I/O; Start and direct trainers retain strict asset
  preflight. The shared early trainer preflight precedes model/FP8 load.
- In stock argparse, `network_dropout=None` means the feature is off,
  equivalently to `0` for **that specific field**. The scope compares a
  local off-state value without mutating arguments and continues to block
  positive dropout and malformed values; unrelated fields such as
  `training_shift` do not gain a general None-to-zero rule.
- Unqualified alternate dataset sources are rejected: `dataset_config`,
  `dataset_class`, `in_json`, `reg_data_dir`,
  `multi_caption_config`, `conditioning_data_dir`, and masked loss.
  Such sources can change the actual Dataset Blueprint `batch_size` and
  `resolution` without modifying top-level CLI values.
- Scope explicitly requires default SD3 conditioning and sampling semantics:
  `pos_emb_random_crop_rate=0` (the real parser spelling), no CLIP/T5
  attention-mask switches, zero encoder dropouts, `training_shift=1`,
  `weighting_scheme=uniform`, `loss_type=l2`,
  `t5xxl_max_token_length=256`. Zero-worker DataLoader cannot enable
  `persistent_data_loader_workers`. The UI already canonicalizes this
  combination; direct Trainer input is checked independently.

The header audit establishes **topology only**, not checkpoint provenance,
the numeric identity of the loaded tensors, the accuracy of training, or
Full-BF16 runtime qualification. C2-B must bind a complete model/input file
identity to the exact-head CUDA evidence and checkpoint lifecycle.

## Only two future-qualified optimizer shapes

| Case | Components | Profiles | Accumulation |
| --- | --- | --- | --- |
| R1 | Attention + MLP + Modulation/Norm | one AdamW with `args={"weight_decay":0.01}`, no fallback | 1 |
| R2 | same three trained Components | Muon primary `args={}` + explicit AdamW fallback `args={}` on every trained route | 2 |

All three trained routes have reference `learning_rate=0.0001`; in R2
the explicit `fallback_learning_rate` is also `0.0001`. Other
Optimizer Profile arguments or learning rates require expanded evidence.

For R2 the live SD3 adapter-target classifier is responsible for Muon
eligibility. Attention/MLP approved hidden 2D weights are candidate Muon
parameters; AdaLN Modulation targets are ineligible and must enter explicit
AdamW fallback. A policy shape alone never proves both children have real
parameters: C2-B must audit **physical** primary/fallback assignments and
both children actually stepping and restoring state.

All further optimizer types/profile arrangements are excluded by this
candidate scope, including the Muon implementation's internal implicit
AdamW fallback.

## Explicit unsupported modes

No trainable text encoders, alternate LoRA network, network_args or preloaded
network/base weights, optimizer_args, dropout, positional embedding scaling
or random cropping, gradient checkpointing, FP8, Full-FP16, compile,
DeepSpeed, fused variants, block swapping, CPU/offload, cached output files,
or other advanced execution modes. The additional `no_half_vae=true` mode
would run the VAE in FP32 instead of the reviewed BF16 reference path
(`train_network.py` computes `vae_dtype` from this flag). It is now
explicitly rejected for R1/R2, including string pseudo-booleans; this does
not change the ordinary SD3 mixed-precision/Standard VAE options.

The host contract additionally limits unreviewed geometry (rank, alpha,
batch, resolution, accumulation, worker count) to the R1/R2 reference cases.
Real effective dataloader batch counts (R1 one batch and R2 two batches per
epoch) must be proven in C2-B; host config inspection alone cannot prove
gradient accumulation semantics.
Single GPU/process and real BF16 dtype/device ownership continue to be
verified by the *existing* execution environment/runtime contracts.

## D1-SD3C2-A acceptance

- C2 scope module remains torch-free, CUDA-free and safe for model-free Preview.
  Start/Trainer perform a read-only `safetensors` header audit; a separate
  pinned CPU test verifies the reader against a tiny actual safetensors file.
- Qualifier is **not released**, no change to
  `FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"]`.
- Host positive and negative tests cover eligible R1/R2 future-qualified
  simulations, each forbidden Component/Optimizer form, missing/invalid
  checkpoint/CLIP/T5/VAE sources, SD3 Medium/Large, malformed fields,
  and unchanged pending/inactive behavior. This includes real UI string and
  post-`prepare_dataset_args` tuple resolution, stock
  `network_dropout=None`, and both R1/R2 bucket-negative cases.
  Every forbidden switch additionally rejects quoted `"false"`, `"0"`,
  `"off"`, `"true"` and `"1"` in both R1 and R2; required-true
  switches also reject strings. Real `ui_custom_params` TOML
  `fp8_base = "false"` is blocked before any asset read.
- Start failure side-effect evidence uses a real configured output/TOML
  path plus a materialization spy; no writes are permitted after a failed
  asset check. Preview remains model-file-free.
- Existing SD1/SDXL/D0 gates and SD3C1 R0 tests still pass on the new HEAD.
- Strict source Review before merging; **no CUDA qualification claim for A**.

## D1-SD3C2-B follow-up (separate PR)

Add evidence-specific SD3 cases to the backend execution-feature matrix,
rather than pretending SD3 belongs to the existing SD1/SDXL
`D1_STOCK_LORA_SPECS` abstraction. Provide a new external fixture
generator with explicit three text encoders, R1 AdamW (accum1), R2
Muon + AdamW fallback (accum2) and two effective batches per R2 epoch.

Require fresh -> Step 1 -> new process Resume -> Step 2, real BF16
parameter **and gradient** dtypes, ownership/device/eligible/fallback counts,
physical optimizer and scheduler state/step lineage, matching model/input
hashes, SD3 forward and epoch-boundary cursor, finite/nonzero LoRA changes,
the D0 shared C1-C4 8/8 regression, and a fresh SD3C1 R0 check on the
**same final candidate HEAD**. Only after these tests and strict Review may
the narrowly guarded SD3 Full-BF16 backend record be released. All other
model families, optimizers, modes and mid-epoch/multi-process Resume remain
pending.
