# Phase D1-SD3B: SD3.5 Medium forward correctness and LoRA resume

Status: **runtime correctness accepted / PR #60 merged**. The exact-head SD3.5 Medium mixed-BF16 GPU fresh/save/resume verification passed on `5011095d30b0f958e62efc537b824825d56abe73`; PR #60 merged as `f749af6bfa14cf37dffd9a201b32c6fa3805d142`. This does **not** qualify full-BF16, FP8, distributed, or generalized mid-epoch resume.

## Scope and math contract

- `scripts/dev/library/sd3_models.py` fixes the SD3.5 Medium `attn2`
  call to pass `attn2.head_dim` (64), not `attn2.num_heads` (24).
  The former uses 24 heads of 64 elements each; the latter incorrectly
  treats 24 as head dimension, giving 64 heads of 24 elements.
- CPU regression extracts real implementation nodes into a minimal Torch-only
  namespace and compares independent PyTorch SDPA forward and backward;
  no real model tensors are required for CPU testing.
- MMDiT extra-attention checkpoints carry `dts_sd3_forward_semantics.json`,
  recording the actual loaded model's algorithm version and its block layout.
  Missing or mismatched sidecars reject `--resume` during Accelerate pre-load
  hooks **before optimizer restoration**.
- Intentional reuse of an old adapter must use `--network_weights` in a **new
  stage**; old optimizer states from the incorrect attention algorithm must
  not silently cross the forward-semantics boundary.
- SD3 Medium and SD3.5 Large variants without extra attention do not need
  a forward-semantics sidecar; SD3 Full has only a forward-semantics guard here,
  **not** a qualified Full resume cursor.

## SD3 LoRA cursor boundary

SD3 LoRA gains `dts_sd3_resume_progress.json` with completed optimizer
steps, checkpoint epoch, local DataLoader batch count, accumulation count,
and process count. On resume the pre-hook compares the saved progress against
`train_state.json` and the current training geometry before optimizer loading.

A scoped SD3 cursor computes absolute optimizer-step, epoch index and remaining
steps without changing legacy Flux/Chroma NetworkTrainer behavior.
For now only **single-process epoch-boundary** resume is accepted:
- Saved `current_step` means optimizer updates already completed.
- `steps_per_epoch = ceil(dataloader_batches / accumulation)`.
- `current_epoch = floor((completed_steps-1)/steps_per_epoch) + 1`.
- `completed_steps % steps_per_epoch` must be zero.
- `initial_step`, `initial_epoch`, `max_train_epochs`,
  `skip_until_initial_step` are prohibited alongside SD3 resume.

The saved geometry is **not** a cryptographic dataset identity. Exact same
per-sample ordering for arbitrary mid-epoch resume is NOT claimed or supported.
The first GPU reference should use a single-item deterministic dataset and an
epoch boundary, so its continuation sample identity is unambiguous.
Mixed BF16 CPU tests do not qualify `full_bf16`, FP8, multi-GPU,
gradient checkpointing, swap, DeepSpeed or the SD3 Full training lifecycle.

## Test matrix

1. `python -m unittest tests.test_sd3_forward_contract tests.test_sd3_resume_cursor tests.test_sd3_runtime_evidence -v`
2. `python -m unittest tests.test_sd3_attention_math -v` (CPU Torch)
3. `python -m unittest discover -s tests -p 'test_sd3_architecture_audit.py' -v`
4. Existing Parameter Policy / Flux / Chroma / SD1 / SDXL regressions.
5. Real exact-head SD3.5 Medium GPU *production trainer* fresh step 1, Save,
   fresh-process Resume from step 1, and exactly one additional update to step 2.

The CPU tests are **necessary but not sufficient**. Do not mark the GPU
criterion PASS without executing the real model and inspecting its evidence.

## WebUI duration / Resume integration (follow-up UI closure)

The SD3 LoRA page (`/lora/sd3.md`) exposes a single explicit training target:

- **Epoch** (fresh default: 20) compiles to `max_train_epochs` for fresh training.
- **Optimizer Steps** compiles to `max_train_steps` for fresh training.
- On **Resume**, both target modes compile to **absolute total**
  `max_train_steps` (including already completed steps), and
  `max_train_epochs` must be absent.
- Epoch Resume computes `steps_per_epoch =
  ceil(saved_dataloader_batches / saved_gradient_accumulation_steps)`
  from the versioned `dts_sd3_resume_progress.json` plus
  `train_state.json`. It then computes
  `total_steps = target_epochs * steps_per_epoch`.
- A read-only preflight checks missing/invalid/old progress JSON,
  Saved Step/Epoch consistency, completed epoch boundary, process count,
  accumulation mismatch, positive remaining steps, and explicit multi-GPU
  selection. The actual SD3 trainer **rechecks** live DataLoader geometry
  and forward-semantics identity before restoring optimizer state.
- The checkpoint sidecars **do not record the original planned total
  optimizer steps**. The UI therefore warns to use the original target;
  it cannot certify scheduler equivalence with uninterrupted training.
- Existing legacy effective TOMLs with both target fields keep
  `max_train_epochs` precedence when imported; re-export uses a single
  canonical target. Rehydrating an epoch-origin Resume configuration yields
  Steps mode because only the absolute total is present in the effective
  trainer TOML.
- `ui_custom_params` is applied before the final SD3 target validation;
  injecting conflicting legacy duration flags is an error.

The WebUI preflight is a **read-only usability guard**, not another
checkpoint format, not a model-identity oracle, and not a new runtime
qualification. Old checkpoints without a PR #60 progress sidecar should
use `network_weights` in a new stage, not `--resume`.

For a UI CUDA smoke, manually select the verified reference configuration
(mixed BF16, AdamW, MMDiT-only rank 4, FP8 and gradient checkpointing off)
and a one-item dataset. The historical default SD3 UI choices can differ;
a successful mixed-BF16 reference does not certify them. Real WebUI
Fresh/Resume GPU acceptance belongs to the new UI-closure PR Head.

## GPU evidence driver (local; requires the actual model and all required assets)

`tools/run_sd3_runtime_gpu_evidence.py` accepts an external manifest with
`schema="dts.sd3b-runtime-evidence"`, `version=1`, and the following:
`fresh_command`, `resume_command` (arrays starting with
`scripts/dev/sd3_train_network.py`), exact `fresh_checkpoint_dir`,
`resume_checkpoint_dir`, `fresh_adapter_path`, `resume_adapter_path`.

Both phases must be the same canonical mixed-BF16 AdamW MMDiT-LoRA
configuration, with `--max_train_steps 2 --save_every_n_steps 1 --save_state`,
`--network_module networks.lora_sd3 --network_train_unet_only`, and
`--gradient_accumulation_steps 1` or `2`. Only `--resume` and
`--output_dir` may differ. No `--config_file` indirection or unsupported
execution modes are permitted for this first reference.

Example evidence preflight on Windows after constructing the external manifest:

```powershell
python tools/run_sd3_runtime_gpu_evidence.py `
  --manifest "C:\DTS-test\sd3b-gpu-manifest.json" `
  --expected-commit "<exact-new-PR-head-sha>" `
  --output "C:\DTS-test\sd3b-gpu-result.json" `
  --preflight
```

Remove `--preflight` **only after** verifying all required text encoders,
VAE, data and model files exist and checking CUDA memory capacity. The tool
requires a clean exact-head checkout, output paths outside the repo and
initially absent, and an actual CUDA device. It then invokes the production
trainer twice in separate subprocesses. Step-1 and Step-2 checkpoints must
carry the correct forward and cursor sidecars, actual Accelerate optimizer
state files, and matching optional Parameter Policy v2 manifests.
The evidence compares optimizer state hashes and finite, changed LoRA tensor
weights. Generated evidence explicitly reports
`qualifies_full_bf16=false` and
`production_qualification_mutated=false`.

**Known restriction:** This reference is explicitly not a general sampler
identity guarantee. A successful two-step run does not certify arbitrary
datasets, multi-worker shuffling, mid-epoch resume, SD3 Full, T5 training or
all optimizer combinations. A full-BF16 backend qualification must remain
a separate, exact-head review and CUDA-gated PR.
