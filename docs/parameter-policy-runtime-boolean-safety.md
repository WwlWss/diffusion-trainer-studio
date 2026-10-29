# Component Runtime Boolean Type Contract — Shared Safety Hotfix

**Status:** Shared host-only safety boundary; no new Full-BF16 or optimizer
qualification is granted. Base `main@bfdac9cb040515b4ff3edb422ea414215df4e825`.
This PR intentionally precedes SD3 D1-SD3C2-A PR #63.

## Root cause and authorization boundary

DTS Component policy historically parsed strings such as `"false"` as
disabled, while legacy trainers directly evaluated the **same raw
`argparse.Namespace` value** with `if args.full_bf16` or
`if args.fp8_base`. In Python `bool("false") is True`.
A quoted TOML pseudo-boolean could therefore masquerade as an inactive
feature in the Host/Session qualification, while activating a different
runtime path. This also affects inherited SD1/SDXL Component qualification,
not just future SD3 Full-BF16.

## Exact contract

`mikazuki/parameter_policy_runtime_bools.py` owns one pure type rule:

- Missing/None, bool, integer 0/1, float 0.0/1.0: accepted, preserving
  existing Python runtime truthiness.
- **All strings** (including `""`, `"false"`, `"0"`, `"true"`),
  other numbers, NaN/Inf, containers or unknown types: rejected.
- No value coercion, rewrites, fallback to Python truthiness, Torch, network
  access, file I/O, or new qualification decisions.
- `full_bf16` feature detection and Execution Identity v1 construction
  share that exact rule. The feature is inactive only for a valid false
  value. Invalid values are blockers on Preview/Start and exceptions on
  contract construction.
- Component v1 semantic compatibility rejects malformed runtime boolean
  values before the historical permissive helpers run. Existing policy,
  backend and optimizer qualification guards are otherwise unchanged.
- Numeric execution fields such as `blocks_to_swap` and
  `fused_optimizer_groups` are *not* reclassified as booleans.

Use **unquoted** TOML booleans:

```toml
full_bf16 = false
fp8_base = false
gradient_checkpointing = false
```

Not `full_bf16 = "false"`.

## Timing and direct trainer defense

Preview and Start use the shared type blocker at Host validation; Preview
does not open the model file. A failed Start cannot materialize sidecars
or begin the job. Direct trainer `create_parameter_policy_session()`
checks the same rules independently.

Because Session construction can occur *after* model/dataset loading, the
stock stable/dev Component trainer entrypoints also invoke a **pure,
lazy-bridged early preflight** before `verify_training_args`, dataset
preparation and model/FP8 initialization. Standard mode does not import
or execute this preflight.

Covered entrypoints:

- Stable SD/SDXL LoRA inherited `NetworkTrainer.train()`.
- Stable SD DreamBooth full and SDXL full `train()`.
- Dev SD3/Flux/Chroma LoRA inherited `NetworkTrainer.train()`.
- Dev Flux finetune `train()`.
- Staged Anima LoRA inherits the patched NetworkTrainer entry.
- Staged Anima Full performs its early check as the **first action in
  `train(args)`**, before `verify_training_args`, dataset creation, Qwen3
  text encoder loading, embedding/latent caching and DiT loading; Anima
  LoRA inherits the patched NetworkTrainer check before its dataset/model
  work. Both paths preserve source-pinned isolated staging.

Other direct standalone scripts without Component Policy integration have
no new checks. This hotfix does not change Standard training semantics.

## Tests and non-regression

Host-only tests cover every registered runtime boolean flag with quoted
`"false"`, `"0"`, `"true"`, `"1"`, the genuine Boolean forms, NaN/Inf,
backend-specific SD1 v2, numerical count independence, TOML-to-Namespace
truthiness reproduction, bridge laziness, and early AST call ordering.
Anima staging tests inspect patched early preflight before network model
work and full-policy preparation. Existing SD1/SDXL/Anima and pending SD3
feature contracts remain unchanged for genuine Boolean values, including
Execution Identity v1 and signature.

No claimed CUDA execution or Full-BF16 promotion. Keep PR #63 Draft until
this shared hotfix is independently reviewed and merged, then apply the same
strict rule to the SD3 R1/R2 scope on the new base. A new exact-head Host/CPU
CI review is required for each PR.
