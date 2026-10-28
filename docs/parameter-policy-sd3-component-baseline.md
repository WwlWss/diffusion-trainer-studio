# D1-SD3C1 — SD3.5 Medium Component mixed-BF16 baseline

Status: **implementation / no real R0 CUDA result yet**.
Base: main@bf22b93e06d978629327eeccbb550ac6942ba4bd (merged PR #61).

## Purpose

PR #60 proved SD3.5 Medium LoRA forward/Resume in mixed BF16.
PR #61 proved WebUI API Fresh/Resume in the same reference range.
Neither proves real Component-wise training with CompositeOptimizer and
CompositeLRScheduler; neither qualifies Component full BF16.

C1 introduces R0: the real SD3.5 Medium production network trainer running
mixed-BF16 Component AdamW, save Step 1, new Python process Resume to Step 2.
The reference runner checks that production Parameter Policy Manifest v2,
policy hash, real optimizer state and SD3-specific progress/forward sidecars
coexist correctly. C1 changes NO production optimizer or trainer semantics.

## R0 scope

- SD3.5 Medium only, DTS-compatible strict MMDiT header and extra-attention
  contract, not SD3 Medium or SD3.5 Large.
- scripts/dev/sd3_train_network.py with networks.lora_sd3, rank 4 alpha 4.
- mixed_precision=bf16, full_bf16 false/absent, no FP8.
- One GPU and one process; batch 1, resolution 512, accum 1.
- Canonical seven-Component Policy: MMDiT Attention and MLP trainable;
  Modulation/Norm may be trainable, MMDiT Other and CLIP-L/G/T5 frozen.
  The generator enables Attention, MLP and Modulation/Norm.
- One explicitly routed AdamW Optimizer Profile; no Muon or fallback yet.
- Exactly one effective DataLoader batch per epoch, making each optimizer
  step an epoch boundary; two total steps, save state every step.
- No gradient checkpointing, offload, swaps, compile, DeepSpeed, fused paths,
  multi-GPU, arbitrary network weights, or unreviewed network_args.
- No Full-BF16 qualification or production backend gate change.

The architecture header audit is shape-only. It is not an authentic model
release/weight provenance check or backend-precision qualification.

## Files added

- tools/prepare_sd3_component_baseline.py: prepare a new external policy JSON
  and manifest with two production subprocess commands; does not train.
- tools/run_sd3_component_gpu_evidence.py: exact-head and clean-tree preflight,
  strict reference options, real MMDiT architecture audit, real GPU Fresh/Resume,
  SD3 forward/cursor checks, canonical Policy Manifest v2 and physical AdamW
  child checks, optimizer/adapter advancement.
- tests/test_sd3_component_gpu_evidence.py: CPU-only negative and positive
  tests for generator, Policy, manifest and qualification isolation.
- .github/workflows/anima-qwen3-review.yml: relevant path triggers and tests.

The runner reuses the existing SD3B checkpoint validator and subprocess
execution path. It does not introduce a second optimizer loop or Resume hook.

## Windows reference (PowerShell)

Run on the exact, clean candidate branch and the same DTS Python/CUDA
environment already used for PR #61.

~~~powershell
cd C:\DTS-test\dts-sd3b-ui-audit
git fetch origin
git switch feature/d1-sd3c1-component-baseline
git pull --ff-only
git submodule update --init --recursive

# Must be empty: the GPU runner refuses dirty HEAD.
git status --porcelain
$sha = (git rev-parse HEAD).Trim()

# NEW location outside the repo, never reuse existing results.
$work = "C:\DTS-test\sd3c1-r0-evidence"

python tools/prepare_sd3_component_baseline.py --model "G:\model\SD35\sd3.5_medium.safetensors" --data-dir "C:\DTS-test\sd3b-data" --work-dir $work --cache-latents --cache-text-encoder-outputs

python tools/run_sd3_component_gpu_evidence.py --manifest "$work\manifest.json" --expected-commit $sha --output "$work\result.json" --preflight

# Only after preflight PASS:
python tools/run_sd3_component_gpu_evidence.py --manifest "$work\manifest.json" --expected-commit $sha --output "$work\result.json"
~~~

If required external CLIP-L, CLIP-G, T5XXL or VAE weights are not bundled
in the checkpoint, supply --clip-l, --clip-g, --t5xxl and/or --vae to the
generator. Inspect the generated manifest before launch.

The dataset must result in one actual DataLoader batch per epoch (after
repeats, bucket handling, etc.). The runner validates the *checkpoint*
DataLoader geometry, not merely the physical image count. If input settings
change, use a completely new work directory.

## Acceptance / non-promotion

The following must all pass on one exact SHA:

1. CI/host tests, clean-tree and exact-head gate.
2. Recognized SD3.5 Medium architecture and real GPU startup.
3. Fresh saved Step 1; new subprocess restores exactly that Step 1.
4. Resumed Step 2: SD3 forward-semantics identity and SD3 cursor valid;
   one process and one batch/epoch.
5. Production Component Manifest v2, correct canonical policy hash,
   trainable/frozen seven-Component ownership, one actual AdamW child and
   stable CompositeScheduler identity.
6. Optimizer state files and finite LoRA adapter weights show advancement.
7. GPU result status=pass and all these remain false:
   qualifies_full_bf16, backend_qualification_eligible,
   production_qualification_mutated.

R0 does not certify per-component Full-BF16 parameter/gradient dtype,
Muon fallback, arbitrary data order, stochastic bitwise reproducibility,
trainable CLIP/T5, or gradient checkpointing.

## Next phase (D1-SD3C2)

Separate PR: implement model- and component-scoped Full-BF16 admission;
R1 AdamW and R2 Muon + explicit AdamW fallback real CUDA cases; accumulation=2,
dtype/device/ownership, checkpoint execution identity and progress lineage;
same-head shared C1-C4 regression; negative checks for unsupported SD3 model
variants and trainable text encoders. Only after strict review and real GPU
evidence can the corresponding restricted Full-BF16 subset be promoted.

The production sd3-lora Full-BF16 row is intentionally still pending.
