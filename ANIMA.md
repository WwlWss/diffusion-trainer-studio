# Anima training in SD-Trainer

This branch adds both **Anima LoRA** and **Anima full finetune** support to the existing SD-Trainer GUI while keeping the original SD / SDXL / SD3 / FLUX / Chroma paths intact. It supports both the standard **28-block Anima** model and the expanded **40-block Anima 2.9B** model.

## What changed

- Adds `sd-scripts` as a pinned git submodule.
- Uses `sd-scripts/anima_train_network.py` for Anima LoRA.
- Uses `sd-scripts/anima_train.py` for Anima full finetune.
- Adds Anima model detection and validation.
- Extends the existing **Flux LoRA expert page** with a `model_type = anima` option. The packaged frontend is still the upstream prebuilt VuePress frontend, so the page title remains Flux; the form itself is provided dynamically by the backend schema and switches to Anima-specific fields.
- When Anima is selected, the GUI exposes an **Anima model variant** selector (`base` / `2.9b`) and a second selector for `LoRA` vs `full finetune`.
- `base` expects the standard 28-block Anima checkpoint; `2.9b` expects the expanded 40-block checkpoint. The launcher validates the checkpoint header before training and rejects a mismatched selection.
- Full finetune exposes Anima component learning rates: self-attention, cross-attention, MLP, AdaLN modulation and LLM Adapter. Because the packaged legacy frontend has a known mode-dependent schema/default issue, older LoRA-only controls may remain visible when full finetune is selected; the backend removes them before launching the trainer.
- Adds Qwen3-0.6B, Qwen-Image VAE, LLM Adapter/T5 tokenizer, timestep, attention, VAE, caching and block-swap controls.
- Adds an optional, default-off **Qwen3 text-encoder joint-finetune** path for Anima full finetune. Its settings are isolated from the existing frozen-Qwen path.
- Adds presets for standard Anima and Anima 2.9B, for both LoRA and full finetune.
- Updates Python dependencies to versions compatible with the current Anima implementation in `sd-scripts`.

## Anima 2.9B loader support

The upstream `kohya-ss/sd-scripts` Anima loader currently hardcodes `num_blocks = 28`, while Anima 2.9B uses 40 blocks. Until upstream PR #2418 is merged, this repository pins the `sd-scripts` submodule to commit `45dddfccb704b6b0591f65d98f0d695c997b3115` from the PR branch. That patch only changes Anima model loading: it probes the safetensors header for block 39 and instantiates 40 blocks for a 2.9B checkpoint, otherwise retaining the existing 28-block behavior.

Once equivalent support lands upstream, the submodule can be rebased onto the corresponding upstream commit while retaining the independent Qwen3 joint-finetune patch.

## Install / update

After checking out this branch, initialize all submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Then run the normal installer again so the upgraded Python dependencies are installed.

Windows:

```powershell
.\install.ps1
```

Linux:

```bash
bash install.bash
```

The GUI also attempts to initialize missing submodules automatically at startup unless environment preparation is skipped.

## Training Anima from the GUI

1. Open **LoRA training -> Flux** (expert page).
2. Set **model architecture** to `anima`.
3. Choose **Anima model variant**:
   - `base`: standard 28-block Anima.
   - `2.9b`: expanded 40-block Anima 2.9B.
4. Explicitly choose **Anima training mode**:
   - `lora`: routes to `sd-scripts/anima_train_network.py`.
   - `finetune`: routes to `sd-scripts/anima_train.py`.
5. Select the matching Anima DiT checkpoint in `pretrained_model_name_or_path`.
6. Select the Qwen3-0.6B text encoder in `qwen3`.
7. Select the Qwen-Image VAE in `vae`.
8. Optionally provide a separate LLM Adapter or T5 tokenizer directory.
9. Configure the dataset and output settings, or load one of the included Anima presets.
10. Start training.

The model-variant selector is a GUI-only routing field and is removed before the config is passed to `sd-scripts`.

The training-mode selector intentionally has no implicit GUI default on this branch. The packaged legacy frontend can evaluate sibling conditional schemas before a default value is materialized into the form model. Requiring an explicit `lora` / `finetune` value makes the new Qwen3 conditional controls deterministic. Existing Anima presets already specify the training mode explicitly.

### LoRA mode

LoRA mode uses `networks.lora_anima` and shows the usual LoRA controls such as rank, alpha, dropout and network arguments.

The included LoRA presets follow the current `sd-scripts` Anima LoRA example closely:

- rank: `8`
- alpha: `1`
- learning rate: `1e-4`
- optimizer: `AdamW8bit`
- scheduler: `constant`
- timestep sampling: `sigmoid`
- mixed precision: `bf16`
- gradient checkpointing: enabled
- latent cache: enabled
- text encoder output cache: enabled
- Qwen-Image 2D VAE: enabled
- VAE chunk size: `64`

Any stale Qwen3 joint-finetune fields are removed at the final launcher boundary when LoRA is selected.

### Full finetune mode

Full finetune directly trains the Anima DiT and does not use `network_module`, `network_dim` or `network_alpha`; stale network fields are removed by the backend before launch.

The normal `anima_train.py` path keeps Qwen3 frozen. The GUI exposes the script's per-component DiT learning rates:

- `self_attn_lr`
- `cross_attn_lr`
- `mlp_lr`
- `mod_lr`
- `llm_adapter_lr`

Leaving one of these blank makes that component use the base `learning_rate`; setting it to `0` freezes that component.

The included full-finetune presets keep Qwen3 frozen and start conservatively with a base learning rate of `1e-5`, BF16 full-precision model weights, gradient checkpointing and both latent/text-output caching enabled. Adjust the learning rate and memory controls for your dataset and GPU.

## Optional Qwen3 joint finetune

When `model_type = anima` and `anima_training_mode = finetune`, the GUI exposes **Train Qwen3 text encoder**. It is default-off. When it is off, all Qwen3-training-only fields are removed before launch and the existing frozen-Qwen `sd-scripts` behavior is preserved.

When enabled, the GUI expands:

- `qwen3_lr` (default `5e-7`)
- `qwen3_gradient_checkpointing` (default enabled)
- optional `qwen3_output_dir`

The first implementation deliberately supports **joint DiT + Qwen3 training**, not Qwen3-only training. The main DiT `learning_rate` and `qwen3_lr` must both be greater than zero.

### Compatibility rules

Qwen3 joint finetuning is rejected when either text-encoder output cache is enabled:

- `cache_text_encoder_outputs`
- `cache_text_encoder_outputs_to_disk`

This is a hard correctness requirement: cached Qwen3 embeddings detach Qwen3 from the training graph. `cache_latents` and `cache_latents_to_disk` remain compatible.

The first implementation also rejects Qwen3 joint finetuning with:

- DeepSpeed
- `fused_backward_pass`
- D-Adaptation optimizers
- Prodigy / ProdigyPlus
- Adafactor

The supported first-pass optimizer set is AdamW, AdamW8bit, PagedAdamW8bit, Lion, Lion8bit, PagedLion8bit, SGDNesterov and SGDNesterov8bit. These modes preserve the independent, much smaller Qwen3 parameter-group learning rate.

These restrictions apply only when Qwen3 training is enabled. Existing frozen-Qwen Anima training keeps the original optimizer/DeepSpeed behavior.

### Qwen3 checkpoint pairing

When Qwen3 is trainable, every saved Anima checkpoint must have a matching Qwen3 sidecar. For example:

```text
model-step00001000.safetensors
model-step00001000_qwen3.safetensors

model-step00002000.safetensors
model-step00002000_qwen3.safetensors
```

The sidecar filename is derived from the main checkpoint automatically. `qwen3_output_dir` changes only the directory; it does not create an independent filename or save cadence. Leaving it blank stores the sidecar beside the main checkpoint.

Checkpoint rotation follows the main checkpoint rotation so old Qwen3 sidecars do not accumulate indefinitely. When Hugging Face upload is enabled, the paired Qwen3 sidecar is uploaded as well.

Saving Qwen3 adds roughly the size of a complete Qwen3-0.6B checkpoint to every model save and therefore increases disk use and checkpoint I/O pauses. This cost exists only while Qwen3 is trainable.

### Resume and two-stage training

A save-state produced with Qwen3 training enabled must be resumed with Qwen3 training enabled. A frozen-Qwen state must not be resumed as a trainable-Qwen state. The patched trainer stores a mode marker in Qwen3-training states and rejects incompatible resumes.

For the recommended "train the text encoder early, freeze it later" workflow, use two training runs rather than changing Qwen3 trainability inside one run:

**Stage 1**

```text
Train Qwen3 = ON
Qwen3 output cache = OFF
DiT + LLM Adapter + Qwen3 train jointly
```

Save the paired DiT and Qwen3 checkpoints.

**Stage 2**

```text
pretrained_model_name_or_path = Stage 1 DiT checkpoint
qwen3 = matching Stage 1 Qwen3 sidecar
Train Qwen3 = OFF
Qwen3 output cache = ON (optional/recommended)
```

Start a new optimizer/scheduler state for Stage 2. Do not use Stage 1's `save_state` as a Qwen-off resume. Once Qwen3 is frozen again, the normal Anima text-output cache path can precompute embeddings and release Qwen3 from VRAM.

## Qwen3 sd-scripts patch staging

The current submodule is a third-party fork used for the unmerged Anima 2.9B loader patch. Qwen3 joint-finetune changes should ultimately live in a **WwlWss-owned `sd-scripts` fork**, with the 2.9B loader change and Qwen3 training change kept as separate commits.

Until that fork exists, this branch includes a fail-closed staging tool:

```bash
python tools/apply_anima_qwen3_sd_scripts_patch.py --check
python tools/apply_anima_qwen3_sd_scripts_patch.py --write
```

The tool is pinned to `sd-scripts` commit `45dddfccb704b6b0591f65d98f0d695c997b3115`. It requires every expected source block to match exactly and parses every patched Python file before writing. If the submodule has moved, the tool refuses to apply and the patch must be reviewed against the new upstream first.

The launcher also checks that both the trainer logic and Qwen3 CLI arguments are present. If the sd-scripts patch is not installed, Qwen3 joint finetuning fails before an `accelerate` process is created; ordinary Anima LoRA and frozen-Qwen full finetuning remain available.

## Memory controls

For lower VRAM cards, `blocks_to_swap` can be increased. The standard 28-block Anima model supports at most 26 swapped blocks, while the 40-block Anima 2.9B model supports at most 38. The launcher validates the limit against the selected model variant.

`blocks_to_swap`, `cpu_offload_checkpointing` and `unsloth_offload_checkpointing` are mutually constrained by upstream `sd-scripts`; do not enable block swap together with either checkpoint-offload mode.

For Qwen3 joint finetuning, Qwen3 gradient checkpointing reduces activation memory at the cost of extra recomputation. On memory-constrained GPUs, BF16/full-BF16, latent caching and an 8-bit optimizer are the preferred starting point. Text-encoder output caching cannot be used while Qwen3 is trainable.

## Updating sd-scripts later

The submodule is deliberately pinned to a known commit so a future `sd-scripts` change cannot silently break the GUI. To update it, advance the `sd-scripts` gitlink deliberately and rerun the Anima/Qwen3 compatibility tests. If the source anchors used by the staging patch no longer match, do not force the patch; re-review the changed upstream code first.
