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
- Full finetune hides LoRA rank/alpha/network settings and exposes Anima component learning rates: self-attention, cross-attention, MLP, AdaLN modulation and LLM Adapter.
- Adds Qwen3-0.6B, Qwen-Image VAE, LLM Adapter/T5 tokenizer, timestep, attention, VAE, caching and block-swap controls.
- Adds presets for standard Anima and Anima 2.9B, for both LoRA and full finetune.
- Updates Python dependencies to versions compatible with the current Anima implementation in `sd-scripts`.

## Anima 2.9B loader support

The upstream `kohya-ss/sd-scripts` Anima loader currently hardcodes `num_blocks = 28`, while Anima 2.9B uses 40 blocks. Until upstream PR #2418 is merged, this repository pins the `sd-scripts` submodule to commit `45dddfccb704b6b0591f65d98f0d695c997b3115` from the PR branch. That patch only changes Anima model loading: it probes the safetensors header for block 39 and instantiates 40 blocks for a 2.9B checkpoint, otherwise retaining the existing 28-block behavior.

Once equivalent support lands upstream, the submodule URL can be switched back to `kohya-ss/sd-scripts` and pinned to the corresponding upstream commit.

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
4. Choose **Anima training mode**:
   - `lora`: routes to `sd-scripts/anima_train_network.py`.
   - `finetune`: routes to `sd-scripts/anima_train.py`.
5. Select the matching Anima DiT checkpoint in `pretrained_model_name_or_path`.
6. Select the Qwen3-0.6B text encoder in `qwen3`.
7. Select the Qwen-Image VAE in `vae`.
8. Optionally provide a separate LLM Adapter or T5 tokenizer directory.
9. Configure the dataset and output settings, or load one of the included Anima presets.
10. Start training.

The model-variant selector is a GUI-only routing field and is removed before the config is passed to `sd-scripts`.

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

### Full finetune mode

Full finetune directly trains the Anima DiT and does not use `network_module`, `network_dim` or `network_alpha`.

The official `anima_train.py` keeps the Qwen3 text encoder frozen. The GUI exposes the script's per-component learning rates:

- `self_attn_lr`
- `cross_attn_lr`
- `mlp_lr`
- `mod_lr`
- `llm_adapter_lr`

Leaving one of these blank makes that component use the base `learning_rate`; setting it to `0` freezes that component.

The included full-finetune presets start conservatively with a base learning rate of `1e-5`, BF16 full-precision model weights, gradient checkpointing and both latent/text-output caching enabled. Adjust the learning rate and memory controls for your dataset and GPU.

## Memory controls

For lower VRAM cards, `blocks_to_swap` can be increased. The standard 28-block Anima model supports at most 26 swapped blocks, while the 40-block Anima 2.9B model supports at most 38. The launcher validates the limit against the selected model variant.

`blocks_to_swap`, `cpu_offload_checkpointing` and `unsloth_offload_checkpointing` are mutually constrained by upstream `sd-scripts`; do not enable block swap together with either checkpoint-offload mode.

## Updating sd-scripts later

The submodule is deliberately pinned to a known commit so a future `sd-scripts` change cannot silently break the GUI. To update it, advance the `sd-scripts` gitlink deliberately and retest the schema/arguments before merging that change.
