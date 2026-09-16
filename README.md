<div align="center">

<img src="assets/dts-logo.webp" width="200" height="200" alt="Diffusion Trainer Studio" style="border-radius: 25px">

# Diffusion Trainer Studio

_✨ A unified training studio for diffusion models. ✨_

**v2.0.0**

</div>

<p align="center">
  <a href="https://github.com/WwlWss/diffusion-trainer-studio">GitHub</a>
  ·
  <a href="https://github.com/WwlWss/diffusion-trainer-studio/releases">Releases</a>
  ·
  <a href="README-zh.md">中文 README</a>
</p>

Diffusion Trainer Studio (DTS) is a WebUI, training-script preset collection and one-click training environment for multiple diffusion-model families. The project has grown from the original SD-Trainer / LoRA-scripts workflow into an independently maintained training studio with dedicated LoRA and full-finetune paths, runtime configuration validation, expanded tagging tools and Anima support.

> [!IMPORTANT]
> Diffusion Trainer Studio is developed from [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) and continues to use training infrastructure from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts). Many parts of the codebase also retain their original upstream licenses and attribution. Thank you to **秋葉 / Akegarasu**, **kohya-ss**, and all upstream contributors for the foundations this project builds on.

## Highlights in v2.0.0

- New project identity: **Diffusion Trainer Studio**.
- Dedicated training pages and backend contracts for SD / SDXL / Flux / Chroma / Anima workflows.
- Anima and Anima 2.9B LoRA and full finetuning.
- Optional Qwen3 text-encoder joint finetuning for Anima full training.
- One authoritative effective-config pipeline shared by Preview, Import, Export and Start.
- Expanded anime taggers including AnimeTimm, DanbooruTagQuery and PixAI while retaining the existing WD / CL workflows.
- Runtime validation for model family, training target, cache, dataset and memory-mode combinations.
- Contract and regression tests for training pages, schemas, APIs and trainer routing.

The historical SD-Trainer v1.x changelog belongs to the upstream project and remains available from [Akegarasu/lora-scripts Releases](https://github.com/Akegarasu/lora-scripts/releases).

## Training workflows

The WebUI currently exposes these main paths:

| Family | LoRA / network training | Full / finetune path |
| --- | --- | --- |
| SD 1.5 / SD2 | Yes | DreamBooth |
| SDXL | Yes | Yes |
| Flux | Yes | Yes |
| Chroma | Yes | — |
| Anima / Anima 2.9B | Yes | Yes |
| SD3 / SD3.5 | Yes | — |

Anima has additional controls documented in [ANIMA.md](ANIMA.md), including 2.9B model selection, block swapping and optional Qwen3 joint training.

## WebUI

Everything is integrated into one browser UI. After installation, run `run_gui.ps1` on Windows or `bash run_gui.sh` on Linux. The default address is:

`http://127.0.0.1:28000`

The repository keeps the pinned prebuilt frontend distribution clean and injects project-specific training pages and behavior at runtime. This reduces divergence from the frontend upstream while allowing DTS to maintain its own backend contracts and UI navigation.

## Installation

### Requirements

- Python 3.11 is the currently tested and recommended version for v2.0.0
- Git
- A CUDA-capable environment for GPU training

### Clone with submodules

```sh
git clone --recurse-submodules https://github.com/WwlWss/diffusion-trainer-studio
cd diffusion-trainer-studio
```

If the repository was cloned without submodules, initialize them with:

```sh
git submodule sync --recursive
git submodule update --init --recursive
```

### Windows

For the normal installation path:

```powershell
.\install.ps1
```

For the existing China-optimized installation path:

```powershell
.\install-cn.ps1
```

Then start the UI with:

```powershell
.\run_gui.ps1
```

### Linux

Run:

```bash
bash install.bash
bash run_gui.sh
```

### Docker

Build the current repository directly rather than relying on historical upstream images:

```bash
docker build -t diffusion-trainer-studio:latest .
docker run --gpus all -p 28000:28000 -p 6006:6006 diffusion-trainer-studio:latest
```

The Mainland China Dockerfile remains available as `Dockerfile-for-Mainland-China`.

## Traditional script workflow

The repository still contains the original script-oriented entry points for users who do not want the WebUI.

On Windows, edit and run `train.ps1`. On Linux, activate the environment, edit `train.sh`, and run it:

```sh
source venv/bin/activate
bash train.sh
```

TensorBoard can be started with the existing TensorBoard helper scripts and is normally available on port `6006`.

## Program arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--host` | str | `127.0.0.1` | Server host |
| `--port` | int | `28000` | WebUI port |
| `--listen` | bool | false | Enable listening mode |
| `--skip-prepare-environment` | bool | false | Skip environment preparation |
| `--disable-tensorboard` | bool | false | Disable TensorBoard |
| `--disable-tageditor` | bool | false | Disable the tag editor |
| `--tensorboard-host` | str | `127.0.0.1` | TensorBoard host |
| `--tensorboard-port` | int | `6006` | TensorBoard port |
| `--localization` | str | — | UI localization |
| `--dev` | bool | false | Developer mode |

## Project lineage and acknowledgements

Diffusion Trainer Studio would not exist without its upstream projects. In particular:

- [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) — original LoRA-scripts / SD-Trainer project by **秋葉 / Akegarasu**.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — training scripts and model-family implementations used throughout the project.
- [hanamizuki-ai/lora-gui-dist](https://github.com/hanamizuki-ai/lora-gui-dist) — pinned prebuilt frontend distribution.
- [Schemastery](https://github.com/shigma/schemastery) — schema-driven UI foundation.

Submodules and vendored/upstream-derived files keep their own license and attribution information. A DTS repository link should not be interpreted as an official support channel for any upstream project.
