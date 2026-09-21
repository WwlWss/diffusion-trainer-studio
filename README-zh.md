<div align="center">

<img src="assets/dts-logo.webp" width="200" height="200" alt="Diffusion Trainer Studio" style="border-radius: 25px">

# Diffusion Trainer Studio

_✨ 多架构 Diffusion 模型训练工作台 ✨_

**v2.0.0**

</div>

<p align="center">
  <a href="https://github.com/WwlWss/diffusion-trainer-studio">GitHub</a>
  ·
  <a href="https://github.com/WwlWss/diffusion-trainer-studio/releases">Releases</a>
  ·
  <a href="README.md">English README</a>
</p>

Diffusion Trainer Studio（DTS）是一套面向多种 Diffusion 模型架构的训练 WebUI、脚本预设和一键训练环境。项目从原有 SD-Trainer / LoRA-scripts 工作流持续发展，现已包含独立的 LoRA 与全参微调页面、统一配置校验、扩展 Tagger，以及完整的 Anima 训练支持。

> [!IMPORTANT]
> Diffusion Trainer Studio 基于 [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) 开发，并继续使用 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 的训练基础设施。仓库中大量上游代码继续保留原有许可证与署名。感谢 **秋葉 / Akegarasu**、**kohya-ss** 以及所有上游贡献者为本项目提供的基础。

## v2.0.0 主要变化

- 正式启用 **Diffusion Trainer Studio** 项目名称与独立品牌。
- 按实际训练后端拆分 SD / SDXL / Flux / Chroma / Anima 的训练页面和配置语义。
- 支持 Anima 与 Anima 2.9B 的 LoRA 和全参微调。
- Anima 全参训练支持可选的 Qwen3 文本编码器联合微调。
- Preview、Import、Export 与 Start 统一使用同一套 effective-config 管线。
- 十个已验证训练后端支持可选的 Component-wise Parameter Policy，可按组件独立设置优化器、学习率与冻结状态，并在 Start 前 fail-closed 检查运行时兼容性。
- Tagger 新增 AnimeTimm、DanbooruTagQuery、PixAI 等模型，同时保留原有 WD / CL 工作流。
- 补充模型类型、训练目标、缓存、数据集与显存模式等组合的后端校验。
- 增加训练页面、schema、API 与 trainer 路由的自动化契约测试。

过去的 SD-Trainer v1.x 更新日志属于上游项目，可前往 [Akegarasu/lora-scripts Releases](https://github.com/Akegarasu/lora-scripts/releases) 查看。本项目从 v2.0.0 起使用 Diffusion Trainer Studio 的版本体系。

## 主要训练工作流

| 模型家族 | LoRA / Network | 全参 / Finetune |
| --- | --- | --- |
| SD 1.5 / SD2 | 支持 | DreamBooth |
| SDXL | 支持 | 支持 |
| Flux | 支持 | 支持 |
| Chroma | 支持 | — |
| Anima / Anima 2.9B | 支持 | 支持 |
| SD3 / SD3.5 | 支持 | — |

Anima 的 2.9B 模型、Block Swap、Qwen3 联合训练等额外说明见 [ANIMA.md](ANIMA.md)。

按组件配置优化器与学习率的使用方法见 [Parameter Policy 指南](docs/parameter-policy.md)。Standard 仍是默认模式；Component 为显式启用，当前支持范围与限制以该指南中的十后端矩阵为准。

## WebUI

训练、配置预览、任务管理、TensorBoard、Tagger 与标签编辑工具都集成在同一个浏览器界面中。安装后在 Windows 运行 `run_gui.ps1`，Linux 运行 `bash run_gui.sh`。默认地址为：

`http://127.0.0.1:28000`

当前项目继续固定使用上游预编译 frontend，并通过运行时注入的方式添加 DTS 自己的训练页面、导航与配置行为，而不是直接修改 frontend 子模块。这可以降低与上游 GUI 更新之间的冲突。

## 安装

### 必要环境

- Python 3.11 是 v2.0.0 当前 CI 实际测试并推荐的版本
- Git
- 进行 GPU 训练时需要可用的 CUDA 环境

### 克隆仓库和子模块

```sh
git clone --recurse-submodules https://github.com/WwlWss/diffusion-trainer-studio
cd diffusion-trainer-studio
```

如果之前没有初始化子模块：

```sh
git submodule sync --recursive
git submodule update --init --recursive
```

### Windows

普通安装路径：

```powershell
.\install.ps1
```

中国大陆优化安装路径：

```powershell
.\install-cn.ps1
```

安装完成后启动 GUI：

```powershell
.\run_gui.ps1
```

### Linux

```bash
bash install.bash
bash run_gui.sh
```

### Docker

建议直接基于当前仓库自行构建镜像，不再把上游历史镜像描述成 DTS 的官方镜像：

```bash
docker build -t diffusion-trainer-studio:latest .
docker run --gpus all -p 28000:28000 -p 6006:6006 diffusion-trainer-studio:latest
```

国内环境仍可使用 `Dockerfile-for-Mainland-China` 进行构建。

## 传统脚本训练方式

仓库仍保留不依赖 WebUI 的传统脚本入口。

Windows 可编辑并运行 `train.ps1`。Linux 可先激活虚拟环境，再编辑并运行 `train.sh`：

```sh
source venv/bin/activate
bash train.sh
```

TensorBoard 默认使用 `6006` 端口，也可以继续使用仓库中的 TensorBoard 辅助脚本。

## 程序参数

| 参数名称 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `--host` | str | `127.0.0.1` | WebUI 主机地址 |
| `--port` | int | `28000` | WebUI 端口 |
| `--listen` | bool | false | 启用监听模式 |
| `--skip-prepare-environment` | bool | false | 跳过环境准备 |
| `--disable-tensorboard` | bool | false | 禁用 TensorBoard |
| `--disable-tageditor` | bool | false | 禁用标签编辑器 |
| `--tensorboard-host` | str | `127.0.0.1` | TensorBoard 主机地址 |
| `--tensorboard-port` | int | `6006` | TensorBoard 端口 |
| `--localization` | str | — | 界面本地化设置 |
| `--dev` | bool | false | 开发者模式 |

## 项目沿革与致谢

Diffusion Trainer Studio 的开发建立在多个上游项目之上：

- [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) — 原 LoRA-scripts / SD-Trainer 项目，作者 **秋葉 / Akegarasu**。
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — 项目大量训练脚本与模型架构实现的基础。
- [hanamizuki-ai/lora-gui-dist](https://github.com/hanamizuki-ai/lora-gui-dist) — 当前固定使用的预编译 frontend 分发。
- [Schemastery](https://github.com/shigma/schemastery) — schema 驱动的界面基础组件。

子模块、上游派生文件和第三方代码继续遵循各自的许可证和版权声明。DTS 的 GitHub Issues 仅用于当前项目，不代表任何上游项目的官方支持渠道。
