"""Runtime branding overlay for the pinned legacy VuePress frontend.

The frontend is intentionally kept as a clean pinned submodule. Project-owned
branding is injected at serve time so we can update the name, links and project
pages without forking or modifying the minified upstream distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import mikazuki.training_pages as training_pages

APP_ASSET = "app.547295de.js"
LAYOUT_ASSET = "layout.96d49288.js"
HOME_CONTENT_ASSET = "index.html.c6ef684b.js"
HOME_DATA_ASSET = "index.html.ec4ace46.js"
ABOUT_CONTENT_ASSET = "about.html.b4807002.js"
ABOUT_DATA_ASSET = "about.html.5b0c0de9.js"

PROJECT_NAME = "Diffusion Trainer Studio"
PROJECT_VERSION = "2.0.0"
PROJECT_REPOSITORY = "https://github.com/WwlWss/diffusion-trainer-studio"
PROJECT_ISSUES = f"{PROJECT_REPOSITORY}/issues"


def _replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"Frontend branding anchor {label!r} expected once, found {count}")
    return content.replace(old, new, 1)


def _replace_between_once(content: str, start: str, end: str, new: str, label: str) -> str:
    """Replace one span while retaining the exact pinned start/end anchors."""
    start_count = content.count(start)
    end_count = content.count(end)
    if start_count != 1 or end_count != 1:
        raise RuntimeError(
            f"Frontend branding span {label!r} expected one start/end anchor, "
            f"found start={start_count}, end={end_count}"
        )
    start_pos = content.index(start) + len(start)
    end_pos = content.index(end, start_pos)
    return content[:start_pos] + new + content[end_pos:]


def patch_branding_app_js(content: str) -> str:
    """Patch project-name strings after the training-page app patch is applied."""
    content = _replace_once(
        content,
        '{"text":"SD-Trainer","link":"/"}',
        '{"text":"Diffusion Trainer Studio","link":"/"}',
        "sidebar project name",
    )
    content = _replace_once(
        content,
        '["v-8daa1a0e","/",{title:"SD-Trainer"},',
        '["v-8daa1a0e","/",{title:"Diffusion Trainer Studio"},',
        "home route title",
    )
    return content


def patch_branding_layout_js(content: str) -> str:
    """Patch only the project-owned GitHub button in the left sidebar."""
    return _replace_once(
        content,
        'href:"https://github.com/Akegarasu/lora-scripts",target:"_blank","aria-label":"GitHub"',
        'href:"https://github.com/WwlWss/diffusion-trainer-studio",target:"_blank","aria-label":"GitHub"',
        "sidebar GitHub link",
    )


def home_html() -> str:
    """Return the DTS homepage body shared by the runtime page and SSR shell."""
    return f"""
<div align="center">
  <h1>Diffusion Trainer Studio</h1>
  <img src="/branding/logo.webp" width="200" height="200" alt="Diffusion Trainer Studio" style="margin:20px;border-radius:25px;object-fit:cover">
  <p><strong>多架构 Diffusion 模型训练工作台 · v{PROJECT_VERSION}</strong></p>
  <p>Maintained by <strong>WwlWss</strong> · <a href="{PROJECT_REPOSITORY}" target="_blank" rel="noopener noreferrer">GitHub</a></p>
  <p>基于 <a href="https://github.com/Akegarasu/lora-scripts" target="_blank" rel="noopener noreferrer">Akegarasu/lora-scripts</a> 与 <a href="https://github.com/kohya-ss/sd-scripts" target="_blank" rel="noopener noreferrer">kohya-ss/sd-scripts</a> 持续开发。感谢秋葉 / Akegarasu、kohya-ss 以及所有上游贡献者。</p>
</div>
<h2>v{PROJECT_VERSION} — Diffusion Trainer Studio</h2>
<p>本版本正式启用 Diffusion Trainer Studio 项目名称，并统一当前项目的界面、文档与仓库入口。</p>
<ul>
  <li>支持 Anima 与 Anima 2.9B 的 LoRA 和全参微调，并提供可选 Qwen3 文本编码器联合微调路径。</li>
  <li>按实际训练后端拆分 SD / SDXL / Flux / Chroma / Anima 的训练页面与配置语义。</li>
  <li>统一 Preview、Import、Export 与 Start 使用的 effective-config 管线。</li>
  <li>扩展 AnimeTimm、DanbooruTagQuery、PixAI 等 Tagger，并保留原有标签工具链。</li>
</ul>
<h3>项目沿革与致谢</h3>
<p>Diffusion Trainer Studio 从 Akegarasu/lora-scripts 的 SD-Trainer 工作流发展而来，并继续使用 kohya-ss/sd-scripts 及相关上游组件。上游项目、许可证与贡献者署名均予以保留。</p>
<p>旧版 SD-Trainer 的历史更新记录请查看 <a href="https://github.com/Akegarasu/lora-scripts/releases" target="_blank" rel="noopener noreferrer">上游 Releases</a>。</p>
""".strip()


def about_html() -> str:
    return f"""
<h2>关于 Diffusion Trainer Studio</h2>
<p>Diffusion Trainer Studio（DTS）是一个面向多种 Diffusion 架构的训练工作台，由 WwlWss 维护。</p>
<p>当前项目仓库：<a href="{PROJECT_REPOSITORY}" target="_blank" rel="noopener noreferrer">WwlWss/diffusion-trainer-studio</a>。问题与功能建议请提交到 <a href="{PROJECT_ISSUES}" target="_blank" rel="noopener noreferrer">GitHub Issues</a>。</p>
<h3>上游项目与致谢</h3>
<ul>
  <li><a href="https://github.com/Akegarasu/lora-scripts" target="_blank" rel="noopener noreferrer">Akegarasu/lora-scripts</a> — 原 SD-Trainer / LoRA-scripts 项目。感谢 <a href="https://space.bilibili.com/12566101" target="_blank" rel="noopener noreferrer">秋葉 / Akegarasu</a> 及其贡献者。</li>
  <li><a href="https://github.com/kohya-ss/sd-scripts" target="_blank" rel="noopener noreferrer">kohya-ss/sd-scripts</a> — 训练器与训练基础设施。</li>
  <li><a href="https://github.com/hanamizuki-ai/lora-gui-dist" target="_blank" rel="noopener noreferrer">hanamizuki-ai/lora-gui-dist</a> — 当前固定使用的预编译 GUI 分发。</li>
  <li><a href="https://github.com/shigma/schemastery" target="_blank" rel="noopener noreferrer">Schemastery</a> — GUI schema 基础组件。</li>
</ul>
<p>本项目保留并遵循仓库内各上游组件的许可证、版权声明与贡献者署名。本页面中的“当前项目”联系方式不代表上述上游项目的官方支持渠道。</p>
""".strip()


def patch_branding_index_html(content: str) -> str:
    """Patch the pinned pre-rendered shell before it reaches the browser.

    VuePress ships an SSR snapshot in ``dist/index.html``. Leaving that snapshot
    untouched would flash the old SD-Trainer identity before hydration and would
    expose stale branding to no-JS clients/crawlers, so the shell is patched with
    the same DTS homepage body used by the runtime page module.
    """
    content = _replace_once(
        content,
        "<title>SD-Trainer | SD 训练 UI</title>",
        "<title>Diffusion Trainer Studio | 多架构 Diffusion 模型训练工作台</title>",
        "document title",
    )
    content = _replace_once(
        content,
        '<meta name="description" content="">',
        '<meta name="description" content="Diffusion Trainer Studio"><link rel="icon" type="image/webp" href="/branding/logo.webp">',
        "document metadata",
    )
    content = _replace_once(
        content,
        'aria-label="SD-Trainer"><!--[--><!--]--> SD-Trainer <!--[--><!--]--></a>',
        'aria-label="Diffusion Trainer Studio"><!--[--><!--]--> Diffusion Trainer Studio <!--[--><!--]--></a>',
        "pre-rendered sidebar project name",
    )
    content = _replace_once(
        content,
        'href="https://github.com/Akegarasu/lora-scripts" target="_blank" aria-label="GitHub"',
        'href="https://github.com/WwlWss/diffusion-trainer-studio" target="_blank" aria-label="GitHub"',
        "pre-rendered sidebar GitHub link",
    )
    content = _replace_between_once(
        content,
        '<div class="theme-default-content"><!--[--><!--]--><div>',
        '</div><!--[--><!--]--></div><footer class="page-meta">',
        home_html(),
        "pre-rendered homepage",
    )

    forbidden = (
        "<title>SD-Trainer | SD 训练 UI</title>",
        'aria-label="SD-Trainer"',
        '<h1 id="sd-trainer"',
        "Stable Diffusion 训练 UI v1.13.0",
    )
    for anchor in forbidden:
        if anchor in content:
            raise RuntimeError(f"Legacy project branding survived frontend shell patch: {anchor}")
    return content


def _static_page_module(html: str, source_name: str) -> str:
    """Build a tiny VuePress page using only exports known to the pinned bundle."""
    html_json = json.dumps(html, ensure_ascii=True)
    source_json = json.dumps(source_name, ensure_ascii=True)
    return (
        'import{_ as n,o as s,c}from"./app.547295de.js";'
        f'const h={{}},i={html_json};'
        'function m(){return s(),c("div",{innerHTML:i})}'
        f'var x=n(h,[["render",m],["__file",{source_json}]]);export{{x as default}};\n'
    )


def home_content_js() -> str:
    return _static_page_module(home_html(), "runtime-dts-home.vue")


def about_content_js() -> str:
    return _static_page_module(about_html(), "runtime-dts-about.vue")


def _page_data_js(*, key: str, path: str, title: str, frontmatter: dict, file_path: str) -> str:
    payload = {
        "key": key,
        "path": path,
        "title": title,
        "lang": "en-US",
        "frontmatter": frontmatter,
        "excerpt": "",
        "headers": [],
        "filePathRelative": file_path,
    }
    compact = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"const e=JSON.parse({json.dumps(compact)});export{{e as data}};\n"


def home_data_js() -> str:
    return _page_data_js(
        key="v-8daa1a0e",
        path="/",
        title=PROJECT_NAME,
        frontmatter={"type": "dashboard"},
        file_path="index.md",
    )


def about_data_js() -> str:
    return _page_data_js(
        key="v-b5471278",
        path="/other/about.html",
        title=f"关于 {PROJECT_NAME}",
        frontmatter={},
        file_path="other/about.md",
    )


def install_frontend_branding_patch() -> None:
    """Compose branding on top of the already-installed training frontend patch."""
    original = training_pages.virtual_asset
    if getattr(original, "_mikazuki_branding_patch", False):
        return
    if not getattr(original, "_mikazuki_effective_config_patch", False):
        raise RuntimeError(
            "Frontend branding must be installed after the effective-config frontend patch"
        )

    def virtual_asset(asset_name: str):
        generated = original(asset_name)

        if asset_name == APP_ASSET:
            if generated is not None:
                raise RuntimeError("Unexpected generated app bundle before branding patch")
            path = Path("frontend/dist/assets") / APP_ASSET
            content = training_pages.patch_frontend_app_js(path.read_text(encoding="utf-8"))
            return patch_branding_app_js(content)

        if asset_name == LAYOUT_ASSET:
            content = generated
            if content is None:
                path = Path("frontend/dist/assets") / LAYOUT_ASSET
                content = path.read_text(encoding="utf-8")
            return patch_branding_layout_js(content)

        if asset_name == HOME_CONTENT_ASSET:
            return home_content_js()
        if asset_name == HOME_DATA_ASSET:
            return home_data_js()
        if asset_name == ABOUT_CONTENT_ASSET:
            return about_content_js()
        if asset_name == ABOUT_DATA_ASSET:
            return about_data_js()

        return generated

    # Preserve the contract marker from the wrapped effective-config layer so
    # future/repeated installer calls cannot mistake the outer wrapper for an
    # unpatched function and stack another effective-config wrapper on top.
    virtual_asset._mikazuki_effective_config_patch = True
    virtual_asset._mikazuki_branding_patch = True
    virtual_asset.__wrapped__ = original
    training_pages.virtual_asset = virtual_asset
