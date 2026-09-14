"""Fail-closed runtime patch for the pinned legacy training layout bundle."""

from __future__ import annotations

from pathlib import Path

import mikazuki.training_pages as training_pages

_LAYOUT_ASSET = "layout.96d49288.js"


def _replace_once(content: str, old: str, new: str, label: str) -> str:
    if old not in content:
        raise RuntimeError(f"Legacy frontend patch anchor missing: {label}")
    return content.replace(old, new, 1)


def patch_training_layout_js(content: str) -> str:
    content = _replace_once(
        content,
        'C=ref([]),d=ref([]),w=["network_args_custom","optimizer_args_custom"]',
        'C=ref([]),d=ref([]),__effectiveToml=ref("Loading..."),__previewTimer=null,w=["network_args_custom","optimizer_args_custom"]',
        "preview state",
    )
    content = _replace_once(
        content,
        'onMounted(async()=>{I(),y()})',
        'onMounted(async()=>{I(),y(),await nextTick(),__refreshPreview()})',
        "initial preview",
    )
    content = _replace_once(
        content,
        'const x=()=>{if(n.value==null)return"Loading...";let _=T(),m=parseParams(_,t),g=checkParams(m);return C.value=g.warnings,d.value=g.errors,stringify(m)},L=computed(()=>{try{return x()}catch(_){console.log(_)}}),I=()=>',
        'const __requestEffective=async()=>{if(n.value==null)return"Loading...";let _=T(),m=parseParams(_,t),g=checkParams(m);C.value=g.warnings,d.value=[];let N=await post("/api/training/preview",JSON.stringify({train_type:t,config:m}),{"Content-Type":"application/json"});if(!N.ok)throw new Error("Training preview request failed");let D=await N.json();if(D.status!="success")throw new Error(D.message||"配置校验失败");return C.value=[...g.warnings,...(D.data&&D.data.warnings||[])],D.data.toml},__refreshPreview=()=>{clearTimeout(__previewTimer),__previewTimer=setTimeout(async()=>{try{__effectiveToml.value=await __requestEffective(),d.value=[]}catch(_){d.value=[_.message||String(_)],__effectiveToml.value="# 配置解析失败\\n# "+(_.message||String(_))}},300)},x=()=>__effectiveToml.value,L=computed(()=>x());watch(a,__refreshPreview,{deep:!0});const I=()=>',
        "effective preview",
    )
    content = _replace_once(
        content,
        'O=async()=>{const _=parseParams(n.value(a.value),t);',
        'O=async()=>{const _=parseParams(n.value(a.value),t),__payload={train_type:t,config:_};',
        "run payload routing",
    )
    content = _replace_once(
        content,
        'try{let m=await post("/api/run",JSON.stringify(_),{"Content-Type":"application/json"});',
        'try{let __check=await post("/api/training/preview",JSON.stringify(__payload),{"Content-Type":"application/json"}),__checkData=await __check.json();if(!__check.ok||__checkData.status!="success"){ElMessage.error(__checkData.message||"参数校验失败");return}let m=await post("/api/run",JSON.stringify(__payload),{"Content-Type":"application/json"});',
        "run validation gate",
    )
    content = _replace_once(
        content,
        'let g=await m.json();g.status=="success"?',
        'let g=await m.json();g.status=="success"&&g.data&&g.data.task_id&&sessionStorage.setItem(`current-task:${t}`,String(g.data.task_id));g.status=="success"?',
        "task id persistence",
    )
    content = _replace_once(
        content,
        'K=async()=>{let _=null;try{let V=',
        'K=async()=>{let _=null;try{const __taskId=sessionStorage.getItem(`current-task:${t}`);if(!__taskId){ElMessage.warning("当前页面没有记录正在运行的训练任务");return}let V=',
        "stop task identity",
    )
    content = _replace_once(
        content,
        '.data.tasks.filter(k=>k.status=="RUNNING");if(V.length==0)',
        '.data.tasks.filter(k=>k.status=="RUNNING"&&String(k.id)===String(__taskId));if(V.length==0)',
        "stop task filter",
    )
    content = _replace_once(
        content,
        'E=()=>{const _=x(),g=`${new Date().getTime()}.toml`;P(g,_)}',
        'E=async()=>{try{const _=await __requestEffective(),g=`${new Date().getTime()}.toml`;P(g,_)}catch(_){ElMessage.error(_.message||String(_))}}',
        "effective config download",
    )
    return content


def install_frontend_training_patch() -> None:
    original = training_pages.virtual_asset
    if getattr(original, "_mikazuki_effective_config_patch", False):
        return

    def virtual_asset(asset_name: str):
        generated = original(asset_name)
        if generated is not None:
            return generated
        if asset_name == _LAYOUT_ASSET:
            path = Path("frontend/dist/assets") / _LAYOUT_ASSET
            return patch_training_layout_js(path.read_text(encoding="utf-8"))
        return None

    virtual_asset._mikazuki_effective_config_patch = True
    training_pages.virtual_asset = virtual_asset
