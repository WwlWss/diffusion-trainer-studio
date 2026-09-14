"""Fail-closed runtime patch for the pinned legacy training layout bundle.

The vendored frontend is intentionally kept at its pinned submodule revision.
Modern training pages are patched at serve time so they behave as a renderer of
raw Schemastery state only: every training semantic decision belongs to Python.
"""

from __future__ import annotations

from pathlib import Path

import mikazuki.training_pages as training_pages

_LAYOUT_ASSET = "layout.96d49288.js"


def _replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"Legacy frontend patch anchor {label!r} expected once, found {count}")
    return content.replace(old, new, 1)


def patch_training_layout_js(content: str) -> str:
    # Schema hot-update must refresh the shared closure as well as this.schemas.
    # Otherwise a newly downloaded page schema can evaluate against the previous
    # SHARED_SCHEMAS until the user refreshes a second time.
    content = _replace_once(
        content,
        'async loadServerSchema(){let t=await(await get("/api/schemas/all")).json();t.status=="success"&&(localStorage.setItem("schemas",JSON.stringify(t.data.schemas)),this.schemas=t.data.schemas)}',
        'async loadServerSchema(){let t=await(await get("/api/schemas/all")).json();t.status=="success"&&(localStorage.setItem("schemas",JSON.stringify(t.data.schemas)),this.schemas=t.data.schemas,this.schemas.forEach(s=>{delete s.schemaObject}),SHARED_SCHEMAS=void 0,this.loadSharedSchema())}',
        "schema hot reload",
    )

    content = _replace_once(
        content,
        'C=ref([]),d=ref([]),w=["network_args_custom","optimizer_args_custom"]',
        'C=ref([]),d=ref([]),__effectiveToml=ref("Loading..."),__previewTimer=null,__startPending=ref(!1),w=["network_args_custom","optimizer_args_custom"]',
        "effective preview state",
    )
    content = _replace_once(
        content,
        'onMounted(async()=>{I(),y()})',
        'onMounted(async()=>{I(),y(),await nextTick(),__refreshPreview()})',
        "initial effective preview",
    )

    # T() is the only input to the backend: schema-normalized raw GUI state.
    # In particular, never call legacy parseParams()/checkParams() here.
    content = _replace_once(
        content,
        'const x=()=>{if(n.value==null)return"Loading...";let _=T(),m=parseParams(_,t),g=checkParams(m);return C.value=g.warnings,d.value=g.errors,stringify(m)},L=computed(()=>{try{return x()}catch(_){console.log(_)}}),I=()=>',
        'const __trainingRequest=async(endpoint,raw)=>{let N=await post(endpoint,JSON.stringify({train_type:t,config:raw}),{"Content-Type":"application/json"});let D=await N.json();if(!N.ok||D.status!="success")throw new Error(D.message||"配置校验失败");return D},__requestEffective=async(raw=T(),endpoint="/api/training/preview")=>{if(n.value==null)return"Loading...";let D=await __trainingRequest(endpoint,raw);return C.value=D.data&&D.data.warnings||[],d.value=[],D.data.toml},__refreshPreview=()=>{clearTimeout(__previewTimer),__previewTimer=setTimeout(async()=>{try{__effectiveToml.value=await __requestEffective(),d.value=[]}catch(_){d.value=[_.message||String(_)],C.value=[],__effectiveToml.value="# 配置解析失败\\n# "+(_.message||String(_))}},300)},x=()=>__effectiveToml.value,L=computed(()=>x());watch(a,__refreshPreview,{deep:!0});const I=()=>',
        "backend-only effective preview",
    )

    # Start posts exactly the same raw T() state. The backend performs the one
    # preparation pass and owns validation + TOML materialization.
    old_start = 'O=async()=>{const _=parseParams(n.value(a.value),t);_.optimizer_type=="DAdaptation"&&ElMessage.warning({message:"DAdaptation \\u8BAD\\u7EC3\\u65F6\\uFF0C\\u6240\\u6709\\u5B66\\u4E60\\u7387\\u5C06\\u88AB\\u8BBE\\u7F6E\\u4E3A 1\\u3002\\u5E76\\u4E14\\u5B66\\u4E60\\u7387\\u8C03\\u5EA6\\u5668\\u5C06\\u88AB\\u8BBE\\u7F6E\\u4E3A constant\\u3002",duration:5e3});try{let m=await post("/api/run",JSON.stringify(_),{"Content-Type":"application/json"});if(!m.ok)throw new Error("Network response was not ok");let g=await m.json();g.status=="success"?ElMessage.success("\\u8BAD\\u7EC3\\u4EFB\\u52A1\\u5DF2\\u63D0\\u4EA4\\u6210\\u529F\\uFF1A"+g.message):ElMessage.error("\\u8BAD\\u7EC3\\u4EFB\\u52A1\\u63D0\\u4EA4\\u5931\\u8D25\\uFF1A"+g.message)}catch(m){ElMessage.error(v("networkError")),console.error("There was a problem with the fetch operation:",m)}}'
    new_start = 'O=async()=>{if(__startPending.value)return;__startPending.value=!0;try{let g=await __trainingRequest("/api/run",T());g.data&&g.data.task_id&&sessionStorage.setItem(`current-task:${t}`,String(g.data.task_id)),ElMessage.success("\\u8BAD\\u7EC3\\u4EFB\\u52A1\\u5DF2\\u63D0\\u4EA4\\u6210\\u529F\\uFF1A"+g.message)}catch(m){ElMessage.error(m.message||v("networkError")),console.error("There was a problem with the fetch operation:",m)}finally{__startPending.value=!1}}'
    content = _replace_once(content, old_start, new_start, "raw start")

    # Stop discovers backend-owned task association. sessionStorage is only a
    # preference when multiple historical task entries exist, never authority.
    old_stop = 'K=async()=>{let _=null;try{let V=(await(await fetch("/api/tasks")).json()).data.tasks.filter(k=>k.status=="RUNNING");if(V.length==0){ElMessage.warning("\\u5F53\\u524D\\u6CA1\\u6709\\u6B63\\u5728\\u8FD0\\u884C\\u7684\\u8BAD\\u7EC3\\u4EFB\\u52A1");return}_=V[0]}catch(g){ElMessage.error(v("networkError")),console.error("There was a problem with the fetch operation:",g);return}'
    new_stop = 'K=async()=>{let _=null;try{const __remembered=sessionStorage.getItem(`current-task:${t}`);let V=(await(await fetch("/api/tasks")).json()).data.tasks.filter(k=>(k.status=="CREATED"||k.status=="RUNNING")&&(k.page_train_type===t||String(k.id)===String(__remembered)));if(V.length==0){ElMessage.warning("\\u5F53\\u524D\\u9875\\u9762\\u6CA1\\u6709\\u6B63\\u5728\\u542F\\u52A8\\u6216\\u8FD0\\u884C\\u7684\\u8BAD\\u7EC3\\u4EFB\\u52A1");return}_=V.find(k=>String(k.id)===String(__remembered))||V[0]}catch(g){ElMessage.error(v("networkError")),console.error("There was a problem with the fetch operation:",g);return}'
    content = _replace_once(content, old_stop, new_stop, "backend task stop")

    # Download uses the dedicated export endpoint, which materializes any
    # content-addressed prompt sidecar before returning the final trainer TOML.
    content = _replace_once(
        content,
        'E=()=>{const _=x(),g=`${new Date().getTime()}.toml`;P(g,_)}',
        'E=async()=>{try{const _=await __requestEffective(T(),"/api/training/export"),g=`${new Date().getTime()}.toml`;P(g,_)}catch(_){ElMessage.error(_.message||String(_))}}',
        "effective config export",
    )

    # A downloaded trainer TOML is not raw GUI state. Rehydrate on the backend,
    # then let the current schema render the semantic state.
    old_import = 'S=()=>{const _=document.createElement("input");_.type="file",_.accept=".toml",_.onchange=m=>{const g=m.target.files[0],N=new FileReader;N.onload=D=>{const V=D.target.result;try{let k=TomlParse(V),U=findChangedDataBySchema(k,n.value);a.value=U,ElMessage.success("\\u5BFC\\u5165\\u6210\\u529F")}catch(k){console.log(k),ElMessage.error("\\u5BFC\\u5165\\u5931\\u8D25")}},N.readAsText(g)},_.click()}'
    new_import = 'S=()=>{const _=document.createElement("input");_.type="file",_.accept=".toml",_.onchange=m=>{const g=m.target.files[0],N=new FileReader;N.onload=async D=>{const V=D.target.result;try{let k=TomlParse(V),U=await __trainingRequest("/api/training/rehydrate",k),B=U.data&&U.data.gui_state;if(!B||typeof B!=="object")throw new Error("导入结果缺少 gui_state");a.value=Object.assign({},n.value(),B),ElMessage.success("\\u5BFC\\u5165\\u6210\\u529F"),await nextTick(),__refreshPreview()}catch(k){console.log(k),ElMessage.error(k.message||"\\u5BFC\\u5165\\u5931\\u8D25")}},N.readAsText(g)},_.click()}'
    content = _replace_once(content, old_import, new_import, "trainer TOML rehydrate")

    # Presets/history remain raw GUI state when applied, but every preview uses
    # the same backend effective-config API rather than stringify/parseParams.
    content = _replace_once(
        content,
        'q=_=>{ElMessageBox.alert(stringify(_),"\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}',
        'q=async _=>{try{let m=n.value(clone(_)),g=await __requestEffective(m);ElMessageBox.alert(g,"\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}catch(m){ElMessage.error(m.message||String(m))}}',
        "preset effective preview",
    )
    content = _replace_once(
        content,
        'Z=(_,m)=>{const g=stringify(parseParams(n.value(clone(m.value)),t));ElMessageBox.alert(g,"\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}',
        'Z=async(_,m)=>{try{const g=await __requestEffective(n.value(clone(m.value)));ElMessageBox.alert(g,"\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}catch(g){ElMessage.error(g.message||String(g))}}',
        "history effective preview",
    )

    # Disable Start while its request is pending. Backend CREATED/RUNNING locking
    # remains the authoritative protection across tabs/processes.
    content = _replace_once(
        content,
        'createVNode(g,{plain:"",class:"max-btn color-btn",type:"primary",onClick:O}',
        'createVNode(g,{plain:"",class:"max-btn color-btn",type:"primary",disabled:__startPending.value,onClick:O}',
        "start pending disable",
    )

    # Fail closed if any modern action still executes parseParams. The function
    # definition is intentionally left in the bundle for unrelated legacy code,
    # but these specific callsites must be gone.
    forbidden = (
        'm=parseParams(_,t),g=checkParams(m)',
        'O=async()=>{const _=parseParams(',
        'stringify(parseParams(n.value(clone(m.value)),t))',
    )
    for anchor in forbidden:
        if anchor in content:
            raise RuntimeError(f"Legacy parseParams callsite survived frontend patch: {anchor}")
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
