"""Fail-closed runtime patch for the pinned legacy training layout bundle."""

from __future__ import annotations

from pathlib import Path
import json
import re

import mikazuki.training_pages as training_pages
from mikazuki.optimizer_profiles import list_optimizer_capabilities

_LAYOUT_ASSET = "layout.96d49288.js"
_ELIGIBILITY_OPTIMIZER_TYPES_JS = json.dumps(
    [
        capability.name.casefold()
        for capability in list_optimizer_capabilities()
        if capability.requires_parameter_eligibility
    ],
    ensure_ascii=False,
    separators=(",", ":"),
)


def _replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"Legacy frontend patch anchor {label!r} expected once, found {count}")
    return content.replace(old, new, 1)


def _replace_span_once(content: str, start: str, end: str, new: str, label: str) -> str:
    pattern = re.escape(start) + r".*?" + re.escape(end)
    replaced, count = re.subn(pattern, lambda _: new + end, content, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Legacy frontend patch span {label!r} expected once, found {count}")
    return replaced


def patch_training_layout_js(content: str) -> str:
    content = _replace_once(
        content,
        'async loadServerSchema(){let t=await(await get("/api/schemas/all")).json();t.status=="success"&&(localStorage.setItem("schemas",JSON.stringify(t.data.schemas)),this.schemas=t.data.schemas)}',
        'async loadServerSchema(){let t=await(await get("/api/schemas/all")).json();t.status=="success"&&(localStorage.setItem("schemas",JSON.stringify(t.data.schemas)),this.schemas=t.data.schemas,this.schemas.forEach(s=>{delete s.schemaObject}),SHARED_SCHEMAS=void 0,this.loadSharedSchema())}',
        "schema hot reload",
    )

    # This is part of the bundle's existing const declaration chain. Mutable
    # preview state therefore lives inside refs instead of being reassigned.
    content = _replace_once(
        content,
        'C=ref([]),d=ref([]),w=["network_args_custom","optimizer_args_custom"]',
        'C=ref([]),d=ref([]),__effectiveToml=ref("Loading..."),__previewTimer=ref(null),__previewGeneration=ref(0),__startPending=ref(!1),__runtimeReady=ref(!0),__runtimeBlockers=ref([]),__policyBootstrapPending=ref(!1),__policyModeGuard=ref(!1),__policyBootstrapGeneration=ref(0),w=["network_args_custom","optimizer_args_custom"]',
        "effective preview state",
    )

    # Never mutate the deep-watched form while compiling a backend request.
    content = _replace_once(
        content,
        'T=()=>{let _=a.value;w.forEach(g=>{_&&_.hasOwnProperty(g)&&_[g]!=null&&(_[g]=_[g].map(N=>N||""))});let m=n.value(_);return w.forEach(g=>{m.hasOwnProperty(g)&&m[g].length==0&&delete m[g]}),m}',
        '__resolveGuiState=_=>{let R=clone(_);w.forEach(g=>{R&&R.hasOwnProperty(g)&&R[g]!=null&&(R[g]=R[g].map(N=>N||""))});let m=n.value(R);return w.forEach(g=>{m.hasOwnProperty(g)&&m[g].length==0&&delete m[g]}),m},T=()=>__resolveGuiState(a.value)',
        "pure raw gui normalization",
    )

    content = _replace_once(
        content,
        'onMounted(async()=>{I(),y()})',
        'onMounted(async()=>{I(),y(),await nextTick(),await __syncPolicyMode()})',
        "initial effective preview",
    )
    content = _replace_once(
        content,
        'onBeforeUnmount(()=>{localStorage.setItem(`configs-${t}-autosave`,JSON.stringify(clone(a.value)))})',
        'onBeforeUnmount(()=>{window.__dtsPolicyProfileHooks===__policyProfileHookOwner&&delete window.__dtsPolicyProfileHooks,localStorage.setItem(`configs-${t}-autosave`,JSON.stringify(clone(a.value)))})',
        "policy profile hook cleanup",
    )

    content = _replace_once(
        content,
        'const x=()=>{if(n.value==null)return"Loading...";let _=T(),m=parseParams(_,t),g=checkParams(m);return C.value=g.warnings,d.value=g.errors,stringify(m)},L=computed(()=>{try{return x()}catch(_){console.log(_)}}),I=()=>',
        'const __trainingRequest=async(endpoint,raw)=>{let N=await post(endpoint,JSON.stringify({train_type:t,config:raw}),{"Content-Type":"application/json"});let D=await N.json();if(!N.ok||D.status!="success")throw new Error(D.message||"配置校验失败");return D},__requestEffective=async(raw=T(),endpoint="/api/training/preview")=>{if(n.value==null)return{toml:"Loading...",warnings:[]};let D=await __trainingRequest(endpoint,raw);return D.data||{}},__isComponentMode=()=>String(a.value&&a.value.optimization_mode||"standard").toLowerCase()=="component",__hasPolicyState=()=>{let P=a.value&&a.value.parameter_policy_profiles,Cc=a.value&&a.value.parameter_policy_components;return!!(P&&typeof P=="object"&&Object.keys(P).length&&Cc&&typeof Cc=="object"&&Object.keys(Cc).length)},__profileNames=()=>Object.keys(a.value&&a.value.parameter_policy_profiles||{}).filter(N=>String(N||"").trim()),__optimizerRequiresEligibility=T=>"+_ELIGIBILITY_OPTIMIZER_TYPES_JS+".includes(String(T||"").trim().toLowerCase()),__renamePolicyProfile=(O,N)=>{if(O===N)return!0;let P=a.value&&a.value.parameter_policy_profiles||{},F=String(N??"").trim();if(String(O||"").trim()&&!F){ElMessage.warning("Optimizer Profile 名称不能为空");return!1}let L=F.toLowerCase();if(L&&Object.keys(P).some(K=>K!==O&&String(K).trim().toLowerCase()===L)){ElMessage.warning("Optimizer Profile 名称已存在");return!1}let C=a.value&&a.value.parameter_policy_components||{};for(const R of Object.values(C)){if(!R||typeof R!="object")continue;R.optimizer_profile===O&&(R.optimizer_profile=N),R.fallback_optimizer_profile===O&&(R.fallback_optimizer_profile=N)}return!0},__canDeletePolicyProfile=N=>{let C=a.value&&a.value.parameter_policy_components||{},R=[];for(const[I,U]of Object.entries(C)){if(!U||typeof U!="object")continue;U.optimizer_profile===N&&R.push(I+": primary"),U.fallback_optimizer_profile===N&&R.push(I+": fallback")}return R.length?(ElMessage.warning("Profile "+N+" 仍被 "+R.length+" 个 Component 引用，请先修改引用。"),!1):!0},__policyProfileNameFromPrefix=P=>{let B="parameter_policy_profiles.",R=String(P||"");return R.startsWith(B)&&R.endsWith(".")?R.slice(B.length,-1):""},__onPolicyOptimizerTypeChanged=(P,O,N)=>{let K=__policyProfileNameFromPrefix(P);if(!K||O===N)return;let C=a.value&&a.value.parameter_policy_components||{},OM=__optimizerRequiresEligibility(O),NM=__optimizerRequiresEligibility(N);for(const R of Object.values(C)){if(!R||typeof R!="object")continue;OM&&!NM&&R.optimizer_profile===K&&(delete R.fallback_optimizer_profile,delete R.fallback_learning_rate),!OM&&NM&&R.fallback_optimizer_profile===K&&(delete R.fallback_optimizer_profile,delete R.fallback_learning_rate)}},__installPolicyProfileHooks=()=>{let H={rename:__renamePolicyProfile,canDelete:__canDeletePolicyProfile,optimizerTypeChanged:__onPolicyOptimizerTypeChanged,profileNames:__profileNames};return window.__dtsPolicyProfileHooks=H,H},__policyProfileHookOwner=__installPolicyProfileHooks(),__presetPolicyMode=_=>{let P=_&&_.parameter_policy_profiles,Cc=_&&_.parameter_policy_components,M=String(_&&_.optimization_mode||"standard").toLowerCase();if(M=="component")return P&&typeof P=="object"&&Object.keys(P).length&&Cc&&typeof Cc=="object"&&Object.keys(Cc).length?"component":"invalid-component";return"standard"},__isComponentPreset=_=>__presetPolicyMode(_)=="component",__policyBootstrapSnapshot=()=>{let R=clone(a.value||{});delete R.parameter_policy_profiles,delete R.parameter_policy_components,R.optimization_mode="standard";return __resolveGuiState(R)},__bootstrapPolicy=async()=>{if(__policyBootstrapPending.value||__policyModeGuard.value||!__isComponentMode()||__hasPolicyState())return;const G=++__policyBootstrapGeneration.value;clearTimeout(__previewTimer.value),++__previewGeneration.value,__policyBootstrapPending.value=!0,__runtimeReady.value=!1,__runtimeBlockers.value=[];try{let U=await __trainingRequest("/api/training/parameter-policy/bootstrap",__policyBootstrapSnapshot()),B=U.data&&U.data.gui_state;if(G!==__policyBootstrapGeneration.value||!__isComponentMode())return;if(!B||typeof B!="object"||!B.parameter_policy_profiles||!B.parameter_policy_components)throw new Error("Parameter Policy bootstrap 返回不完整 gui_state");__policyModeGuard.value=!0,Object.assign(a.value,{parameter_policy_profiles:clone(B.parameter_policy_profiles),parameter_policy_components:clone(B.parameter_policy_components)}),await nextTick()}catch(_){if(G!==__policyBootstrapGeneration.value||!__isComponentMode())return;__runtimeReady.value=!1,__runtimeBlockers.value=[_.message||String(_)],d.value=[_.message||String(_)]}finally{if(G===__policyBootstrapGeneration.value){__policyModeGuard.value=!1,__policyBootstrapPending.value=!1;if(__isComponentMode()&&__hasPolicyState())__refreshPreview()}}},__syncPolicyMode=async()=>{if(__policyModeGuard.value)return;if(!__isComponentMode()){++__policyBootstrapGeneration.value,__policyBootstrapPending.value=!1,__runtimeReady.value=!0,__runtimeBlockers.value=[],__refreshPreview();return}__runtimeReady.value=!1;if(__hasPolicyState()){++__policyBootstrapGeneration.value,__policyBootstrapPending.value=!1,__refreshPreview();return}await __bootstrapPolicy()},__refreshPreview=()=>{clearTimeout(__previewTimer.value);const __generation=++__previewGeneration.value;if(__isComponentMode())__runtimeReady.value=!1;if(__policyBootstrapPending.value)return;__previewTimer.value=setTimeout(async()=>{try{let R=await __requestEffective();if(__generation!==__previewGeneration.value)return;C.value=R.warnings||[],__effectiveToml.value=R.toml||"";if(__isComponentMode()){__runtimeReady.value=R.runtime_ready===!0,__runtimeBlockers.value=Array.isArray(R.runtime_blockers)?R.runtime_blockers:[],d.value=__runtimeReady.value?[]:(__runtimeBlockers.value.length?__runtimeBlockers.value:["Component runtime is not ready."])}else __runtimeReady.value=!0,__runtimeBlockers.value=[],d.value=[]}catch(_){if(__generation!==__previewGeneration.value)return;let M=_.message||String(_);if(__isComponentMode())__runtimeReady.value=!1,__runtimeBlockers.value=[M];d.value=[M],C.value=[],__effectiveToml.value="# 配置解析失败\\n# "+M}},300)},x=()=>__effectiveToml.value,L=computed(()=>x());watch(a,__refreshPreview,{deep:!0});watch(()=>a.value&&a.value.optimization_mode,__syncPolicyMode);const I=()=>',
        "backend-only effective preview",
    )

    new_start = 'O=async()=>{if(__startPending.value||__policyBootstrapPending.value||__isComponentMode()&&!__runtimeReady.value)return;__startPending.value=!0;try{let g=await __trainingRequest("/api/run",T());g.data&&g.data.task_id&&sessionStorage.setItem(`current-task:${t}`,String(g.data.task_id)),ElMessage.success("\\u8BAD\\u7EC3\\u4EFB\\u52A1\\u5DF2\\u63D0\\u4EA4\\u6210\\u529F\\uFF1A"+g.message)}catch(m){ElMessage.error(m.message||v("networkError")),console.error("There was a problem with the fetch operation:",m)}finally{__startPending.value=!1}}'
    content = _replace_span_once(
        content,
        'O=async()=>{const _=parseParams(n.value(a.value),t);',
        ',K=async()=>',
        new_start,
        "raw start",
    )

    new_stop_prefix = 'K=async()=>{let _=null;try{const __remembered=sessionStorage.getItem(`current-task:${t}`);let V=(await(await fetch("/api/tasks")).json()).data.tasks.filter(k=>(k.status=="CREATED"||k.status=="RUNNING")&&(k.page_train_type===t||String(k.id)===String(__remembered)));if(V.length==0){ElMessage.warning("\\u5F53\\u524D\\u9875\\u9762\\u6CA1\\u6709\\u6B63\\u5728\\u542F\\u52A8\\u6216\\u8FD0\\u884C\\u7684\\u8BAD\\u7EC3\\u4EFB\\u52A1");return}_=V.find(k=>String(k.id)===String(__remembered))||V[0]'
    content = _replace_span_once(
        content,
        'K=async()=>{let _=null;try{let V=',
        '}catch(g){',
        new_stop_prefix,
        "backend task stop",
    )

    content = _replace_once(
        content,
        'E=()=>{const _=x(),g=`${new Date().getTime()}.toml`;P(g,_)}',
        'E=async()=>{try{const R=await __requestEffective(T(),"/api/training/export"),H=R.sidecars&&R.sidecars.length>0,g=`${new Date().getTime()}${H?".dts.json":".toml"}`;P(g,H?(R.bundle||JSON.stringify({format:"dts-training-bundle-v1",train_type:t,toml:R.toml||"",sidecars:{}})):(R.toml||""))}catch(_){ElMessage.error(_.message||String(_))}}',
        "effective config export",
    )

    new_import = 'S=()=>{const _=document.createElement("input");_.type="file",_.accept=".toml,.json",_.onchange=m=>{const g=m.target.files[0],N=new FileReader;N.onload=async D=>{const V=D.target.result;try{let k=g.name.toLowerCase().endsWith(".json")?JSON.parse(V):TomlParse(V),U=await __trainingRequest("/api/training/rehydrate",k),B=U.data&&U.data.gui_state;if(!B||typeof B!=="object")throw new Error("导入结果缺少 gui_state");++__policyBootstrapGeneration.value,a.value=clone(B),ElMessage.success("\\u5BFC\\u5165\\u6210\\u529F"),await nextTick(),await __syncPolicyMode()}catch(k){console.log(k),ElMessage.error(k.message||"\\u5BFC\\u5165\\u5931\\u8D25")}},N.readAsText(g)},_.click()}'
    content = _replace_span_once(
        content,
        'S=()=>{const _=document.createElement("input");',
        ',$=_=>',
        new_import,
        "trainer TOML rehydrate",
    )

    content = _replace_once(
        content,
        '$=_=>{let m=findChangedDataBySchema(_,n.value);a.value==null?a.value=clone(m):a.value=Object.assign({},a.value,m),console.log(a.value)}',
        '$=async _=>{const Pm=__presetPolicyMode(_);if(Pm=="invalid-component"){ElMessage.error("Component preset 缺少完整 Parameter Policy");return}++__policyBootstrapGeneration.value,__policyBootstrapPending.value=!1;const Pc=Pm=="component";let m=Pc?clone(_):findChangedDataBySchema(_,n.value);a.value==null?a.value=clone(m):a.value=Object.assign({},a.value,m),a.value.optimization_mode=Pm,__runtimeReady.value=!Pc,__runtimeBlockers.value=[],await nextTick(),await __syncPolicyMode(),console.log(a.value)}',
        "Preset policy-mode transition",
    )

    content = _replace_once(
        content,
        'Y=(_,m)=>{a.value=clone(m.value),i.value=!1,ElMessage.success("\\u5DF2\\u5C06\\u5386\\u53F2\\u53C2\\u6570\\u5E94\\u7528\\u81F3\\u5F53\\u524D\\u53C2\\u6570")}',
        'Y=async(_,m)=>{++__policyBootstrapGeneration.value,__policyBootstrapPending.value=!1,a.value=clone(m.value),__runtimeReady.value=!__isComponentMode(),__runtimeBlockers.value=[],i.value=!1,ElMessage.success("\\u5DF2\\u5C06\\u5386\\u53F2\\u53C2\\u6570\\u5E94\\u7528\\u81F3\\u5F53\\u524D\\u53C2\\u6570"),await nextTick(),await __syncPolicyMode()}',
        "history policy-mode transition",
    )

    content = _replace_once(
        content,
        'q=_=>{ElMessageBox.alert(stringify(_),"\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}',
        'q=async _=>{try{let m=n.value(clone(_)),R=await __requestEffective(m);ElMessageBox.alert(R.toml||"","\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}catch(m){ElMessage.error(m.message||String(m))}}',
        "preset effective preview",
    )
    content = _replace_once(
        content,
        'Z=(_,m)=>{const g=stringify(parseParams(n.value(clone(m.value)),t));ElMessageBox.alert(g,"\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}',
        'Z=async(_,m)=>{try{const R=await __requestEffective(n.value(clone(m.value)));ElMessageBox.alert(R.toml||"","\\u9884\\u89C8",{confirmButtonText:"\\u786E\\u5B9A",customStyle:{whiteSpace:"pre-line"}})}catch(g){ElMessage.error(g.message||String(g))}}',
        "history effective preview",
    )

    content = _replace_once(
        content,
        'createVNode(g,{plain:"",class:"max-btn color-btn",type:"primary",onClick:O}',
        'createVNode(g,{plain:"",class:"max-btn color-btn",type:"primary",disabled:__startPending.value||__policyBootstrapPending.value||__isComponentMode()&&!__runtimeReady.value,onClick:O}',
        "start pending disable",
    )

    forbidden = (
        'm=parseParams(_,t),g=checkParams(m)',
        'O=async()=>{const _=parseParams(',
        'stringify(parseParams(n.value(clone(m.value)),t))',
        'a.value=Object.assign({},n.value(),B)',
        'T=()=>{let _=a.value;',
        '__policyBootstrapSnapshot=()=>{let R=clone(a.value||{});delete R.parameter_policy_profiles,delete R.parameter_policy_components,R.optimization_mode="standard";return R}',
        '__previewTimer=null',
        '__previewGeneration=0',
        '++__previewGeneration;',
        '__previewTimer=setTimeout',
        '__trainingRequest("/api/training/parameter-policy/bootstrap",T())',
    )
    for anchor in forbidden:
        if anchor in content:
            raise RuntimeError(f"Legacy/stale training callsite survived frontend patch: {anchor}")
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
    virtual_asset.__wrapped__ = original
    training_pages.virtual_asset = virtual_asset
