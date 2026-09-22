"""Runtime page definitions for the prebuilt VuePress frontend.

The frontend submodule is a compiled distribution, so backend-specific training
pages are injected into its route/page-data maps at serve time. Keeping the page
contract here makes navigation, trainType/schema names and backend intent
reviewable without maintaining a fork of minified frontend sources.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


@dataclass(frozen=True)
class TrainingPage:
    key: str
    path: str
    aliases: tuple[str, ...]
    title: str
    train_type: str
    file_path: str
    description: str

    @property
    def slug(self) -> str:
        return self.path.strip("/").replace("/", "-").replace(".html", "")

    @property
    def data_asset(self) -> str:
        return f"training-{self.slug}.data.js"

    @property
    def content_asset(self) -> str:
        return f"training-{self.slug}.page.js"


VIRTUAL_TRAINING_PAGES = (
    TrainingPage(
        key="v-train-chroma-lora",
        path="/lora/chroma.html",
        aliases=("/lora/chroma", "/lora/chroma.md"),
        title="Chroma LoRA 训练",
        train_type="chroma-lora",
        file_path="lora/chroma.md",
        description="Chroma LoRA：固定使用 flux_train_network.py 的 Chroma/T5 路径，不显示 Flux CLIP-L 参数。",
    ),
    TrainingPage(
        key="v-train-anima-lora",
        path="/lora/anima.html",
        aliases=("/lora/anima", "/lora/anima.md"),
        title="Anima LoRA 训练",
        train_type="anima-lora",
        file_path="lora/anima.md",
        description="Anima LoRA：固定使用 sd-scripts/anima_train_network.py。",
    ),
    TrainingPage(
        key="v-train-sdxl-finetune",
        path="/finetune/sdxl.html",
        aliases=("/finetune/sdxl", "/finetune/sdxl.md"),
        title="SDXL 全参微调",
        train_type="sdxl-full",
        file_path="finetune/sdxl.md",
        description="SDXL 全参微调：页面使用独立 sdxl-full schema，后端固定 scripts/stable/sdxl_train.py，不是 DreamBooth trainer。",
    ),
    TrainingPage(
        key="v-train-flux-finetune",
        path="/finetune/flux.html",
        aliases=("/finetune/flux", "/finetune/flux.md"),
        title="Flux 全参微调",
        train_type="flux-finetune",
        file_path="finetune/flux.md",
        description="Flux 全参微调：固定使用 scripts/dev/flux_train.py。",
    ),
    TrainingPage(
        key="v-train-anima-finetune",
        path="/finetune/anima.html",
        aliases=("/finetune/anima", "/finetune/anima.md"),
        title="Anima 全参微调",
        train_type="anima-finetune",
        file_path="finetune/anima.md",
        description="Anima 全参微调：固定使用 sd-scripts/anima_train.py。",
    ),
)

# Chroma deliberately has no full-finetune page: this repository currently has
# no Chroma full trainer. SD1.5/SD2 DreamBooth keeps its historical /dreambooth/
# URL for compatibility, but appears under the renamed 全参微调 sidebar group.


def _json_string(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def page_data_js(page: TrainingPage) -> str:
    payload = {
        "key": page.key,
        "path": page.path,
        "title": page.title,
        "lang": "en-US",
        "frontmatter": {"example": True, "trainType": page.train_type},
        "excerpt": "",
        "headers": [],
        "filePathRelative": page.file_path,
    }
    return f"const e=JSON.parse({_json_string(_json_string(payload))});export{{e as data}};\n"


def page_content_js(page: TrainingPage) -> str:
    title = _json_string(" " + page.title)
    description = _json_string(page.description)
    anchor = re.sub(r"[^a-z0-9-]+", "-", page.slug.lower()).strip("-") or "training"
    anchor_json = _json_string("#" + anchor)
    anchor_id = _json_string(anchor)
    return (
        'import{_ as o,o as t,c as a,a as e,b as l}from"./app.547295de.js";'
        f'const r={{}},s=e("h1",{{id:{anchor_id},tabindex:"-1"}},'
        f'[e("a",{{class:"header-anchor",href:{anchor_json},"aria-hidden":"true"}},"#"),l({title})],-1),'
        f'c=e("p",null,{description},-1),n=[s,c];'
        'function d(u,i){return t(),a("div",null,n)}'
        'var h=o(r,[["render",d],["__file","runtime-training-page.vue"]]);export{h as default};\n'
    )


def virtual_asset(asset_name: str) -> str | None:
    for page in VIRTUAL_TRAINING_PAGES:
        if asset_name == page.data_asset:
            return page_data_js(page)
        if asset_name == page.content_asset:
            return page_content_js(page)
    return None


def _route_tuple(page: TrainingPage) -> str:
    aliases = ",".join(_json_string(alias) for alias in page.aliases)
    return f'[{_json_string(page.key)},{_json_string(page.path)},{{title:{_json_string(page.title)}}},[{aliases}]]'


def _data_loader(page: TrainingPage) -> str:
    return f'{_json_string(page.key)}:()=>wt(()=>import("./{page.data_asset}"),[]).then(({{data:e}})=>e)'


def _content_loader(page: TrainingPage) -> str:
    return f'{_json_string(page.key)}:Jt(()=>wt(()=>import("./{page.content_asset}"),[]))'


def patch_frontend_app_js(content: str) -> str:
    """Patch navigation and inject virtual pages into the compiled app bundle.

    Every replacement is anchored to the pinned frontend build and fails closed
    if upstream changes invalidate an assumption. Silent partial navigation
    patches are more dangerous than refusing startup after a frontend update.

    The pinned Schemastery renderer flattens children of Schema.intersect by
    forcing extra.foldable=false. That makes a child object's collapse metadata
    unreachable: the heading renders, but its fields stay permanently expanded.
    DTS keeps the flat intersect shape so trainer keys remain top-level TOML
    keys. Only children that explicitly declare meta.collapse may inherit their
    own foldability; conditional Schema.union branches must remain non-foldable
    or union.vue renders empty selector/collapse rows.
    """

    # Schemastery's pinned union renderer normally selects the first branch for
    # which the *entire* branch validates. That is hostile to editable forms:
    # one temporarily-invalid child can make an explicit discriminator such as
    # optimization_mode=component, caption_mode=multi, type=Muon, or train=true
    # fall through to a broad fallback branch and make the visible editor
    # disappear. Prefer an explicit top-level const discriminator when present;
    # if no branch has an exact discriminator match, retain upstream validation
    # and default behavior unchanged.
    discriminator_anchor = (
        'function Zx(e,t){try{return Hr(e)(t),!0}catch{return!1}}'
        'function md(e,t){'
    )
    discriminator_replacement = (
        'function Zx(e,t){try{return Hr(e)(t),!0}catch{return!1}}'
        'function __dtsUnionDiscriminator(e,t){'
        'if(!t||typeof t!="object"||Array.isArray(t))return!1;'
        'const n=[];'
        'function r(o){'
        'if(!o)return;'
        'if(o.type==="transform")return r(o.inner);'
        'if(o.type==="intersect"){for(const a of o.list||[])r(a);return}'
        'if(o.type!=="object")return;'
        'for(const[a,l]of Object.entries(o.dict||{})){'
        'let i=l;for(;i&&i.type==="transform";)i=i.inner;'
        'i&&i.type==="const"&&n.push([a,i.value])'
        '}'
        '}'
        'return r(e),n.length>0&&n.every(([o,a])=>'
        'Object.prototype.hasOwnProperty.call(t,o)&&t[o]===a)'
        '}'
        'function md(e,t){'
    )
    discriminator_count = content.count(discriminator_anchor)
    if discriminator_count != 1:
        raise RuntimeError(
            "Frontend Schemastery discriminator helper anchor expected once, "
            f"found {discriminator_count}"
        )
    content = content.replace(
        discriminator_anchor,
        discriminator_replacement,
        1,
    )

    union_input_anchor = (
        'const l=oo({input(d){a.value=null;let f=!0,h=0;'
        'for(;!a.value&&f&&++h<10;){'
    )
    union_input_replacement = (
        'const l=oo({input(d){a.value=null;'
        'const __dtsPinned=t.schema.list.find(v=>'
        '!v.meta.hidden&&__dtsUnionDiscriminator(v,d));'
        'if(__dtsPinned)a.value=__dtsPinned;'
        'let f=!a.value,h=0;for(;!a.value&&f&&++h<10;){'
    )
    union_input_count = content.count(union_input_anchor)
    if union_input_count != 1:
        raise RuntimeError(
            "Frontend Schemastery union input anchor expected once, "
            f"found {union_input_count}"
        )
    content = content.replace(
        union_input_anchor,
        union_input_replacement,
        1,
    )

    foldable_anchor = 'extra:{foldable:!1}'
    foldable_count = content.count(foldable_anchor)
    if foldable_count != 1:
        raise RuntimeError(
            "Frontend Schemastery foldability anchor expected once, "
            f"found {foldable_count}"
        )
    content = content.replace(
        foldable_anchor,
        # Only explicit child groups with meta.collapse should inherit their
        # own foldability. Keep conditional Schema.union branches non-foldable;
        # union.vue otherwise renders an empty branch selector/collapse row.
        'extra:{foldable:h.meta.collapse?void 0:!1}',
        1,
    )

    # The pinned renderer only shows a visible "expand" button while a group is
    # collapsed. Once expanded, collapsing again is hidden inside the ellipsis
    # menu. Keep the same control visible in both states so the interaction is
    # symmetric: collapsed -> 展开以编辑, expanded -> 收起.
    collapsed_control_anchor = (
        'e.collapsible?(x(),U(Pe,{key:1},[n.value?'
        '(x(),ce(u,{key:0,onClick:i[0]||(i[0]=h=>n.value=!1)},'
        '{default:G(()=>[Je(Ee(c(o)("expand")),1)]),_:1}))'
        ':ye("",!0)],64)):ye("",!0)'
    )
    collapsed_control_replacement = (
        'e.collapsible?(x(),U(Pe,{key:1},[n.value?'
        '(x(),ce(u,{key:0,onClick:i[0]||(i[0]=h=>n.value=!1)},'
        '{default:G(()=>[Je(Ee(c(o)("expand")),1)]),_:1}))'
        ':(x(),ce(u,{key:1,onClick:h=>n.value=!0},'
        '{default:G(()=>[Je(Ee(c(o)("collapse")),1)]),_:1}))],64)):ye("",!0)'
    )
    control_count = content.count(collapsed_control_anchor)
    if control_count != 1:
        raise RuntimeError(
            "Frontend Schemastery visible collapse-control anchor expected once, "
            f"found {control_count}"
        )
    content = content.replace(
        collapsed_control_anchor,
        collapsed_control_replacement,
        1,
    )

    zh_collapse_anchor = 'collapse:"\\u6298\\u53E0\\u5B50\\u9879"'
    zh_collapse_count = content.count(zh_collapse_anchor)
    if zh_collapse_count != 1:
        raise RuntimeError(
            "Frontend Schemastery Chinese collapse label anchor expected once, "
            f"found {zh_collapse_count}"
        )
    content = content.replace(
        zh_collapse_anchor,
        'collapse:"\\u6536\\u8D77"',
        1,
    )
    old_lora_children = (
        '{"text":"LoRA\\u8BAD\\u7EC3","link":"/lora/index.md","collapsible":false,"children":['
        '{"text":"\\u65B0\\u624B\\uFF08SD1.5\\uFF09","link":"/lora/basic.md"},'
        '{"text":"\\u4E13\\u5BB6","link":"/lora/master.md"},'
        '{"text":"Flux","link":"/lora/flux.md"},'
        '{"text":"SD3.5","link":"/lora/sd3.md"},'
        '{"text":"\\u5DE5\\u5177","link":"/lora/tools.md"},'
        '{"text":"\\u53C2\\u6570\\u8BE6\\u89E3","link":"/lora/params.md"}]},'
        '{"text":"Dreambooth \\u8BAD\\u7EC3","link":"/dreambooth/index.md"}'
    )
    new_lora_children = (
        '{"text":"LoRA\\u8BAD\\u7EC3","collapsible":true,"children":['
        '{"text":"LoRA \\u6982\\u89C8","link":"/lora/index.md"},'
        '{"text":"\\u65B0\\u624B\\uFF08SD1.5\\uFF09","link":"/lora/basic.md"},'
        '{"text":"SD1.5 / SD2 LoRA","link":"/lora/master.md"},'
        '{"text":"SDXL LoRA","link":"/lora/sdxl.md"},'
        '{"text":"Flux LoRA","link":"/lora/flux.md"},'
        '{"text":"Chroma LoRA","link":"/lora/chroma.md"},'
        '{"text":"Anima LoRA","link":"/lora/anima.md"},'
        '{"text":"SD3.5","link":"/lora/sd3.md"},'
        '{"text":"\\u5DE5\\u5177","link":"/lora/tools.md"},'
        '{"text":"\\u53C2\\u6570\\u8BE6\\u89E3","link":"/lora/params.md"}]},'
        '{"text":"\\u5168\\u53C2\\u5FAE\\u8C03","collapsible":true,"children":['
        '{"text":"SD1.5 / SD2 DreamBooth","link":"/dreambooth/index.md"},'
        '{"text":"SDXL Finetune","link":"/finetune/sdxl.md"},'
        '{"text":"Flux Finetune","link":"/finetune/flux.md"},'
        '{"text":"Anima Finetune","link":"/finetune/anima.md"}]}'
    )
    if old_lora_children not in content:
        raise RuntimeError("Frontend sidebar anchor not found; pinned frontend layout changed")
    content = content.replace(old_lora_children, new_lora_children, 1)

    content = content.replace("Dreambooth \\u8BAD\\u7EC3 \\u4E13\\u5BB6\\u6A21\\u5F0F", "SD1.5 / SD2 DreamBooth \\u5168\\u53C2\\u5FAE\\u8C03")
    content = content.replace("Flux LoRA \\u8BAD\\u7EC3 \\u4E13\\u5BB6\\u6A21\\u5F0F", "Flux LoRA \\u8BAD\\u7EC3")
    content = content.replace("LoRA \\u8BAD\\u7EC3 \\u4E13\\u5BB6\\u6A21\\u5F0F", "SD1.5 / SD2 LoRA \\u8BAD\\u7EC3")

    data_anchor = '"v-3706649a":()=>wt(()=>import("./404.html.686caba0.js"),[]).then(({data:e})=>e)'
    if data_anchor not in content:
        raise RuntimeError("Frontend page-data loader anchor not found")
    data_insert = ",".join(_data_loader(page) for page in VIRTUAL_TRAINING_PAGES)
    content = content.replace(data_anchor, data_insert + "," + data_anchor, 1)

    content_anchor = '"v-3706649a":Jt(()=>wt(()=>import("./404.html.7a24a487.js"),[]))'
    if content_anchor not in content:
        raise RuntimeError("Frontend content loader anchor not found")
    content_insert = ",".join(_content_loader(page) for page in VIRTUAL_TRAINING_PAGES)
    content = content.replace(content_anchor, content_insert + "," + content_anchor, 1)

    route_anchor = '["v-3706649a","/404.html",{title:""},["/404"]]'
    if route_anchor not in content:
        raise RuntimeError("Frontend route-table anchor not found")
    route_insert = ",".join(_route_tuple(page) for page in VIRTUAL_TRAINING_PAGES)
    content = content.replace(route_anchor, route_insert + "," + route_anchor, 1)
    return content


def virtual_page_paths() -> set[str]:
    paths = set()
    for page in VIRTUAL_TRAINING_PAGES:
        paths.add(page.path)
        paths.update(page.aliases)
    return paths
