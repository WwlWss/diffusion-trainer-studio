"""Generate the public Schemastery editor for Parameter Policy Step 7B.

The editor is generated from the Step 7A host metadata contract.  This module
does not own optimizer capabilities or model-component definitions; it only
turns those registry-derived values into a Schemastery expression that can be
intersected with an existing training-page schema.
"""

from __future__ import annotations

import json

from mikazuki.parameter_policy_editor import parameter_policy_editor_metadata


PARAMETER_POLICY_EDITOR_MARKER = "DTS_PARAMETER_POLICY_EDITOR_V1"


def _js_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _component_row(component: dict) -> str:
    component_id = str(component["id"])
    label = str(component.get("label") or component_id)
    description = str(component.get("description") or "")
    translated = {
        "dit.self_attention": ("DiT 自注意力", "Anima DiT 自注意力参数。"),
        "dit.cross_attention": ("DiT 交叉注意力", "Anima DiT 交叉注意力参数。"),
        "dit.mlp": ("DiT MLP", "Anima DiT MLP 参数。"),
        "dit.modulation": ("DiT 调制层", "Anima AdaLN 调制参数。"),
        "dit.llm_adapter": ("LLM Adapter", "可选的 Anima LLM Adapter 桥接参数。"),
        "dit.base_other": ("DiT 其他参数", "其余 Anima DiT embedding、输出层和外围参数。"),
        "qwen3": ("Qwen3", "启用联合训练时的 Qwen3 文本编码器参数。"),
    }.get(component_id)
    if translated:
        label, description = translated
    row_description = label if not description else f"{label}：{description}"

    # Keep one real object instead of an intersect+union wrapper.  The pinned
    # Schemastery renderer otherwise repeats the same group heading.  Frozen
    # routes may keep stale LR/profile fields because the backend canonicalizer
    # already makes Train=false authoritative.
    return f"""Schema.object({{
        train: Schema.boolean().default(false).description("是否训练该组件；关闭后后端会忽略残留的学习率和 Profile。"),
        learning_rate: Schema.string().description("该组件的学习率，例如 1e-4。"),
        optimizer_profile: Schema.union(["main", "legacy_main", "muon", "fallback", Schema.string().description("自定义 Profile 名称")]).description("主 Optimizer Profile；常用名称可直接选择，也可切换到自定义名称。"),
        fallback_optimizer_profile: Schema.union(["fallback", "main", "legacy_main", Schema.string().description("自定义回退 Profile 名称")]).description("可选回退 Profile；常用名称可直接选择，也可使用自定义名称。"),
        fallback_learning_rate: Schema.string().description("可选回退学习率；设置回退 Profile 时使用。")
    }}).description({_js_string(row_description)}).collapse()"""

def parameter_policy_schema_fragment(train_type: str) -> str:
    """Return the Parameter Policy editor as one Schemastery expression."""

    metadata = parameter_policy_editor_metadata(train_type)
    optimizer_types = [row["type"] for row in metadata["optimizer_types"]]
    optimizer_capabilities = list(metadata["optimizer_capabilities"])
    if not optimizer_types:
        raise RuntimeError(
            f"Parameter Policy editor has no supported optimizer choices for {metadata['train_type']!r}."
        )
    if "AdamW" not in optimizer_types:
        raise RuntimeError("Parameter Policy editor requires AdamW as the safe default profile type.")

    optimizer_choices = []
    for row in optimizer_capabilities:
        option = f"Schema.const({_js_string(row['type'])})"
        if row["component_support"] != "supported":
            detail = row.get("restriction") or (
                f"Component support is {row['component_support']}."
            )
            option += (
                f".disabled().description({_js_string(row['component_support'].title() + ': ' + detail)})"
            )
        optimizer_choices.append(option)
    optimizer_choice_expr = "[" + ",".join(optimizer_choices) + "]"
    component_lines = []
    for component in metadata["components"]:
        component_lines.append(
            f"            {_js_string(component['id'])}: {_component_row(component)}"
        )
    components = ",\n".join(component_lines)

    return f"""Schema.intersect([
    Schema.object({{
        optimization_mode: Schema.union(["standard", "component"]).default("standard").description("优化模式：Standard 保持原有 DTS 优化器/学习率逻辑；Component 启用按组件 Parameter Policy。")
    }}).description("Parameter Policy"),
    Schema.union([
        Schema.intersect([
            Schema.object({{
                optimization_mode: Schema.const("component").required()
            }}),
            Schema.object({{
                parameter_policy_profiles: Schema.dict(Schema.object({{
                    type: Schema.union({optimizer_choice_expr}).default("AdamW").description("Supported optimizers are selectable; restricted/planned entries remain readable for imported policies but cannot be newly selected."),
                    args: Schema.dict(Schema.string()).description("优化器构造参数。左侧填写参数名，右侧只填写参数值；支持 0.95、false、[1, 2]、(0.9, 0.95) 等 literal。空白行会被忽略。")
                }})).description("Optimizer Profiles；字典 key 是自定义 Profile 名称，例如 main、muon、fallback。"),
                parameter_policy_components: Schema.object({{
{components}
                }}).description("后端组件；Train=false 为最终语义，关闭的组件会忽略残留的学习率/Profile。")
            }}).description("按组件 Parameter Policy")
        ]),
        Schema.object({{}})
    ])
])"""


def wrap_parameter_policy_editor(schema: str, train_type: str) -> str:
    """Intersect exactly one generated editor with an existing schema."""

    if PARAMETER_POLICY_EDITOR_MARKER in schema:
        return schema

    base = schema.strip()
    if base.endswith(";"):
        base = base[:-1].rstrip()
    fragment = parameter_policy_schema_fragment(train_type)

    return (
        f"/* {PARAMETER_POLICY_EDITOR_MARKER} */\n"
        "Schema.intersect([\n"
        f"    ({base}),\n"
        f"    ({fragment})\n"
        "])"
    )


__all__ = [
    "PARAMETER_POLICY_EDITOR_MARKER",
    "parameter_policy_schema_fragment",
    "wrap_parameter_policy_editor",
]
