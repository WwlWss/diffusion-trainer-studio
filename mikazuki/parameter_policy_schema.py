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

    # Keep conditional visibility without attaching the same description to the
    # outer intersect.  In the pinned renderer that avoids duplicate headings,
    # while Train=false still hides stale LR/profile controls.
    return f"""Schema.intersect([
        Schema.object({{
            train: Schema.boolean().default(false).description({_js_string(row_description + " 是否训练该组件；关闭后后端会忽略残留的学习率和 Profile。")})
        }}),
        Schema.union([
            Schema.intersect([
                Schema.object({{
                    train: Schema.const(true).required()
                }}),
                Schema.object({{
                    learning_rate: Schema.string().description("该组件的学习率，例如 1e-4。"),
                    optimizer_profile: Schema.string().description("主 Optimizer Profile；下拉选项动态来自当前 Optimizer Profiles，也可直接输入自定义名称。"),
                    fallback_optimizer_profile: Schema.string().description("可选回退 Profile；下拉选项动态来自当前 Optimizer Profiles，也可直接输入自定义名称。"),
                    fallback_learning_rate: Schema.string().description("可选回退学习率；设置回退 Profile 时使用。")
                }})
            ]),
            Schema.object({{}})
        ])
    ])"""

def _optimizer_profile_branch(row: dict) -> str:
    """Return one self-contained optimizer branch for the legacy union renderer.

    The pinned frontend replaces the whole union model when the user changes
    branches.  Every branch therefore owns and defaults its discriminator
    (`type`) so switching optimizer can never erase the profile type.
    """

    optimizer_type = str(row["type"])
    type_literal = _js_string(optimizer_type)
    if optimizer_type == "Muon":
        args_schema = """Schema.object({
            momentum: Schema.number().min(0).description("Muon 动量，例如 0.95；必须小于 1，最终范围由后端严格校验。"),
            weight_decay: Schema.number().min(0).description("Muon 权重衰减，例如 0.01。"),
            weight_decouple: Schema.boolean().description("使用 decoupled weight decay。"),
            nesterov: Schema.boolean().description("启用 Nesterov momentum。"),
            ns_steps: Schema.number().min(1).step(1).description("Newton-Schulz 迭代次数，例如 5。"),
            ns_coeffs: Schema.union(["original", "quintic", "polar_express", "polar_express_safer"]).description("Newton-Schulz 系数预设。"),
            use_adjusted_lr: Schema.boolean().description("按 Muon 实现启用 adjusted LR。")
        }).description("Muon 参数；这些是 DTS 当前明确支持的 Muon 参数，直接填写即可，无需手工添加 key/value。")"""
    else:
        args_schema = (
            'Schema.dict(Schema.string()).description("优化器构造参数。左侧填写参数名，右侧只填写参数值；'
            '支持 0.95、false、[1, 2]、(0.9, 0.95) 等 literal。空白行会被忽略。")'
        )

    description = optimizer_type
    if row["component_support"] != "supported":
        detail = row.get("restriction") or (
            f"Component support is {row['component_support']}."
        )
        description = f"{optimizer_type} — {row['component_support'].title()}: {detail}"

    branch = f"""Schema.object({{
        type: Schema.const({type_literal}).required(),
        args: {args_schema}
    }}).default({{ type: {type_literal}, args: {{}} }}).description({_js_string(description)})"""
    if row["component_support"] != "supported":
        branch += ".disabled()"
    return branch

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

    optimizer_profile_branches = [
        _optimizer_profile_branch(row)
        for row in optimizer_capabilities
    ]
    optimizer_profile_expr = "[\n                    " + ",\n                    ".join(optimizer_profile_branches) + "\n                ]"
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
                parameter_policy_profiles: Schema.dict(
                    Schema.union({optimizer_profile_expr})
                ).description("Optimizer Profiles；每个分支自己保存 optimizer type，切换类型不会丢失 Profile；字典 key 是自定义 Profile 名称。"),
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
