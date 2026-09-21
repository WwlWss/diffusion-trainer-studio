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
    row_description = label if not description else f"{label}: {description}"

    return f"""Schema.intersect([
        Schema.object({{
            train: Schema.boolean().default(false).description("Train this component; disabled components ignore stale LR/profile fields.")
        }}),
        Schema.union([
            Schema.intersect([
                Schema.object({{
                    train: Schema.const(true).required()
                }}),
                Schema.object({{
                    learning_rate: Schema.string().description("Component learning rate, for example 1e-4."),
                    optimizer_profile: Schema.string().description("Optimizer Profile name defined above."),
                    fallback_optimizer_profile: Schema.string().description("Optional fallback Profile for eligibility-gated optimizers such as Muon."),
                    fallback_learning_rate: Schema.string().description("Optional fallback learning rate; requires a fallback Profile.")
                }})
            ]),
            Schema.object({{}})
        ])
    ]).description({_js_string(row_description)})"""


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
        optimization_mode: Schema.union(["standard", "component"]).default("standard").description("Optimization ownership: Standard keeps historical DTS optimizer/LR controls; Component enables Parameter Policy.")
    }}).description("Parameter Policy"),
    Schema.union([
        Schema.intersect([
            Schema.object({{
                optimization_mode: Schema.const("component").required()
            }}),
            Schema.object({{
                parameter_policy_profiles: Schema.dict(Schema.object({{
                    type: Schema.union({optimizer_choice_expr}).default("AdamW").description("Supported optimizers are selectable; restricted/planned entries remain readable for imported policies but cannot be newly selected."),
                    args: Schema.dict(Schema.string()).description("Optimizer constructor args. Values accept literals such as 0.95, false, [1, 2], or (0.9, 0.95).")
                }})).description("Optimizer Profiles; dictionary key is the user-defined Profile name."),
                parameter_policy_components: Schema.object({{
{components}
                }}).description("Backend components. Train=false is authoritative; stale hidden LR/profile fields are ignored by the backend canonicalizer.")
            }}).description("Component-wise Parameter Policy")
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
