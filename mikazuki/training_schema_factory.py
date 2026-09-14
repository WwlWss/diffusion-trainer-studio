"""Pure schema specialization helpers for backend-specific training pages.

The shipped frontend is an old prebuilt VuePress/Schemastery application.  Its
form renderer does not reliably materialize defaults from one sibling schema
node before resolving another sibling ``Schema.union``.  Therefore changing
only the visible backend selector to a disabled/default value is *not* enough:
all of the old conditional branches still see a missing discriminator and the
form can fall through to the wrong backend (or to an empty object).

The helpers below keep the legacy source templates for compatibility, but wrap
them in a small runtime structural specializer.  It prunes unions by the fixed
page discriminators and removes those discriminator fields from the rendered
tree.  Hidden fixed fields are then prepended so the submitted config still
contains the concrete backend keys expected by the Python API.
"""

from __future__ import annotations

import json
from typing import Any


def _runtime_specialized_schema(
    template: str,
    fixed: dict[str, Any],
    *,
    drop_keys: tuple[str, ...] = (),
    hidden_defaults: dict[str, Any] | None = None,
) -> str:
    """Return a Schemastery expression specialized to ``fixed`` values.

    ``fixed`` keys drive union pruning.  ``drop_keys`` remove page-irrelevant
    controls without influencing branch selection.  ``hidden_defaults`` are
    submitted to the backend but are also deliberately excluded from branch
    selection; Chroma's forced T5 attention-mask flag is the main example.
    """

    hidden_defaults = hidden_defaults or {}
    fixed_json = json.dumps(fixed, ensure_ascii=True, separators=(",", ":"))
    drop_json = json.dumps(list(drop_keys), ensure_ascii=True, separators=(",", ":"))
    hidden_json = json.dumps(hidden_defaults, ensure_ascii=True, separators=(",", ":"))

    # Repository .ts schema files are expression statements and normally end
    # with a semicolon.  They are embedded below inside parentheses, where that
    # statement terminator would produce invalid JavaScript: ``(expr;)``.
    # Remove only the final statement terminator; do not rewrite the template.
    template_expression = template.strip()
    if template_expression.endswith(";"):
        template_expression = template_expression[:-1].rstrip()

    # ``template_expression`` is a trusted repository schema expression.
    # Parenthesizing it lets the wrapper consume either Schema.intersect(...)
    # or any future single-expression schema without changing the source file.
    return f"""(() => {{
    const __fixed = {fixed_json};
    const __dropKeys = new Set({drop_json});
    const __hiddenDefaults = {hidden_json};
    const __fixedKeys = Object.keys(__fixed);
    const __hiddenKeys = new Set(Object.keys(__hiddenDefaults));
    const __source = ({template_expression});

    // 0 = this schema does not constrain the key, 1 = accepts the fixed value,
    // -1 = explicitly rejects it.
    const __fieldConstraint = (field, value) => {{
        if (!field) return 0;
        if (field.type === \"const\") return field.value === value ? 1 : -1;
        if (field.type === \"union\" && field.list && field.list.length && field.list.every((item) => item.type === \"const\")) {{
            return field.list.some((item) => item.value === value) ? 1 : -1;
        }}
        return 0;
    }};

    const __constraint = (schema, key) => {{
        if (!schema) return 0;
        if (schema.type === \"object\" && schema.dict) {{
            return __fieldConstraint(schema.dict[key], __fixed[key]);
        }}
        if (schema.type === \"intersect\" && schema.list) {{
            let accepted = false;
            for (const child of schema.list) {{
                const state = __constraint(child, key);
                if (state < 0) return -1;
                if (state > 0) accepted = true;
            }}
            return accepted ? 1 : 0;
        }}
        return 0;
    }};

    const __branchState = (schema) => {{
        let constrained = false;
        for (const key of __fixedKeys) {{
            const state = __constraint(schema, key);
            if (state < 0) return {{ constrained: true, matches: false }};
            if (state > 0) constrained = true;
        }}
        return {{ constrained, matches: true }};
    }};

    const __walk = (schema) => {{
        if (!schema) return schema;

        if (schema.type === \"union\" && schema.list) {{
            const states = schema.list.map(__branchState);
            if (states.some((state) => state.constrained)) {{
                // When a discriminator-specific branch matches, the historical
                // empty-object fallback must be discarded.  Keeping it would
                // let the legacy renderer choose the fallback before defaults
                // have been materialized.
                const matched = schema.list.filter((_, index) => states[index].constrained && states[index].matches);
                if (matched.length === 1) return __walk(matched[0]);
                if (matched.length > 1) {{
                    schema.list = matched.map(__walk);
                    return schema;
                }}

                const fallback = schema.list.filter((_, index) => !states[index].constrained);
                if (fallback.length === 1) return __walk(fallback[0]);
                schema.list = fallback.map(__walk);
                return schema;
            }}

            schema.list = schema.list.map(__walk);
            return schema;
        }}

        if (schema.type === \"intersect\" && schema.list) {{
            schema.list = schema.list.map(__walk);
            return schema;
        }}

        if (schema.type === \"object\" && schema.dict) {{
            for (const key of __fixedKeys) delete schema.dict[key];
            for (const key of __dropKeys) delete schema.dict[key];
            for (const key of __hiddenKeys) delete schema.dict[key];
            for (const key of Object.keys(schema.dict)) schema.dict[key] = __walk(schema.dict[key]);
            return schema;
        }}

        if ((schema.type === \"array\" || schema.type === \"dict\" || schema.type === \"transform\") && schema.inner) {{
            schema.inner = __walk(schema.inner);
        }}
        if (schema.type === \"tuple\" && schema.list) schema.list = schema.list.map(__walk);
        return schema;
    }};

    const __fixedFields = {{}};
    for (const [key, value] of Object.entries({{ ...__fixed, ...__hiddenDefaults }})) {{
        __fixedFields[key] = Schema.const(value).default(value).hidden();
    }}

    return Schema.intersect([
        Schema.object(__fixedFields),
        __walk(__source),
    ]);
}})()"""


def fixed_sd_schema(template: str, train_type: str) -> str:
    """Build a real fixed SD/SDXL page from the historical shared template."""
    if train_type not in {"sd-lora", "sdxl-lora", "sd-dreambooth", "sdxl-finetune"}:
        raise ValueError(f"Unsupported SD schema backend: {train_type}")
    return _runtime_specialized_schema(template, {"model_train_type": train_type})


def fixed_flux_family_schema(
    template: str,
    model_type: str,
    train_type: str,
    anima_mode: str | None = None,
) -> str:
    """Build one concrete Flux/Chroma/Anima schema with foreign branches pruned."""
    if model_type not in {"flux", "chroma", "anima"}:
        raise ValueError(f"Unsupported Flux-family model type: {model_type}")

    fixed: dict[str, Any] = {
        "model_type": model_type,
        "model_train_type": train_type,
    }
    if model_type == "anima":
        if anima_mode not in {"lora", "finetune"}:
            raise ValueError("Anima schema requires anima_mode=lora or finetune")
        fixed["anima_training_mode"] = anima_mode
    elif anima_mode is not None:
        raise ValueError("anima_mode is only valid for Anima schemas")

    if model_type == "chroma":
        # Chroma uses the shared Flux trainer but is T5-only.  CLIP-L is a real
        # Flux control and a Chroma no-op, while the backend requires the T5
        # attention mask enabled.  Keep the forced flag hidden so old presets
        # and the right-side parameter preview receive the correct value.
        return _runtime_specialized_schema(
            template,
            fixed,
            drop_keys=("clip_l",),
            hidden_defaults={"apply_t5_attn_mask": True},
        )

    return _runtime_specialized_schema(template, fixed)
