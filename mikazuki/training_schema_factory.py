"""Pure schema specialization helpers for backend-specific training pages.

The shipped frontend is an old prebuilt VuePress/Schemastery application. Its
form state is reused while navigating between training pages. Backend selector
fields therefore must not remain in the rendered schema: a stale value from the
previous page can make Schemastery reject the new page before the request ever
reaches the Python backend (for example ``expected flux but got anima``).

The helpers below use fixed discriminator values only while pruning conditional
unions. They then remove those discriminator fields entirely. The page's outer
``train_type`` is the routing source of truth, and the backend materializes any
trainer values (such as Flux ``model_type``) into the final effective config.
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

    ``fixed`` keys drive union pruning and are removed from the rendered form.
    ``drop_keys`` remove page-irrelevant controls without influencing branch
    selection. ``hidden_defaults`` are also removed from the form; the backend
    is responsible for applying those page-forced values to the effective
    trainer config. Keeping them as hidden ``Schema.const`` fields is unsafe in
    the pinned frontend because hidden defaults are not reliably materialized
    before validation and stale cross-page values win over defaults.
    """

    hidden_defaults = hidden_defaults or {}
    fixed_json = json.dumps(fixed, ensure_ascii=True, separators=(",", ":"))
    drop_json = json.dumps(list(drop_keys), ensure_ascii=True, separators=(",", ":"))
    hidden_json = json.dumps(hidden_defaults, ensure_ascii=True, separators=(",", ":"))

    template_expression = template.strip()
    if template_expression.endswith(";"):
        template_expression = template_expression[:-1].rstrip()

    return f"""(() => {{
    const __fixed = {fixed_json};
    const __dropKeys = new Set({drop_json});
    const __forcedBackendValues = {hidden_json};
    const __fixedKeys = Object.keys(__fixed);
    const __forcedKeys = new Set(Object.keys(__forcedBackendValues));
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
            for (const key of __forcedKeys) delete schema.dict[key];
            for (const key of Object.keys(schema.dict)) schema.dict[key] = __walk(schema.dict[key]);
            return schema;
        }}

        if ((schema.type === \"array\" || schema.type === \"dict\" || schema.type === \"transform\") && schema.inner) {{
            schema.inner = __walk(schema.inner);
        }}
        if (schema.type === \"tuple\" && schema.list) schema.list = schema.list.map(__walk);
        return schema;
    }};

    return __walk(__source);
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
        return _runtime_specialized_schema(
            template,
            fixed,
            drop_keys=("clip_l",),
            hidden_defaults={"apply_t5_attn_mask": True},
        )

    return _runtime_specialized_schema(template, fixed)
