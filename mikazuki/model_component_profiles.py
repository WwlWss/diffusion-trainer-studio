from __future__ import annotations

"""Pure model-component and training-target contracts for Parameter Policy.

This module is intentionally host-side and dependency-light. It does not import
PyTorch or trainer model classes, construct/load models, mutate parameters, or
perform any optimizer work. Classification consumes metadata prepared by the
Step 3 parameter scanner and explicit LoRA original-target metadata supplied by
Step 4.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol


ParameterClass = Literal[
    "matrix_weight",
    "bias",
    "norm_weight",
    "embedding_weight",
    "conv_weight",
    "other",
]

EligibilityDecision = tuple[bool, str]


class AdapterTargetLike(Protocol):
    root: str
    module_path: str
    module_type: str


class ParameterAliasLike(Protocol):
    root: str
    module_path: str
    module_type: str
    module_class: str
    ancestor_module_types: tuple[str, ...]
    ancestor_module_classes: tuple[str, ...]
    parameter_role: str
    parameter_class: ParameterClass
    adapter_target: AdapterTargetLike | None


@dataclass(frozen=True)
class ComponentDefinition:
    component_id: str
    display_name: str
    description: str
    roots: frozenset[str]


@dataclass(frozen=True)
class TrainingTargetProfile:
    train_type: str
    available_components: frozenset[str]
    unavailable_reasons: Mapping[str, str]

    def is_available(self, component_id: str) -> bool:
        return component_id in self.available_components


@dataclass(frozen=True)
class ModelComponentProfile:
    train_type: str
    components: Mapping[str, ComponentDefinition]
    classify_alias: Callable[[ParameterAliasLike], str | None]
    eligibility_check: Callable[[str, ParameterAliasLike, str], EligibilityDecision]


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_NORMALIZATION_CLASSES = {
    "LayerNorm",
    "GroupNorm",
    "RMSNorm",
    "QKNorm",
    "LLMAdapterRMSNorm",
}


def _bool(value: object, *, field: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"{field} must be boolean-like 0/1, got {value!r}.")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}.")


def _attr(obj: object, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _path(alias: ParameterAliasLike) -> str:
    return str(_attr(alias, "module_path", "") or "").strip(".")


def _root(alias: ParameterAliasLike) -> str:
    return str(_attr(alias, "root", "") or "")


def _parameter_class(alias: ParameterAliasLike) -> str:
    return str(_attr(alias, "parameter_class", "other") or "other")


def _class_names(alias: ParameterAliasLike) -> tuple[str, ...]:
    ancestors = tuple(str(item) for item in (_attr(alias, "ancestor_module_classes", ()) or ()))
    current = str(_attr(alias, "module_class", "") or "")
    return ancestors + ((current,) if current else ())


def _has_class(alias: ParameterAliasLike, *names: str) -> bool:
    classes = set(_class_names(alias))
    return any(name in classes for name in names)


def _is_norm_alias(alias: ParameterAliasLike) -> bool:
    if _parameter_class(alias) == "norm_weight":
        return True
    return any(name in _NORMALIZATION_CLASSES or name.endswith("Norm") for name in _class_names(alias))


def _adapter_target(alias: ParameterAliasLike) -> AdapterTargetLike | None:
    return _attr(alias, "adapter_target", None)


def _target_root(alias: ParameterAliasLike) -> str:
    target = _adapter_target(alias)
    return str(_attr(target, "root", "") or "") if target is not None else ""


def _target_path(alias: ParameterAliasLike) -> str:
    target = _adapter_target(alias)
    return str(_attr(target, "module_path", "") or "").strip(".") if target is not None else ""


def _target_type(alias: ParameterAliasLike) -> str:
    target = _adapter_target(alias)
    if target is None:
        return ""
    return str(_attr(target, "module_type", "") or "")


def _short_type(type_name: str) -> str:
    return type_name.rsplit(".", 1)[-1]


def _network_args(config: Mapping[str, Any]) -> dict[str, str | None]:
    raw = config.get("network_args")
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError("network_args must be a string or list of strings in effective config.")

    parsed: dict[str, str | None] = {}
    for raw_item in items:
        item = str(raw_item).strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key.strip().lower()] = value.strip()
        else:
            parsed[item.lower()] = None
    return parsed


def _network_bool(config: Mapping[str, Any], key: str) -> bool:
    args = _network_args(config)
    if key.lower() not in args:
        return False
    value = args[key.lower()]
    return True if value is None else _bool(value, field=f"network_args[{key}]")


def _components(*items: tuple[str, str, str, tuple[str, ...]]) -> Mapping[str, ComponentDefinition]:
    result: dict[str, ComponentDefinition] = {}
    for component_id, display_name, description, roots in items:
        result[component_id] = ComponentDefinition(
            component_id=component_id,
            display_name=display_name,
            description=description,
            roots=frozenset(roots),
        )
    return MappingProxyType(result)


def _target_profile(
    profile: ModelComponentProfile,
    available: set[str] | frozenset[str],
    *,
    reasons: Mapping[str, str] | None = None,
) -> TrainingTargetProfile:
    known = set(profile.components)
    available_set = set(available)
    unknown = available_set.difference(known)
    if unknown:
        raise ValueError(
            f"Training target for {profile.train_type} referenced unknown component(s): "
            + ", ".join(sorted(unknown))
        )
    unavailable: dict[str, str] = {}
    supplied = dict(reasons or {})
    for component_id in sorted(known.difference(available_set)):
        unavailable[component_id] = supplied.get(
            component_id,
            "Disabled by the current trainer target/effective configuration.",
        )
    return TrainingTargetProfile(
        train_type=profile.train_type,
        available_components=frozenset(available_set),
        unavailable_reasons=MappingProxyType(unavailable),
    )


def _sd_full_classifier(alias: ParameterAliasLike, *, sdxl: bool) -> str | None:
    root = _root(alias)
    if root == "unet":
        if _has_class(alias, "Transformer2DModel", "BasicTransformerBlock"):
            return "unet.transformer"
        if _has_class(alias, "ResnetBlock2D", "Downsample2D", "Upsample2D") or _parameter_class(alias) == "conv_weight":
            return "unet.conv_resnet"
        if _is_norm_alias(alias) or _parameter_class(alias) == "bias":
            return "unet.norm_bias_other"
        return "unet.base_other"
    if sdxl and root == "text_encoder_1":
        return "text_encoder_1"
    if sdxl and root == "text_encoder_2":
        return "text_encoder_2"
    if not sdxl and root == "text_encoder":
        return "text_encoder"
    return None


def _sd_dreambooth_classifier(alias: ParameterAliasLike) -> str | None:
    return _sd_full_classifier(alias, sdxl=False)


def _sdxl_full_classifier(alias: ParameterAliasLike) -> str | None:
    return _sd_full_classifier(alias, sdxl=True)


def _sd_lora_classifier(alias: ParameterAliasLike, *, sdxl: bool) -> str | None:
    target = _adapter_target(alias)
    if target is None:
        return None
    root = _target_root(alias)
    path = _target_path(alias).lower()
    type_name = _short_type(_target_type(alias))

    if root == "unet":
        dotted = f".{path}."
        if ".attn1." in dotted or ".attn2." in dotted or ".attention." in dotted:
            return "unet.attention.adapter"
        if ".ff." in dotted or ".feed_forward." in dotted:
            return "unet.feed_forward.adapter"
        if type_name == "Conv2d":
            return "unet.conv.adapter"
        return "unet.other.adapter"

    if not sdxl and root == "text_encoder":
        return "text_encoder.adapter"
    if sdxl and root == "text_encoder_1":
        return "text_encoder_1.adapter"
    if sdxl and root == "text_encoder_2":
        return "text_encoder_2.adapter"
    return None


def _sd_lora_classifier_v1(alias: ParameterAliasLike) -> str | None:
    return _sd_lora_classifier(alias, sdxl=False)


def _sdxl_lora_classifier(alias: ParameterAliasLike) -> str | None:
    return _sd_lora_classifier(alias, sdxl=True)


def _flux_is_mod_or_norm(path: str, alias: ParameterAliasLike) -> bool:
    dotted = f".{path.lower()}."
    if any(token in dotted for token in (".img_mod.", ".txt_mod.", ".modulation.", ".adaln_modulation.")):
        return True
    return _is_norm_alias(alias) or _has_class(alias, "Modulation")


def _flux_full_classifier(alias: ParameterAliasLike) -> str | None:
    if _root(alias) not in {"transformer", "flux"}:
        return None
    path = _path(alias)
    lower = path.lower()

    if _flux_is_mod_or_norm(path, alias):
        return "transformer.modulation_norm_other"
    if lower.startswith("double_blocks."):
        return "transformer.double_stream"
    if lower.startswith("single_blocks."):
        return "transformer.single_stream"
    if lower == "img_in" or lower.startswith("img_in.") or lower == "txt_in" or lower.startswith("txt_in."):
        return "transformer.input_conditioning"
    if any(lower == prefix or lower.startswith(prefix + ".") for prefix in ("time_in", "vector_in", "guidance_in")):
        return "transformer.input_conditioning"
    if lower == "final_layer" or lower.startswith("final_layer."):
        return "transformer.final"
    return "transformer.other"


def _flux_lora_classifier(alias: ParameterAliasLike, *, chroma: bool) -> str | None:
    target = _adapter_target(alias)
    if target is None:
        return None
    root = _target_root(alias)
    path = _target_path(alias).lower()

    if root in {"transformer", "flux"}:
        if path.startswith("double_blocks."):
            return "transformer.double_stream.adapter"
        if path.startswith("single_blocks."):
            return "transformer.single_stream.adapter"
        if any(path == prefix or path.startswith(prefix + ".") for prefix in ("img_in", "time_in", "vector_in", "guidance_in", "txt_in")):
            return "transformer.input_conditioning.adapter"
        return None
    if not chroma and root in {"clip_l", "text_encoder_1"}:
        return "clip_l.adapter"
    if root in {"t5xxl", "text_encoder_2", "text_encoder_3"}:
        return "t5xxl.adapter"
    return None


def _flux_lora_classifier_v1(alias: ParameterAliasLike) -> str | None:
    return _flux_lora_classifier(alias, chroma=False)


def _chroma_lora_classifier(alias: ParameterAliasLike) -> str | None:
    return _flux_lora_classifier(alias, chroma=True)


def _anima_full_classifier(alias: ParameterAliasLike) -> str | None:
    root = _root(alias)
    if root == "qwen3":
        return "qwen3"
    if root not in {"dit", "anima"}:
        return None

    path = f".{_path(alias).lower()}."
    if ".llm_adapter." in path:
        return "dit.llm_adapter"
    if ".adaln_modulation" in path:
        return "dit.modulation"
    if ".self_attn." in path:
        return "dit.self_attention"
    if ".cross_attn." in path:
        return "dit.cross_attention"
    if ".mlp." in path:
        return "dit.mlp"
    return "dit.base_other"


def _anima_lora_classifier(alias: ParameterAliasLike) -> str | None:
    target = _adapter_target(alias)
    if target is None:
        return None
    root = _target_root(alias)
    path = f".{_target_path(alias).lower()}."

    if root in {"qwen3", "text_encoder"}:
        return "qwen3.adapter"
    if root not in {"dit", "anima", "llm_adapter"}:
        return None
    if root == "llm_adapter" or ".llm_adapter." in path:
        return "llm_adapter.adapter"
    if ".adaln_modulation" in path or "_modulation." in path:
        return "dit.modulation.adapter"
    if ".self_attn." in path:
        return "dit.self_attention.adapter"
    if ".cross_attn." in path:
        return "dit.cross_attention.adapter"
    if ".mlp." in path:
        return "dit.mlp.adapter"
    return "dit.other.adapter"


def _sd3_lora_classifier(alias: ParameterAliasLike) -> str | None:
    target = _adapter_target(alias)
    if target is None:
        return None
    root = _target_root(alias)
    path = f".{_target_path(alias).lower()}."

    if root in {"mmdit", "transformer"}:
        if ".attn." in path or ".attn2." in path:
            return "mmdit.attention.adapter"
        if ".mlp." in path:
            return "mmdit.mlp.adapter"
        if ".adaln_modulation." in path or ".norm" in path:
            return "mmdit.modulation_norm.adapter"
        return "mmdit.other.adapter"
    if root in {"clip_l", "text_encoder_1"}:
        return "clip_l.adapter"
    if root in {"clip_g", "text_encoder_2"}:
        return "clip_g.adapter"
    if root in {"t5xxl", "text_encoder_3"}:
        return "t5xxl.adapter"
    return None


def _matrix(alias: ParameterAliasLike) -> bool:
    return _parameter_class(alias) == "matrix_weight"


def _text_hidden_path(path: str) -> bool:
    lower = f".{path.lower()}."
    if ".encoder.layers." in lower and (".self_attn." in lower or ".mlp." in lower):
        return True
    if ".layers." in lower and (".self_attn." in lower or ".mlp." in lower):
        return True
    if ".block." in lower and ".layer." in lower:
        return True
    return False


def _sd_transformer_hidden(alias: ParameterAliasLike) -> bool:
    return _matrix(alias) and _has_class(alias, "BasicTransformerBlock") and not _is_norm_alias(alias)


def _flux_hidden_path(path: str) -> bool:
    lower = f".{path.lower()}."
    if any(token in lower for token in (".img_mod.", ".txt_mod.", ".modulation.", ".norm.")):
        return False
    return lower.startswith(".double_blocks.") or lower.startswith(".single_blocks.")


def _anima_llm_hidden_path(path: str) -> bool:
    lower = f".{path.lower()}."
    if ".llm_adapter.blocks." not in lower:
        return False
    if ".norm" in lower:
        return False
    return any(token in lower for token in (".self_attn.", ".cross_attn.", ".mlp."))


def _target_linear_hidden(alias: ParameterAliasLike, *, family: str) -> bool:
    target = _adapter_target(alias)
    if target is None or not _matrix(alias):
        return False
    path = _target_path(alias)
    target_type = _short_type(_target_type(alias))
    if target_type not in {"Linear", "Conv1D"}:
        return False

    if family == "sd":
        lower = f".{path.lower()}."
        return any(token in lower for token in (".attn1.", ".attn2.", ".ff."))
    if family == "flux":
        return _flux_hidden_path(path)
    if family == "anima":
        lower = f".{path.lower()}."
        if ".adaln_modulation" in lower:
            return False
        return any(token in lower for token in (".self_attn.", ".cross_attn.", ".mlp.")) or _anima_llm_hidden_path(path)
    if family == "sd3":
        lower = f".{path.lower()}."
        return (".attn." in lower or ".attn2." in lower or ".mlp." in lower) and ".norm" not in lower and ".adaln_modulation." not in lower
    if family == "text":
        return _text_hidden_path(path)
    return False


def _eligibility_for_profile(
    family: str,
    policy: str,
    alias: ParameterAliasLike,
    component_id: str,
) -> EligibilityDecision:
    if policy != "model_hidden_2d_weight":
        return False, f"Unknown parameter eligibility policy {policy!r}."
    if not _matrix(alias):
        return False, "Parameter is not a 2D matrix weight."

    if ".adapter" in component_id:
        if component_id in {
            "text_encoder.adapter",
            "text_encoder_1.adapter",
            "text_encoder_2.adapter",
            "clip_l.adapter",
            "clip_g.adapter",
            "t5xxl.adapter",
            "qwen3.adapter",
        }:
            ok = _target_linear_hidden(alias, family="text")
        else:
            ok = _target_linear_hidden(alias, family=family)
        return (True, "Original LoRA target is an approved hidden linear module.") if ok else (
            False,
            "Original LoRA target is not an approved hidden linear module.",
        )

    if component_id == "unet.transformer":
        ok = _sd_transformer_hidden(alias)
    elif component_id in {"transformer.double_stream", "transformer.single_stream"}:
        ok = _flux_hidden_path(_path(alias))
    elif component_id in {"dit.self_attention", "dit.cross_attention", "dit.mlp"}:
        ok = True
    elif component_id == "dit.llm_adapter":
        ok = _anima_llm_hidden_path(_path(alias))
    elif component_id in {"qwen3", "text_encoder", "text_encoder_1", "text_encoder_2"}:
        ok = _text_hidden_path(_path(alias))
    else:
        ok = False

    return (True, "Model profile approved this hidden 2D weight.") if ok else (
        False,
        "Model profile does not authorize this matrix as a hidden-layer Muon weight.",
    )


def _eligibility_sd(policy: str, alias: ParameterAliasLike, component_id: str) -> EligibilityDecision:
    return _eligibility_for_profile("sd", policy, alias, component_id)


def _eligibility_flux(policy: str, alias: ParameterAliasLike, component_id: str) -> EligibilityDecision:
    return _eligibility_for_profile("flux", policy, alias, component_id)


def _eligibility_anima(policy: str, alias: ParameterAliasLike, component_id: str) -> EligibilityDecision:
    return _eligibility_for_profile("anima", policy, alias, component_id)


def _eligibility_sd3(policy: str, alias: ParameterAliasLike, component_id: str) -> EligibilityDecision:
    return _eligibility_for_profile("sd3", policy, alias, component_id)


_SD_FULL_COMPONENTS = _components(
    ("unet.transformer", "U-Net Transformer", "Attention/FFN transformer subtree inside the diffusion U-Net.", ("unet",)),
    ("unet.conv_resnet", "U-Net Conv / ResNet", "Convolution, resnet and sampling blocks outside the transformer subtree.", ("unet",)),
    ("unet.norm_bias_other", "U-Net Norm / Bias", "Outer normalization and bias parameters not owned by transformer/resnet components.", ("unet",)),
    ("unet.base_other", "U-Net Other", "Remaining U-Net embeddings/projections and uncategorized outer parameters.", ("unet",)),
)

_SD_LORA_COMPONENTS = _components(
    ("unet.attention.adapter", "U-Net Attention Adapters", "LoRA modules attached to U-Net attention projections.", ("unet",)),
    ("unet.feed_forward.adapter", "U-Net FFN Adapters", "LoRA modules attached to U-Net feed-forward projections.", ("unet",)),
    ("unet.conv.adapter", "U-Net Conv Adapters", "LoRA modules attached to U-Net convolutions.", ("unet",)),
    ("unet.other.adapter", "U-Net Other Adapters", "Other U-Net LoRA targets that are not attention/FFN/conv.", ("unet",)),
)

_FLUX_TRANSFORMER_COMPONENTS = _components(
    ("transformer.double_stream", "Double Stream", "Flux double-stream attention/MLP parameters.", ("transformer", "flux")),
    ("transformer.single_stream", "Single Stream", "Flux fused single-stream attention/MLP parameters.", ("transformer", "flux")),
    ("transformer.modulation_norm_other", "Modulation / Norm", "Flux modulation and normalization parameters.", ("transformer", "flux")),
    ("transformer.input_conditioning", "Input / Conditioning", "Flux image/text/time/vector/guidance input projections.", ("transformer", "flux")),
    ("transformer.final", "Final Layer", "Flux final output layer.", ("transformer", "flux")),
    ("transformer.other", "Transformer Other", "Remaining Flux transformer parameters.", ("transformer", "flux")),
)

_FLUX_LORA_TRANSFORMER_COMPONENTS = _components(
    ("transformer.double_stream.adapter", "Double Stream Adapters", "LoRA attached to Flux double-stream modules.", ("transformer", "flux")),
    ("transformer.single_stream.adapter", "Single Stream Adapters", "LoRA attached to Flux single-stream modules.", ("transformer", "flux")),
    ("transformer.input_conditioning.adapter", "Input / Conditioning Adapters", "LoRA attached to Flux input/conditioning projections.", ("transformer", "flux")),
)

_ANIMA_FULL_COMPONENTS = _components(
    ("dit.self_attention", "DiT Self Attention", "Anima DiT self-attention parameters.", ("dit", "anima")),
    ("dit.cross_attention", "DiT Cross Attention", "Anima DiT cross-attention parameters.", ("dit", "anima")),
    ("dit.mlp", "DiT MLP", "Anima DiT MLP parameters.", ("dit", "anima")),
    ("dit.modulation", "DiT Modulation", "Anima AdaLN modulation parameters.", ("dit", "anima")),
    ("dit.llm_adapter", "LLM Adapter", "Optional Anima LLM Adapter bridge parameters.", ("dit", "anima")),
    ("dit.base_other", "DiT Other", "Remaining Anima DiT embedding/final/outer parameters.", ("dit", "anima")),
    ("qwen3", "Qwen3", "Qwen3 text-encoder parameters when joint training is enabled.", ("qwen3",)),
)

_ANIMA_LORA_COMPONENTS = _components(
    ("dit.self_attention.adapter", "DiT Self Attention Adapters", "Anima LoRA targeting DiT self-attention.", ("dit", "anima")),
    ("dit.cross_attention.adapter", "DiT Cross Attention Adapters", "Anima LoRA targeting DiT cross-attention.", ("dit", "anima")),
    ("dit.mlp.adapter", "DiT MLP Adapters", "Anima LoRA targeting DiT MLP modules.", ("dit", "anima")),
    ("dit.modulation.adapter", "DiT Modulation Adapters", "Anima LoRA targeting AdaLN modulation modules.", ("dit", "anima")),
    ("dit.other.adapter", "DiT Other Adapters", "Other Anima DiT LoRA targets.", ("dit", "anima")),
    ("llm_adapter.adapter", "LLM Adapter LoRA", "LoRA attached to the optional Anima LLM Adapter.", ("llm_adapter", "dit", "anima")),
    ("qwen3.adapter", "Qwen3 Adapters", "LoRA attached to Qwen3 attention/MLP modules.", ("qwen3",)),
)

_SD3_LORA_COMPONENTS = _components(
    ("mmdit.attention.adapter", "MMDiT Attention Adapters", "SD3 LoRA targeting MMDiT attention projections.", ("mmdit", "transformer")),
    ("mmdit.mlp.adapter", "MMDiT MLP Adapters", "SD3 LoRA targeting MMDiT MLP projections.", ("mmdit", "transformer")),
    ("mmdit.modulation_norm.adapter", "MMDiT Modulation / Norm Adapters", "SD3 LoRA targeting modulation/norm locations.", ("mmdit", "transformer")),
    ("mmdit.other.adapter", "MMDiT Other Adapters", "Other SD3 MMDiT LoRA targets.", ("mmdit", "transformer")),
    ("clip_l.adapter", "CLIP-L Adapters", "SD3 CLIP-L LoRA parameters.", ("clip_l", "text_encoder_1")),
    ("clip_g.adapter", "CLIP-G Adapters", "SD3 CLIP-G LoRA parameters.", ("clip_g", "text_encoder_2")),
    ("t5xxl.adapter", "T5XXL Adapters", "SD3 T5XXL LoRA parameters.", ("t5xxl", "text_encoder_3")),
)


def _merge_components(*mappings: Mapping[str, ComponentDefinition]) -> Mapping[str, ComponentDefinition]:
    merged: dict[str, ComponentDefinition] = {}
    for mapping in mappings:
        overlap = set(merged).intersection(mapping)
        if overlap:
            raise RuntimeError("Duplicate component definition(s): " + ", ".join(sorted(overlap)))
        merged.update(mapping)
    return MappingProxyType(merged)


_PROFILES: Mapping[str, ModelComponentProfile] = MappingProxyType(
    {
        "sd-dreambooth": ModelComponentProfile(
            "sd-dreambooth",
            _merge_components(
                _SD_FULL_COMPONENTS,
                _components(("text_encoder", "Text Encoder", "Stable Diffusion text encoder.", ("text_encoder",))),
            ),
            _sd_dreambooth_classifier,
            _eligibility_sd,
        ),
        "sdxl-finetune": ModelComponentProfile(
            "sdxl-finetune",
            _merge_components(
                _SD_FULL_COMPONENTS,
                _components(
                    ("text_encoder_1", "Text Encoder 1", "SDXL CLIP-L text encoder.", ("text_encoder_1",)),
                    ("text_encoder_2", "Text Encoder 2", "SDXL OpenCLIP text encoder.", ("text_encoder_2",)),
                ),
            ),
            _sdxl_full_classifier,
            _eligibility_sd,
        ),
        "sd-lora": ModelComponentProfile(
            "sd-lora",
            _merge_components(
                _SD_LORA_COMPONENTS,
                _components(("text_encoder.adapter", "Text Encoder Adapters", "Stable Diffusion text-encoder LoRA.", ("text_encoder",))),
            ),
            _sd_lora_classifier_v1,
            _eligibility_sd,
        ),
        "sdxl-lora": ModelComponentProfile(
            "sdxl-lora",
            _merge_components(
                _SD_LORA_COMPONENTS,
                _components(
                    ("text_encoder_1.adapter", "Text Encoder 1 Adapters", "SDXL CLIP-L LoRA.", ("text_encoder_1",)),
                    ("text_encoder_2.adapter", "Text Encoder 2 Adapters", "SDXL OpenCLIP LoRA.", ("text_encoder_2",)),
                ),
            ),
            _sdxl_lora_classifier,
            _eligibility_sd,
        ),
        "flux-finetune": ModelComponentProfile(
            "flux-finetune",
            _FLUX_TRANSFORMER_COMPONENTS,
            _flux_full_classifier,
            _eligibility_flux,
        ),
        "flux-lora": ModelComponentProfile(
            "flux-lora",
            _merge_components(
                _FLUX_LORA_TRANSFORMER_COMPONENTS,
                _components(
                    ("clip_l.adapter", "CLIP-L Adapters", "Flux CLIP-L LoRA.", ("clip_l", "text_encoder_1")),
                    ("t5xxl.adapter", "T5XXL Adapters", "Flux T5XXL LoRA.", ("t5xxl", "text_encoder_2")),
                ),
            ),
            _flux_lora_classifier_v1,
            _eligibility_flux,
        ),
        "chroma-lora": ModelComponentProfile(
            "chroma-lora",
            _merge_components(
                _FLUX_LORA_TRANSFORMER_COMPONENTS,
                _components(("t5xxl.adapter", "T5XXL Adapters", "Chroma T5XXL LoRA.", ("t5xxl", "text_encoder_2"))),
            ),
            _chroma_lora_classifier,
            _eligibility_flux,
        ),
        "anima-finetune": ModelComponentProfile(
            "anima-finetune",
            _ANIMA_FULL_COMPONENTS,
            _anima_full_classifier,
            _eligibility_anima,
        ),
        "anima-lora": ModelComponentProfile(
            "anima-lora",
            _ANIMA_LORA_COMPONENTS,
            _anima_lora_classifier,
            _eligibility_anima,
        ),
        "sd3-lora": ModelComponentProfile(
            "sd3-lora",
            _SD3_LORA_COMPONENTS,
            _sd3_lora_classifier,
            _eligibility_sd3,
        ),
    }
)


def list_model_component_profiles() -> tuple[ModelComponentProfile, ...]:
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


def get_model_component_profile(train_type: str) -> ModelComponentProfile:
    key = str(train_type or "").strip().lower()
    profile = _PROFILES.get(key)
    if profile is None:
        raise ValueError(f"No Parameter Policy Model Component Profile is registered for {train_type!r}.")
    return profile


def _resolve_sd_lora_target(
    profile: ModelComponentProfile,
    config: Mapping[str, Any],
) -> TrainingTargetProfile:
    unet_only = _bool(config.get("network_train_unet_only"), field="network_train_unet_only")
    te_only = _bool(config.get("network_train_text_encoder_only"), field="network_train_text_encoder_only")
    if unet_only and te_only:
        raise ValueError(
            "LoRA target cannot enable both network_train_unet_only and network_train_text_encoder_only."
        )

    unet_components = {component_id for component_id in profile.components if component_id.startswith("unet.")}
    te_components = set(profile.components).difference(unet_components)
    if unet_only:
        available = unet_components
    elif te_only:
        available = te_components
    else:
        available = set(profile.components)
    return _target_profile(profile, available)


def _resolve_flux_lora_target(
    profile: ModelComponentProfile,
    config: Mapping[str, Any],
    *,
    chroma: bool,
) -> TrainingTargetProfile:
    unet_only = _bool(config.get("network_train_unet_only"), field="network_train_unet_only")
    te_only = _bool(
        config.get("network_train_text_encoder_only"),
        field="network_train_text_encoder_only",
    )
    if unet_only and te_only:
        raise ValueError(
            "Flux/Chroma LoRA target cannot enable both unet-only and text-encoder-only modes."
        )

    available: set[str] = set()
    if not te_only:
        available.update(
            component_id
            for component_id in profile.components
            if component_id.startswith("transformer.")
        )
    if not chroma and not unet_only:
        available.add("clip_l.adapter")
    if _network_bool(config, "train_t5xxl"):
        available.add("t5xxl.adapter")
    return _target_profile(profile, available)


def _resolve_anima_lora_target(
    profile: ModelComponentProfile,
    config: Mapping[str, Any],
) -> TrainingTargetProfile:
    dit_only = _bool(config.get("network_train_unet_only"), field="network_train_unet_only")
    qwen_only = _bool(
        config.get("network_train_text_encoder_only"),
        field="network_train_text_encoder_only",
    )
    if dit_only and qwen_only:
        raise ValueError("Anima LoRA target cannot enable both unet-only and text-encoder-only modes.")

    available: set[str] = set()
    dit_components = {
        component_id for component_id in profile.components if component_id.startswith("dit.")
    }
    if not qwen_only:
        available.update(dit_components)
        if _network_bool(config, "train_llm_adapter"):
            available.add("llm_adapter.adapter")
    if not dit_only:
        available.add("qwen3.adapter")
    return _target_profile(profile, available)


def _resolve_sd3_lora_target(
    profile: ModelComponentProfile,
    config: Mapping[str, Any],
) -> TrainingTargetProfile:
    unet_only = _bool(config.get("network_train_unet_only"), field="network_train_unet_only")
    te_only = _bool(
        config.get("network_train_text_encoder_only"),
        field="network_train_text_encoder_only",
    )
    if unet_only and te_only:
        raise ValueError(
            "SD3 LoRA target cannot enable both unet-only and text-encoder-only modes."
        )

    available: set[str] = set()
    if not te_only:
        available.update(
            component_id
            for component_id in profile.components
            if component_id.startswith("mmdit.")
        )
    if not unet_only:
        available.update({"clip_l.adapter", "clip_g.adapter"})
    if _network_bool(config, "train_t5xxl"):
        available.add("t5xxl.adapter")
    return _target_profile(profile, available)


def resolve_training_target_profile(
    train_type: str,
    effective_config: Mapping[str, Any],
) -> TrainingTargetProfile:
    profile = get_model_component_profile(train_type)
    config = effective_config

    if profile.train_type in {"sd-lora", "sdxl-lora"}:
        return _resolve_sd_lora_target(profile, config)
    if profile.train_type == "flux-lora":
        return _resolve_flux_lora_target(profile, config, chroma=False)
    if profile.train_type == "chroma-lora":
        return _resolve_flux_lora_target(profile, config, chroma=True)
    if profile.train_type == "anima-lora":
        return _resolve_anima_lora_target(profile, config)
    if profile.train_type == "sd3-lora":
        return _resolve_sd3_lora_target(profile, config)

    if profile.train_type == "sd-dreambooth":
        available = set(profile.components)
        stop = config.get("stop_text_encoder_training")
        if stop not in (None, ""):
            try:
                if int(stop) < 0:
                    available.discard("text_encoder")
            except (TypeError, ValueError) as exc:
                raise ValueError("stop_text_encoder_training must be an integer.") from exc
        return _target_profile(profile, available)

    if profile.train_type == "sdxl-finetune":
        available = {
            component_id for component_id in profile.components if component_id.startswith("unet.")
        }
        if _bool(config.get("train_text_encoder"), field="train_text_encoder"):
            available.update({"text_encoder_1", "text_encoder_2"})
        return _target_profile(profile, available)

    if profile.train_type == "flux-finetune":
        return _target_profile(profile, set(profile.components))

    if profile.train_type == "anima-finetune":
        available = set(profile.components)
        if not _bool(
            config.get("train_qwen3_text_encoder"),
            field="train_qwen3_text_encoder",
        ):
            available.discard("qwen3")
        return _target_profile(profile, available)

    raise AssertionError(f"Unhandled Model Component Profile {profile.train_type!r}.")
