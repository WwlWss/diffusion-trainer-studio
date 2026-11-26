"""Isolated GPU-evidence-only construction for tiny synthetic Full-BF16 models.

This module belongs ONLY to tools/run_parameter_policy_execution_gpu_matrix.py.
It is not a Trainer, Start, Launcher, or API entry point and cannot authorize
an unqualified backend.  Never expose it as a production Admission bypass.

The synthetic Flux scaffold exercises genuine DTS low-level routing, optimizer,
dtype, accelerator and checkpoint contracts without claiming Flux Full-BF16
backend qualification.  The real create_parameter_policy_session() must retain
its mandatory, fail-closed early/late Admission protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


# Exactly the small, source-owned scaffold geometries used by the C0-C4 GPU
# evidence runner.  Do not expand this allowlist to arbitrary model weights.
_SYNTHETIC_SHAPES: dict[str, dict[str, tuple[int, ...]]] = {
    "_TinyFlux": {
        "double_blocks.0.weight": (8, 8),
        "double_blocks.0.bias": (8,),
    },
    "_TinyMuonFlux": {
        "double_blocks.0.weight": (12, 8),
        "double_blocks.1.weight": (8, 12),
    },
    "_TinyMuonAdamWFallbackFlux": {
        "double_blocks.0.weight": (12, 8),
        "double_blocks.0.bias": (12,),
        "double_blocks.1.weight": (8, 12),
        "double_blocks.1.bias": (8,),
    },
}
_SYNTHETIC_PROFILES = {
    "_TinyFlux": {"main": "AdamW"},
    "_TinyMuonFlux": {"main": "Muon"},
    "_TinyMuonAdamWFallbackFlux": {
        "muon": "Muon",
        "adamw_fallback": "AdamW",
    },
}
_SYNTHETIC_ROUTES = {
    "_TinyFlux": ("main", None, 1e-2, None),
    "_TinyMuonFlux": ("main", None, 2e-2, None),
    "_TinyMuonAdamWFallbackFlux": (
        "muon", "adamw_fallback", 2e-2, 1e-2,
    ),
}


def build_gpu_evidence_full_bf16_session(
    *,
    args: object,
    train_type: str,
    roots: Mapping[str, Any],
    structural_frozen_parameters: Iterable[Any] = (),
):
    """Construct a genuine low-level Session for the fixed tiny GPU scaffold.

    Production Host/Late Admission is deliberately *not* called: it correctly
    blocks flux-finetune Full-BF16.  This helper therefore tests execution
    mechanics only, never claims or grants production backend qualification.
    """
    from mikazuki import parameter_policy_trainer as t

    config = t._effective_config(args)
    if (
        type(train_type) is not str
        or train_type != "flux-finetune"
        or config.get("full_bf16") is not True
        or config.get("mixed_precision") != "bf16"
    ):
        raise ValueError(
            "GPU evidence helper requires exact synthetic flux-finetune "
            "with real full_bf16=True and mixed_precision='bf16'."
        )
    if not isinstance(roots, Mapping) or set(roots) != {"transformer"}:
        raise ValueError("GPU evidence helper accepts only the transformer scaffold root.")

    model = roots["transformer"]
    model_kind = type(model).__name__
    if not isinstance(model, t.torch.nn.Module) or model_kind not in _SYNTHETIC_SHAPES:
        raise ValueError("GPU evidence helper rejects non-synthetic model classes.")
    actual_shapes = {
        name: tuple(int(dim) for dim in parameter.shape)
        for name, parameter in model.named_parameters()
    }
    if actual_shapes != _SYNTHETIC_SHAPES[model_kind]:
        raise ValueError("GPU evidence helper rejects changed synthetic model geometry.")

    structural = t._unique_parameters(structural_frozen_parameters)
    if structural:
        raise ValueError("GPU evidence helper does not admit extra frozen parameters.")

    policy, policy_hash = t.load_parameter_policy_file(t._policy_path(args))
    actual_profiles = {
        name: profile["type"]
        for name, profile in policy["optimizer_profiles"].items()
    }
    if actual_profiles != _SYNTHETIC_PROFILES[model_kind]:
        raise ValueError("GPU evidence helper rejects unreviewed optimizer profiles.")
    if set(policy["components"]) != {"transformer.double_stream"}:
        raise ValueError("GPU evidence helper rejects non-scaffold Component policies.")
    route = policy["components"]["transformer.double_stream"]
    actual_route = (
        route.get("optimizer_profile"),
        route.get("fallback_optimizer_profile"),
        route.get("learning_rate"),
        route.get("fallback_learning_rate"),
    )
    if route.get("train") is not True or actual_route != _SYNTHETIC_ROUTES[model_kind]:
        raise ValueError("GPU evidence helper rejects unreviewed optimizer routing.")

    execution_contract = t.build_parameter_policy_execution_contract(
        train_type=train_type,
        effective_config=config,
    )
    if execution_contract is None or execution_contract.features != ("full_bf16",):
        raise ValueError("GPU evidence helper requires a genuine Full-BF16 execution contract.")

    # Retain the actual production graph/optimizer/Session implementation;
    # only the unqualified *Admission* layer is isolated for these GPU probes.
    scan_roots = dict(roots)
    descriptors = t.scan_parameter_roots(scan_roots)
    execution_root_refs = t._capture_execution_root_refs(scan_roots)
    routed_descriptors = tuple(descriptors)
    plan = t.build_parameter_routing_plan(
        policy,
        train_type=train_type,
        effective_config=config,
        descriptors=routed_descriptors,
    )
    if not plan.is_valid:
        raise t.ParameterPolicyTrainerRuntimeError(t._routing_error_message(plan))

    runtime_spec = t.compile_parameter_policy_runtime_spec(policy, plan)
    optimizer = t.build_parameter_policy_optimizer(runtime_spec)
    session = t.ParameterPolicyTrainerSession(
        train_type=train_type,
        policy=policy,
        policy_hash=policy_hash,
        descriptors=descriptors,
        routed_descriptors=routed_descriptors,
        routing_plan=plan,
        runtime_spec=runtime_spec,
        optimizer=optimizer,
        structural_frozen_parameters=structural,
        execution_contract=execution_contract,
        execution_root_refs=execution_root_refs,
        resume_requested=bool(
            str(config.get("resume") or "").strip()
            or config.get("resume_from_huggingface")
        ),
    )
    session.apply_requires_grad_contract()
    session.assert_requires_grad_contract()
    return session


__all__ = ["build_gpu_evidence_full_bf16_session"]
