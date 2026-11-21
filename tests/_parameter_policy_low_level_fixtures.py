"""Test-only low-level Full-BF16 session construction.

IMPORTANT: This is NOT a Direct Trainer/Launcher admission alternative.
Only CPU smoke tests may call this helper to exercise historical parameter,
optimizer, scheduler, root identity and checkpoint invariants for backend
families whose Full-BF16 training remains explicitly unqualified.

It builds real production objects using the existing low-level APIs.  Do not
import or copy this module into production or add a bypass flag to the live
create_parameter_policy_session() entry.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def build_test_only_full_bf16_session(
    *, args: object, train_type: str,
    roots: Mapping[str, Any],
    structural_frozen_parameters: Iterable[Any] = (),
):
    # Import torch-heavy internals only when a CPU runtime test uses them.
    from mikazuki import parameter_policy_trainer as t

    config = t._effective_config(args)
    if (
        train_type != "flux-finetune"
        or config.get("full_bf16") is not True
        or config.get("mixed_precision") != "bf16"
    ):
        raise ValueError(
            "Isolated Full-BF16 fixture is restricted to synthetic flux-finetune "
            "with explicit BF16 mode; not for real training."
        )
    policy, policy_hash = t.load_parameter_policy_file(t._policy_path(args))
    execution_contract = t.build_parameter_policy_execution_contract(
        train_type=train_type,
        effective_config=config,
    )
    if execution_contract is None or execution_contract.features != ("full_bf16",):
        raise ValueError("Isolated test must have a genuine Full-BF16 execution contract.")

    # Preserve the production parameter graph and runtime specifications.
    scan_roots = dict(roots)
    descriptors = t.scan_parameter_roots(scan_roots)
    execution_root_refs = t._capture_execution_root_refs(scan_roots)
    structural = t._unique_parameters(structural_frozen_parameters)
    descriptor_ids = {descriptor.parameter_id for descriptor in descriptors}
    structural_ids = {id(parameter) for parameter in structural}
    if not structural_ids <= descriptor_ids:
        raise t.ParameterPolicyTrainerRuntimeError(
            "Isolated Full-BF16 fixture referenced unscanned frozen Parameters."
        )
    routed_descriptors = tuple(
        descriptor for descriptor in descriptors
        if descriptor.parameter_id not in structural_ids
    )
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


__all__ = ["build_test_only_full_bf16_session"]
