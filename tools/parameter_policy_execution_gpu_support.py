"""CPU-testable support helpers for Phase C execution GPU evidence.

This module is tooling-only. Production trainer/request code must never import it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from mikazuki import parameter_policy_execution as execution


class ExecutionGpuMatrixError(RuntimeError):
    pass


SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID = (
    execution.FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID
)
SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID = (
    "phase-d:shared-adamw-muon-full-bf16-regression:v1"
)

ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID = "phase-c:adamw-full-bf16:v1"
MUON_FULL_BF16_EVIDENCE_BUNDLE_ID = "phase-c:muon-full-bf16:v1"
MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID = (
    "phase-c:muon-adamw-explicit-fallback-full-bf16:v1"
)

MUON_FULL_BF16_ARGUMENT_FAMILY = (
    "momentum",
    "nesterov",
    "ns_coeffs",
    "ns_steps",
    "use_adjusted_lr",
    "weight_decay",
    "weight_decouple",
)

ADAMW_FULL_BF16_CASE_IDS = (
    "optimizer:adamw:full-bf16:accum1:v1",
    "optimizer:adamw:full-bf16:accum2:v1",
)

MUON_FULL_BF16_CASE_IDS = (
    "optimizer:muon:full-bf16:accum1:v1",
    "optimizer:muon:full-bf16:accum2:v1",
)

MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS = (
    "optimizer:muon-adamw-explicit-fallback:full-bf16:accum1:v1",
    "optimizer:muon-adamw-explicit-fallback:full-bf16:accum2:v1",
)

EXECUTION_INFRA_CASE_IDS = (
    "infra:cuda-bf16-capability:v1",
    "infra:full-bf16-session:v1",
)

EXECUTION_GPU_CASE_PHASES: dict[str, tuple[str, ...]] = {
    "infra:cuda-bf16-capability:v1": ("probe",),
    "infra:full-bf16-session:v1": ("probe",),
    ADAMW_FULL_BF16_CASE_IDS[0]: (
        "train_save",
        "resume_second_step",
    ),
    ADAMW_FULL_BF16_CASE_IDS[1]: (
        "train_save",
        "resume_second_step",
    ),
    MUON_FULL_BF16_CASE_IDS[0]: (
        "train_save",
        "resume_second_step",
    ),
    MUON_FULL_BF16_CASE_IDS[1]: (
        "train_save",
        "resume_second_step",
    ),
    MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS[0]: (
        "train_save",
        "resume_second_step",
    ),
    MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS[1]: (
        "train_save",
        "resume_second_step",
    ),
}


def summarize_adamw_full_bf16_bundle(
    case_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    rows = {
        row.get("case_id"): row
        for row in case_rows
        if row.get("case_id") in ADAMW_FULL_BF16_CASE_IDS
    }
    if not rows:
        return None

    missing_cases = sorted(set(ADAMW_FULL_BF16_CASE_IDS).difference(rows))
    if missing_cases:
        status = "incomplete"
    elif all(row.get("status") == "pass" for row in rows.values()):
        status = "pass"
    else:
        status = "fail"

    return {
        "scope": "optimizer",
        "optimizer": "AdamW",
        "feature": "full_bf16",
        "required_cases": list(ADAMW_FULL_BF16_CASE_IDS),
        "missing_cases": missing_cases,
        "status": status,
        "optimizer_evidence_complete": status == "pass",
        "promotion_eligible": False,
        "optimizer_qualification_eligible": False,
        "production_qualification_mutated": False,
    }


def summarize_muon_full_bf16_bundle(
    case_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    rows = {
        row.get("case_id"): row
        for row in case_rows
        if row.get("case_id") in MUON_FULL_BF16_CASE_IDS
    }
    if not rows:
        return None

    missing_cases = sorted(set(MUON_FULL_BF16_CASE_IDS).difference(rows))
    if missing_cases:
        status = "incomplete"
    elif all(row.get("status") == "pass" for row in rows.values()):
        status = "pass"
    else:
        status = "fail"

    return {
        "scope": "optimizer",
        "optimizer": "Muon",
        "feature": "full_bf16",
        "required_cases": list(MUON_FULL_BF16_CASE_IDS),
        "missing_cases": missing_cases,
        "status": status,
        "optimizer_evidence_complete": status == "pass",
        "promotion_eligible": False,
        "optimizer_qualification_eligible": False,
        "production_qualification_mutated": False,
    }


def summarize_muon_adamw_fallback_full_bf16_bundle(
    case_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    rows = {
        row.get("case_id"): row
        for row in case_rows
        if row.get("case_id") in MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS
    }
    if not rows:
        return None

    missing_cases = sorted(
        set(MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS).difference(rows)
    )
    if missing_cases:
        status = "incomplete"
    elif all(row.get("status") == "pass" for row in rows.values()):
        status = "pass"
    else:
        status = "fail"

    return {
        "scope": "optimizer_topology",
        "feature": "full_bf16",
        "topology": "muon_primary_adamw_explicit_fallback",
        "optimizers": ["Muon", "AdamW"],
        "required_cases": list(MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS),
        "missing_cases": missing_cases,
        "status": status,
        "topology_evidence_complete": status == "pass",
        "promotion_eligible": False,
        "optimizer_qualification_eligible": False,
        "production_qualification_mutated": False,
        "prerequisite_bundles": [
            ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
            MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
        ],
    }


def summarize_shared_full_bf16_promotion(
    case_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    qualification_snapshot: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    required_cases = tuple(EXECUTION_GPU_CASE_PHASES)
    rows = {
        row.get("case_id"): row
        for row in case_rows
        if row.get("case_id") in required_cases
    }
    missing_cases = sorted(set(required_cases).difference(rows))
    failed_cases = sorted(
        case_id
        for case_id, row in rows.items()
        if row.get("status") != "pass"
    )

    optimizers = qualification_snapshot.get("optimizers", {})
    backends = qualification_snapshot.get("backends", {})
    expected_targets = ("AdamW", "Muon")
    target_rows_match = all(
        optimizers.get(name, {}).get("status") == "qualified"
        and optimizers.get(name, {}).get("evidence_case_id")
        == SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID
        for name in expected_targets
    )
    unexpected_optimizer_promotions = sorted(
        name
        for name, row in optimizers.items()
        if name not in expected_targets and row.get("status") == "qualified"
    )
    unexpected_backend_promotions = sorted(
        name
        for name, row in backends.items()
        if row.get("status") == "qualified"
    )

    complete = not missing_cases and not failed_cases
    source_scope_valid = (
        target_rows_match
        and not unexpected_optimizer_promotions
        and not unexpected_backend_promotions
    )
    status = "pass" if complete and source_scope_valid else "fail"

    return {
        "id": SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID,
        "scope": "shared_optimizer_promotion",
        "feature": "full_bf16",
        "targets": list(expected_targets),
        "required_cases": list(required_cases),
        "required_bundles": [
            ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
            MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
            MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID,
        ],
        "missing_cases": missing_cases,
        "failed_cases": failed_cases,
        "target_rows_match": target_rows_match,
        "unexpected_optimizer_promotions": unexpected_optimizer_promotions,
        "unexpected_backend_promotions": unexpected_backend_promotions,
        "infra_complete": all(
            rows.get(case_id, {}).get("status") == "pass"
            for case_id in EXECUTION_INFRA_CASE_IDS
        ),
        "status": status,
        "promotion_eligible": status == "pass",
        "optimizer_qualification_eligible": status == "pass",
        "runtime_qualification_mutated": False,
    }


def summarize_shared_full_bf16_regression(
    case_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    qualification_snapshot: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    required_cases = tuple(EXECUTION_GPU_CASE_PHASES)
    rows = {
        row.get("case_id"): row
        for row in case_rows
        if row.get("case_id") in required_cases
    }
    missing_cases = sorted(set(required_cases).difference(rows))
    failed_cases = sorted(
        case_id
        for case_id, row in rows.items()
        if row.get("status") != "pass"
    )

    optimizers = qualification_snapshot.get("optimizers", {})
    expected_targets = ("AdamW", "Muon")
    target_rows_match = all(
        optimizers.get(name, {}).get("status") == "qualified"
        and optimizers.get(name, {}).get("evidence_case_id")
        == SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID
        for name in expected_targets
    )
    complete = not missing_cases and not failed_cases
    status = "pass" if complete and target_rows_match else "fail"

    return {
        "id": SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
        "scope": "shared_optimizer_regression",
        "feature": "full_bf16",
        "targets": list(expected_targets),
        "required_cases": list(required_cases),
        "required_bundles": [
            ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID,
            MUON_FULL_BF16_EVIDENCE_BUNDLE_ID,
            MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID,
        ],
        "missing_cases": missing_cases,
        "failed_cases": failed_cases,
        "target_rows_match": target_rows_match,
        "infra_complete": all(
            rows.get(case_id, {}).get("status") == "pass"
            for case_id in EXECUTION_INFRA_CASE_IDS
        ),
        "status": status,
        "promotion_eligible": False,
        "optimizer_qualification_eligible": False,
        "runtime_qualification_mutated": False,
    }


@contextmanager
def temporary_execution_qualification(
    *,
    backend: str,
    optimizers: tuple[str, ...],
    evidence_case_id: str,
) -> Iterator[dict[str, Any]]:
    evidence_id = str(evidence_case_id or "").strip()
    if not evidence_id:
        raise ExecutionGpuMatrixError(
            "Temporary execution qualification requires a non-empty evidence_case_id."
        )

    originals: list[
        tuple[
            dict[str, execution.ExecutionFeatureQualification],
            str,
            execution.ExecutionFeatureQualification,
        ]
    ] = []
    records: list[dict[str, Any]] = []

    targets = [
        (execution.FULL_BF16_BACKEND_QUALIFICATIONS, backend, "backend"),
        *[
            (
                execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
                optimizer,
                "optimizer",
            )
            for optimizer in optimizers
        ],
    ]

    try:
        for table, name, kind in targets:
            original = table.get(name)
            if original is None:
                raise ExecutionGpuMatrixError(
                    f"{kind}={name!r} has no full-BF16 qualification record."
                )
            originals.append((table, name, original))

            if original.status == "qualified":
                if not str(original.evidence_case_id or "").strip():
                    raise ExecutionGpuMatrixError(
                        f"{kind}={name!r} is marked qualified without an "
                        "evidence_case_id; qualification remains fail-closed."
                    )
                lease_applied = False
            elif original.status == "pending":
                lease_applied = True
                table[name] = execution.ExecutionFeatureQualification(
                    "qualified",
                    "Phase C exact-head GPU evidence worker temporary lease.",
                    evidence_id,
                )
            elif original.status == "unsupported":
                raise ExecutionGpuMatrixError(
                    f"{kind}={name!r} is explicitly unsupported: {original.reason}"
                )
            else:
                raise ExecutionGpuMatrixError(
                    f"{kind}={name!r} has invalid qualification status "
                    f"{original.status!r}."
                )

            records.append(
                {
                    "kind": kind,
                    "name": name,
                    "before_status": original.status,
                    "before_evidence_case_id": original.evidence_case_id,
                    "lease_applied": lease_applied,
                }
            )

        yield {"records": records}
    finally:
        for table, name, original in reversed(originals):
            table[name] = original
        for table, name, original in originals:
            if table.get(name) != original:
                raise ExecutionGpuMatrixError(
                    f"Temporary execution qualification lease did not restore {name!r}."
                )


__all__ = [
    "SHARED_FULL_BF16_PROMOTION_EVIDENCE_ID",
    "SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID",
    "EXECUTION_INFRA_CASE_IDS",
    "ADAMW_FULL_BF16_CASE_IDS",
    "ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID",
    "EXECUTION_GPU_CASE_PHASES",
    "ExecutionGpuMatrixError",
    "MUON_FULL_BF16_ARGUMENT_FAMILY",
    "MUON_FULL_BF16_CASE_IDS",
    "MUON_FULL_BF16_EVIDENCE_BUNDLE_ID",
    "MUON_ADAMW_FALLBACK_FULL_BF16_CASE_IDS",
    "MUON_ADAMW_FALLBACK_FULL_BF16_EVIDENCE_BUNDLE_ID",
    "summarize_adamw_full_bf16_bundle",
    "summarize_muon_full_bf16_bundle",
    "summarize_muon_adamw_fallback_full_bf16_bundle",
    "summarize_shared_full_bf16_promotion",
    "summarize_shared_full_bf16_regression",
    "temporary_execution_qualification",
]
