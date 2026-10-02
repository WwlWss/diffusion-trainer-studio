"""CPU-testable support helpers for Phase C execution GPU evidence.

This module is tooling-only. Production trainer/request code must never import it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from mikazuki import parameter_policy_execution as execution


class ExecutionGpuMatrixError(RuntimeError):
    pass


ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID = "phase-c:adamw-full-bf16:v1"
MUON_FULL_BF16_EVIDENCE_BUNDLE_ID = "phase-c:muon-full-bf16:v1"

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
    "ADAMW_FULL_BF16_CASE_IDS",
    "ADAMW_FULL_BF16_EVIDENCE_BUNDLE_ID",
    "EXECUTION_GPU_CASE_PHASES",
    "ExecutionGpuMatrixError",
    "MUON_FULL_BF16_ARGUMENT_FAMILY",
    "MUON_FULL_BF16_CASE_IDS",
    "MUON_FULL_BF16_EVIDENCE_BUNDLE_ID",
    "summarize_adamw_full_bf16_bundle",
    "summarize_muon_full_bf16_bundle",
    "temporary_execution_qualification",
]
