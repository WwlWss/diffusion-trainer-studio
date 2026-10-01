"""CPU-testable support helpers for Phase C execution GPU evidence.

This module is tooling-only. Production trainer/request code must never import it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from mikazuki import parameter_policy_execution as execution


class ExecutionGpuMatrixError(RuntimeError):
    pass


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
    "ExecutionGpuMatrixError",
    "temporary_execution_qualification",
]
