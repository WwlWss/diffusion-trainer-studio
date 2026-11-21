"""First-party B3-C Session admission fixtures, not production authorization.

This test-only module uses exactly the public policy loader, first-party
classifier and typed one-shot intent.  No Torch dependency at import time.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_admission import prepare_direct_trainer_admission


def make_test_admission_kwargs(
    args: Any,
    train_type: str,
    *,
    experimental_ack: bool = False,
    expected_status: str = "inactive",
) -> dict[str, object]:
    """Return matching explicit Session control inputs for a test's args.

    This intentionally fails rather than inventing approval or bypassing
    source-owned readiness.  A test of held status can request that status
    explicitly but the real Late Session still rejects it before scanning.
    """
    if type(experimental_ack) is not bool:
        raise TypeError("Test acknowledgement must be an exact bool.")
    if isinstance(args, Mapping):
        effective_config = dict(args)
    else:
        effective_config = dict(vars(args))
    policy_path = effective_config.get("parameter_policy_config")
    if not isinstance(policy_path, str) or not policy_path.strip():
        raise ValueError("Test needs an actual Parameter Policy sidecar path.")
    intent = DirectTrainerAdmissionIntent(experimental_ack)
    snapshot = prepare_direct_trainer_admission(
        policy_path,
        train_type=train_type,
        effective_config=effective_config,
        admission_intent=intent,
    )
    if snapshot.admission.status != expected_status:
        raise AssertionError(
            "Test does not have the requested real first-party admission: "
            f"expected={expected_status!r}; got={snapshot.admission.status!r}; "
            f"blockers={snapshot.admission.runtime_blockers!r}"
        )
    return {
        "admission_snapshot": snapshot,
        "admission_intent": intent,
    }


__all__ = ["make_test_admission_kwargs"]
