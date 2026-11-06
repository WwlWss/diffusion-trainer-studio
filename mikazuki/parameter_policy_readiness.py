"""Pure Parameter Policy execution-readiness diagnostics (QG-1A2).

This module describes a caller-established capability, runtime findings and
read-only evidence coverage.  It does not establish implementation support,
authorize a training job, certify CUDA, sign anything, or alter execution or
checkpoint identities.  Runtime enforcement remains in existing trainer paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal


ReadinessDecision = Literal["allow", "warn", "block"]
CapabilityStatus = Literal[
    "implemented", "experimental_candidate", "not_implemented",
    "known_incompatible",
]
EvidenceTier = Literal[
    "formally_qualified", "observed_real_gpu", "host_cpu_checked",
    "unverified", "failed",
]

_CAPABILITIES = frozenset(
    {"implemented", "experimental_candidate", "not_implemented",
     "known_incompatible"}
)
_EVIDENCE_TIERS = frozenset(
    {"formally_qualified", "observed_real_gpu", "host_cpu_checked",
     "unverified", "failed"}
)
_DECISIONS = frozenset({"allow", "warn", "block"})
_UNSUPPORTED = frozenset({"not_implemented", "known_incompatible"})
_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")
_SOURCE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")


def _require_choice(value: object, *, name: str, choices: frozenset[str]) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(f"Unknown Parameter Policy readiness {name}.")


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    """A typed diagnostic; the message is never an authorization key."""

    code: str
    message: str
    source: str

    def __post_init__(self) -> None:
        if type(self.code) is not str or _CODE_PATTERN.fullmatch(self.code) is None:
            raise ValueError("ReadinessIssue requires a stable lowercase dotted code.")
        if type(self.message) is not str or not self.message.strip():
            raise ValueError("ReadinessIssue requires a nonempty message.")
        if type(self.source) is not str or _SOURCE_PATTERN.fullmatch(self.source) is None:
            raise ValueError("ReadinessIssue requires a stable source identifier.")


def _issue_tuple(values: Sequence[ReadinessIssue], *, name: str) -> tuple[ReadinessIssue, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"Readiness {name} must be a sequence of ReadinessIssue.")
    result = tuple(values)
    if any(type(issue) is not ReadinessIssue for issue in result):
        raise ValueError(f"Readiness {name} must contain only ReadinessIssue.")
    return result


def _validate_unique(
    hard_errors: tuple[ReadinessIssue, ...],
    warnings: tuple[ReadinessIssue, ...],
) -> None:
    codes = [issue.code for issue in (*hard_errors, *warnings)]
    if len(codes) != len(set(codes)):
        raise ValueError("Readiness issues cannot repeat a code across severities.")


def _expected_decision(
    capability: str,
    hard_errors: tuple[ReadinessIssue, ...],
    warnings: tuple[ReadinessIssue, ...],
) -> ReadinessDecision:
    if hard_errors or capability in _UNSUPPORTED:
        return "block"
    if capability == "experimental_candidate" or warnings:
        return "warn"
    return "allow"


@dataclass(frozen=True, slots=True)
class ExecutionReadiness:
    """Immutable diagnostic result; can_start is NOT a launch authorization."""

    decision: ReadinessDecision
    capability: CapabilityStatus
    hard_errors: tuple[ReadinessIssue, ...]
    warnings: tuple[ReadinessIssue, ...]
    evidence_tier: EvidenceTier

    def __post_init__(self) -> None:
        _require_choice(self.decision, name="decision", choices=_DECISIONS)
        _require_choice(self.capability, name="capability", choices=_CAPABILITIES)
        _require_choice(self.evidence_tier, name="evidence tier", choices=_EVIDENCE_TIERS)
        hard = _issue_tuple(self.hard_errors, name="hard errors")
        warnings = _issue_tuple(self.warnings, name="warnings")
        _validate_unique(hard, warnings)

        if self.capability in _UNSUPPORTED and not hard:
            raise ValueError("Unsupported capability requires a hard error.")
        if self.capability == "experimental_candidate" and not hard and not warnings:
            raise ValueError("Experimental candidate requires an advisory warning.")
        if self.decision != _expected_decision(self.capability, hard, warnings):
            raise ValueError("Readiness decision conflicts with capability or issues.")

        # Defensively discard mutable caller lists, including direct construction.
        object.__setattr__(self, "hard_errors", hard)
        object.__setattr__(self, "warnings", warnings)

    @property
    def can_start(self) -> bool:
        """Advisory only; real Start must still recheck runtime invariants."""
        return self.decision != "block"

    def to_diagnostics(self) -> dict[str, object]:
        """Create an independent JSON-safe diagnostic, never a certificate."""
        return {
            "decision": self.decision,
            "capability": self.capability,
            "hard_errors": [
                {"code": issue.code, "message": issue.message, "source": issue.source}
                for issue in self.hard_errors
            ],
            "warnings": [
                {"code": issue.code, "message": issue.message, "source": issue.source}
                for issue in self.warnings
            ],
            "evidence_tier": self.evidence_tier,
        }


def evaluate_execution_readiness(
    *,
    capability: str,
    hard_errors: Sequence[ReadinessIssue] = (),
    warnings: Sequence[ReadinessIssue] = (),
    evidence_tier: str = "unverified",
) -> ExecutionReadiness:
    """Reduce explicit capability and findings; evidence cannot grant support.

    The caller must determine capability from concrete implementation and
    runtime semantics, independently of formal evidence status.  This pure
    reducer does not check a training backend or issue a trusted permission.
    """
    _require_choice(capability, name="capability", choices=_CAPABILITIES)
    _require_choice(evidence_tier, name="evidence tier", choices=_EVIDENCE_TIERS)
    hard = _issue_tuple(hard_errors, name="hard errors")
    advisory = _issue_tuple(warnings, name="warnings")
    _validate_unique(hard, advisory)

    if capability in _UNSUPPORTED and not hard:
        hard = (
            ReadinessIssue(
                code=f"capability.{capability}",
                message="The selected capability is not implemented or is incompatible.",
                source="G01",
            ),
        )
    elif capability == "experimental_candidate" and not hard and not advisory:
        advisory = (
            ReadinessIssue(
                code="capability.experimental",
                message="This capability has an experimental execution contract.",
                source="E02",
            ),
        )

    # Fail on ambiguous codes, including auto-generated capability findings.
    _validate_unique(hard, advisory)
    return ExecutionReadiness(
        decision=_expected_decision(capability, hard, advisory),
        capability=capability,
        hard_errors=hard,
        warnings=advisory,
        evidence_tier=evidence_tier,
    )


__all__ = ["ReadinessIssue", "ExecutionReadiness", "evaluate_execution_readiness"]
