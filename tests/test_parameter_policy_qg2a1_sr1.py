"""QG-2-A1-SR1: genuine Strict Review RED repairs and old GREEN baselines.

The absence of a safety invariant produces an exact failure marker; no
dummy assertFalse, skip, xfail or blanket stubbing of real source guards.
Only AST-isolated Request integration is mocked to expose the independent
Admission decision even when a legacy Step 6F function returns [].
"""
from __future__ import annotations

from typing import get_args, get_type_hints
import unittest
from unittest.mock import patch

from mikazuki.parameter_policy_admission import (
    ComponentAdmission,
    assess_component_training_admission,
)
from mikazuki.parameter_policy_execution import FULL_BF16_BACKEND_QUALIFICATIONS
from mikazuki.training_config import PreparedTrainingConfig, RequestAdmissionIntent
from tests.test_parameter_policy_admission_red import _config, _pending, _policy
from tests.test_parameter_policy_request_contract import (
    _load_prepare_request_config, _prepared,
)

MARKER = "QG-2-A1-SR1 REPAIR CONTRACT ABSENT"


def _first_party_admission(status: str, *, purpose: str = "start"):
    """Create a *real typed* first-party diagnostic, never a forged status."""
    if status == "inactive":
        result = assess_component_training_admission(
            _policy(),
            train_type="sd-lora",
            effective_config=_config(full_bf16=False),
            purpose=purpose,
        )
    elif status == "blocked":
        result = assess_component_training_admission(
            _policy(),
            train_type="sd-lora",
            effective_config=_config(mixed_precision="fp16"),
            purpose=purpose,
            experimental_ack=True,
        )
    elif status in {"ack_required", "experimental_host_candidate"}:
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {
            "sd-lora": _pending(),
        }):
            result = assess_component_training_admission(
                _policy(),
                train_type="sd-lora",
                effective_config=_config(),
                purpose=purpose,
                experimental_ack=(status == "experimental_host_candidate"),
            )
    else:
        result = assess_component_training_admission(
            _policy(),
            train_type="sd-lora",
            effective_config=_config(),
            purpose=purpose,
        )
    if result.status != status:
        raise AssertionError(
            f"Invalid SR1 typed fixture: expected={status}, actual={result.status}"
        )
    return result


def _prepare_fixture(
    *,
    admission=None,
    component=True,
    gpu_ids=None,
):
    """Return Request from actual source, with recorded side effects.

    Legacy Step6F and GPU are intentionally stubbed to [] to test whether
    first-party Admission independently blocks Start.  Other real tests
    continue to cover the actual Step6F producer/Start sequencing.
    """
    events: list[tuple] = []

    def stage_sidecar(config, page):
        if component:
            return "policy.json", {"policy.json": "{}"}, {"version": 1}
        return None, {}, None

    def compile_request(raw, **kw):
        events.append(("prepare", kw["launch"]))
        value = (
            {"parameter_policy_config": "policy.json"}
            if component else {}
        )
        result = _prepared(value)
        result.gpu_ids = gpu_ids
        return result

    prepare = _load_prepare_request_config(
        build_parameter_policy_sidecar=stage_sidecar,
        prepare_training_config=compile_request,
        assess_component_training_admission=lambda policy, **kw: admission,
        parameter_policy_runtime_blockers=lambda policy, **kw: [],
        parameter_policy_gpu_selection_blockers=lambda ids: [],
        materialize_sidecars=lambda sidecars: events.append(
            ("materialize", dict(sidecars))
        ),
    )
    return prepare, events


class QG2A1SR1TargetedRed(unittest.TestCase):
    def _requires_denial(
        self, prepare, events, *, fragment: str, **kwargs
    ):
        try:
            prepare({}, "sd-lora", **kwargs)
        except ValueError as exc:
            self.assertIn(fragment, str(exc))
        else:
            self.fail(
                f"{MARKER}: this request returned without independent "
                f"fail-closed rejection: {fragment}"
            )
        self.assertFalse(
            any(e[0] == "materialize" for e in events),
            "Rejected request must not write staged sidecars.",
        )

    def test_sr101_explicit_ack_blocked_still_rejected_when_step6f_empty(self):
        prepare, events = _prepare_fixture(admission=_first_party_admission("blocked"))
        self._requires_denial(
            prepare, events,
            fragment="runtime.bf16.precision_mismatch",
            launch=True, materialize=True,
            admission_intent=RequestAdmissionIntent(True, "webui_start"),
        )

    def test_sr102_no_ack_blocked_still_rejected_when_step6f_empty(self):
        prepare, events = _prepare_fixture(admission=_first_party_admission("blocked"))
        self._requires_denial(
            prepare, events,
            fragment="runtime.bf16.precision_mismatch",
            launch=True, materialize=True,
        )

    def test_sr103_missing_ack_independently_rejected_when_step6f_empty(self):
        prepare, events = _prepare_fixture(admission=_first_party_admission("ack_required"))
        self._requires_denial(
            prepare, events,
            fragment="experimental_full_bf16_ack",
            launch=True, materialize=True,
        )

    def test_sr104_experimental_host_candidate_is_staged_hold_even_if_legacy_empty(self):
        prepare, events = _prepare_fixture(
            admission=_first_party_admission("experimental_host_candidate")
        )
        self._requires_denial(
            prepare, events,
            fragment="QG-2-A2",
            launch=True, materialize=True,
            admission_intent=RequestAdmissionIntent(True, "webui_start"),
        )

    def test_sr105_none_from_admission_is_invalid_even_if_legacy_empty(self):
        prepare, events = _prepare_fixture(admission=None)
        self._requires_denial(
            prepare, events,
            fragment="typed ComponentAdmission",
            launch=True, materialize=True,
        )

    def test_sr106_preview_cannot_request_sidecar_materialization(self):
        prepare, events = _prepare_fixture(component=False)
        self._requires_denial(
            prepare, events, fragment="Preview",
            launch=False, admission_purpose="preview", materialize=True,
        )
        self.assertEqual(events, [], "Reject Preview materialize before preparation.")

    def test_sr107_materialize_requires_strict_boolean(self):
        # Freeze one test identity / one RED failure. SubTest would emit four
        # separate unittest failures and corrupt the exact RED count.
        violations = []
        for value in ("false", 1, 0, None):
            prepare, events = _prepare_fixture(component=False)
            try:
                prepare(
                    {}, "sd-lora", launch=False,
                    admission_purpose="preview", materialize=value,
                )
            except ValueError as exc:
                if "materialize" not in str(exc):
                    violations.append(f"{value!r}: wrong error {exc}")
            else:
                violations.append(f"{value!r}: invalid value was accepted")
            if events:
                violations.append(f"{value!r}: preparation/writes occurred {events}")
        if violations:
            self.fail(f"{MARKER}: " + "; ".join(violations))

    def test_sr108_prepared_type_hints_resolve_real_component_admission(self):
        try:
            hints = get_type_hints(PreparedTrainingConfig)
        except (NameError, TypeError) as exc:
            self.fail(f"{MARKER}: runtime get_type_hints failed: {exc}")
        types = get_args(hints["admission"])
        self.assertIn(ComponentAdmission, types, hints["admission"])
        self.assertIn(type(None), types, hints["admission"])


class QG2A1SR1TrueBaselines(unittest.TestCase):
    def test_sr901_existing_d1_legacy_approved_start_remains_allowed(self):
        prepare, events = _prepare_fixture(
            admission=_first_party_admission("legacy_approved")
        )
        result = prepare({}, "sd-lora", launch=True, materialize=True)
        self.assertEqual(result.admission.status, "legacy_approved")
        self.assertEqual([a[0] for a in events], ["prepare", "materialize"])

    def test_sr902_inactive_component_start_remains_allowed(self):
        prepare, events = _prepare_fixture(
            admission=_first_party_admission("inactive")
        )
        result = prepare({}, "sd-lora", launch=True, materialize=True)
        self.assertEqual(result.admission.status, "inactive")
        self.assertEqual([a[0] for a in events], ["prepare", "materialize"])

    def test_sr903_preview_blocked_is_diagnostic_without_writes(self):
        prepare, events = _prepare_fixture(
            admission=_first_party_admission("blocked", purpose="preview")
        )
        result = prepare({}, "sd-lora", launch=False, materialize=False)
        self.assertEqual(result.admission.status, "blocked")
        self.assertEqual([a[0] for a in events], ["prepare"])

    def test_sr904_export_and_standard_start_keep_deliberate_materialization(self):
        export, events = _prepare_fixture(
            admission=_first_party_admission("legacy_approved", purpose="export")
        )
        prepared = export(
            {}, "sd-lora", launch=False, materialize=True,
            admission_purpose="export",
        )
        self.assertEqual(prepared.admission.purpose, "export")
        self.assertEqual([a[0] for a in events], ["prepare", "materialize"])
        standard, standard_events = _prepare_fixture(component=False)
        standard({}, "sd-lora", launch=True, materialize=True)
        self.assertEqual(
            [a[0] for a in standard_events],
            ["prepare", "materialize"],
        )


if __name__ == "__main__":
    unittest.main()
