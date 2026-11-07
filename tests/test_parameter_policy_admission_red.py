"""QG-1B3/QG-2 admission contract: 12 targeted RED + 4 real-source baselines.

The new admission adapter and Preview/Export/Start/Direct Trainer wiring do not
exist yet. The 12 failures must identify exactly those absent contracts.
No xfail/skip, no side effects, no physical CUDA or credential issuance.
"""
from __future__ import annotations

import ast
from dataclasses import is_dataclass
import importlib
import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
    build_parameter_policy_execution_contract,
    parameter_policy_execution_blockers,
)
from mikazuki.parameter_policy_readiness import ExecutionReadiness

MARKER = "QG-1B3-QG2 ADMISSION CONTRACT ABSENT"
ROOT = Path(__file__).resolve().parents[1]
ADAPTER = "mikazuki.parameter_policy_admission"


def _source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def _source_fn(path, func):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    node = next(
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == func
    )
    return ast.get_source_segment(source, node) or "", node


def _called_names(node):
    return {
        sub.func.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }


def _policy(train_type="sd-lora", optimizer="AdamW"):
    component_ids = get_model_component_profile(train_type).components
    selected = (
        "mmdit.attention.adapter" if train_type == "sd3-lora"
        else "unet.attention.adapter"
    )
    if selected not in component_ids:
        raise AssertionError("Invalid test fixture target")
    components = {name: {"train": False} for name in component_ids}
    components[selected] = {
        "train": True, "optimizer_profile": "main", "learning_rate": 1e-4,
    }
    return {
        "version": 1,
        "optimizer_profiles": {"main": {"type": optimizer, "args": {}}},
        "components": components,
    }


def _config(**kw):
    return {"full_bf16": True, "mixed_precision": "bf16", **kw}


def _pending():
    return ExecutionFeatureQualification(
        "pending", "Test-only missing release evidence", None
    )


class QG1B3QG2TargetedRed(unittest.TestCase):
    def _adapter(self):
        if importlib.util.find_spec(ADAPTER) is None:
            self.fail(f"{MARKER}: first-party admission module is absent")
        module = importlib.import_module(ADAPTER)
        func = getattr(module, "assess_component_training_admission", None)
        if not callable(func):
            self.fail(f"{MARKER}: admission assessment entrypoint is absent")
        return module, func

    def _assess(self, policy=None, *, train_type="sd-lora", config=None,
                purpose="preview", experimental_ack=False, **kw):
        _module, func = self._adapter()
        return func(
            _policy(train_type) if policy is None else policy,
            train_type=train_type,
            effective_config=_config() if config is None else config,
            purpose=purpose, experimental_ack=experimental_ack, **kw,
        )

    def test_b301_immutable_admission_source_contract_without_caller_authority(self):
        module, fn = self._adapter()
        params = inspect.signature(fn).parameters
        self.assertEqual(
            list(params),
            ["canonical_policy", "train_type", "effective_config",
             "purpose", "experimental_ack"],
        )
        self.assertIs(params["experimental_ack"].default, False)
        self.assertNotIn("**", str(inspect.signature(fn)))
        kind = getattr(module, "ComponentAdmission", None)
        if kind is None or not is_dataclass(kind) or not kind.__dataclass_params__.frozen:
            self.fail(f"{MARKER}: frozen ComponentAdmission is absent")
        result = self._assess()
        for name in ("status", "purpose", "readiness", "runtime_blockers",
                     "warnings", "requires_experimental_ack", "host_admissible"):
            self.assertTrue(hasattr(result, name), name)
        self.assertIsInstance(result.readiness, ExecutionReadiness)

    def test_b302_existing_qualified_sd1_is_still_host_admissible_without_ack(self):
        r = self._assess(purpose="start")
        self.assertEqual(r.status, "legacy_approved")
        self.assertEqual(r.readiness.capability, "implemented")
        self.assertEqual(r.readiness.evidence_tier, "unverified")
        self.assertTrue(r.host_admissible)
        self.assertFalse(r.requires_experimental_ack)
        self.assertFalse(r.runtime_blockers)

    def test_b303_pending_only_certificate_is_warning_and_preview_ack_required(self):
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            r = self._assess(purpose="preview")
        self.assertEqual(r.status, "ack_required")
        self.assertTrue(r.requires_experimental_ack)
        self.assertFalse(r.host_admissible)
        self.assertEqual(r.readiness.capability, "implemented")
        self.assertFalse(r.readiness.hard_errors)
        self.assertEqual(r.readiness.evidence_tier, "unverified")
        self.assertIn(
            "qualification.backend.pending",
            {x.code for x in r.readiness.warnings},
        )
        self.assertTrue(r.warnings)
        self.assertTrue(r.runtime_blockers)

    def test_b304_start_requires_explicit_ack_for_pending_supported_candidate(self):
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            blocked = self._assess(purpose="start", experimental_ack=False)
            approved_host = self._assess(purpose="start", experimental_ack=True)
        self.assertEqual(blocked.status, "ack_required")
        self.assertFalse(blocked.host_admissible)
        self.assertEqual(approved_host.status, "experimental_host_candidate")
        self.assertTrue(approved_host.host_admissible)
        self.assertFalse(approved_host.requires_experimental_ack)
        self.assertEqual(approved_host.readiness.evidence_tier, "unverified")
        self.assertFalse(approved_host.runtime_blockers)
        self.assertTrue(approved_host.warnings)

    def test_b305_fake_qualification_and_user_claims_cannot_authorize_sd3(self):
        forged = ExecutionFeatureQualification(
            "qualified", "fake worker", "fake:evidence-id"
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd3-lora": forged}):
            r = self._assess(
                train_type="sd3-lora", purpose="direct_trainer",
                experimental_ack=True,
                config=_config(
                    can_start=True, decision="allow", capability="implemented",
                    evidence_tier="formally_qualified", physical_cuda_qualified=True,
                    claimed_warnings={"runtime.dtype.mismatch": "warning"},
                ),
            )
        self.assertEqual(r.status, "blocked")
        self.assertFalse(r.host_admissible)
        self.assertEqual(r.readiness.decision, "block")
        self.assertIn(
            "capability.backend.audit_required",
            {x.code for x in r.readiness.hard_errors},
        )

    def test_b306_ack_never_turns_bad_precision_or_pseudo_bool_into_allow(self):
        for cfg, code in (
            (_config(mixed_precision="fp16"), "runtime.bf16.precision_mismatch"),
            (_config(full_bf16="false"), "runtime.bool.invalid"),
        ):
            with self.subTest(cfg=cfg):
                r = self._assess(
                    config=cfg, purpose="start", experimental_ack=True
                )
                self.assertEqual(r.status, "blocked")
                self.assertFalse(r.host_admissible)
                self.assertIn(
                    code, {x.code for x in r.readiness.hard_errors}
                )

    def test_b307_ack_never_overrides_sdxl_ownership_risk_or_planned_optimizer(self):
        risky = self._assess(
            train_type="sdxl-lora",
            config=_config(gradient_checkpointing=True),
            purpose="start", experimental_ack=True,
        )
        self.assertEqual(risky.status, "blocked")
        self.assertFalse(risky.host_admissible)
        self.assertIn(
            "runtime.sdxl.gradient_checkpointing_unreviewed",
            {x.code for x in risky.readiness.hard_errors},
        )
        planned = self._assess(
            policy=_policy(optimizer="pytorch_optimizer.CAME"),
            purpose="start", experimental_ack=True,
        )
        self.assertEqual(planned.status, "blocked")
        self.assertIn(
            "capability.optimizer.not_implemented",
            {x.code for x in planned.readiness.hard_errors},
        )

    def test_b308_inactive_standard_and_strict_explicit_ack_inputs(self):
        for cfg in (_config(full_bf16=False), {"mixed_precision": "bf16"}):
            r = self._assess(config=cfg, purpose="start")
            self.assertEqual(r.status, "inactive")
            self.assertIsNone(r.readiness)
            self.assertTrue(r.host_admissible)
            self.assertFalse(r.warnings)
            self.assertFalse(r.runtime_blockers)
        with self.assertRaises(ValueError):
            self._assess(experimental_ack="true")
        with self.assertRaises(ValueError):
            self._assess(purpose="unrecognized")
        with self.assertRaises(TypeError):
            self._assess(
                can_start=True, capability="implemented",
                warnings=[], evidence_tier="formally_qualified",
            )

    def test_b309_legacy_runtime_gate_uses_trusted_source_without_dropping_safety(self):
        _code, fn = _source_fn(
            "mikazuki/parameter_policy.py", "parameter_policy_runtime_blockers"
        )
        required = {
            "assess_component_training_admission",
            "parameter_policy_v1_semantic_blockers",
            "_non_anima_policy_structure_and_target_blockers",
            "_non_anima_text_encoder_cache_blockers",
        }
        missing = required - _called_names(fn)
        if missing:
            self.fail(
                f"{MARKER}: old certificate-only gate not replaced with "
                f"trusted admission + genuine semantic checks: {sorted(missing)}"
            )

    def test_b310_request_uses_post_override_admission_and_ephemeral_intent(self):
        source, fn = _source_fn(
            "mikazuki/training_request.py", "prepare_request_config"
        )
        prepared = _source("mikazuki/training_config.py")
        if (
            "assess_component_training_admission" not in _called_names(fn)
            or "experimental_full_bf16_ack" not in source
            or "admission_intent" not in prepared
        ):
            self.fail(
                f"{MARKER}: Request lacks post-override first-party "
                "admission or separately carried per-job acknowledgement"
            )
        self.assertLess(
            source.index("assess_component_training_admission"),
            source.index("materialize_sidecars(prepared.sidecars)"),
        )

    def test_b311_preview_export_start_expose_typed_admission_and_keep_asset_order(self):
        api = "mikazuki/app/training_api.py"
        payload, _ = _source_fn(api, "_prepared_payload")
        preview, _ = _source_fn(api, "preview_training_config")
        export, _ = _source_fn(api, "export_training_config")
        start, _ = _source_fn(api, "create_toml_file")
        if (
            "admission" not in payload
            or "prepare_request_config" not in preview
            or "prepare_request_config" not in export
            or "prepare_request_config" not in start
            or "admission_intent" not in start
        ):
            self.fail(
                f"{MARKER}: Preview/Export/Start lack a shared typed "
                "admission/acknowledgement boundary"
            )
        self.assertLess(
            start.index("validate_prepared_config(prepared, True"),
            start.index("materialize_sidecars(prepared.sidecars)"),
        )

    def test_b312_direct_trainer_uses_same_admission_and_explicit_cli_intent(self):
        code, fn = _source_fn(
            "mikazuki/parameter_policy_trainer.py",
            "create_parameter_policy_session",
        )
        if (
            "assess_component_training_admission" not in _called_names(fn)
            or "experimental_full_bf16_ack" not in code
            or "load_parameter_policy_file" not in code
        ):
            self.fail(
                f"{MARKER}: Direct Trainer has no shared first-party "
                "admission / explicit direct-CLI acknowledgement"
            )
        entire = _source("mikazuki/parameter_policy_trainer.py")
        for name in (
            "assert_execution_dtype_contract",
            "assert_live_root_identity_contract",
            "register_checkpoint_manifest",
        ):
            self.assertIn(name, entire)
        self.assertLess(
            code.index("load_parameter_policy_file"), code.index("scan_roots:")
        )


class QG1B3QG2ExistingBaselines(unittest.TestCase):
    def test_b901_current_sd1_qualified_d1_source_execution_passes(self):
        policy = _policy()
        self.assertEqual(
            parameter_policy_execution_blockers(
                policy, train_type="sd-lora",
                effective_config=_config(), require_assets=False,
            ), [],
        )
        contract = build_parameter_policy_execution_contract(
            train_type="sd-lora", effective_config=_config()
        )
        self.assertEqual(contract.features, ("full_bf16",))
        self.assertEqual(
            contract.execution_identity()["train_type"], "sd-lora"
        )

    def test_b902_current_sd3_pending_gate_is_still_closed(self):
        blockers = parameter_policy_execution_blockers(
            _policy("sd3-lora"), train_type="sd3-lora",
            effective_config=_config(), require_assets=False,
        )
        self.assertEqual(len(blockers), 1, blockers)
        self.assertIn("pending", blockers[0])

    def test_b903_inactive_and_quoted_false_legacy_semantics_unchanged(self):
        self.assertEqual(
            parameter_policy_execution_blockers(
                _policy(), train_type="sd-lora",
                effective_config=_config(full_bf16=False), require_assets=False,
            ), [],
        )
        errors = parameter_policy_execution_blockers(
            _policy(), train_type="sd-lora",
            effective_config=_config(full_bf16="false"), require_assets=False,
        )
        self.assertTrue(errors)
        self.assertIn("full_bf16", errors[0])
        self.assertIn("boolean", errors[0])

    def test_b904_current_start_and_direct_gates_precede_any_materialization(self):
        request = _source("mikazuki/training_request.py")
        self.assertLess(
            request.index("if launch and blockers:"),
            request.index("if materialize:"),
        )
        start, _ = _source_fn(
            "mikazuki/app/training_api.py", "create_toml_file"
        )
        self.assertLess(
            start.index("validate_prepared_config(prepared, True"),
            start.index("materialize_sidecars(prepared.sidecars)"),
        )
        direct, _ = _source_fn(
            "mikazuki/parameter_policy_trainer.py",
            "create_parameter_policy_session",
        )
        self.assertLess(
            direct.index("parameter_policy_execution_blockers"),
            direct.index("scan_roots:"),
        )


if __name__ == "__main__":
    unittest.main()
