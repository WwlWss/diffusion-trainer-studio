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
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
    build_parameter_policy_execution_contract,
    parameter_policy_execution_blockers,
)
from mikazuki.parameter_policy_admission import assess_component_training_admission
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


def _maybe_source_fn(path, func):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    node = next(
        (
            n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == func
        ),
        None,
    )
    if node is None:
        return "", None
    return ast.get_source_segment(source, node) or "", node


def _source_method(path, class_name, method):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    owner = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n for n in owner.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == method
    )
    return ast.get_source_segment(source, node) or "", node


def _main_guard_source(path):
    source = _source(path)
    tree = ast.parse(source, filename=path)
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if (
            isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and any(
                isinstance(item, ast.Constant) and item.value == "__main__"
                for item in node.test.comparators
            )
        ):
            return ast.get_source_segment(source, node) or ""
    return ""


def _call_lines(node, name):
    lines = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        function = item.func
        if (
            isinstance(function, ast.Name) and function.id == name
        ) or (
            isinstance(function, ast.Attribute) and function.attr == name
        ):
            lines.append(item.lineno)
    return sorted(lines)


def _argument_names(node):
    return {
        item.arg
        for item in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
    }


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
        # Legacy permission is not proof that this particular model/target
        # matches the physical D1 vector. A fabricated release case ID must
        # never create an additional "legacy approved" exemption.
        forged = ExecutionFeatureQualification(
            "qualified", "Test-only wrong source case", "fake:other-case"
        )
        with patch.dict(
            FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": forged}
        ):
            fake = self._assess(purpose="start")
        self.assertNotEqual(fake.status, "legacy_approved")
        self.assertEqual(fake.readiness.evidence_tier, "unverified")

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
        # One test = one targeted RED. Still inspect BOTH negative cases once
        # the absent adapter is implemented; avoid two failing subTest records.
        cases = (
            (_config(mixed_precision="fp16"), "runtime.bf16.precision_mismatch"),
            (_config(full_bf16="false"), "runtime.bool.invalid"),
        )
        problems = []
        for cfg, code in cases:
            r = self._assess(
                config=cfg, purpose="start", experimental_ack=True
            )
            if (
                r.status != "blocked"
                or r.host_admissible
                or code not in {x.code for x in r.readiness.hard_errors}
            ):
                problems.append((cfg, code, r.status))
        if problems:
            self.fail(f"{MARKER}: explicit acknowledgement bypasses {problems!r}")

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
        missing = []
        admission_source = _source("mikazuki/parameter_policy_admission.py")
        if "class DirectTrainerAdmissionSnapshot" not in admission_source:
            missing.append("immutable Policy/config admission snapshot")
        direct_admission, direct_admission_fn = _maybe_source_fn(
            "mikazuki/parameter_policy_admission.py",
            "prepare_direct_trainer_admission",
        )
        if not direct_admission:
            missing.append("pure first-party Direct Trainer admission preparation")
        elif (
            "assess_component_training_admission" not in _called_names(direct_admission_fn)
            or 'purpose="direct_trainer"' not in direct_admission
            or "experimental_ack" not in direct_admission
            or "load_parameter_policy_sidecar" not in direct_admission
        ):
            missing.append("Direct Trainer purpose + strict acknowledgement + pure canonical Policy loader")
        policy_source = _source("mikazuki/parameter_policy.py")
        if (
            "def load_parameter_policy_sidecar" not in policy_source
            or "import torch" in policy_source
            or "from torch" in policy_source
        ):
            missing.append("shared torch-free canonical Policy sidecar loader/hash")

        # The B2 CLI/parser/TOML contract is independently GREEN in the
        # QG-2-B2 suite. B312 remains focused on the still-missing Admission
        # preflight, model-aware Session and Launcher parity.

        for trainer_path in (
            "scripts/stable/train_network.py",
            "scripts/dev/train_network.py",
        ):
            train_code, train_fn = _source_method(
                trainer_path, "NetworkTrainer", "train"
            )
            if "admission_intent" not in _argument_names(train_fn):
                missing.append(f"{trainer_path}: Trainer train() admission-intent parameter")
            preflights = _call_lines(train_fn, "preflight_component_training_admission")
            dataset = _call_lines(train_fn, "prepare_dataset_args")
            extra = _call_lines(train_fn, "assert_extra_args")
            accelerator = _call_lines(train_fn, "prepare_accelerator")
            if (
                len(preflights) < 2 or not dataset or not extra or not accelerator
                or not (preflights[0] < min(dataset))
                or not (preflights[1] > max(extra) and preflights[1] < min(accelerator))
            ):
                missing.append(
                    f"{trainer_path}: early Host gate plus final-config gate before Accelerator/model load"
                )

        # The early bridge remains torch-free and must not import the late,
        # model-aware trainer module.
        for bridge_path in (
            "scripts/stable/library/dts_parameter_policy_bridge.py",
            "scripts/dev/library/dts_parameter_policy_bridge.py",
        ):
            bridge_code, bridge_fn = _maybe_source_fn(
                bridge_path, "preflight_component_training_admission"
            )
            if (
                not bridge_code
                or "prepare_direct_trainer_admission" not in _called_names(bridge_fn)
                or "parameter_policy_trainer" in bridge_code
            ):
                missing.append(f"{bridge_path}: pure lazy Direct Trainer admission bridge")

        # Late Session must revalidate the exact snapshot before scanning live
        # roots, and retain the original semantic/assets/runtime contracts.
        session_code, session_fn = _source_fn(
            "mikazuki/parameter_policy_trainer.py",
            "create_parameter_policy_session",
        )
        session_args = _argument_names(session_fn)
        if not {"admission_intent", "admission_snapshot"}.issubset(session_args):
            missing.append("late Session typed intent + immutable preflight snapshot")
        if "assess_component_training_admission" not in _called_names(session_fn):
            missing.append("late Session first-party Direct Trainer Admission")
        policy_loader = next(
            (
                name for name in ("load_parameter_policy_file", "load_parameter_policy_sidecar")
                if name in session_code
            ),
            None,
        )
        if (
            policy_loader is None
            or "policy_hash" not in session_code
            or "effective_config_fingerprint" not in session_code
            or "admission_snapshot" not in session_code
            or "require_assets=True" not in session_code
            or "scan_parameter_roots" not in session_code
            or "purpose=\"direct_trainer\"" not in session_code
            or "experimental_ack" not in session_code
        ):
            missing.append("late Session Policy/config identity + Direct Trainer Admission + mandatory asset gate")
        elif not (
            session_code.index(policy_loader)
            < session_code.index("assess_component_training_admission")
            < session_code.index("scan_parameter_roots")
        ):
            missing.append("late Admission and identity comparison must precede live root scan")
        if (
            "parameter_policy_v1_semantic_blockers" not in _called_names(session_fn)
            or not (
                "parameter_policy_execution_blockers" in _called_names(session_fn)
                or "parameter_policy_runtime_blockers" in _called_names(session_fn)
            )
        ):
            missing.append("late Session semantic + hard execution/asset blockers")

        trainer_source = _source("mikazuki/parameter_policy_trainer.py")
        for name in (
            "assert_execution_dtype_contract",
            "assert_live_root_identity_contract",
            "assert_execution_environment_contract",
            "_audit_prepared_optimizer_and_device",
            "register_checkpoint_manifest",
            "validate_checkpoint_manifest",
        ):
            if name not in trainer_source:
                missing.append(f"preserve live runtime invariant: {name}")
        manifest_code, _manifest_fn = _source_method(
            "mikazuki/parameter_policy_trainer.py",
            "ParameterPolicyTrainerSession",
            "checkpoint_manifest",
        )
        if "experimental_full_bf16_ack" in manifest_code:
            missing.append("acknowledgement must never enter Checkpoint identity")

        # API -> Launcher -> Accelerate -> child CLI: per-job only. The flag
        # must follow trainer_file and must not enter env or task metadata.
        launcher_code, launcher_fn = _source_fn(
            "mikazuki/training_launcher.py", "run_prepared_train"
        )
        if (
            not launcher_code
            or "admission_intent" not in _argument_names(launcher_fn)
            or "--experimental_full_bf16_ack" not in launcher_code
            or "experimental_full_bf16_ack" not in launcher_code
            or "admission_intent.origin" not in launcher_code
            or "webui_start" not in launcher_code
            or "admission_intent.experimental_full_bf16_ack is True" not in launcher_code
        ):
            missing.append("typed strict-positive WebUI per-job Launcher intent and child CLI flag")
        elif "args = [" in launcher_code:
            args_start = launcher_code.index("args = [")
            trainer_index = launcher_code.find("trainer_file", args_start)
            flag_index = launcher_code.find('"--experimental_full_bf16_ack"', args_start)
            if trainer_index < 0 or flag_index < 0 or trainer_index > flag_index:
                missing.append("Launcher acknowledgement argument must follow trainer_file")
        if 'env["experimental_full_bf16_ack"]' in launcher_code:
            missing.append("Launcher must not transport acknowledgement through environment")
        for item in ast.walk(launcher_fn):
            if isinstance(item, ast.Dict):
                keys = {
                    key.value for key in item.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if "experimental_full_bf16_ack" in keys or "admission_intent" in keys:
                    missing.append("Launcher task metadata must not persist admission intent")
                    break

        api_start, _api_start_fn = _source_fn(
            "mikazuki/app/training_api.py", "create_toml_file"
        )
        if "admission_intent=prepared.admission_intent" not in api_start:
            missing.append("API must pass PreparedTrainingConfig intent to Launcher")
        if 'prepared.config["experimental_full_bf16_ack"]' in api_start:
            missing.append("API must not serialize acknowledgement into Trainer TOML")

        if missing:
            self.fail(
                f"{MARKER}: Direct Trainer / Launcher parity missing: "
                + "; ".join(dict.fromkeys(missing))
            )

        # Once the source contract exists, verify that the typed intent is
        # strict and that the actual first-party classifier retains D1 legacy,
        # pending acknowledgement and unsupported-backend distinctions.
        legacy = assess_component_training_admission(
            _policy(), train_type="sd-lora", effective_config=_config(),
            purpose="direct_trainer", experimental_ack=False,
        )
        self.assertEqual(legacy.status, "legacy_approved")
        with patch.dict(
            FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}
        ):
            no_ack = assess_component_training_admission(
                _policy(), train_type="sd-lora", effective_config=_config(),
                purpose="direct_trainer", experimental_ack=False,
            )
            with_ack = assess_component_training_admission(
                _policy(), train_type="sd-lora", effective_config=_config(),
                purpose="direct_trainer", experimental_ack=True,
            )
        self.assertEqual(no_ack.status, "ack_required")
        self.assertEqual(with_ack.status, "experimental_host_candidate")
        unsupported = assess_component_training_admission(
            _policy("sd3-lora"), train_type="sd3-lora",
            effective_config=_config(), purpose="direct_trainer",
            experimental_ack=True,
        )
        self.assertEqual(unsupported.status, "blocked")



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
        direct = assess_component_training_admission(
            policy, train_type="sd-lora", effective_config=_config(),
            purpose="direct_trainer", experimental_ack=False,
        )
        self.assertEqual(direct.status, "legacy_approved")
        self.assertTrue(direct.host_admissible)

    def test_b902_current_sd3_pending_gate_is_still_closed(self):
        blockers = parameter_policy_execution_blockers(
            _policy("sd3-lora"), train_type="sd3-lora",
            effective_config=_config(), require_assets=False,
        )
        self.assertEqual(len(blockers), 1, blockers)
        self.assertIn("pending", blockers[0])
        direct = assess_component_training_admission(
            _policy("sd3-lora"), train_type="sd3-lora",
            effective_config=_config(), purpose="direct_trainer",
            experimental_ack=True,
        )
        self.assertEqual(direct.status, "blocked")
        self.assertFalse(direct.host_admissible)

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
        # Standard mode must keep the torch-heavy Trainer behind the existing
        # opt-in bridge; importing either bridge alone stays lightweight.
        for bridge_path in (
            "scripts/stable/library/dts_parameter_policy_bridge.py",
            "scripts/dev/library/dts_parameter_policy_bridge.py",
        ):
            source = _source(bridge_path)
            tree = ast.parse(source, filename=bridge_path)
            top_level_imports = [
                node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            imported = ast.dump(ast.Module(body=top_level_imports, type_ignores=[]))
            self.assertNotIn("parameter_policy_trainer", imported)
            self.assertNotIn("torch", imported)

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
        session = _source("mikazuki/parameter_policy_trainer.py")
        for invariant in (
            "assert_execution_dtype_contract",
            "assert_live_root_identity_contract",
            "assert_execution_environment_contract",
            "_audit_prepared_optimizer_and_device",
            "register_checkpoint_manifest",
            "validate_checkpoint_manifest",
        ):
            self.assertIn(invariant, session)
        manifest, _ = _source_method(
            "mikazuki/parameter_policy_trainer.py",
            "ParameterPolicyTrainerSession",
            "checkpoint_manifest",
        )
        self.assertNotIn("experimental_full_bf16_ack", manifest)


if __name__ == "__main__":
    unittest.main()
