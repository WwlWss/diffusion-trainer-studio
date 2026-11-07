"""QG-1B3-B: typed certificate-only runtime migration, separate from QG-2."""
from __future__ import annotations

import ast
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from mikazuki.parameter_policy import parameter_policy_runtime_blockers
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
    FULL_BF16_OPTIMIZER_QUALIFICATIONS,
    parameter_policy_execution_blockers,
)
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from tests.test_parameter_policy_admission_red import _config, _policy

MARKER = "QG-1B3-B TYPED CERTIFICATE MIGRATION ABSENT"
ROOT = Path(__file__).resolve().parents[1]


def _run(*, backend="sd-lora", config=None, policy=None, require_assets=False):
    return parameter_policy_runtime_blockers(
        _policy(backend) if policy is None else policy,
        train_type=backend,
        effective_config=_config() if config is None else config,
        integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
        require_assets=require_assets,
    )


def _pending():
    return ExecutionFeatureQualification("pending", "test certificate pending", None)


def _runtime_owner_ast():
    source = (ROOT / "mikazuki" / "parameter_policy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "parameter_policy_runtime_blockers"
    )
    return fn


def _called_names(fn):
    return {
        node.func.id for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _assert_migrated(case):
    fn = _runtime_owner_ast()
    names = _called_names(fn)
    required = {
        "assess_component_training_admission",
        "parameter_policy_v1_semantic_blockers",
        "_non_anima_policy_structure_and_target_blockers",
        "_non_anima_text_encoder_cache_blockers",
    }
    if not required.issubset(names):
        case.fail(f"{MARKER}: runtime owner has no source-owned typed admission; missing={sorted(required - names)}")


class QG1B3BTypedMigrationRed(TestCase):
    def test_b3b101_owner_uses_source_owned_typed_admission(self):
        _assert_migrated(self)
        # No user-provided severity/can_start may become the authority.
        fn = _runtime_owner_ast()
        self.assertNotIn("getattr", _called_names(fn))

    def test_b3b102_pending_backend_requires_ack_not_certificate_hard_error(self):
        _assert_migrated(self)
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            blockers = _run()
        self.assertTrue(any("experimental" in b.lower() and "ack" in b.lower() for b in blockers), blockers)
        self.assertFalse(any("test certificate pending" in b for b in blockers), blockers)

    def test_b3b103_pending_referenced_optimizer_uses_ack_and_preserves_coverage(self):
        _assert_migrated(self)
        with patch.dict(FULL_BF16_OPTIMIZER_QUALIFICATIONS, {"AdamW": _pending()}):
            blockers = _run()
        self.assertTrue(any("experimental" in b.lower() and "ack" in b.lower() for b in blockers), blockers)
        self.assertFalse(any("test certificate pending" in b for b in blockers), blockers)

    def test_b3b104_wrong_release_id_never_becomes_legacy_approved(self):
        _assert_migrated(self)
        fake = ExecutionFeatureQualification("qualified", "forged", "fake:other-case")
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": fake}):
            blockers = _run()
        self.assertTrue(any("ack" in b.lower() for b in blockers), blockers)

    def test_b3b105_forged_sd3_release_cannot_overrule_missing_implementation_audit(self):
        _assert_migrated(self)
        fake = ExecutionFeatureQualification("qualified", "forged", "fake:other-case")
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd3-lora": fake}):
            blockers = _run(backend="sd3-lora")
        self.assertTrue(any("audit" in b.lower() or "审查" in b for b in blockers), blockers)

    def test_b3b106_real_d1_compatibility_and_inactive_mode_unchanged(self):
        _assert_migrated(self)
        self.assertEqual(_run(), [])
        self.assertEqual(_run(config=_config(full_bf16=False)), [])

    def test_b3b107_semantic_and_optimizer_hard_errors_never_swallowed(self):
        _assert_migrated(self)
        blocked = _run(config=_config(deepspeed=True))
        self.assertTrue(blocked)
        self.assertTrue(any("deepspeed" in x.lower() for x in blocked), blocked)
        bad = _run(config=_config(mixed_precision="fp16"))
        self.assertTrue(any("bf16" in x.lower() for x in bad), bad)

    def test_b3b108_no_config_ack_bypass_and_legacy_unaudited_pending_diagnostic(self):
        _assert_migrated(self)
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {"sd-lora": _pending()}):
            blockers = _run(config=_config(experimental_full_bf16_ack=True))
        self.assertTrue(any("ack" in b.lower() for b in blockers), blockers)
        unaudited = _run(backend="sd3-lora")
        self.assertTrue(any("pending" in b.lower() for b in unaudited), unaudited)
        self.assertTrue(any("audit" in b.lower() or "审查" in b for b in unaudited), unaudited)


class QG1B3BTrueExistingBaselines(TestCase):
    def test_b3b901_existing_d1_approved_sd_lora_is_legacy_green(self):
        self.assertEqual(_run(), [])
        self.assertEqual(parameter_policy_execution_blockers(
            _policy(), train_type="sd-lora",
            effective_config=_config(), require_assets=False,
        ), [])

    def test_b3b902_existing_sd3_unqualified_path_really_stops(self):
        self.assertTrue(_run(backend="sd3-lora"))
        self.assertTrue(any("pending" in x.lower() for x in _run(backend="sd3-lora")))

    def test_b3b903_inactive_stays_noop_but_quoted_false_fails_closed(self):
        self.assertEqual(_run(config=_config(full_bf16=False)), [])
        bad = _run(config=_config(full_bf16="false"))
        self.assertTrue(bad)
        self.assertTrue(any("full_bf16" in x.lower() for x in bad))

    def test_b3b904_semantic_and_referenced_optimizer_restrictions_exist(self):
        bad = _run(config=_config(deepspeed=True))
        self.assertTrue(any("deepspeed" in x.lower() for x in bad), bad)
        bad_precision = _run(config=_config(mixed_precision="fp16"))
        self.assertTrue(any("bf16" in x.lower() for x in bad_precision), bad_precision)
