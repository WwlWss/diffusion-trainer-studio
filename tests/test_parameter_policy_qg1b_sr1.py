"""QG-1B2-SR1: exact 12 targeted F1-F4 RED + four existing baselines.

Tests intentionally fail on the reviewed QG-1B2 producer until scoped
F1 evidence coverage, F2 assets boundary, F3 diagnostic aggregation and
F4 status semantics are implemented. This does NOT grant a Start or CUDA
qualification and it does not modify the old legacy gate.
"""
from __future__ import annotations

from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy_execution import (
    FULL_BF16_BACKEND_QUALIFICATIONS,
    FULL_BF16_OPTIMIZER_QUALIFICATIONS,
    ExecutionFeatureQualification,
    build_parameter_policy_execution_contract,
    parameter_policy_execution_blockers,
)
from mikazuki.parameter_policy_readiness_producer import (
    classify_component_full_bf16_readiness,
)
from tools.parameter_policy_backend_feature_gpu_support import (
    D1_STOCK_LORA_SPECS,
    SD_LORA_FULL_BF16_CASE_IDS,
    SDXL_LORA_FULL_BF16_CASE_IDS,
)

MARKER = "QG-1B2-SR1 CONTRACT NOT SATISFIED"


def _config(**kw):
    return {"full_bf16": True, "mixed_precision": "bf16", **kw}


def _policy(train_type="sd-lora", assigned=None, profiles=None):
    components = {
        name: {"train": False}
        for name in get_model_component_profile(train_type).components
    }
    if assigned is None:
        assigned = {"unet.attention.adapter": "main"}
    for component_id, profile in assigned.items():
        components[component_id] = {
            "train": True,
            "optimizer_profile": profile,
            "learning_rate": 0.0001,
        }
    return {
        "version": 1,
        "optimizer_profiles": profiles or {"main": {"type": "AdamW", "args": {}}},
        "components": components,
    }


def _classify(policy=None, *, train_type="sd-lora", config=None, **kw):
    return classify_component_full_bf16_readiness(
        _policy(train_type) if policy is None else policy,
        train_type=train_type,
        effective_config=_config() if config is None else config,
        **kw,
    )


def _codes(report, severity="warnings"):
    return {x.code for x in getattr(report, severity)}


def _assert_unverified_but_not_block(test, report, code="qualification.variant.outside_scope"):
    if (
        report.evidence_tier != "unverified"
        or report.decision != "warn"
        or not report.can_start
        or report.hard_errors
        or code not in _codes(report)
    ):
        test.fail(
            f"{MARKER}: source-audited but unproven variant must remain WARN "
            f"with unverified evidence and {code}"
        )


class QG1B2SR1TargetedRed(TestCase):
    def test_sr101_incomplete_host_reference_does_not_mint_formal_certification(self):
        _assert_unverified_but_not_block(self, _classify())

    def test_sr102_sdxl_te1_only_is_not_d1_formal_scope(self):
        policy = _policy(
            "sdxl-lora", {"text_encoder_1.adapter": "main"}
        )
        _assert_unverified_but_not_block(
            self, _classify(policy, train_type="sdxl-lora")
        )

    def test_sr103_nonreference_rank_and_resolution_are_unverified(self):
        cfg = _config(
            network_dim=8, network_alpha=8,
            resolution="768,768",
            gradient_accumulation_steps=1,
            network_module="networks.lora",
        )
        _assert_unverified_but_not_block(self, _classify(config=cfg))

    def test_sr104_nonreference_accumulation_is_unverified(self):
        cfg = _config(
            network_dim=4, network_alpha=4,
            resolution="512,512",
            gradient_accumulation_steps=3,
            network_module="networks.lora",
        )
        _assert_unverified_but_not_block(self, _classify(config=cfg))

    def test_sr105_muon_nonreference_ns_steps_is_unverified(self):
        profile = {
            "muon": {"type": "Muon", "args": {"ns_steps": 11}},
            "fallback": {"type": "AdamW", "args": {}},
        }
        policy = _policy(
            assigned={"unet.attention.adapter": "muon"},
            profiles=profile,
        )
        policy["components"]["unet.attention.adapter"].update({
            "fallback_optimizer_profile": "fallback",
            "fallback_learning_rate": 1e-5,
        })
        cfg = _config(
            network_dim=4, network_alpha=4, resolution="512,512",
            gradient_accumulation_steps=2,
            network_module="networks.lora",
            network_args=["conv_dim=4", "conv_alpha=4"],
        )
        _assert_unverified_but_not_block(self, _classify(policy, config=cfg))

    def test_sr106_active_asset_check_request_must_not_succeed_without_check(self):
        try:
            _classify(require_assets=True)
        except ValueError as exc:
            self.assertIn("asset", str(exc).lower())
        else:
            self.fail(
                f"{MARKER}: require_assets=True returned unverified asset success"
            )

    def test_sr107_multiple_optimizer_types_are_all_named_in_hard_diagnostics(self):
        policy = _policy(
            assigned={
                "unet.attention.adapter": "a",
                "unet.feed_forward.adapter": "b",
            },
            profiles={
                "a": {"type": "AdamW8bit", "args": {}},
                "b": {"type": "Lion", "args": {}},
            },
        )
        report = _classify(policy)
        messages = " ".join(x.message for x in report.hard_errors)
        if report.decision != "block" or not all(
            name in messages for name in ("AdamW8bit", "Lion", "a", "b")
        ):
            self.fail(f"{MARKER}: both referenced optimizer failures must remain")

    def test_sr108_multiple_pending_optimizers_both_named_in_warning(self):
        policy = _policy(
            assigned={
                "unet.attention.adapter": "a",
                "unet.feed_forward.adapter": "b",
            },
            profiles={
                "a": {"type": "AdamW", "args": {}},
                "b": {"type": "Muon", "args": {}},
            },
        )
        pending = ExecutionFeatureQualification("pending", "test only", None)
        with patch.dict(
            FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": pending, "Muon": pending},
        ):
            report = _classify(policy)
        messages = " ".join(x.message for x in report.warnings)
        if (
            report.decision != "warn"
            or not all(name in messages for name in ("AdamW", "Muon", "a", "b"))
            or "qualification.optimizer.pending" not in _codes(report)
        ):
            self.fail(f"{MARKER}: both pending optimizer profiles must be reported")

    def test_sr109_same_optimizer_two_profiles_are_both_named(self):
        policy = _policy(
            assigned={
                "unet.attention.adapter": "first",
                "unet.feed_forward.adapter": "second",
            },
            profiles={
                "first": {"type": "AdamW8bit", "args": {}},
                "second": {"type": "AdamW8bit", "args": {}},
            },
        )
        report = _classify(policy)
        text = " ".join(x.message for x in report.hard_errors)
        if not all(label in text for label in ("first", "second", "AdamW8bit")):
            self.fail(f"{MARKER}: two same-type optimizer profiles were collapsed")

    def test_sr110_sd3_requires_review_but_is_not_marked_unimplemented(self):
        policy = _policy("sd3-lora", {
            "mmdit.attention.adapter": "main"
        })
        report = _classify(policy, train_type="sd3-lora")
        if (
            report.capability != "experimental_candidate"
            or report.decision != "block"
            or report.can_start
            or "capability.backend.audit_required" not in _codes(report, "hard_errors")
        ):
            self.fail(f"{MARKER}: audited-needed SD3 is blocked experimental_candidate")

    def test_sr111_sd2_v2_requires_review_but_is_not_marked_unimplemented(self):
        report = _classify(config=_config(v2=True))
        if (
            report.capability != "experimental_candidate"
            or report.decision != "block"
            or report.can_start
            or "capability.backend.audit_required" not in _codes(report, "hard_errors")
        ):
            self.fail(f"{MARKER}: SD2 v2 must remain review-blocked, not absent")

    def test_sr112_known_incompatible_wins_over_planned_optimizer(self):
        policy = _policy(
            "sdxl-lora",
            profiles={"main": {"type": "pytorch_optimizer.CAME", "args": {}}},
        )
        report = _classify(
            policy, train_type="sdxl-lora",
            config=_config(gradient_checkpointing=True),
        )
        if (
            report.capability != "known_incompatible"
            or report.decision != "block"
            or report.can_start
            or not {
                "runtime.sdxl.gradient_checkpointing_unreviewed",
                "capability.optimizer.not_implemented",
            }.issubset(_codes(report, "hard_errors"))
        ):
            self.fail(
                f"{MARKER}: known incompatible must survive multiple hard reasons"
            )


class QG1B2SR1ExistingBaselines(TestCase):
    def test_sr901_original_d1_positive_case_identifiers_remain_registered(self):
        self.assertEqual(
            D1_STOCK_LORA_SPECS["sd-lora"]["case_ids"],
            SD_LORA_FULL_BF16_CASE_IDS,
        )
        self.assertEqual(
            D1_STOCK_LORA_SPECS["sdxl-lora"]["case_ids"],
            SDXL_LORA_FULL_BF16_CASE_IDS,
        )
        self.assertEqual(
            D1_STOCK_LORA_SPECS["sdxl-lora"]["text_components"],
            ("text_encoder_1.adapter", "text_encoder_2.adapter"),
        )

    def test_sr902_inactive_full_bf16_is_noop(self):
        self.assertIsNone(_classify(config=_config(full_bf16=False)))

    def test_sr903_genuinely_planned_came_remains_blocked(self):
        report = _classify(
            _policy(profiles={
                "main": {"type": "pytorch_optimizer.CAME", "args": {}}
            })
        )
        self.assertEqual(report.decision, "block")
        self.assertFalse(report.can_start)
        self.assertIn(
            "capability.optimizer.not_implemented", _codes(report, "hard_errors")
        )

    def test_sr904_original_execution_identity_and_certificate_rows_unchanged(self):
        policy = _policy()
        cfg = _config()
        before_policy = deepcopy(policy)
        before_cfg = deepcopy(cfg)
        before_backend = deepcopy(FULL_BF16_BACKEND_QUALIFICATIONS)
        before_optimizer = deepcopy(FULL_BF16_OPTIMIZER_QUALIFICATIONS)
        contract = build_parameter_policy_execution_contract(
            train_type="sd-lora", effective_config=cfg
        )
        identity = contract.execution_identity()
        signature = contract.execution_signature()
        _classify(policy, config=cfg)
        self.assertEqual(contract.execution_identity(), identity)
        self.assertEqual(contract.execution_signature(), signature)
        self.assertEqual(before_policy, policy)
        self.assertEqual(before_cfg, cfg)
        self.assertEqual(before_backend, FULL_BF16_BACKEND_QUALIFICATIONS)
        self.assertEqual(before_optimizer, FULL_BF16_OPTIMIZER_QUALIFICATIONS)
        legacy = parameter_policy_execution_blockers(
            _policy("sd3-lora", {"mmdit.attention.adapter": "main"}),
            train_type="sd3-lora", effective_config=_config(), require_assets=False,
        )
        self.assertEqual(len(legacy), 1)
        self.assertIn("pending", legacy[0])


if __name__ == "__main__":
    unittest.main()
