"""QG-1B2-SR1 D1 reference-shape positive/negative host-only regression.

A matching D1 shape is NOT proof of this job's local model SHA, exact-head,
optimizer state or Checkpoint lineage. No asset or GPU operation occurs.
"""
from __future__ import annotations

from copy import deepcopy
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy_evidence_coverage import (
    EvidenceCoverage,
    match_full_bf16_evidence_coverage,
)
from mikazuki.parameter_policy_readiness_producer import (
    classify_component_full_bf16_readiness,
)
from tools.parameter_policy_backend_feature_gpu_support import (
    SD_LORA_FULL_BF16_CASE_IDS,
    SDXL_LORA_FULL_BF16_CASE_IDS,
)


def _case(backend: str, muon: bool = False):
    all_ids = get_model_component_profile(backend).components
    required = {
        "unet.attention.adapter",
        "unet.feed_forward.adapter",
        "unet.conv.adapter",
        *(
            ("text_encoder.adapter",) if backend == "sd-lora"
            else ("text_encoder_1.adapter", "text_encoder_2.adapter")
        ),
    }
    policy = {
        "version": 1,
        "optimizer_profiles": (
            {"muon": {"type": "Muon", "args": {}},
             "fallback": {"type": "AdamW", "args": {}}}
            if muon else {"main": {"type": "AdamW", "args": {}}}
        ),
        "components": {},
    }
    for key in all_ids:
        if key not in required:
            policy["components"][key] = {"train": False}
            continue
        route = {
            "train": True,
            "optimizer_profile": "muon" if muon else "main",
            "learning_rate": 1e-4,
        }
        if muon and key == "unet.conv.adapter":
            route["fallback_optimizer_profile"] = "fallback"
            route["fallback_learning_rate"] = 1e-5
        policy["components"][key] = route
    cfg = {
        "full_bf16": True,
        "mixed_precision": "bf16",
        "network_module": "networks.lora",
        "network_dim": 4,
        "network_alpha": 4,
        "network_args": ["conv_dim=4", "conv_alpha=4"],
        "resolution": "512,512",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 2 if muon else 1,
        "save_model_as": "safetensors",
    }
    return policy, cfg


class QG1B2D1CoverageHost(unittest.TestCase):
    def test_d1_sd1_adamw_positive_reference_shape(self):
        p, c = _case("sd-lora")
        matched = match_full_bf16_evidence_coverage(
            train_type="sd-lora", canonical_policy=p, effective_config=c,
        )
        self.assertIsInstance(matched, EvidenceCoverage)
        self.assertTrue(matched.covered, matched.reasons)
        self.assertEqual(matched.case_id, SD_LORA_FULL_BF16_CASE_IDS[0])
        self.assertFalse(matched.reasons)

    def test_d1_sdxl_adamw_positive_reference_shape(self):
        p, c = _case("sdxl-lora")
        matched = match_full_bf16_evidence_coverage(
            train_type="sdxl-lora", canonical_policy=p, effective_config=c,
        )
        self.assertTrue(matched.covered, matched.reasons)
        self.assertEqual(matched.case_id, SDXL_LORA_FULL_BF16_CASE_IDS[0])

    def test_d1_sd1_muon_fallback_positive_reference_shape(self):
        p, c = _case("sd-lora", muon=True)
        matched = match_full_bf16_evidence_coverage(
            train_type="sd-lora", canonical_policy=p, effective_config=c,
        )
        self.assertTrue(matched.covered, matched.reasons)
        self.assertEqual(matched.case_id, SD_LORA_FULL_BF16_CASE_IDS[1])

    def test_d1_sdxl_muon_fallback_positive_reference_shape(self):
        p, c = _case("sdxl-lora", muon=True)
        matched = match_full_bf16_evidence_coverage(
            train_type="sdxl-lora", canonical_policy=p, effective_config=c,
        )
        self.assertTrue(matched.covered, matched.reasons)
        self.assertEqual(matched.case_id, SDXL_LORA_FULL_BF16_CASE_IDS[1])

    def test_missing_reference_fields_cannot_match(self):
        p, c = _case("sdxl-lora")
        c.clear()
        matched = match_full_bf16_evidence_coverage(
            train_type="sdxl-lora", canonical_policy=p, effective_config=c,
        )
        self.assertFalse(matched.covered)
        self.assertIsNone(matched.case_id)
        self.assertIn("network.rank_unconfirmed", matched.reasons)
        self.assertIn("dataset.resolution_unconfirmed", matched.reasons)

    def test_variant_and_target_changes_fail_shape_match_without_hard_errors(self):
        p, c = _case("sdxl-lora")
        p["components"]["text_encoder_2.adapter"] = {"train": False}
        c["network_dim"] = 8
        c["resolution"] = "1024,1024"
        matched = match_full_bf16_evidence_coverage(
            train_type="sdxl-lora", canonical_policy=p, effective_config=c,
        )
        self.assertFalse(matched.covered)
        self.assertIn("components.outside_d1_scope", matched.reasons)
        self.assertIn("network.rank_unconfirmed", matched.reasons)
        self.assertIn("dataset.resolution_unconfirmed", matched.reasons)

    def test_positive_d1_shape_still_does_not_mint_formal_gpu_proof(self):
        p, c = _case("sdxl-lora")
        report = classify_component_full_bf16_readiness(
            p, train_type="sdxl-lora", effective_config=c,
        )
        self.assertEqual(report.capability, "implemented")
        self.assertEqual(report.decision, "warn")
        self.assertTrue(report.can_start)
        self.assertFalse(report.hard_errors)
        self.assertEqual(report.evidence_tier, "unverified")
        self.assertIn(
            "qualification.variant.identity_unverified",
            {item.code for item in report.warnings},
        )

    def test_case_input_and_identity_not_mutated(self):
        p, c = _case("sd-lora")
        p0, c0 = deepcopy(p), deepcopy(c)
        report = classify_component_full_bf16_readiness(
            p, train_type="sd-lora", effective_config=c,
        )
        self.assertEqual(report.evidence_tier, "unverified")
        self.assertEqual(p, p0)
        self.assertEqual(c, c0)

    def test_real_implementation_guards_stay_separate_from_evidence_coverage(self):
        p, c = _case("sdxl-lora")
        c["gradient_checkpointing"] = True
        cover = match_full_bf16_evidence_coverage(
            train_type="sdxl-lora", canonical_policy=p, effective_config=c,
        )
        self.assertFalse(cover.covered)
        report = classify_component_full_bf16_readiness(
            p, train_type="sdxl-lora", effective_config=c,
        )
        self.assertEqual(report.decision, "block")
        self.assertIn(
            "runtime.sdxl.gradient_checkpointing_unreviewed",
            {item.code for item in report.hard_errors},
        )

    def test_unknown_backend_never_claims_d1_coverage(self):
        p, c = _case("sd-lora")
        matched = match_full_bf16_evidence_coverage(
            train_type="sd3-lora", canonical_policy=p, effective_config=c,
        )
        self.assertFalse(matched.covered)
        self.assertEqual(matched.reasons, ("backend.outside_d1_scope",))


if __name__ == "__main__":
    unittest.main()
