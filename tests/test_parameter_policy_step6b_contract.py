from __future__ import annotations

import unittest
from pathlib import Path

from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers


ROOT = Path(__file__).resolve().parents[1]


class ParameterPolicyStep6BContractTests(unittest.TestCase):
    def test_lora_max_norm_is_fail_closed(self):
        for train_type in ("sd-lora", "sdxl-lora", "flux-lora", "chroma-lora", "sd3-lora"):
            with self.subTest(train_type=train_type):
                blockers = parameter_policy_v1_semantic_blockers(
                    {"scale_weight_norms": 1.0},
                    train_type,
                )
                self.assertTrue(
                    any("scale_weight_norms" in item for item in blockers),
                    blockers,
                )

    def test_stable_network_trainer_owns_component_runtime_without_legacy_fallback(self):
        source = (ROOT / "scripts" / "stable" / "train_network.py").read_text(encoding="utf-8")
        self.assertIn('return "sdxl-lora" if self.is_sdxl else "sd-lora"', source)
        self.assertIn("parameter_policy_runtime_enabled=parameter_policy_train_type is not None", source)
        self.assertIn('roots={"network": network}', source)
        self.assertIn("optimizer = parameter_policy_session.optimizer", source)
        self.assertIn("parameter_policy_session.build_scheduler", source)
        self.assertIn("parameter_policy_session.audit_after_prepare", source)
        self.assertIn("parameter_policy_session.register_checkpoint_manifest", source)
        self.assertIn("parameter_policy_session.trainable_parameters", source)
        self.assertIn("parameter_policy_session.assert_requires_grad_contract()", source)

        component_branch = source[source.index("if parameter_policy_session is None:") :]
        self.assertIn("network.prepare_optimizer_params", component_branch)
        self.assertIn("train_util.get_optimizer", component_branch)
        self.assertIn("else:\n            lr_descriptions = None", component_branch)

    def test_stable_sdxl_defers_cache_conflict_to_final_policy_flags(self):
        source = (ROOT / "scripts" / "stable" / "sdxl_train_network.py").read_text(encoding="utf-8")
        self.assertIn('if not str(getattr(args, "parameter_policy_config", "") or "").strip():', source)

    def test_dev_base_is_closed_and_only_flux_chroma_sd3_opt_in(self):
        base = (ROOT / "scripts" / "dev" / "train_network.py").read_text(encoding="utf-8")
        flux = (ROOT / "scripts" / "dev" / "flux_train_network.py").read_text(encoding="utf-8")
        sd3 = (ROOT / "scripts" / "dev" / "sd3_train_network.py").read_text(encoding="utf-8")

        self.assertIn("def get_parameter_policy_train_type(self, args):\n        return None", base)
        self.assertIn("has not integrated DTS Parameter Policy runtime", base)
        self.assertIn('return "chroma-lora" if args.model_type == "chroma" else "flux-lora"', flux)
        self.assertIn('return "sd3-lora"', sd3)

    def test_flux_and_sd3_use_policy_final_text_encoder_flags(self):
        flux = (ROOT / "scripts" / "dev" / "flux_train_network.py").read_text(encoding="utf-8")
        sd3 = (ROOT / "scripts" / "dev" / "sd3_train_network.py").read_text(encoding="utf-8")

        self.assertIn('session.trains_component("clip_l.adapter")', flux)
        self.assertIn('session.trains_component("t5xxl.adapter")', flux)
        self.assertIn('session.trains_prefix("transformer.")', flux)

        self.assertIn('session.trains_component("clip_l.adapter")', sd3)
        self.assertIn('session.trains_component("clip_g.adapter")', sd3)
        self.assertIn('session.trains_component("t5xxl.adapter")', sd3)
        self.assertIn('session.trains_prefix("mmdit.")', sd3)
        self.assertIn(
            "return [self.train_clip_l, self.train_clip_g, self.train_t5xxl]",
            sd3,
        )

    def test_component_logging_and_metadata_do_not_use_legacy_optimizer_identity(self):
        for path in (
            ROOT / "scripts" / "stable" / "train_network.py",
            ROOT / "scripts" / "dev" / "train_network.py",
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("parameter_policy_session.component_lr_logs", source)
                self.assertIn('optimizer_name = "DTSParameterPolicy"', source)
                self.assertIn('"ss_dts_parameter_policy_hash"', source)
                self.assertIn('"ss_dts_parameter_policy_topology"', source)
                self.assertIn('"ss_dts_parameter_policy_profiles"', source)

    def test_request_gate_remains_closed_in_step6b(self):
        source = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")
        self.assertIn(
            "PARAMETER_POLICY_RUNTIME_TRAIN_TYPES: frozenset[str] = frozenset()",
            source,
        )


if __name__ == "__main__":
    unittest.main()
