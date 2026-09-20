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

    def test_preloaded_lora_with_text_encoder_cache_is_fail_closed(self):
        affected = ("sdxl-lora", "flux-lora", "chroma-lora", "sd3-lora")
        for train_type in affected:
            with self.subTest(train_type=train_type, cache="memory"):
                blockers = parameter_policy_v1_semantic_blockers(
                    {
                        "network_weights": "existing.safetensors",
                        "cache_text_encoder_outputs": True,
                    },
                    train_type,
                )
                self.assertTrue(
                    any("network_weights" in item and "Text Encoder" in item for item in blockers),
                    blockers,
                )

            with self.subTest(train_type=train_type, cache="disk"):
                blockers = parameter_policy_v1_semantic_blockers(
                    {
                        "network_weights": "existing.safetensors",
                        "cache_text_encoder_outputs": False,
                        "cache_text_encoder_outputs_to_disk": True,
                    },
                    train_type,
                )
                self.assertTrue(
                    any("network_weights" in item and "Text Encoder" in item for item in blockers),
                    blockers,
                )

    def test_preloaded_lora_cache_blocker_does_not_overreach(self):
        cases = (
            (
                "sdxl-lora",
                {
                    "network_weights": "",
                    "cache_text_encoder_outputs": True,
                },
            ),
            (
                "flux-lora",
                {
                    "network_weights": "existing.safetensors",
                    "cache_text_encoder_outputs": False,
                    "cache_text_encoder_outputs_to_disk": False,
                },
            ),
            (
                "sd-lora",
                {
                    "network_weights": "existing.safetensors",
                    "cache_text_encoder_outputs": True,
                },
            ),
        )
        for train_type, config in cases:
            with self.subTest(train_type=train_type, config=config):
                blockers = parameter_policy_v1_semantic_blockers(config, train_type)
                self.assertFalse(
                    any("network_weights" in item and "Text Encoder" in item for item in blockers),
                    blockers,
                )

    def test_stable_network_trainer_owns_component_runtime_without_legacy_fallback(self):
        source = (ROOT / "scripts" / "stable" / "train_network.py").read_text(encoding="utf-8")
        self.assertIn('return "sdxl-lora" if self.is_sdxl else "sd-lora"', source)
        self.assertIn("parameter_policy_runtime_enabled=parameter_policy_train_type is not None", source)
        self.assertIn('roots={"network": network}', source)
        self.assertIn("optimizer = parameter_policy_session.optimizer", source)
        self.assertIn("parameter_policy_session.build_scheduler", source)
        self.assertIn("parameter_policy_session.finalize_after_prepare", source)
        self.assertIn('phase="post_resume"', source)
        self.assertIn('phase="epoch_start"', source)
        self.assertIn("parameter_policy_session.trainable_parameters", source)

        component_branch = source[source.index("if parameter_policy_session is None:") :]
        self.assertIn("network.prepare_optimizer_params", component_branch)
        self.assertIn("train_util.get_optimizer", component_branch)
        self.assertIn("else:\n            lr_descriptions = None", component_branch)

    def test_dev_component_metadata_lr_is_always_initialized(self):
        source = (ROOT / "scripts" / "dev" / "train_network.py").read_text(encoding="utf-8")
        marker = 'optimizer_name = "DTSParameterPolicy"'
        branch = source[source.rfind("else:", 0, source.index(marker)):source.index("# prepare dataloader")]
        self.assertIn("text_encoder_lr = None", branch)
        self.assertIn('"ss_text_encoder_lr": text_encoder_lr', source)

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

    def test_component_te_cache_is_staged_as_full_cache_before_routing(self):
        flux = (ROOT / "scripts" / "dev" / "flux_train_network.py").read_text(encoding="utf-8")
        sd3 = (ROOT / "scripts" / "dev" / "sd3_train_network.py").read_text(encoding="utf-8")

        self.assertIn("and args.cache_text_encoder_outputs", flux)
        self.assertIn("self.train_clip_l = False", flux)
        self.assertIn("self.train_t5xxl = False", flux)

        self.assertIn("and args.cache_text_encoder_outputs", sd3)
        self.assertIn("self.train_clip_l = False", sd3)
        self.assertIn("self.train_clip_g = False", sd3)
        self.assertIn("self.train_t5xxl = False", sd3)

    def test_component_logging_and_metadata_do_not_use_legacy_optimizer_identity(self):
        for path in (
            ROOT / "scripts" / "stable" / "train_network.py",
            ROOT / "scripts" / "dev" / "train_network.py",
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("parameter_policy_session.component_lr_logs", source)
                self.assertIn('optimizer_name = "DTSParameterPolicy"', source)
                self.assertIn(
                    "metadata.update(parameter_policy_session.model_metadata())",
                    source,
                )

    def test_request_gate_remains_closed_in_step6b(self):
        source = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")
        self.assertIn(
            "PARAMETER_POLICY_RUNTIME_TRAIN_TYPES: frozenset[str] = frozenset()",
            source,
        )


if __name__ == "__main__":
    unittest.main()
