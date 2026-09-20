from __future__ import annotations

import unittest
from pathlib import Path

from mikazuki.full_trainer_contract import normalize_validate_sdxl_full
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers


ROOT = Path(__file__).resolve().parents[1]


class ParameterPolicyStep6CContractTests(unittest.TestCase):
    def test_torch_compile_is_fail_closed_for_component_runtime(self):
        for train_type in ("sd-dreambooth", "sdxl-finetune", "flux-finetune", "sd-lora"):
            with self.subTest(train_type=train_type):
                blockers = parameter_policy_v1_semantic_blockers(
                    {"torch_compile": True},
                    train_type,
                )
                self.assertTrue(any("torch_compile" in item for item in blockers), blockers)

    def test_dreambooth_dynamic_text_encoder_stop_remains_fail_closed(self):
        blockers = parameter_policy_v1_semantic_blockers(
            {"stop_text_encoder_training": 100},
            "sd-dreambooth",
        )
        self.assertTrue(any("stop_text_encoder_training" in item for item in blockers), blockers)

    def test_sdxl_full_component_mode_defers_text_encoder_cache_conflict(self):
        config = {
            "parameter_policy_config": "policy.json",
            "train_text_encoder": True,
            "cache_text_encoder_outputs": True,
        }
        normalize_validate_sdxl_full(config)

        with self.assertRaisesRegex(ValueError, "文本编码器"):
            normalize_validate_sdxl_full(
                {
                    "train_text_encoder": True,
                    "cache_text_encoder_outputs": True,
                }
            )

    def test_full_trainers_explicitly_opt_in_and_use_session_owned_clipping(self):
        cases = (
            ("scripts/stable/train_db.py", '"sd-dreambooth"'),
            ("scripts/stable/sdxl_train.py", '"sdxl-finetune"'),
            ("scripts/dev/flux_train.py", '"flux-finetune"'),
        )
        for relpath, train_type in cases:
            with self.subTest(path=relpath):
                source = (ROOT / relpath).read_text(encoding="utf-8")
                self.assertIn("parameter_policy_runtime_enabled=parameter_policy_requested", source)
                self.assertIn(train_type, source)
                self.assertIn("parameter_policy_session.optimizer", source)
                self.assertIn("parameter_policy_session.build_scheduler", source)
                self.assertIn("parameter_policy_session.audit_after_prepare", source)
                self.assertIn("parameter_policy_session.register_checkpoint_manifest", source)
                self.assertIn("parameter_policy_session.trainable_parameters", source)
                self.assertIn("parameter_policy_session.component_lr_logs", source)

    def test_dreambooth_component_mode_exits_dynamic_stop_state_machine(self):
        source = (ROOT / "scripts" / "stable" / "train_db.py").read_text(encoding="utf-8")
        self.assertIn(
            "if parameter_policy_session is None and global_step == args.stop_text_encoder_training:",
            source,
        )
        self.assertIn(
            "train_text_encoder\n                    if parameter_policy_session is not None",
            source,
        )

    def test_sdxl_structural_freeze_is_passed_before_runtime_build(self):
        source = (ROOT / "scripts" / "stable" / "sdxl_train.py").read_text(encoding="utf-8")
        self.assertIn("structural_frozen_parameters = tuple(", source)
        self.assertIn("text_encoder1.text_model.encoder.layers[-1].parameters()", source)
        self.assertIn("text_encoder1.text_model.final_layer_norm.parameters()", source)
        self.assertIn("structural_frozen_parameters=structural_frozen_parameters", source)

    def test_flux_unsafe_runtime_modes_remain_blocked(self):
        blockers = parameter_policy_v1_semantic_blockers(
            {
                "blockwise_fused_optimizers": True,
                "blocks_to_swap": 2,
                "cpu_offload_checkpointing": True,
            },
            "flux-finetune",
        )
        joined = "\n".join(blockers)
        self.assertIn("blockwise_fused_optimizers", joined)
        self.assertIn("blocks_to_swap", joined)
        self.assertIn("cpu_offload_checkpointing", joined)

    def test_step6c_does_not_open_global_start_gate(self):
        source = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")
        self.assertIn(
            "PARAMETER_POLICY_RUNTIME_TRAIN_TYPES: frozenset[str] = frozenset()",
            source,
        )


if __name__ == "__main__":
    unittest.main()
