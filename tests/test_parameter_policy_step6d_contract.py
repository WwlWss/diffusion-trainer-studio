from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_qwen_config import normalize_qwen_training_config
from mikazuki.parameter_policy import parameter_policy_runtime_blockers
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers


class ParameterPolicyStep6DContractTests(unittest.TestCase):
    def _policy(self, *, qwen_train=False, dit_train=True):
        components = {
            "dit.self_attention": {"train": dit_train, "optimizer_profile": "main", "learning_rate": 1e-4} if dit_train else {"train": False},
            "dit.cross_attention": {"train": False},
            "dit.mlp": {"train": False},
            "dit.modulation": {"train": False},
            "dit.llm_adapter": {"train": False},
            "dit.base_other": {"train": False},
            "qwen3": {"train": qwen_train, "optimizer_profile": "main", "learning_rate": 5e-7} if qwen_train else {"train": False},
        }
        return {
            "version": 1,
            "optimizer_profiles": {"main": {"type": "AdamW", "args": {}}},
            "components": components,
        }

    def test_component_qwen_normalizer_defers_lr_optimizer_and_cache_to_policy(self):
        config = {
            "parameter_policy_config": "policy.json",
            "train_qwen3_text_encoder": True,
            "qwen3_lr": 5e-7,
            "cache_text_encoder_outputs": True,
            "optimizer_type": "DefinitelyLegacyOnly",
        }
        self.assertTrue(normalize_qwen_training_config(config, "finetune"))
        self.assertNotIn("qwen3_lr", config)
        self.assertTrue(config["cache_text_encoder_outputs"])

    def test_host_blocks_qwen_train_without_target_permission(self):
        blockers = parameter_policy_runtime_blockers(
            self._policy(qwen_train=True),
            train_type="anima-finetune",
            effective_config={"parameter_policy_config": "policy.json"},
            integrated_train_types={"anima-finetune"},
        )
        self.assertTrue(any("target permission" in item for item in blockers), blockers)

    def test_host_blocks_qwen_train_with_text_encoder_cache(self):
        blockers = parameter_policy_runtime_blockers(
            self._policy(qwen_train=True),
            train_type="anima-finetune",
            effective_config={
                "parameter_policy_config": "policy.json",
                "train_qwen3_text_encoder": True,
                "cache_text_encoder_outputs_to_disk": True,
            },
            integrated_train_types={"anima-finetune"},
        )
        self.assertTrue(any("缓存 Text Encoder" in item for item in blockers), blockers)

    def test_host_blocks_qwen_only_component_training(self):
        blockers = parameter_policy_runtime_blockers(
            self._policy(qwen_train=True, dit_train=False),
            train_type="anima-finetune",
            effective_config={
                "parameter_policy_config": "policy.json",
                "train_qwen3_text_encoder": True,
            },
            integrated_train_types={"anima-finetune"},
        )
        self.assertTrue(any("Qwen3-only" in item for item in blockers), blockers)

    def test_anima_compile_is_fail_closed(self):
        for train_type in ("anima-lora", "anima-finetune"):
            with self.subTest(train_type=train_type):
                blockers = parameter_policy_v1_semantic_blockers(
                    {"compile": True},
                    train_type,
                )
                self.assertTrue(any("Anima compile" in item for item in blockers), blockers)

    def test_anima_preloaded_lora_text_cache_is_fail_closed(self):
        blockers = parameter_policy_v1_semantic_blockers(
            {
                "network_weights": "existing.safetensors",
                "cache_text_encoder_outputs": True,
            },
            "anima-lora",
        )
        self.assertTrue(any("network_weights" in item for item in blockers), blockers)

    def test_anima_base_weights_with_text_encoder_cache_is_fail_closed(self):
        for field in ("cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"):
            with self.subTest(cache_field=field):
                blockers = parameter_policy_v1_semantic_blockers(
                    {
                        "base_weights": ["qwen_lora.safetensors"],
                        field: True,
                    },
                    "anima-lora",
                )
                self.assertTrue(
                    any("base_weights" in item and "Text Encoder" in item for item in blockers),
                    blockers,
                )

    def test_anima_base_weights_cache_blocker_does_not_overreach(self):
        self.assertFalse(
            any(
                "base_weights" in item
                for item in parameter_policy_v1_semantic_blockers(
                    {
                        "base_weights": [],
                        "cache_text_encoder_outputs": True,
                    },
                    "anima-lora",
                )
            )
        )
        self.assertFalse(
            any(
                "base_weights" in item
                for item in parameter_policy_v1_semantic_blockers(
                    {
                        "base_weights": ["dit_only.safetensors"],
                        "cache_text_encoder_outputs": False,
                        "cache_text_encoder_outputs_to_disk": False,
                    },
                    "anima-lora",
                )
            )
        )

    def test_anima_integrations_are_present_in_step6f_matrix(self):
        from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES

        self.assertTrue(
            {"anima-lora", "anima-finetune"} <= PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
        )


if __name__ == "__main__":
    unittest.main()
