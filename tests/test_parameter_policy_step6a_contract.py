from __future__ import annotations

import ast
import unittest
from pathlib import Path

from mikazuki.anima_finetune_config import validate_anima_finetune_config
from mikazuki.parameter_policy import parameter_policy_runtime_blockers
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers


ROOT = Path(__file__).resolve().parents[1]


def _policy():
    return {
        "version": 1,
        "optimizer_profiles": {
            "adam": {"type": "AdamW", "args": {}},
        },
        "components": {
            "transformer.double_stream": {
                "train": True,
                "optimizer_profile": "adam",
                "learning_rate": 1e-4,
            },
        },
    }


class ParameterPolicyStep6AContractTests(unittest.TestCase):
    def test_host_gate_is_backend_aware_but_still_closed(self):
        blockers = parameter_policy_runtime_blockers(
            _policy(),
            train_type="flux-finetune",
            effective_config={},
            integrated_train_types=(),
        )
        self.assertTrue(blockers)
        self.assertIn("flux-finetune", blockers[0])
        self.assertIn("尚未接入", blockers[0])

    def test_semantic_runtime_gate_rejects_optimizer_ownership_conflicts(self):
        for config, marker in (
            ({"deepspeed": True}, "DeepSpeed"),
            ({"fused_backward_pass": True}, "fused_backward_pass"),
            ({"blockwise_fused_optimizers": True}, "blockwise_fused_optimizers"),
            ({"blocks_to_swap": 2}, "blocks_to_swap"),
            ({"cpu_offload_checkpointing": True}, "cpu_offload_checkpointing"),
        ):
            with self.subTest(config=config):
                blockers = parameter_policy_v1_semantic_blockers(
                    config,
                    "flux-finetune",
                )
                self.assertTrue(any(marker in item for item in blockers), blockers)

    def test_anima_component_mode_does_not_require_legacy_global_lr(self):
        config = {"parameter_policy_config": "policy.json"}
        validate_anima_finetune_config(config, "finetune")

    def test_stable_and_dev_trainers_expose_one_policy_cli_path(self):
        for path in (
            ROOT / "scripts" / "stable" / "library" / "train_util.py",
            ROOT / "scripts" / "dev" / "library" / "train_util.py",
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertEqual(source.count('"--parameter_policy_config"'), 1)

    def test_trainer_bridges_are_lazy_and_do_not_import_torch_runtime_at_module_load(self):
        for path in (
            ROOT / "scripts" / "stable" / "library" / "dts_parameter_policy_bridge.py",
            ROOT / "scripts" / "dev" / "library" / "dts_parameter_policy_bridge.py",
        ):
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                top_imports = []
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        top_imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top_imports.append(node.module)
                self.assertNotIn("mikazuki.parameter_policy_trainer", top_imports)
                self.assertNotIn("torch", top_imports)

    def test_training_config_guards_legacy_adaptive_lr_normalization(self):
        source = (ROOT / "mikazuki" / "training_config.py").read_text(encoding="utf-8")
        self.assertIn("def _normalize_legacy_optimizer_learning_rates", source)
        self.assertIn('config.get("parameter_policy_config")', source)

    def test_step6a_does_not_open_request_launch_gate(self):
        source = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")
        self.assertIn("PARAMETER_POLICY_RUNTIME_TRAIN_TYPES: frozenset[str] = frozenset()", source)
        self.assertIn("effective_launch = launch and policy is None", source)


if __name__ == "__main__":
    unittest.main()
