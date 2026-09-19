from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "mikazuki/parameter_policy_torch.py"


class ParameterPolicyTorchSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SOURCE_PATH))

    def test_runtime_isolated_from_trainer_integration(self):
        for forbidden in (
            "training_request",
            "training_launcher",
            "accelerator.prepare",
            "requires_grad_",
            ".requires_grad =",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)

    def test_composite_is_real_torch_optimizer(self):
        composite = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CompositeOptimizer"
        )
        self.assertTrue(
            any(
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Attribute)
                and isinstance(base.value.value, ast.Name)
                and base.value.value.id == "torch"
                and base.value.attr == "optim"
                and base.attr == "Optimizer"
                for base in composite.bases
            )
        )

    def test_optional_optimizer_dependencies_are_lazy(self):
        top_level_imports = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.add(node.module)
        for forbidden in (
            "bitsandbytes",
            "lion_pytorch",
            "pytorch_optimizer",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, top_level_imports)
        self.assertIn("importlib.import_module", self.source)
        self.assertIn("resolve_muon_class()", self.source)

    def test_muon_groups_are_forced_to_true_and_never_receive_internal_fallback(self):
        self.assertIn('payload["use_muon"] = True', self.source)
        self.assertIn("fallback parameters must be routed to a separate Optimizer Profile", self.source)
        for forbidden in ("adamw_lr", "adamw_betas", "adamw_wd", "adamw_eps"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(f'"{forbidden}"', self.source)

    def test_sgd_nesterov_preserves_legacy_defaults(self):
        self.assertIn('kwargs.setdefault("momentum", 0.9)', self.source)
        self.assertIn("nesterov=True", self.source)
        self.assertIn('"sets nesterov=False."', self.source)

    def test_composite_base_init_guard_allows_only_pytorch_initialization(self):
        self.assertIn("self._initializing_composite_base = True", self.source)
        self.assertIn("super().__init__(flat_parameters, defaults={})", self.source)
        self.assertIn("self._initializing_composite_base = False", self.source)
        self.assertIn(
            'if getattr(self, "_initializing_composite_base", False):',
            self.source,
        )

    def test_composite_param_groups_are_child_group_objects_not_copies(self):
        expected = (
            "self.param_groups = [\n"
            "            group\n"
            "            for entry in self.entries\n"
            "            for group in entry.optimizer.param_groups\n"
            "        ]"
        )
        self.assertGreaterEqual(self.source.count(expected), 2)

    def test_composite_topology_is_immutable(self):
        self.assertIn("def add_param_group(", self.source)
        self.assertIn("topology is immutable after construction", self.source)

    def test_state_schema_is_versioned_and_nested_by_profile(self):
        self.assertIn(
            'COMPOSITE_OPTIMIZER_STATE_KIND = "dts_parameter_policy_composite_optimizer"',
            self.source,
        )
        self.assertIn("COMPOSITE_OPTIMIZER_STATE_VERSION = 1", self.source)
        self.assertIn('"topology_fingerprint": self.runtime_spec.topology_fingerprint', self.source)
        self.assertIn('"children": {', self.source)
        self.assertIn('"state_dict": entry.optimizer.state_dict()', self.source)

    def test_load_validates_all_child_metadata_before_any_child_load(self):
        validate_pos = self.source.index("validated_payloads.append")
        load_pos = self.source.index("entry.optimizer.load_state_dict(child_state)")
        self.assertLess(validate_pos, load_pos)
        self.assertIn(
            "# All composite-level metadata is validated before mutating any child.",
            self.source,
        )

    def test_supported_constructor_map_matches_current_capability_foundation(self):
        for optimizer_type in (
            "AdamW",
            "AdamW8bit",
            "PagedAdamW8bit",
            "PagedAdamW",
            "PagedAdamW32bit",
            "Lion",
            "Lion8bit",
            "PagedLion8bit",
            "SGDNesterov",
            "SGDNesterov8bit",
            "Muon",
        ):
            with self.subTest(optimizer_type=optimizer_type):
                self.assertIn(f'"{optimizer_type}"', self.source)


if __name__ == "__main__":
    unittest.main()
