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
            "schedulefree",
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

    def test_schedulefree_factory_requires_lifecycle_and_is_optimizer_managed(self):
        self.assertIn('"schedulefree"', self.source)
        for optimizer_type in (
            "RAdamScheduleFree",
            "AdamWScheduleFree",
            "SGDScheduleFree",
        ):
            with self.subTest(optimizer_type=optimizer_type):
                self.assertIn(f'"{optimizer_type}"', self.source)
        self.assertIn("the required train()/eval() lifecycle", self.source)
        self.assertIn('mode="optimizer_managed"', self.source)
        self.assertIn('spec.lr_semantics != "optimizer_managed"', self.source)

    def test_optimizer_load_restores_runtime_lifecycle_after_checkpoint_state(self):
        self.assertIn("desired_lifecycle_mode = self._lifecycle_mode", self.source)
        self.assertIn(
            "ScheduleFree state may be checkpointed while the trainer is in eval",
            self.source,
        )
        self.assertIn('if desired_lifecycle_mode == "train":', self.source)
        self.assertIn("self.train()", self.source)
        self.assertIn('elif desired_lifecycle_mode == "eval":', self.source)
        self.assertIn("self.eval()", self.source)

    def test_composite_optimizer_forwards_train_and_eval(self):
        self.assertIn("def train(self):", self.source)
        self.assertIn('train_fn = getattr(entry.optimizer, "train", None)', self.source)
        self.assertIn("def eval(self):", self.source)
        self.assertIn('eval_fn = getattr(entry.optimizer, "eval", None)', self.source)
        self.assertIn("composite.train()", self.source)

    def test_composite_scheduler_is_real_lrscheduler_without_base_init(self):
        scheduler = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CompositeLRScheduler"
        )
        self.assertTrue(
            any(
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Attribute)
                and isinstance(base.value.value, ast.Attribute)
                and isinstance(base.value.value.value, ast.Name)
                and base.value.value.value.id == "torch"
                and base.value.value.attr == "optim"
                and base.value.attr == "lr_scheduler"
                and base.attr == "LRScheduler"
                for base in scheduler.bases
            )
        )
        init = next(
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "__init__"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "super"
                for node in ast.walk(init)
            )
        )
        self.assertIn("has already executed its own provider-defined initialization", self.source)

    def test_external_scheduler_factory_receives_each_child_optimizer(self):
        self.assertIn(
            "scheduler = scheduler_factory(spec, optimizer_entry.optimizer)",
            self.source,
        )
        self.assertIn("is not attached ", self.source)
        self.assertIn("to that child optimizer.", self.source)
        self.assertIn("not torch.optim.lr_scheduler.LRScheduler", self.source)

    def test_scheduler_factory_receives_profile_spec_for_legacy_context(self):
        self.assertIn(
            "[OptimizerInstanceSpec, torch.optim.Optimizer]",
            self.source,
        )
        self.assertIn(
            "scheduler = scheduler_factory(spec, optimizer_entry.optimizer)",
            self.source,
        )

    def test_scheduler_step_count_propagates_accelerate_direct_increment(self):
        self.assertIn("@_step_count.setter", self.source)
        self.assertIn("delta = value - old", self.source)
        self.assertIn("entry.scheduler._step_count = child_value + delta", self.source)

    def test_optimizer_managed_scheduler_has_no_child_scheduler_state(self):
        self.assertIn('mode="optimizer_managed"', self.source)
        self.assertIn("scheduler=None", self.source)
        self.assertIn("must not ", self.source)
        self.assertIn("carry external scheduler state.", self.source)

    def test_composite_scheduler_state_is_versioned_by_profile_and_class(self):
        self.assertIn(
            'COMPOSITE_SCHEDULER_STATE_KIND = "dts_parameter_policy_composite_scheduler"',
            self.source,
        )
        self.assertIn("COMPOSITE_SCHEDULER_STATE_VERSION = 1", self.source)
        self.assertIn('"scheduler_type": entry.scheduler_type', self.source)
        self.assertIn('"step_count": self._step_count', self.source)
        self.assertIn('"last_epoch": self.last_epoch', self.source)

    def test_scheduler_lr_logging_preserves_profile_group_order(self):
        self.assertIn("def get_last_lr_by_profile(self)", self.source)
        self.assertIn("for entry in self.entries:", self.source)
        self.assertIn("len(values) != len(self.optimizer.param_groups)", self.source)

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
            "RAdamScheduleFree",
            "AdamWScheduleFree",
            "SGDScheduleFree",
            "Muon",
        ):
            with self.subTest(optimizer_type=optimizer_type):
                self.assertIn(f'"{optimizer_type}"', self.source)


if __name__ == "__main__":
    unittest.main()
