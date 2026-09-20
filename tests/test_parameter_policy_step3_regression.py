from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STEP3_DEPENDENCY_LIGHT_MODULES = (
    "mikazuki/model_component_profiles.py",
    "mikazuki/parameter_policy.py",
    "mikazuki/parameter_routing.py",
    "mikazuki/parameter_policy_bootstrap.py",
    "mikazuki/parameter_policy_compat.py",
)

BANNED_HEAVY_IMPORT_ROOTS = {
    "accelerate",
    "diffusers",
    "pytorch_optimizer",
    "torch",
    "transformers",
}

BANNED_PARAMETER_CALLS = {
    "clone",
    "cpu",
    "cuda",
    "detach",
    "requires_grad_",
    "to",
}


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _tree(relative_path: str) -> ast.AST:
    return ast.parse(_source(relative_path), filename=relative_path)


def _import_roots(relative_path: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(_tree(relative_path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class ParameterPolicyStep3RegressionTests(unittest.TestCase):
    def test_step3_host_modules_do_not_import_heavy_ml_runtime_dependencies(self):
        for relative_path in STEP3_DEPENDENCY_LIGHT_MODULES:
            with self.subTest(path=relative_path):
                imported = _import_roots(relative_path)
                self.assertFalse(
                    imported.intersection(BANNED_HEAVY_IMPORT_ROOTS),
                    f"{relative_path} imported heavy runtime dependency: "
                    f"{sorted(imported.intersection(BANNED_HEAVY_IMPORT_ROOTS))}",
                )

    def test_parameter_router_has_no_tensor_device_or_requires_grad_mutation_calls(self):
        tree = _tree("mikazuki/parameter_routing.py")
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in BANNED_PARAMETER_CALLS
        }
        self.assertEqual(calls, set())

        assigned_requires_grad = []
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                else:
                    targets = [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if (
                        isinstance(child, ast.Attribute)
                        and child.attr == "requires_grad"
                    ):
                        assigned_requires_grad.append(child)

        self.assertEqual(
            assigned_requires_grad,
            [],
            "Step 3 router must only observe requires_grad, never assign it.",
        )

    def test_bootstrap_and_compatibility_helpers_do_not_scan_or_route_models(self):
        combined = (
            _source("mikazuki/parameter_policy_bootstrap.py")
            + "\n"
            + _source("mikazuki/parameter_policy_compat.py")
        )
        for forbidden in (
            "scan_parameter_roots",
            "build_parameter_routing_plan",
            "ParameterDescriptor",
            "RoutingPlan",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)

    def test_training_request_does_not_wire_step3_router_or_bootstrap_into_runtime(self):
        request = _source("mikazuki/training_request.py")
        for forbidden in (
            "parameter_routing",
            "scan_parameter_roots",
            "build_parameter_routing_plan",
            "parameter_policy_bootstrap",
            "bootstrap_parameter_policy_from_standard",
            "parameter_policy_compat",
            "parameter_policy_compatibility_blockers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, request)

    def test_component_start_guard_remains_in_request_layer(self):
        request = _source("mikazuki/training_request.py")
        self.assertIn("effective_launch = launch and policy is None", request)
        self.assertIn("parameter_policy_runtime_blockers(", request)
        self.assertIn("train_type=prepared.train_type", request)
        self.assertIn(
            "integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES",
            request,
        )
        self.assertIn("if launch:", request)

    def test_standard_request_launch_contract_is_still_covered_behaviorally(self):
        request_test = _source("tests/test_parameter_policy_request_contract.py")
        self.assertIn(
            "test_standard_start_keeps_launch_semantics_and_materializes_when_requested",
            request_test,
        )
        self.assertIn(
            "test_component_start_disables_launch_staging_and_fails_before_materialization",
            request_test,
        )

    def test_optimizer_registry_provider_resolution_is_covered_separately(self):
        capability_test = _source("tests/test_optimizer_profile_capabilities.py")
        self.assertIn(
            "test_muon_resolver_returns_exact_pinned_provider_class",
            capability_test,
        )
        self.assertIn(
            "test_missing_pinned_muon_provider_fails_closed_without_fallback",
            capability_test,
        )

    def test_step3_modules_do_not_construct_optimizer_runtime_objects(self):
        for relative_path in (
            "mikazuki/model_component_profiles.py",
            "mikazuki/parameter_policy.py",
            "mikazuki/parameter_routing.py",
            "mikazuki/parameter_policy_bootstrap.py",
            "mikazuki/parameter_policy_compat.py",
        ):
            source = _source(relative_path)
            with self.subTest(path=relative_path):
                self.assertNotIn("torch.optim", source)
                self.assertNotIn("pytorch_optimizer.", source)
                self.assertNotIn("accelerator.prepare", source)
                self.assertNotIn("optimizer.step(", source)
                self.assertNotIn("optimizer.zero_grad(", source)


if __name__ == "__main__":
    unittest.main()
