import types
import unittest

from mikazuki.optimizer_profiles import (
    MUON_ADJUST_LR_MODES,
    canonical_optimizer_type,
    get_optimizer_capability,
    list_optimizer_capabilities,
    normalize_optimizer_profile,
    require_component_optimizer_support,
    resolve_native_muon_class,
    validate_muon_arguments,
)


class OptimizerProfileCapabilityTests(unittest.TestCase):
    def test_registry_names_are_unique_case_insensitively(self):
        names = [item.name.casefold() for item in list_optimizer_capabilities()]
        self.assertEqual(len(names), len(set(names)))

    def test_muon_is_component_only_capability_with_explicit_eligibility(self):
        muon = get_optimizer_capability("muon")
        self.assertEqual(muon.name, "Muon")
        self.assertEqual(muon.component_support, "supported")
        self.assertTrue(muon.supports_group_lr)
        self.assertTrue(muon.uses_external_scheduler)
        self.assertTrue(muon.requires_parameter_eligibility)
        self.assertEqual(muon.dependency, "torch.optim.Muon")

    def test_optimizer_names_are_canonicalized_without_constructing_runtime(self):
        self.assertEqual(canonical_optimizer_type(" adamw "), "AdamW")
        self.assertEqual(
            normalize_optimizer_profile({"type": "muon", "args": {"momentum": 0.9}}),
            {"type": "Muon", "args": {"momentum": 0.9}},
        )

    def test_muon_profile_keeps_lr_out_of_optimizer_args(self):
        with self.assertRaisesRegex(ValueError, "may not define lr"):
            normalize_optimizer_profile({"type": "Muon", "args": {"lr": 1e-3}})

        with self.assertRaisesRegex(ValueError, "may not define params"):
            normalize_optimizer_profile({"type": "Muon", "args": {"params": []}})

    def test_muon_argument_contract_accepts_native_api_shape(self):
        args = validate_muon_arguments({
            "momentum": 0.95,
            "weight_decay": 0.01,
            "nesterov": True,
            "ns_coefficients": [3.4445, -4.7750, 2.0315],
            "eps": 1e-7,
            "ns_steps": 5,
            "adjust_lr_fn": "match_rms_adamw",
        })
        self.assertEqual(args["momentum"], 0.95)
        self.assertEqual(validate_muon_arguments({"momentum": 1.0})["momentum"], 1.0)
        self.assertEqual(args["weight_decay"], 0.01)
        self.assertEqual(args["ns_coefficients"], (3.4445, -4.775, 2.0315))
        self.assertEqual(args["ns_steps"], 5)
        self.assertIn(args["adjust_lr_fn"], MUON_ADJUST_LR_MODES)

    def test_muon_arguments_fail_closed(self):
        bad = (
            {"unknown": 1},
            {"momentum": float("nan")},
            {"weight_decay": -0.1},
            {"nesterov": 1},
            {"ns_coefficients": [1.0, 2.0]},
            {"ns_steps": 1.5},
            {"ns_steps": 0},
            {"ns_steps": 100},
            {"eps": 0.0},
            {"adjust_lr_fn": "invented-mode"},
        )
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_muon_arguments(args)

    def test_missing_native_muon_fails_closed_without_fallback(self):
        fake_torch = types.SimpleNamespace(
            __version__="test-no-muon",
            optim=types.SimpleNamespace(),
        )
        with self.assertRaisesRegex(RuntimeError, "will not auto-upgrade PyTorch"):
            resolve_native_muon_class(fake_torch)

    def test_native_muon_resolver_returns_exact_runtime_class(self):
        class FakeMuon:
            pass

        fake_torch = types.SimpleNamespace(
            __version__="test-with-muon",
            optim=types.SimpleNamespace(Muon=FakeMuon),
        )
        self.assertIs(resolve_native_muon_class(fake_torch), FakeMuon)

    def test_planned_optimizer_is_not_accidentally_enabled(self):
        with self.assertRaisesRegex(ValueError, "not enabled"):
            require_component_optimizer_support("pytorch_optimizer.CAME")

    def test_restricted_optimizer_remains_explicitly_restricted(self):
        capability = require_component_optimizer_support("AdaFactor")
        self.assertEqual(capability.component_support, "restricted")
        self.assertIn("relative_step=False", capability.restriction)


if __name__ == "__main__":
    unittest.main()
