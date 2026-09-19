from pathlib import Path
import types
import unittest

from mikazuki.optimizer_profiles import (
    MUON_NS_PRESETS,
    canonical_optimizer_type,
    get_optimizer_capability,
    list_optimizer_capabilities,
    normalize_optimizer_profile,
    require_component_optimizer_candidate,
    require_unrestricted_component_optimizer,
    resolve_muon_class,
    validate_muon_arguments,
)


class OptimizerProfileCapabilityTests(unittest.TestCase):
    def test_registry_names_are_unique_case_insensitively(self):
        names = [item.name.casefold() for item in list_optimizer_capabilities()]
        self.assertEqual(len(names), len(set(names)))

    def test_registry_covers_existing_named_runtime_optimizers(self):
        expected = {
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
            "DAdaptation",
            "DAdaptAdamPreprint",
            "DAdaptAdam",
            "DAdaptAdaGrad",
            "DAdaptAdan",
            "DAdaptAdanIP",
            "DAdaptLion",
            "DAdaptSGD",
            "AdaFactor",
            "Prodigy",
            "prodigyplus.ProdigyPlusScheduleFree",
            "pytorch_optimizer.CAME",
            "Custom",
            "Muon",
        }
        actual = {item.name for item in list_optimizer_capabilities()}
        self.assertTrue(expected.issubset(actual))

    def test_muon_provider_dependency_is_pinned_by_dts(self):
        requirements = (
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("pytorch-optimizer==3.10.0", requirements)

    def test_muon_uses_already_pinned_pytorch_optimizer_provider(self):
        muon = get_optimizer_capability("muon")
        self.assertEqual(muon.name, "Muon")
        self.assertEqual(muon.component_support, "supported")
        self.assertTrue(muon.supports_group_lr)
        self.assertTrue(muon.uses_external_scheduler)
        self.assertTrue(muon.requires_parameter_eligibility)
        self.assertEqual(muon.dependency, "pytorch-optimizer==3.10.0")
        self.assertEqual(muon.implementation, "pytorch_optimizer.Muon")

    def test_optimizer_names_are_canonicalized_without_constructing_runtime(self):
        self.assertEqual(canonical_optimizer_type(" adamw "), "AdamW")
        self.assertEqual(
            normalize_optimizer_profile({"type": "muon", "args": {"momentum": 0.9}}),
            {"type": "Muon", "args": {"momentum": 0.9}},
        )

    def test_muon_profile_keeps_policy_owned_values_out_of_optimizer_args(self):
        for key, value in (
            ("lr", 1e-3),
            ("params", []),
            ("use_muon", True),
            ("adamw_lr", 3e-4),
            ("adamw_betas", [0.9, 0.95]),
            ("adamw_wd", 0.01),
            ("adamw_eps", 1e-8),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "may not define"):
                normalize_optimizer_profile({"type": "Muon", "args": {key: value}})

    def test_muon_argument_contract_matches_pinned_provider_subset(self):
        args = validate_muon_arguments({
            "momentum": 0.95,
            "weight_decay": 0.01,
            "weight_decouple": True,
            "nesterov": True,
            "ns_steps": 5,
            "ns_coeffs": "polar_express_safer",
            "use_adjusted_lr": True,
        })
        self.assertEqual(args["momentum"], 0.95)
        self.assertEqual(args["weight_decay"], 0.01)
        self.assertEqual(args["ns_steps"], 5)
        self.assertIn(args["ns_coeffs"], MUON_NS_PRESETS)
        self.assertTrue(args["use_adjusted_lr"])

    def test_muon_arguments_fail_closed(self):
        bad = (
            {"unknown": 1},
            {"momentum": float("nan")},
            {"momentum": 1.0},
            {"weight_decay": -0.1},
            {"weight_decouple": 1},
            {"nesterov": 1},
            {"ns_steps": 1.5},
            {"ns_steps": 0},
            {"ns_coeffs": "invented"},
            {"use_adjusted_lr": 1},
        )
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_muon_arguments(args)

    def test_missing_pinned_muon_provider_fails_closed_without_fallback(self):
        fake_module = types.SimpleNamespace(__version__="test-no-muon")
        with self.assertRaisesRegex(RuntimeError, "will not auto-upgrade"):
            resolve_muon_class(fake_module)

    def test_muon_resolver_returns_exact_pinned_provider_class(self):
        class FakeMuon:
            pass

        fake_module = types.SimpleNamespace(__version__="3.10.0", Muon=FakeMuon)
        self.assertIs(resolve_muon_class(fake_module), FakeMuon)

    def test_planned_optimizer_is_not_accidentally_enabled(self):
        with self.assertRaisesRegex(ValueError, "not enabled"):
            require_component_optimizer_candidate("pytorch_optimizer.CAME")
        with self.assertRaisesRegex(ValueError, "not enabled"):
            require_component_optimizer_candidate("Custom")

    def test_restricted_optimizer_requires_explicit_semantic_gate(self):
        capability = require_component_optimizer_candidate("AdaFactor")
        self.assertEqual(capability.component_support, "restricted")
        self.assertIn("relative_step=False", capability.restriction)

        with self.assertRaisesRegex(ValueError, "optimizer-specific"):
            require_unrestricted_component_optimizer("AdaFactor")

    def test_supported_optimizer_can_pass_unrestricted_gate(self):
        self.assertEqual(
            require_unrestricted_component_optimizer("Muon").name,
            "Muon",
        )

    def test_non_muon_args_are_canonical_json_safe(self):
        normalized = normalize_optimizer_profile({
            "type": "AdamW",
            "args": {"betas": (0.9, 0.999), "weight_decay": 0.01},
        })
        self.assertEqual(normalized["args"]["betas"], [0.9, 0.999])

        with self.assertRaisesRegex(ValueError, "NaN"):
            normalize_optimizer_profile({
                "type": "AdamW",
                "args": {"weight_decay": float("nan")},
            })


if __name__ == "__main__":
    unittest.main()
