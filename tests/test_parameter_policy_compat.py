import copy
import unittest

from mikazuki.parameter_policy_compat import (
    _parse_network_args,
    parameter_policy_compatibility_blockers,
)


_CANONICAL_LORA_MODULES = {
    "sd-lora": "networks.lora",
    "sdxl-lora": "networks.lora",
    "flux-lora": "networks.lora_flux",
    "chroma-lora": "networks.lora_flux",
    "sd3-lora": "networks.lora_sd3",
    "anima-lora": "networks.lora_anima",
}


def _ordinary_config(train_type):
    module = _CANONICAL_LORA_MODULES.get(train_type)
    return {"network_module": module} if module else {}


class ParameterPolicyCompatibilityTests(unittest.TestCase):
    def test_ordinary_reviewed_configs_have_no_blockers(self):
        for train_type in (
            "sd-lora",
            "sdxl-lora",
            "sd-dreambooth",
            "sdxl-finetune",
            "flux-lora",
            "chroma-lora",
            "sd3-lora",
            "flux-finetune",
            "anima-lora",
            "anima-finetune",
        ):
            with self.subTest(train_type=train_type):
                self.assertEqual(
                    parameter_policy_compatibility_blockers(
                        _ordinary_config(train_type),
                        train_type,
                    ),
                    [],
                )

    def test_input_config_is_not_mutated(self):
        config = {
            "network_module": "networks.lora_flux",
            "network_args": [
                "network_reg_dims=foo=8",
                "my_loraplus_lr_ratio_backup=2",
            ],
            "fused_backward_pass": False,
        }
        before = copy.deepcopy(config)
        parameter_policy_compatibility_blockers(config, "flux-lora")
        self.assertEqual(config, before)

    def test_blocker_order_and_dedup_are_deterministic(self):
        config = {
            "network_module": "custom.network",
            "network_args": [
                "loraplus_lr_ratio=2",
                "loraplus_unet_lr_ratio=4",
                "network_reg_lrs=foo=1e-4",
            ],
            "fused_backward_pass": True,
            "deepspeed": True,
        }
        first = parameter_policy_compatibility_blockers(config, "flux-lora")
        second = parameter_policy_compatibility_blockers(config, "flux-lora")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 5)
        self.assertIn("fused_backward_pass", first[0])
        self.assertIn("DeepSpeed", first[1])
        self.assertIn("LoRA+", first[2])
        self.assertIn("network_reg_lrs", first[3])
        self.assertIn("network_module", first[4])

    def test_network_args_use_exact_keys_not_substring_matching(self):
        config = {
            "network_module": "networks.lora",
            "network_args": [
                "my_loraplus_lr_ratio_backup=2",
                "down_lr_weight_backup=1,1,1",
            ],
        }
        self.assertEqual(
            parameter_policy_compatibility_blockers(config, "sd-lora"),
            [],
        )

    def test_network_args_duplicate_keys_use_later_value(self):
        parsed = _parse_network_args(
            ["Foo=first=payload", "foo=second", "BAR = value"]
        )
        self.assertEqual(parsed, {"foo": "second", "bar": "value"})

    def test_network_args_string_supports_line_separated_items(self):
        parsed = _parse_network_args("foo=1\nbar=a=b\n")
        self.assertEqual(parsed, {"foo": "1", "bar": "a=b"})

    def test_malformed_network_args_fail_closed(self):
        for raw in (
            ["broken"],
            ["=value"],
            [123],
            {"loraplus_lr_ratio": 2},
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError,
                "network_args",
            ):
                _parse_network_args(raw)

    def test_sdxl_full_block_lr_is_blocked(self):
        blockers = parameter_policy_compatibility_blockers(
            {"block_lr": ",".join(["1e-5"] * 23)},
            "sdxl-finetune",
        )
        self.assertEqual(len(blockers), 1)
        self.assertIn("block_lr", blockers[0])

    def test_dreambooth_nonnegative_stop_is_blocked(self):
        for stop in (0, 10, "20"):
            with self.subTest(stop=stop):
                blockers = parameter_policy_compatibility_blockers(
                    {"stop_text_encoder_training": stop},
                    "sd-dreambooth",
                )
                self.assertEqual(len(blockers), 1)
                self.assertIn("stop_text_encoder_training", blockers[0])

    def test_dreambooth_none_or_negative_stop_is_not_blocked(self):
        for stop in (None, "", -1, "-1"):
            with self.subTest(stop=stop):
                self.assertEqual(
                    parameter_policy_compatibility_blockers(
                        {"stop_text_encoder_training": stop},
                        "sd-dreambooth",
                    ),
                    [],
                )

    def test_dreambooth_malformed_stop_fails_closed(self):
        for stop in (True, 1.5, "1.5", "later", float("inf")):
            with self.subTest(stop=stop), self.assertRaisesRegex(
                ValueError,
                "stop_text_encoder_training",
            ):
                parameter_policy_compatibility_blockers(
                    {"stop_text_encoder_training": stop},
                    "sd-dreambooth",
                )

    def test_sd_block_weight_lr_is_one_blocker(self):
        for train_type in ("sd-lora", "sdxl-lora"):
            for key in ("down_lr_weight", "mid_lr_weight", "up_lr_weight"):
                with self.subTest(train_type=train_type, key=key):
                    blockers = parameter_policy_compatibility_blockers(
                        {
                            "network_module": "networks.lora",
                            "network_args": [
                                f"{key}=1,1,1",
                                "block_lr_zero_threshold=0.01",
                            ],
                        },
                        train_type,
                    )
                    self.assertEqual(len(blockers), 1)
                    self.assertIn("block LR", blockers[0])

    def test_lone_block_lr_zero_threshold_is_inert(self):
        for train_type in ("sd-lora", "sdxl-lora"):
            with self.subTest(train_type=train_type):
                self.assertEqual(
                    parameter_policy_compatibility_blockers(
                        {
                            "network_module": "networks.lora",
                            "network_args": ["block_lr_zero_threshold=0.01"],
                        },
                        train_type,
                    ),
                    [],
                )

    def test_all_loraplus_keys_are_blocked_for_reviewed_lora_families(self):
        keys = (
            "loraplus_lr_ratio",
            "loraplus_unet_lr_ratio",
            "loraplus_text_encoder_lr_ratio",
        )
        for train_type, module in _CANONICAL_LORA_MODULES.items():
            for key in keys:
                with self.subTest(train_type=train_type, key=key):
                    blockers = parameter_policy_compatibility_blockers(
                        {
                            "network_module": module,
                            "network_args": [f"{key}=1"],
                        },
                        train_type,
                    )
                    self.assertEqual(len(blockers), 1)
                    self.assertIn("LoRA+", blockers[0])

    def test_regex_specific_lr_is_blocked_only_where_reviewed(self):
        for train_type in ("flux-lora", "chroma-lora", "anima-lora"):
            with self.subTest(train_type=train_type):
                blockers = parameter_policy_compatibility_blockers(
                    {
                        "network_module": _CANONICAL_LORA_MODULES[train_type],
                        "network_args": ["network_reg_lrs=foo=1e-4"],
                    },
                    train_type,
                )
                self.assertEqual(len(blockers), 1)
                self.assertIn("network_reg_lrs", blockers[0])

        self.assertEqual(
            parameter_policy_compatibility_blockers(
                {
                    "network_module": "networks.lora_sd3",
                    "network_args": ["network_reg_lrs=foo=1e-4"],
                },
                "sd3-lora",
            ),
            [],
        )

    def test_network_reg_dims_is_not_an_lr_blocker(self):
        for train_type in ("flux-lora", "chroma-lora", "anima-lora"):
            with self.subTest(train_type=train_type):
                self.assertEqual(
                    parameter_policy_compatibility_blockers(
                        {
                            "network_module": _CANONICAL_LORA_MODULES[train_type],
                            "network_args": ["network_reg_dims=foo=16"],
                        },
                        train_type,
                    ),
                    [],
                )

    def test_unqualified_full_precision_and_fp8_modes_are_blocked(self):
        cases = (
            ("full_fp16", True, "full_fp16"),
            ("full_bf16", True, "full_bf16"),
            ("fp8_base", True, "fp8_base"),
            ("fp8_base_unet", True, "fp8_base_unet"),
        )
        for field, value, marker in cases:
            with self.subTest(field=field):
                blockers = parameter_policy_compatibility_blockers(
                    {field: value},
                    "flux-finetune",
                )
                self.assertEqual(len(blockers), 1)
                self.assertIn(marker, blockers[0])

    def test_ordinary_mixed_precision_remains_allowed(self):
        for precision in ("no", "fp16", "bf16"):
            with self.subTest(precision=precision):
                self.assertEqual(
                    parameter_policy_compatibility_blockers(
                        {
                            "mixed_precision": precision,
                            "full_fp16": False,
                            "full_bf16": False,
                            "fp8_base": False,
                            "fp8_base_unet": False,
                        },
                        "flux-finetune",
                    ),
                    [],
                )

    def test_global_optimizer_runtime_semantics_are_blocked(self):
        cases = (
            ("fused_backward_pass", True, "fused_backward_pass"),
            ("fused_optimizer_groups", 2, "fused_optimizer_groups"),
            ("blockwise_fused_optimizers", True, "blockwise_fused_optimizers"),
            ("deepspeed", True, "DeepSpeed"),
        )
        for field, value, marker in cases:
            with self.subTest(field=field):
                blockers = parameter_policy_compatibility_blockers(
                    {field: value},
                    "flux-finetune",
                )
                self.assertEqual(len(blockers), 1)
                self.assertIn(marker, blockers[0])

    def test_false_or_zero_global_runtime_fields_are_inert(self):
        config = {
            "fused_backward_pass": False,
            "fused_optimizer_groups": 0,
            "blockwise_fused_optimizers": "false",
            "deepspeed": 0,
            "full_fp16": False,
            "full_bf16": "false",
            "fp8_base": 0,
            "fp8_base_unet": False,
        }
        self.assertEqual(
            parameter_policy_compatibility_blockers(config, "flux-finetune"),
            [],
        )

    def test_invalid_global_runtime_field_values_fail_closed(self):
        cases = (
            ("fused_backward_pass", "maybe"),
            ("fused_optimizer_groups", True),
            ("fused_optimizer_groups", 1.5),
            ("fused_optimizer_groups", float("inf")),
            ("blockwise_fused_optimizers", 2),
            ("deepspeed", []),
            ("full_fp16", "maybe"),
            ("full_bf16", 2),
            ("fp8_base", []),
            ("fp8_base_unet", "maybe"),
        )
        for field, value in cases:
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                field,
            ):
                parameter_policy_compatibility_blockers(
                    {field: value},
                    "flux-finetune",
                )

    def test_noncanonical_explicit_network_module_is_blocked(self):
        for train_type in _CANONICAL_LORA_MODULES:
            with self.subTest(train_type=train_type):
                blockers = parameter_policy_compatibility_blockers(
                    {"network_module": "lycoris.kohya"},
                    train_type,
                )
                self.assertEqual(len(blockers), 1)
                self.assertIn("network_module", blockers[0])

    def test_missing_canonical_network_module_is_allowed(self):
        for train_type in _CANONICAL_LORA_MODULES:
            with self.subTest(train_type=train_type):
                self.assertEqual(
                    parameter_policy_compatibility_blockers({}, train_type),
                    [],
                )

    def test_unrelated_network_args_remain_allowed(self):
        for train_type, module in _CANONICAL_LORA_MODULES.items():
            with self.subTest(train_type=train_type):
                self.assertEqual(
                    parameter_policy_compatibility_blockers(
                        {
                            "network_module": module,
                            "network_args": [
                                "rank_dropout=0.1",
                                "module_dropout=0.05",
                                "network_reg_dims=foo=8",
                                "verbose=true",
                            ],
                        },
                        train_type,
                    ),
                    [],
                )

    def test_unknown_backend_fails_closed(self):
        with self.assertRaises(ValueError):
            parameter_policy_compatibility_blockers({}, "unknown-backend")


if __name__ == "__main__":
    unittest.main()
