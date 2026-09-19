from __future__ import annotations

import ast
import unittest
from pathlib import Path

from mikazuki.parameter_routing import (
    ADAPTER_TARGET_MARKER_ATTR,
    AdapterTargetMetadata,
    ParameterScanError,
    scan_parameter_roots,
)


class FakeParameter:
    def __init__(self, shape=(2, 2)):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = "float32"
        self.requires_grad = True


class FakeModule:
    def __init__(self, *, parameters=None, children=None):
        self._local_parameters = list(parameters or [])
        self._children = list(children or [])
        self.named_modules_calls = 0

    def named_modules(self, *, remove_duplicate=True):
        if remove_duplicate is not False:
            raise AssertionError("scanner must request remove_duplicate=False")
        self.named_modules_calls += 1

        def walk(module, prefix):
            yield prefix, module
            for child_name, child in module._children:
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from walk(child, child_prefix)

        return walk(self, "")

    def named_parameters(self, *, recurse=True, remove_duplicate=True):
        if recurse is not False:
            raise AssertionError("scanner must request recurse=False")
        if remove_duplicate is not False:
            raise AssertionError("scanner must request remove_duplicate=False")
        return iter(self._local_parameters)


Linear = type("Linear", (FakeModule,), {})
LoRAModule = type("LoRAModule", (FakeModule,), {})


def module(*, parameters=None, children=None, cls=FakeModule):
    return cls(parameters=parameters, children=children)


def marked_adapter(marker):
    adapter = module(
        cls=LoRAModule,
        children=[
            (
                "lora_down",
                module(
                    cls=Linear,
                    parameters=[("weight", FakeParameter((4, 8)))],
                ),
            ),
            (
                "lora_up",
                module(
                    cls=Linear,
                    parameters=[("weight", FakeParameter((8, 4)))],
                ),
            ),
        ],
    )
    setattr(adapter, ADAPTER_TARGET_MARKER_ATTR, marker)
    return adapter


class AdapterMarkerTransportTests(unittest.TestCase):
    def test_marker_constant_is_versioned_and_private(self):
        self.assertEqual(
            ADAPTER_TARGET_MARKER_ATTR,
            "_dts_parameter_policy_target_v1",
        )

    def test_attached_marker_is_auto_discovered_and_inherited_by_descendants(self):
        adapter = marked_adapter(
            (
                "TRANSFORMER",
                "double_blocks.0.img_attn.qkv",
                "torch.nn.modules.linear.Linear",
            )
        )
        root = module(children=[("adapter_0", adapter)])

        descriptors = scan_parameter_roots({"network": root})

        self.assertEqual(len(descriptors), 2)
        expected = AdapterTargetMetadata(
            "transformer",
            "double_blocks.0.img_attn.qkv",
            "torch.nn.modules.linear.Linear",
        )
        for descriptor in descriptors:
            self.assertEqual(descriptor.canonical_alias.adapter_target, expected)

    def test_split_style_nested_descendants_inherit_parent_marker(self):
        adapter = module(
            cls=LoRAModule,
            children=[
                (
                    "lora_down",
                    module(
                        children=[
                            ("0", module(cls=Linear, parameters=[("weight", FakeParameter())])),
                            ("1", module(cls=Linear, parameters=[("weight", FakeParameter())])),
                            ("2", module(cls=Linear, parameters=[("weight", FakeParameter())])),
                        ]
                    ),
                )
            ],
        )
        setattr(
            adapter,
            ADAPTER_TARGET_MARKER_ATTR,
            ("mmdit", "joint_blocks.0.x_block.attn.qkv", "torch.nn.Linear"),
        )
        root = module(children=[("adapter_qkv", adapter)])

        descriptors = scan_parameter_roots({"network": root})

        self.assertEqual(len(descriptors), 3)
        self.assertEqual(
            {
                descriptor.canonical_alias.adapter_target.module_path
                for descriptor in descriptors
            },
            {"joint_blocks.0.x_block.attn.qkv"},
        )

    def test_scanner_uses_existing_module_enumeration_for_marker_index(self):
        adapter = marked_adapter(
            ("unet", "input_blocks.1.attn1.to_q", "torch.nn.Linear")
        )
        root = module(children=[("adapter", adapter)])

        scan_parameter_roots({"network": root})

        self.assertEqual(root.named_modules_calls, 1)

    def test_no_marker_preserves_existing_none_behavior(self):
        root = module(
            children=[
                (
                    "linear",
                    module(
                        cls=Linear,
                        parameters=[("weight", FakeParameter())],
                    ),
                )
            ]
        )

        descriptors = scan_parameter_roots({"network": root})
        self.assertIsNone(descriptors[0].canonical_alias.adapter_target)

    def test_identical_explicit_and_attached_metadata_are_allowed(self):
        adapter = marked_adapter(
            ("unet", "input_blocks.1.attn1.to_q", "torch.nn.Linear")
        )
        root = module(children=[("adapter", adapter)])
        explicit = AdapterTargetMetadata(
            "unet",
            "input_blocks.1.attn1.to_q",
            "torch.nn.Linear",
        )

        descriptors = scan_parameter_roots(
            {"network": root},
            adapter_targets={id(adapter): explicit},
        )

        self.assertEqual(descriptors[0].canonical_alias.adapter_target, explicit)

    def test_conflicting_explicit_and_attached_metadata_fail_closed(self):
        adapter = marked_adapter(
            ("unet", "input_blocks.1.attn1.to_q", "torch.nn.Linear")
        )
        root = module(children=[("adapter", adapter)])

        with self.assertRaisesRegex(
            ParameterScanError,
            "conflicting explicit and attached",
        ):
            scan_parameter_roots(
                {"network": root},
                adapter_targets={
                    id(adapter): AdapterTargetMetadata(
                        "unet",
                        "input_blocks.1.attn2.to_q",
                        "torch.nn.Linear",
                    )
                },
            )

    def test_conflicting_nested_attached_markers_fail_closed(self):
        inner = marked_adapter(
            ("unet", "input_blocks.1.attn2.to_q", "torch.nn.Linear")
        )
        outer = module(cls=LoRAModule, children=[("inner", inner)])
        setattr(
            outer,
            ADAPTER_TARGET_MARKER_ATTR,
            ("unet", "input_blocks.1.attn1.to_q", "torch.nn.Linear"),
        )
        root = module(children=[("outer", outer)])

        with self.assertRaisesRegex(ParameterScanError, "conflicting nested adapter"):
            scan_parameter_roots({"network": root})

    def test_malformed_marker_shape_and_types_fail_closed(self):
        bad_markers = (
            ["unet", "path", "torch.nn.Linear"],
            ("unet", "path"),
            ("unet", 123, "torch.nn.Linear"),
            ("", "path", "torch.nn.Linear"),
            ("unet", "path", ""),
        )
        for marker in bad_markers:
            with self.subTest(marker=marker):
                adapter = marked_adapter(marker)
                root = module(children=[("adapter", adapter)])
                with self.assertRaisesRegex(
                    ParameterScanError,
                    "malformed|invalid",
                ):
                    scan_parameter_roots({"network": root})


class NativeLoRAMetadataEmitterContractTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    FILES = {
        "stable": ROOT / "scripts/stable/networks/lora.py",
        "flux": ROOT / "scripts/dev/networks/lora_flux.py",
        "sd3": ROOT / "scripts/dev/networks/lora_sd3.py",
    }

    def _source(self, key):
        return self.FILES[key].read_text(encoding="utf-8")

    def _tree(self, key):
        return ast.parse(self._source(key), filename=str(self.FILES[key]))

    def test_native_emitters_use_exact_versioned_marker_without_host_import(self):
        for key in self.FILES:
            with self.subTest(key=key):
                source = self._source(key)
                self.assertIn(
                    'DTS_PARAMETER_POLICY_TARGET_ATTR = "_dts_parameter_policy_target_v1"',
                    source,
                )
                self.assertNotIn("mikazuki.parameter_routing", source)
                self.assertNotIn("AdapterTargetMetadata", source)

    def test_native_emitters_keep_lora_constructor_signatures_unchanged(self):
        expected = {
            "stable": [
                "self",
                "lora_name",
                "org_module",
                "multiplier",
                "lora_dim",
                "alpha",
                "dropout",
                "rank_dropout",
                "module_dropout",
            ],
            "flux": [
                "self",
                "lora_name",
                "org_module",
                "multiplier",
                "lora_dim",
                "alpha",
                "dropout",
                "rank_dropout",
                "module_dropout",
                "split_dims",
                "ggpo_beta",
                "ggpo_sigma",
            ],
        }
        for key, expected_args in expected.items():
            with self.subTest(key=key):
                tree = self._tree(key)
                module_cls = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "LoRAModule"
                )
                init = next(
                    node
                    for node in module_cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                )
                self.assertEqual(
                    [arg.arg for arg in init.args.args],
                    expected_args,
                )

        sd3 = self._source("sd3")
        self.assertIn(
            "from networks.lora_flux import LoRAModule, LoRAInfModule",
            sd3,
        )
        self.assertFalse(
            any(
                isinstance(node, ast.ClassDef) and node.name == "LoRAModule"
                for node in self._tree("sd3").body
            )
        )

    def test_marker_payload_contains_only_root_path_and_original_type_strings(self):
        for key in self.FILES:
            with self.subTest(key=key):
                source = self._source(key)
                self.assertIn(
                    "(target_root, target_path, target_type)",
                    source,
                )
                self.assertIn(
                    'target_path = (name + "." if name else "") + child_name',
                    source,
                )
                self.assertNotIn(
                    "DTS_PARAMETER_POLICY_TARGET_ATTR, child_module",
                    source,
                )

    def test_stable_roots_cover_sd_and_sdxl_without_guessing_lora_name(self):
        source = self._source("stable")
        for marker in (
            'target_root = "unet"',
            'target_root = "text_encoder"',
            'target_root = "text_encoder_1"',
            'target_root = "text_encoder_2"',
        ):
            self.assertIn(marker, source)
        self.assertIn(
            "_attach_dts_parameter_policy_target(\n                                lora,\n                                target_root,\n                                target_path,\n                                child_module,",
            source,
        )

    def test_flux_roots_cover_transformer_clip_l_and_t5(self):
        source = self._source("flux")
        for marker in ('"transformer"', '"clip_l"', '"t5xxl"'):
            self.assertIn(marker, source)
        self.assertNotIn('"flux"\n                if is_flux', source)

    def test_sd3_roots_cover_mmdit_and_all_three_text_encoders(self):
        source = self._source("sd3")
        for marker in ('"mmdit"', '"clip_l"', '"clip_g"', '"t5xxl"'):
            self.assertIn(marker, source)

    def test_marker_attachment_occurs_after_adapter_construction_before_append(self):
        for key in self.FILES:
            with self.subTest(key=key):
                source = self._source(key)
                construct = source.index("lora = module_class(")
                attach = source.index("_attach_dts_parameter_policy_target(", construct)
                append = source.index("loras.append(lora)", construct)
                self.assertLess(construct, attach)
                self.assertLess(attach, append)

    def test_emitters_do_not_register_marker_as_parameter_or_buffer(self):
        for key in self.FILES:
            with self.subTest(key=key):
                source = self._source(key)
                self.assertNotIn(
                    'register_buffer("DTS_PARAMETER_POLICY_TARGET_ATTR"',
                    source,
                )
                self.assertNotIn(
                    "register_parameter(DTS_PARAMETER_POLICY_TARGET_ATTR",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
