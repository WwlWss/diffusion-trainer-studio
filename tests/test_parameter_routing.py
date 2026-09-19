from __future__ import annotations

import unittest

from mikazuki.parameter_routing import (
    AdapterTargetMetadata,
    ParameterScanError,
    classify_parameter_class,
    scan_parameter_roots,
)


class FakeParameter:
    def __init__(
        self,
        shape=(2, 2),
        *,
        ndim=None,
        dtype="float32",
        requires_grad=True,
    ):
        self.shape = tuple(shape)
        self.ndim = len(self.shape) if ndim is None else ndim
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.device_calls = []

    def _forbidden(self, name):
        self.device_calls.append(name)
        raise AssertionError(f"scanner must not call parameter.{name}()")

    def cpu(self):
        return self._forbidden("cpu")

    def cuda(self):
        return self._forbidden("cuda")

    def to(self, *args, **kwargs):
        return self._forbidden("to")

    def clone(self):
        return self._forbidden("clone")

    def detach(self):
        return self._forbidden("detach")


class FakeModule:
    def __init__(self, *, parameters=None, children=None):
        self._local_parameters = list(parameters or [])
        self._children = list(children or [])

    def named_modules(self, *, remove_duplicate=True):
        if remove_duplicate is not False:
            raise AssertionError("scanner must request remove_duplicate=False")

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
LayerNorm = type("LayerNorm", (FakeModule,), {})
Embedding = type("Embedding", (FakeModule,), {})
Conv2d = type("Conv2d", (FakeModule,), {})
GenericBlock = type("GenericBlock", (FakeModule,), {})
BasicTransformerBlock = type("BasicTransformerBlock", (FakeModule,), {})
ResnetBlock2D = type("ResnetBlock2D", (FakeModule,), {})
LoRAModule = type("LoRAModule", (FakeModule,), {})


def module(*, parameters=None, children=None, cls=FakeModule):
    return cls(parameters=parameters, children=children)


class ParameterScannerTests(unittest.TestCase):
    def test_unique_scan_builds_metadata_only_descriptor(self):
        weight = FakeParameter((4, 8), dtype="bfloat16", requires_grad=False)
        bias = FakeParameter((4,), dtype="bfloat16", requires_grad=True)
        root = module(
            children=[
                (
                    "linear",
                    module(
                        cls=Linear,
                        parameters=[("weight", weight), ("bias", bias)],
                    ),
                )
            ]
        )

        descriptors = scan_parameter_roots({"UNET": root})
        self.assertEqual([item.canonical_name for item in descriptors], [
            "unet.linear.bias",
            "unet.linear.weight",
        ])

        by_name = {item.canonical_name: item for item in descriptors}
        matrix = by_name["unet.linear.weight"]
        self.assertIs(matrix.parameter, weight)
        self.assertEqual(matrix.parameter_id, id(weight))
        self.assertEqual(matrix.shape, (4, 8))
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.numel, 32)
        self.assertEqual(matrix.dtype, "bfloat16")
        self.assertFalse(matrix.requires_grad)
        self.assertEqual(matrix.canonical_alias.parameter_class, "matrix_weight")

        bias_desc = by_name["unet.linear.bias"]
        self.assertEqual(bias_desc.canonical_alias.parameter_class, "bias")

        self.assertEqual(weight.device_calls, [])
        self.assertEqual(bias.device_calls, [])

    def test_shared_parameter_is_deduplicated_by_identity_and_aliases_are_preserved(self):
        shared = FakeParameter((8, 8))
        root = module(
            children=[
                ("a", module(cls=Linear, parameters=[("weight", shared)])),
                ("b", module(cls=Linear, parameters=[("weight", shared)])),
            ]
        )

        descriptors = scan_parameter_roots({"transformer": root})
        self.assertEqual(len(descriptors), 1)
        descriptor = descriptors[0]
        self.assertEqual(descriptor.parameter_id, id(shared))
        self.assertEqual(
            [alias.qualified_name for alias in descriptor.aliases],
            ["transformer.a.weight", "transformer.b.weight"],
        )
        self.assertEqual(descriptor.canonical_name, "transformer.a.weight")

    def test_descriptor_order_depends_on_semantic_names_not_object_creation_order(self):
        z_param = FakeParameter((2, 2))
        a_param = FakeParameter((2, 2))
        root = module(
            children=[
                ("z", module(cls=Linear, parameters=[("weight", z_param)])),
                ("a", module(cls=Linear, parameters=[("weight", a_param)])),
            ]
        )

        descriptors = scan_parameter_roots({"unet": root})
        self.assertEqual(
            [item.canonical_name for item in descriptors],
            ["unet.a.weight", "unet.z.weight"],
        )

    def test_shared_leaf_preserves_path_specific_ancestry(self):
        shared_leaf = module(
            cls=Linear,
            parameters=[("weight", FakeParameter((3, 3)))],
        )
        transformer_parent = module(
            cls=BasicTransformerBlock,
            children=[("proj", shared_leaf)],
        )
        resnet_parent = module(
            cls=ResnetBlock2D,
            children=[("proj", shared_leaf)],
        )
        root = module(
            children=[
                ("transformer", transformer_parent),
                ("resnet", resnet_parent),
            ]
        )

        descriptors = scan_parameter_roots({"unet": root})
        self.assertEqual(len(descriptors), 1)
        aliases = {alias.module_path: alias for alias in descriptors[0].aliases}

        self.assertEqual(
            aliases["transformer.proj"].ancestor_module_classes,
            ("FakeModule", "BasicTransformerBlock"),
        )
        self.assertEqual(
            aliases["resnet.proj"].ancestor_module_classes,
            ("FakeModule", "ResnetBlock2D"),
        )

    def test_structural_parameter_classes_cover_bias_norm_embedding_conv_matrix_other(self):
        params = {
            "bias": FakeParameter((4,)),
            "norm": FakeParameter((4,)),
            "embedding": FakeParameter((100, 8)),
            "conv": FakeParameter((8, 4, 3, 3)),
            "matrix": FakeParameter((8, 4)),
            "other": FakeParameter((4,)),
        }
        root = module(
            children=[
                ("linear", module(cls=Linear, parameters=[
                    ("weight", params["matrix"]),
                    ("bias", params["bias"]),
                ])),
                ("norm", module(cls=LayerNorm, parameters=[("weight", params["norm"])])),
                ("embedding", module(cls=Embedding, parameters=[("weight", params["embedding"])])),
                ("conv", module(cls=Conv2d, parameters=[("weight", params["conv"])])),
                ("generic", module(cls=GenericBlock, parameters=[("scale", params["other"])])),
            ]
        )

        descriptors = scan_parameter_roots({"dit": root})
        classes = {
            item.canonical_name: item.canonical_alias.parameter_class
            for item in descriptors
        }
        self.assertEqual(classes["dit.linear.bias"], "bias")
        self.assertEqual(classes["dit.norm.weight"], "norm_weight")
        self.assertEqual(classes["dit.embedding.weight"], "embedding_weight")
        self.assertEqual(classes["dit.conv.weight"], "conv_weight")
        self.assertEqual(classes["dit.linear.weight"], "matrix_weight")
        self.assertEqual(classes["dit.generic.scale"], "other")

        self.assertEqual(
            classify_parameter_class(
                parameter_role="projection_bias",
                module_class="Linear",
                ndim=1,
            ),
            "bias",
        )
        self.assertEqual(
            classify_parameter_class(
                parameter_role="weight",
                module_class="TimestepEmbedding",
                ndim=2,
            ),
            "matrix_weight",
        )

    def test_adapter_metadata_is_inherited_from_registered_adapter_ancestor(self):
        down_weight = FakeParameter((4, 8))
        up_weight = FakeParameter((8, 4))
        adapter = module(
            cls=LoRAModule,
            children=[
                ("lora_down", module(cls=Linear, parameters=[("weight", down_weight)])),
                ("lora_up", module(cls=Linear, parameters=[("weight", up_weight)])),
            ],
        )
        root = module(children=[("adapter_0", adapter)])
        target = AdapterTargetMetadata(
            root="TRANSFORMER",
            module_path="double_blocks.0.img_attn.qkv",
            module_type="torch.nn.modules.linear.Linear",
        )

        descriptors = scan_parameter_roots(
            {"network": root},
            adapter_targets={id(adapter): target},
        )

        self.assertEqual(len(descriptors), 2)
        for descriptor in descriptors:
            self.assertEqual(descriptor.canonical_alias.adapter_target, target)
        self.assertEqual(
            descriptors[0].canonical_alias.adapter_target.root,
            "transformer",
        )

    def test_identical_nested_adapter_metadata_is_allowed_and_nearest_registration_wins(self):
        parameter = FakeParameter((4, 4))
        inner = module(cls=LoRAModule, parameters=[("weight", parameter)])
        outer = module(cls=LoRAModule, children=[("inner", inner)])
        root = module(children=[("outer", outer)])
        target = AdapterTargetMetadata(
            root="unet",
            module_path="input_blocks.1.attn1.to_q",
            module_type="torch.nn.Linear",
        )

        descriptors = scan_parameter_roots(
            {"network": root},
            adapter_targets={id(outer): target, id(inner): target},
        )
        self.assertEqual(descriptors[0].canonical_alias.adapter_target, target)

    def test_conflicting_nested_adapter_metadata_fails_closed(self):
        parameter = FakeParameter((4, 4))
        inner = module(cls=LoRAModule, parameters=[("weight", parameter)])
        outer = module(cls=LoRAModule, children=[("inner", inner)])
        root = module(children=[("outer", outer)])

        with self.assertRaisesRegex(ParameterScanError, "conflicting nested adapter"):
            scan_parameter_roots(
                {"network": root},
                adapter_targets={
                    id(outer): AdapterTargetMetadata(
                        "unet",
                        "attn1.to_q",
                        "torch.nn.Linear",
                    ),
                    id(inner): AdapterTargetMetadata(
                        "unet",
                        "attn2.to_q",
                        "torch.nn.Linear",
                    ),
                },
            )

    def test_scanner_rejects_duplicate_removing_legacy_api_instead_of_degrading(self):
        class LegacyModule:
            def named_modules(self):
                return [("", self)]

            def named_parameters(self):
                return []

        with self.assertRaisesRegex(ParameterScanError, "remove_duplicate=False"):
            scan_parameter_roots({"unet": LegacyModule()})

    def test_scanner_rejects_bad_ndim_and_never_touches_device_methods(self):
        bad = FakeParameter((2, 3), ndim=1)
        root = module(parameters=[("weight", bad)])

        with self.assertRaisesRegex(ParameterScanError, "reports ndim=1"):
            scan_parameter_roots({"unet": root})

        self.assertEqual(bad.device_calls, [])

    def test_requires_grad_is_snapshotted_without_mutation(self):
        frozen = FakeParameter((2, 2), requires_grad=False)
        trainable = FakeParameter((2, 2), requires_grad=True)
        root = module(parameters=[("frozen", frozen), ("trainable", trainable)])

        descriptors = scan_parameter_roots({"qwen3": root})
        states = {item.canonical_name: item.requires_grad for item in descriptors}
        self.assertEqual(
            states,
            {
                "qwen3.frozen": False,
                "qwen3.trainable": True,
            },
        )
        self.assertFalse(frozen.requires_grad)
        self.assertTrue(trainable.requires_grad)

    def test_normalized_duplicate_roots_fail_closed(self):
        left = module()
        right = module()
        with self.assertRaisesRegex(ParameterScanError, "duplicated after normalization"):
            scan_parameter_roots({" UNET ": left, "unet": right})

    def test_semantic_parameter_name_cannot_refer_to_two_objects(self):
        class BrokenModule(FakeModule):
            def named_modules(self, *, remove_duplicate=True):
                if remove_duplicate is not False:
                    raise AssertionError
                return iter([("", self), ("x", self), ("x", module())])

        with self.assertRaisesRegex(ParameterScanError, "duplicate module path"):
            scan_parameter_roots({"unet": BrokenModule()})

    def test_empty_root_mapping_and_invalid_adapter_registry_fail_closed(self):
        with self.assertRaises(ParameterScanError):
            scan_parameter_roots({})

        root = module()
        with self.assertRaisesRegex(ParameterScanError, "integer object IDs"):
            scan_parameter_roots(
                {"unet": root},
                adapter_targets={"not-an-id": AdapterTargetMetadata(
                    "unet",
                    "attn",
                    "torch.nn.Linear",
                )},
            )


if __name__ == "__main__":
    unittest.main()
