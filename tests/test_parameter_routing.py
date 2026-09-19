from __future__ import annotations

import unittest

from mikazuki.parameter_routing import (
    AdapterTargetMetadata,
    ParameterScanError,
    _resolve_parameter_ownership_pass,
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


def _policy(components, *, profiles=None):
    return {
        "version": 1,
        "optimizer_profiles": profiles
        or {"main": {"type": "AdamW", "args": {}}},
        "components": components,
    }


def _train_route(profile="main", lr=1e-4):
    return {
        "train": True,
        "optimizer_profile": profile,
        "learning_rate": lr,
    }


def _nested_linear(parts, parameter, *, role="weight", leaf_cls=Linear):
    node = module(cls=leaf_cls, parameters=[(role, parameter)])
    for part in reversed(parts):
        node = module(children=[(part, node)])
    return node


def _flux_single_descriptor(
    *,
    component="double",
    parameter=None,
    role="weight",
    requires_grad=True,
):
    parameter = parameter or FakeParameter((4, 4), requires_grad=requires_grad)
    prefix = "double_blocks" if component == "double" else "single_blocks"
    root = _nested_linear((prefix, "0", "linear1"), parameter, role=role)
    return scan_parameter_roots({"transformer": root}), parameter


class ParameterOwnershipTests(unittest.TestCase):
    def test_train_true_available_component_becomes_trainable_candidate(self):
        descriptors, parameter = _flux_single_descriptor()
        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream": _train_route()}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(result.assignments, ())
        self.assertEqual(len(result.trainable), 1)
        candidate = result.trainable[0]
        self.assertIs(candidate.descriptor.parameter, parameter)
        self.assertEqual(candidate.component_id, "transformer.double_stream")
        self.assertEqual(candidate.parameter_class, "matrix_weight")
        self.assertTrue(candidate.route["train"])
        self.assertEqual(candidate.route["optimizer_profile"], "main")
        self.assertFalse(any(issue.severity == "error" for issue in result.issues))

    def test_train_false_available_component_routes_frozen_without_mutation(self):
        descriptors, parameter = _flux_single_descriptor(requires_grad=True)
        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(result.trainable, ())
        self.assertEqual(len(result.assignments), 1)
        assignment = result.assignments[0]
        self.assertIs(assignment.parameter, parameter)
        self.assertEqual(assignment.route_kind, "frozen")
        self.assertIsNone(assignment.optimizer_profile)
        self.assertIsNone(assignment.learning_rate)
        self.assertTrue(parameter.requires_grad)

    def test_target_unavailable_parameter_routes_unavailable_without_policy_row(self):
        parameter = FakeParameter((4, 4))
        root = module(
            children=[("encoder", module(cls=Linear, parameters=[("weight", parameter)]))]
        )
        descriptors = scan_parameter_roots({"text_encoder_1": root})

        result = _resolve_parameter_ownership_pass(
            _policy({"unet.transformer": {"train": False}}),
            train_type="sdxl-finetune",
            effective_config={"train_text_encoder": False},
            descriptors=descriptors,
        )

        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(result.assignments[0].route_kind, "unavailable")
        self.assertEqual(result.assignments[0].component_id, "text_encoder_1")
        self.assertFalse(
            any(issue.code == "missing_component_policy" for issue in result.issues)
        )

    def test_target_unavailable_train_true_is_error_but_parameter_stays_unavailable(self):
        parameter = FakeParameter((4, 4))
        root = module(
            children=[("encoder", module(cls=Linear, parameters=[("weight", parameter)]))]
        )
        descriptors = scan_parameter_roots({"text_encoder_1": root})

        result = _resolve_parameter_ownership_pass(
            _policy({"text_encoder_1": _train_route()}),
            train_type="sdxl-finetune",
            effective_config={"train_text_encoder": False},
            descriptors=descriptors,
        )

        self.assertEqual(result.assignments[0].route_kind, "unavailable")
        self.assertIn(
            "target_unavailable_train_enabled",
            {issue.code for issue in result.issues},
        )

    def test_missing_policy_row_is_aggregated_per_available_component(self):
        first = FakeParameter((4, 4))
        second = FakeParameter((4, 4))
        root = module(
            children=[
                (
                    "double_blocks",
                    module(
                        children=[
                            (
                                "0",
                                module(
                                    children=[
                                        ("a", module(cls=Linear, parameters=[("weight", first)])),
                                        ("b", module(cls=Linear, parameters=[("weight", second)])),
                                    ]
                                ),
                            )
                        ]
                    ),
                )
            ]
        )
        descriptors = scan_parameter_roots({"transformer": root})

        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.final": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        issues = [issue for issue in result.issues if issue.code == "missing_component_policy"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].component_id, "transformer.double_stream")
        self.assertEqual(issues[0].count, 2)
        self.assertEqual(len(result.unassigned), 2)

    def test_unknown_policy_component_fails_closed_even_when_frozen(self):
        descriptors, _ = _flux_single_descriptor()
        result = _resolve_parameter_ownership_pass(
            _policy(
                {
                    "transformer.double_stream": {"train": False},
                    "made.up.component": {"train": False},
                }
            ),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        issue = next(
            issue for issue in result.issues if issue.code == "unknown_policy_component"
        )
        self.assertEqual(issue.component_id, "made.up.component")

    def test_shared_parameter_with_component_conflict_is_not_routed(self):
        shared = FakeParameter((4, 4))
        root = module(
            children=[
                (
                    "double_blocks",
                    module(
                        children=[
                            (
                                "0",
                                module(
                                    children=[
                                        ("linear1", module(cls=Linear, parameters=[("weight", shared)]))
                                    ]
                                ),
                            )
                        ]
                    ),
                ),
                (
                    "single_blocks",
                    module(
                        children=[
                            (
                                "0",
                                module(
                                    children=[
                                        ("linear1", module(cls=Linear, parameters=[("weight", shared)]))
                                    ]
                                ),
                            )
                        ]
                    ),
                ),
            ]
        )
        descriptors = scan_parameter_roots({"transformer": root})
        result = _resolve_parameter_ownership_pass(
            _policy(
                {
                    "transformer.double_stream": {"train": False},
                    "transformer.single_stream": {"train": False},
                }
            ),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(result.assignments, ())
        self.assertEqual(result.trainable, ())
        self.assertEqual(len(result.conflicts), 1)
        self.assertIn("alias_component_conflict", {issue.code for issue in result.issues})
        self.assertEqual(
            result.observed_components,
            frozenset({"transformer.double_stream", "transformer.single_stream"}),
        )

    def test_shared_parameter_with_parameter_class_conflict_is_not_routed(self):
        shared = FakeParameter((4, 4))
        root = module(
            children=[
                (
                    "double_blocks",
                    module(
                        children=[
                            (
                                "0",
                                module(
                                    children=[
                                        ("a", module(cls=Linear, parameters=[("weight", shared)])),
                                        ("b", module(cls=Linear, parameters=[("bias", shared)])),
                                    ]
                                ),
                            )
                        ]
                    ),
                )
            ]
        )
        descriptors = scan_parameter_roots({"transformer": root})
        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(len(result.conflicts), 1)
        self.assertIn(
            "alias_parameter_class_conflict",
            {issue.code for issue in result.issues},
        )

    def test_shared_lora_parameter_with_adapter_target_conflict_is_not_routed(self):
        shared = FakeParameter((4, 4))
        left_adapter = module(
            cls=LoRAModule,
            children=[("lora_down", module(cls=Linear, parameters=[("weight", shared)]))],
        )
        right_adapter = module(
            cls=LoRAModule,
            children=[("lora_down", module(cls=Linear, parameters=[("weight", shared)]))],
        )
        root = module(children=[("left", left_adapter), ("right", right_adapter)])
        descriptors = scan_parameter_roots(
            {"network": root},
            adapter_targets={
                id(left_adapter): AdapterTargetMetadata(
                    "transformer",
                    "double_blocks.0.img_attn.qkv",
                    "torch.nn.Linear",
                ),
                id(right_adapter): AdapterTargetMetadata(
                    "transformer",
                    "double_blocks.1.img_attn.qkv",
                    "torch.nn.Linear",
                ),
            },
        )

        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream.adapter": {"train": False}}),
            train_type="flux-lora",
            effective_config={"network_train_unet_only": True},
            descriptors=descriptors,
        )

        self.assertEqual(len(result.conflicts), 1)
        self.assertIn(
            "alias_adapter_target_conflict",
            {issue.code for issue in result.issues},
        )

    def test_lora_without_original_target_metadata_is_unassigned_not_guessed(self):
        parameter = FakeParameter((4, 4))
        adapter = module(
            cls=LoRAModule,
            children=[("lora_down", module(cls=Linear, parameters=[("weight", parameter)]))],
        )
        descriptors = scan_parameter_roots(
            {"network": module(children=[("x", adapter)])}
        )

        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream.adapter": {"train": False}}),
            train_type="flux-lora",
            effective_config={"network_train_unet_only": True},
            descriptors=descriptors,
        )

        self.assertEqual(len(result.unassigned), 1)
        self.assertEqual(result.assignments, ())
        self.assertIn("unassigned_parameter", {issue.code for issue in result.issues})

    def test_duplicate_descriptor_identity_is_rejected_before_routing(self):
        descriptors, _ = _flux_single_descriptor()
        descriptor = descriptors[0]

        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=[descriptor, descriptor],
        )

        self.assertIn(
            "duplicate_descriptor_identity",
            {issue.code for issue in result.issues},
        )
        self.assertEqual(result.assignments, ())
        self.assertEqual(result.trainable, ())

    def test_requires_grad_does_not_suppress_trainable_ownership(self):
        descriptors, parameter = _flux_single_descriptor(requires_grad=False)
        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream": _train_route()}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(len(result.trainable), 1)
        self.assertFalse(result.trainable[0].descriptor.requires_grad)
        self.assertFalse(parameter.requires_grad)

    def test_assignment_order_is_semantic_not_object_identity(self):
        first = FakeParameter((4, 4))
        second = FakeParameter((4, 4))
        root = module(
            children=[
                (
                    "single_blocks",
                    module(
                        children=[
                            (
                                "0",
                                module(
                                    children=[
                                        ("z", module(cls=Linear, parameters=[("weight", second)])),
                                        ("a", module(cls=Linear, parameters=[("weight", first)])),
                                    ]
                                ),
                            )
                        ]
                    ),
                )
            ]
        )
        descriptors = scan_parameter_roots({"transformer": root})
        result = _resolve_parameter_ownership_pass(
            _policy({"transformer.single_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )
        self.assertEqual(
            [item.canonical_name for item in result.assignments],
            [
                "transformer.single_blocks.0.a.weight",
                "transformer.single_blocks.0.z.weight",
            ],
        )


if __name__ == "__main__":
    unittest.main()
