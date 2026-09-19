from __future__ import annotations

import unittest

from mikazuki.parameter_routing import (
    AdapterTargetMetadata,
    ParameterDescriptor,
    ParameterScanError,
    RoutingAssignment,
    _audit_final_assignments,
    _resolve_parameter_ownership_pass,
    _route_trainable_ownership,
    build_parameter_routing_plan,
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


def _muon_profiles():
    return {
        "muon": {"type": "Muon", "args": {}},
        "fallback": {"type": "AdamW", "args": {}},
    }


def _flux_double_descriptors(*, include_bias=True):
    weight = FakeParameter((4, 4))
    params = [("weight", weight)]
    bias = None
    if include_bias:
        bias = FakeParameter((4,))
        params.append(("bias", bias))
    root = _nested_linear(
        ("double_blocks", "0", "linear1"),
        weight,
    )
    leaf = root._children[0][1]._children[0][1]._children[0][1]
    leaf._local_parameters = params
    return scan_parameter_roots({"transformer": root}), weight, bias


def _route_3d(policy, *, train_type, effective_config, descriptors):
    ownership = _resolve_parameter_ownership_pass(
        policy,
        train_type=train_type,
        effective_config=effective_config,
        descriptors=descriptors,
    )
    return ownership, _route_trainable_ownership(ownership)


class OptimizerEligibilityRoutingTests(unittest.TestCase):
    def test_ordinary_optimizer_routes_primary_without_eligibility(self):
        descriptors, parameter = _flux_single_descriptor()
        ownership, routed = _route_3d(
            _policy({"transformer.double_stream": _train_route()}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertFalse(any(issue.severity == "error" for issue in ownership.issues))
        self.assertEqual(len(routed.assignments), 1)
        assignment = routed.assignments[0]
        self.assertIs(assignment.parameter, parameter)
        self.assertEqual(assignment.route_kind, "primary")
        self.assertEqual(assignment.optimizer_profile, "main")
        self.assertEqual(assignment.learning_rate, 1e-4)
        self.assertEqual(routed.unroutable, ())

    def test_pure_muon_component_needs_no_fallback_when_all_real_parameters_are_eligible(self):
        descriptors, _, _ = _flux_double_descriptors(include_bias=False)
        policy = _policy(
            {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-4,
                }
            },
            profiles=_muon_profiles(),
        )
        _, routed = _route_3d(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(len(routed.assignments), 1)
        self.assertEqual(routed.assignments[0].route_kind, "primary")
        self.assertEqual(routed.assignments[0].optimizer_profile, "muon")
        self.assertEqual(routed.assignments[0].learning_rate, 2e-4)
        self.assertFalse(any(issue.code == "missing_fallback" for issue in routed.issues))

    def test_mixed_muon_component_splits_weight_to_primary_and_bias_to_fallback(self):
        descriptors, weight, bias = _flux_double_descriptors(include_bias=True)
        policy = _policy(
            {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-4,
                    "fallback_optimizer_profile": "fallback",
                }
            },
            profiles=_muon_profiles(),
        )
        _, routed = _route_3d(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertEqual(len(routed.assignments), 2)
        by_parameter = {item.parameter_id: item for item in routed.assignments}
        self.assertEqual(by_parameter[id(weight)].route_kind, "primary")
        self.assertEqual(by_parameter[id(weight)].optimizer_profile, "muon")
        self.assertEqual(by_parameter[id(bias)].route_kind, "fallback")
        self.assertEqual(by_parameter[id(bias)].optimizer_profile, "fallback")
        self.assertEqual(by_parameter[id(bias)].learning_rate, 2e-4)
        self.assertEqual(
            dict(routed.fallback_usage),
            {"transformer.double_stream": 1},
        )
        self.assertEqual(routed.unroutable, ())

    def test_explicit_fallback_learning_rate_overrides_component_lr(self):
        descriptors, _, bias = _flux_double_descriptors(include_bias=True)
        policy = _policy(
            {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-4,
                    "fallback_optimizer_profile": "fallback",
                    "fallback_learning_rate": 7e-5,
                }
            },
            profiles=_muon_profiles(),
        )
        _, routed = _route_3d(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )
        fallback = next(
            item for item in routed.assignments if item.parameter_id == id(bias)
        )
        self.assertEqual(fallback.route_kind, "fallback")
        self.assertEqual(fallback.learning_rate, 7e-5)

    def test_missing_fallback_errors_only_for_real_ineligible_parameters(self):
        descriptors, weight, bias = _flux_double_descriptors(include_bias=True)
        policy = _policy(
            {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-4,
                }
            },
            profiles=_muon_profiles(),
        )
        _, routed = _route_3d(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        by_parameter = {item.parameter_id: item for item in routed.assignments}
        self.assertIn(id(weight), by_parameter)
        self.assertNotIn(id(bias), by_parameter)
        self.assertEqual(len(routed.unroutable), 1)
        self.assertIs(routed.unroutable[0].parameter, bias)
        issue = next(issue for issue in routed.issues if issue.code == "missing_fallback")
        self.assertEqual(issue.component_id, "transformer.double_stream")
        self.assertEqual(issue.count, 1)

        eligible_only, _, _ = _flux_double_descriptors(include_bias=False)
        _, eligible_routed = _route_3d(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=eligible_only,
        )
        self.assertFalse(
            any(issue.code == "missing_fallback" for issue in eligible_routed.issues)
        )
        self.assertEqual(eligible_routed.unroutable, ())

    def test_alias_eligibility_conflict_keeps_one_physical_parameter_out_of_all_routes(self):
        shared = FakeParameter((4, 4))
        hidden = _nested_linear(
            ("model", "layers", "0", "self_attn", "q_proj"),
            shared,
        )
        embedding = _nested_linear(("embed_tokens",), shared)
        qwen_root = module(
            children=[
                ("hidden", hidden),
                ("embedding_alias", embedding),
            ]
        )
        descriptors = scan_parameter_roots({"qwen3": qwen_root})
        policy = _policy(
            {
                "qwen3": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 1e-4,
                    "fallback_optimizer_profile": "fallback",
                }
            },
            profiles=_muon_profiles(),
        )

        ownership, routed = _route_3d(
            policy,
            train_type="anima-finetune",
            effective_config={"train_qwen3_text_encoder": True},
            descriptors=descriptors,
        )

        self.assertEqual(len(ownership.trainable), 1)
        self.assertEqual(routed.assignments, ())
        self.assertEqual(routed.unroutable, ())
        self.assertEqual(len(routed.conflicts), 1)
        issue = next(
            issue for issue in routed.issues if issue.code == "alias_eligibility_conflict"
        )
        self.assertEqual(issue.parameter_id, id(shared))
        self.assertEqual(issue.count, 2)
        self.assertTrue(any("eligible" in example for example in issue.examples))
        self.assertTrue(any("ineligible" in example for example in issue.examples))

    def test_frozen_and_unavailable_assignments_never_enter_optimizer_routing(self):
        frozen_descriptors, _ = _flux_single_descriptor()
        frozen_ownership = _resolve_parameter_ownership_pass(
            _policy({"transformer.double_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=frozen_descriptors,
        )
        frozen_routed = _route_trainable_ownership(frozen_ownership)
        self.assertEqual(frozen_routed.assignments, ())
        self.assertEqual(frozen_routed.unroutable, ())

        te_param = FakeParameter((4, 4))
        te_root = module(
            children=[("encoder", module(cls=Linear, parameters=[("weight", te_param)]))]
        )
        unavailable_descriptors = scan_parameter_roots({"text_encoder_1": te_root})
        unavailable_ownership = _resolve_parameter_ownership_pass(
            _policy({"unet.transformer": {"train": False}}),
            train_type="sdxl-finetune",
            effective_config={"train_text_encoder": False},
            descriptors=unavailable_descriptors,
        )
        unavailable_routed = _route_trainable_ownership(unavailable_ownership)
        self.assertEqual(unavailable_routed.assignments, ())
        self.assertEqual(unavailable_routed.unroutable, ())


class FinalRoutingPlanTests(unittest.TestCase):
    def test_public_plan_combines_primary_fallback_and_stats(self):
        descriptors, weight, bias = _flux_double_descriptors(include_bias=True)
        policy = _policy(
            {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-4,
                    "fallback_optimizer_profile": "fallback",
                }
            },
            profiles=_muon_profiles(),
        )

        plan = build_parameter_routing_plan(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertTrue(plan.is_valid)
        by_id = {item.parameter_id: item for item in plan.assignments}
        self.assertEqual(by_id[id(weight)].route_kind, "primary")
        self.assertEqual(by_id[id(bias)].route_kind, "fallback")
        self.assertEqual(plan.stats.total.tensors, 2)
        self.assertEqual(plan.stats.total.numel, 20)
        self.assertEqual(plan.stats.assigned.tensors, 2)
        self.assertEqual(plan.stats.unassigned.tensors, 0)
        self.assertEqual(plan.stats.conflicts.tensors, 0)
        self.assertEqual(plan.stats.unroutable.tensors, 0)
        self.assertEqual(plan.stats.by_component["transformer.double_stream"].tensors, 2)
        self.assertEqual(plan.stats.by_route["primary"].tensors, 1)
        self.assertEqual(plan.stats.by_route["fallback"].tensors, 1)
        self.assertEqual(plan.stats.by_parameter_class["matrix_weight"].tensors, 1)
        self.assertEqual(plan.stats.by_parameter_class["bias"].tensors, 1)

    def test_train_true_absent_component_is_hard_error(self):
        plan = build_parameter_routing_plan(
            _policy({"transformer.double_stream": _train_route()}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=(),
        )

        self.assertFalse(plan.is_valid)
        issue = next(issue for issue in plan.issues if issue.code == "component_absent")
        self.assertEqual(issue.component_id, "transformer.double_stream")
        self.assertEqual(issue.count, 0)
        self.assertEqual(plan.stats.total.tensors, 0)

    def test_train_false_absent_component_is_allowed(self):
        plan = build_parameter_routing_plan(
            _policy({"transformer.double_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=(),
        )

        self.assertTrue(plan.is_valid)
        self.assertFalse(any(issue.code == "component_absent" for issue in plan.issues))

    def test_target_unavailable_absent_component_does_not_add_absent_noise(self):
        plan = build_parameter_routing_plan(
            _policy({"text_encoder_1": {"train": False}}),
            train_type="sdxl-finetune",
            effective_config={"train_text_encoder": False},
            descriptors=(),
        )
        self.assertTrue(plan.is_valid)
        self.assertFalse(any(issue.code == "component_absent" for issue in plan.issues))

        illegal = build_parameter_routing_plan(
            _policy({"text_encoder_1": _train_route()}),
            train_type="sdxl-finetune",
            effective_config={"train_text_encoder": False},
            descriptors=(),
        )
        self.assertFalse(illegal.is_valid)
        self.assertIn(
            "target_unavailable_train_enabled",
            {issue.code for issue in illegal.issues},
        )
        self.assertNotIn("component_absent", {issue.code for issue in illegal.issues})

    def test_conflicting_candidate_suppresses_redundant_component_absent(self):
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
        plan = build_parameter_routing_plan(
            _policy({"transformer.double_stream": _train_route()}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertFalse(plan.is_valid)
        self.assertIn(
            "alias_parameter_class_conflict",
            {issue.code for issue in plan.issues},
        )
        self.assertNotIn("component_absent", {issue.code for issue in plan.issues})
        self.assertEqual(plan.stats.conflicts.tensors, 1)

    def test_unused_fallback_is_warning_only_when_every_parameter_is_primary_eligible(self):
        descriptors, _, _ = _flux_double_descriptors(include_bias=False)
        policy = _policy(
            {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-4,
                    "fallback_optimizer_profile": "fallback",
                }
            },
            profiles=_muon_profiles(),
        )
        plan = build_parameter_routing_plan(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertTrue(plan.is_valid)
        warning = next(issue for issue in plan.issues if issue.code == "unused_fallback")
        self.assertEqual(warning.severity, "warning")
        self.assertEqual(warning.component_id, "transformer.double_stream")

        mixed, _, _ = _flux_double_descriptors(include_bias=True)
        mixed_plan = build_parameter_routing_plan(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=mixed,
        )
        self.assertFalse(any(issue.code == "unused_fallback" for issue in mixed_plan.issues))

    def test_missing_fallback_is_unroutable_not_conflict(self):
        descriptors, _, bias = _flux_double_descriptors(include_bias=True)
        plan = build_parameter_routing_plan(
            _policy(
                {
                    "transformer.double_stream": {
                        "train": True,
                        "optimizer_profile": "muon",
                        "learning_rate": 2e-4,
                    }
                },
                profiles=_muon_profiles(),
            ),
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )

        self.assertFalse(plan.is_valid)
        self.assertEqual(plan.stats.unroutable.tensors, 1)
        self.assertEqual(plan.stats.conflicts.tensors, 0)
        self.assertEqual(plan.stats.assigned.tensors, 1)
        self.assertNotIn(
            id(bias),
            {assignment.parameter_id for assignment in plan.assignments},
        )

    def test_eligibility_disagreement_is_conflict_not_unroutable(self):
        shared = FakeParameter((4, 4))
        hidden = _nested_linear(
            ("model", "layers", "0", "self_attn", "q_proj"),
            shared,
        )
        embedding = _nested_linear(("embed_tokens",), shared)
        descriptors = scan_parameter_roots(
            {
                "qwen3": module(
                    children=[
                        ("hidden", hidden),
                        ("embedding_alias", embedding),
                    ]
                )
            }
        )
        plan = build_parameter_routing_plan(
            _policy(
                {
                    "qwen3": {
                        "train": True,
                        "optimizer_profile": "muon",
                        "learning_rate": 1e-4,
                        "fallback_optimizer_profile": "fallback",
                    }
                },
                profiles=_muon_profiles(),
            ),
            train_type="anima-finetune",
            effective_config={"train_qwen3_text_encoder": True},
            descriptors=descriptors,
        )

        self.assertFalse(plan.is_valid)
        self.assertEqual(plan.stats.conflicts.tensors, 1)
        self.assertEqual(plan.stats.unroutable.tensors, 0)
        self.assertEqual(plan.assignments, ())

    def test_unassigned_stats_cover_missing_lora_metadata(self):
        parameter = FakeParameter((4, 4))
        adapter = module(
            cls=LoRAModule,
            children=[("lora_down", module(cls=Linear, parameters=[("weight", parameter)]))],
        )
        descriptors = scan_parameter_roots(
            {"network": module(children=[("x", adapter)])}
        )
        plan = build_parameter_routing_plan(
            _policy({"transformer.double_stream.adapter": {"train": False}}),
            train_type="flux-lora",
            effective_config={"network_train_unet_only": True},
            descriptors=descriptors,
        )

        self.assertFalse(plan.is_valid)
        self.assertEqual(plan.stats.unassigned.tensors, 1)
        self.assertEqual(plan.stats.assigned.tensors, 0)

    def test_frozen_and_unavailable_are_counted_as_assigned_routes(self):
        frozen_descriptors, _ = _flux_single_descriptor()
        frozen_plan = build_parameter_routing_plan(
            _policy({"transformer.double_stream": {"train": False}}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=frozen_descriptors,
        )
        self.assertTrue(frozen_plan.is_valid)
        self.assertEqual(frozen_plan.stats.assigned.tensors, 1)
        self.assertEqual(frozen_plan.stats.by_route["frozen"].tensors, 1)

        te_param = FakeParameter((4, 4))
        te_root = module(
            children=[("encoder", module(cls=Linear, parameters=[("weight", te_param)]))]
        )
        unavailable_descriptors = scan_parameter_roots({"text_encoder_1": te_root})
        unavailable_plan = build_parameter_routing_plan(
            _policy({"unet.transformer": {"train": False}}),
            train_type="sdxl-finetune",
            effective_config={"train_text_encoder": False},
            descriptors=unavailable_descriptors,
        )
        self.assertTrue(unavailable_plan.is_valid)
        self.assertEqual(unavailable_plan.stats.assigned.tensors, 1)
        self.assertEqual(unavailable_plan.stats.by_route["unavailable"].tensors, 1)

    def test_assignment_audit_removes_duplicate_physical_ownership(self):
        parameter = FakeParameter((2, 2))
        descriptor = ParameterDescriptor(
            parameter=parameter,
            parameter_id=id(parameter),
            aliases=(),
            shape=(2, 2),
            ndim=2,
            numel=4,
            dtype="float32",
            requires_grad=True,
        )
        first = RoutingAssignment(
            parameter=parameter,
            parameter_id=id(parameter),
            canonical_name="x.weight",
            parameter_class="matrix_weight",
            component_id="x",
            route_kind="primary",
            optimizer_profile="a",
            learning_rate=1e-4,
            reason="test",
        )
        second = RoutingAssignment(
            parameter=parameter,
            parameter_id=id(parameter),
            canonical_name="x.weight",
            parameter_class="matrix_weight",
            component_id="x",
            route_kind="fallback",
            optimizer_profile="b",
            learning_rate=1e-4,
            reason="test",
        )

        valid, issues, conflicts = _audit_final_assignments(
            (first, second),
            (descriptor,),
        )
        self.assertEqual(valid, ())
        self.assertEqual(len(conflicts), 1)
        self.assertIn(
            "duplicate_parameter_assignment",
            {issue.code for issue in issues},
        )

    def test_public_assignment_order_and_stats_maps_are_deterministic(self):
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
        plan = build_parameter_routing_plan(
            _policy({"transformer.single_stream": _train_route()}),
            train_type="flux-finetune",
            effective_config={},
            descriptors=tuple(reversed(descriptors)),
        )

        self.assertTrue(plan.is_valid)
        self.assertEqual(
            [item.canonical_name for item in plan.assignments],
            [
                "transformer.single_blocks.0.a.weight",
                "transformer.single_blocks.0.z.weight",
            ],
        )
        self.assertEqual(
            list(plan.stats.by_component),
            sorted(plan.stats.by_component),
        )
        self.assertEqual(
            list(plan.stats.by_route),
            sorted(plan.stats.by_route),
        )
        self.assertEqual(
            list(plan.stats.by_parameter_class),
            sorted(plan.stats.by_parameter_class),
        )


if __name__ == "__main__":
    unittest.main()
