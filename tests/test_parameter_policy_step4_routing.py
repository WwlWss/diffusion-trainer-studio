from __future__ import annotations

import unittest

from mikazuki.parameter_routing import (
    ADAPTER_TARGET_MARKER_ATTR,
    build_parameter_routing_plan,
    scan_parameter_roots,
)


class FakeParameter:
    def __init__(self, shape=(4, 4)):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = "float32"
        self.requires_grad = True


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
LoRAModule = type("LoRAModule", (FakeModule,), {})


def module(*, parameters=None, children=None, cls=FakeModule):
    return cls(parameters=parameters, children=children)


def marked_adapter(
    *,
    target_root,
    target_path,
    target_type="torch.nn.modules.linear.Linear",
    shape=(4, 4),
):
    parameter = FakeParameter(shape)
    adapter = module(
        cls=LoRAModule,
        children=[
            (
                "lora_down",
                module(
                    cls=Linear,
                    parameters=[("weight", parameter)],
                ),
            )
        ],
    )
    setattr(
        adapter,
        ADAPTER_TARGET_MARKER_ATTR,
        (target_root, target_path, target_type),
    )
    network = module(children=[("adapter", adapter)])
    return network, parameter


def muon_policy(component_id, *, lr=2e-4, fallback=True):
    route = {
        "train": True,
        "optimizer_profile": "muon",
        "learning_rate": lr,
    }
    profiles = {"muon": {"type": "Muon", "args": {}}}
    if fallback:
        profiles["fallback"] = {"type": "AdamW", "args": {}}
        route["fallback_optimizer_profile"] = "fallback"
        route["fallback_learning_rate"] = 7e-5
    return {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": {component_id: route},
    }


def route_marked_adapter(
    *,
    train_type,
    component_id,
    target_root,
    target_path,
    target_type="torch.nn.modules.linear.Linear",
    effective_config=None,
    fallback=True,
):
    network, parameter = marked_adapter(
        target_root=target_root,
        target_path=target_path,
        target_type=target_type,
    )
    descriptors = scan_parameter_roots({"network": network})
    plan = build_parameter_routing_plan(
        muon_policy(component_id, fallback=fallback),
        train_type=train_type,
        effective_config=effective_config or {},
        descriptors=descriptors,
    )
    return plan, parameter, descriptors[0]


class Step4EndToEndRoutingTests(unittest.TestCase):
    def assert_route(
        self,
        *,
        train_type,
        component_id,
        target_root,
        target_path,
        expected_route,
        target_type="torch.nn.modules.linear.Linear",
        effective_config=None,
    ):
        plan, parameter, descriptor = route_marked_adapter(
            train_type=train_type,
            component_id=component_id,
            target_root=target_root,
            target_path=target_path,
            target_type=target_type,
            effective_config=effective_config,
        )
        self.assertTrue(
            plan.is_valid,
            [(issue.code, issue.message) for issue in plan.issues],
        )
        self.assertEqual(len(plan.assignments), 1)
        assignment = plan.assignments[0]
        self.assertEqual(assignment.parameter_id, id(parameter))
        self.assertEqual(assignment.component_id, component_id)
        self.assertEqual(assignment.route_kind, expected_route)
        self.assertEqual(
            assignment.optimizer_profile,
            "muon" if expected_route == "primary" else "fallback",
        )
        self.assertEqual(plan.stats.total.tensors, 1)
        self.assertEqual(plan.stats.assigned.tensors, 1)
        self.assertEqual(
            plan.stats.by_component[component_id].tensors,
            1,
        )
        self.assertEqual(
            plan.stats.by_route[expected_route].tensors,
            1,
        )
        target = descriptor.canonical_alias.adapter_target
        self.assertIsNotNone(target)
        self.assertEqual(target.root, target_root)
        self.assertEqual(target.module_path, target_path)
        return plan

    def test_sd_attention_marker_routes_to_muon_primary(self):
        self.assert_route(
            train_type="sd-lora",
            component_id="unet.attention.adapter",
            target_root="unet",
            target_path="input_blocks.1.1.transformer_blocks.0.attn1.to_q",
            expected_route="primary",
            effective_config={"network_train_unet_only": True},
        )

    def test_sd_conv_marker_routes_to_fallback_even_when_lora_weight_is_2d(self):
        self.assert_route(
            train_type="sd-lora",
            component_id="unet.conv.adapter",
            target_root="unet",
            target_path="input_blocks.0.0",
            target_type="torch.nn.modules.conv.Conv2d",
            expected_route="fallback",
            effective_config={"network_train_unet_only": True},
        )

    def test_sdxl_text_encoder_marker_uses_original_hidden_path_for_muon(self):
        self.assert_route(
            train_type="sdxl-lora",
            component_id="text_encoder_1.adapter",
            target_root="text_encoder_1",
            target_path="text_model.encoder.layers.0.self_attn.q_proj",
            expected_route="primary",
            effective_config={"network_train_text_encoder_only": True},
        )

    def test_flux_single_stream_marker_routes_to_primary(self):
        self.assert_route(
            train_type="flux-lora",
            component_id="transformer.single_stream.adapter",
            target_root="transformer",
            target_path="single_blocks.0.linear1",
            expected_route="primary",
            effective_config={"network_train_unet_only": True},
        )

    def test_flux_modulation_target_routes_to_fallback(self):
        self.assert_route(
            train_type="flux-lora",
            component_id="transformer.double_stream.adapter",
            target_root="transformer",
            target_path="double_blocks.0.img_mod.lin",
            expected_route="fallback",
            effective_config={"network_train_unet_only": True},
        )

    def test_chroma_t5_marker_routes_as_text_hidden_primary(self):
        self.assert_route(
            train_type="chroma-lora",
            component_id="t5xxl.adapter",
            target_root="t5xxl",
            target_path="encoder.block.0.layer.0.SelfAttention.q",
            expected_route="primary",
            effective_config={"network_args": ["train_t5xxl=True"]},
        )

    def test_sd3_attention_marker_routes_to_primary(self):
        self.assert_route(
            train_type="sd3-lora",
            component_id="mmdit.attention.adapter",
            target_root="mmdit",
            target_path="joint_blocks.0.x_block.attn.qkv",
            expected_route="primary",
            effective_config={"network_train_unet_only": True},
        )

    def test_sd3_adaln_marker_routes_to_modulation_component_and_fallback(self):
        self.assert_route(
            train_type="sd3-lora",
            component_id="mmdit.modulation_norm.adapter",
            target_root="mmdit",
            target_path="joint_blocks.0.x_block.adaLN_modulation.1",
            expected_route="fallback",
            effective_config={"network_train_unet_only": True},
        )

    def test_anima_self_attention_marker_routes_to_primary(self):
        self.assert_route(
            train_type="anima-lora",
            component_id="dit.self_attention.adapter",
            target_root="dit",
            target_path="blocks.0.self_attn.qkv",
            expected_route="primary",
            effective_config={"network_train_unet_only": True},
        )

    def test_anima_llm_adapter_marker_routes_to_primary_when_target_enabled(self):
        self.assert_route(
            train_type="anima-lora",
            component_id="llm_adapter.adapter",
            target_root="dit",
            target_path="llm_adapter.blocks.0.self_attn.q_proj",
            expected_route="primary",
            effective_config={
                "network_train_unet_only": True,
                "network_args": ["train_llm_adapter=true"],
            },
        )

    def test_anima_qwen_marker_routes_to_text_hidden_primary(self):
        self.assert_route(
            train_type="anima-lora",
            component_id="qwen3.adapter",
            target_root="qwen3",
            target_path="model.layers.0.self_attn.q_proj",
            expected_route="primary",
            effective_config={"network_train_text_encoder_only": True},
        )

    def test_higher_level_target_still_caps_correct_metadata(self):
        network, parameter = marked_adapter(
            target_root="clip_l",
            target_path="text_model.encoder.layers.0.self_attn.q_proj",
        )
        descriptors = scan_parameter_roots({"network": network})
        plan = build_parameter_routing_plan(
            muon_policy("clip_l.adapter"),
            train_type="flux-lora",
            effective_config={"network_train_unet_only": True},
            descriptors=descriptors,
        )

        self.assertFalse(plan.is_valid)
        self.assertEqual(len(plan.assignments), 1)
        assignment = plan.assignments[0]
        self.assertEqual(assignment.parameter_id, id(parameter))
        self.assertEqual(assignment.component_id, "clip_l.adapter")
        self.assertEqual(assignment.route_kind, "unavailable")
        self.assertIsNone(assignment.optimizer_profile)
        self.assertIn(
            "target_unavailable_train_enabled",
            {issue.code for issue in plan.issues},
        )

    def test_missing_marker_still_fails_closed_in_public_plan(self):
        parameter = FakeParameter((4, 4))
        network = module(
            children=[
                (
                    "adapter",
                    module(
                        cls=LoRAModule,
                        children=[
                            (
                                "lora_down",
                                module(
                                    cls=Linear,
                                    parameters=[("weight", parameter)],
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
        descriptors = scan_parameter_roots({"network": network})
        plan = build_parameter_routing_plan(
            {
                "version": 1,
                "optimizer_profiles": {"main": {"type": "AdamW", "args": {}}},
                "components": {
                    "transformer.single_stream.adapter": {
                        "train": False,
                    }
                },
            },
            train_type="flux-lora",
            effective_config={"network_train_unet_only": True},
            descriptors=descriptors,
        )

        self.assertFalse(plan.is_valid)
        self.assertEqual(plan.stats.unassigned.tensors, 1)
        self.assertEqual(plan.assignments, ())


if __name__ == "__main__":
    unittest.main()
