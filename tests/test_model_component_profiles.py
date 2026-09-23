from dataclasses import dataclass
import unittest

from mikazuki.model_component_profiles import (
    get_model_component_profile,
    list_model_component_profiles,
    resolve_training_target_profile,
)


@dataclass(frozen=True)
class Target:
    root: str
    module_path: str
    module_type: str = "torch.nn.modules.linear.Linear"


@dataclass(frozen=True)
class Alias:
    root: str
    module_path: str
    parameter_class: str = "matrix_weight"
    module_type: str = "torch.nn.modules.linear.Linear"
    module_class: str = "Linear"
    ancestor_module_types: tuple[str, ...] = ()
    ancestor_module_classes: tuple[str, ...] = ()
    parameter_role: str = "weight"
    adapter_target: Target | None = None


class ModelComponentProfileTests(unittest.TestCase):
    def test_registry_covers_every_current_effective_backend(self):
        expected = {
            "sd-lora",
            "sdxl-lora",
            "sd-dreambooth",
            "sdxl-finetune",
            "flux-lora",
            "chroma-lora",
            "anima-lora",
            "flux-finetune",
            "anima-finetune",
            "sd3-lora",
        }
        actual = {profile.train_type for profile in list_model_component_profiles()}
        self.assertEqual(actual, expected)

        with self.assertRaisesRegex(ValueError, "No Parameter Policy Model Component Profile"):
            get_model_component_profile("unknown-backend")

    def test_sd_full_classification_uses_module_ancestry(self):
        profile = get_model_component_profile("sdxl-finetune")
        transformer = Alias(
            "unet",
            "input_blocks.4.1.transformer_blocks.0.attn1.to_q",
            ancestor_module_classes=(
                "SdxlUNet2DConditionModel",
                "Transformer2DModel",
                "BasicTransformerBlock",
                "CrossAttention",
            ),
        )
        transformer_bias = Alias(
            "unet",
            "input_blocks.4.1.transformer_blocks.0.attn1.to_out.0",
            parameter_class="bias",
            parameter_role="bias",
            ancestor_module_classes=(
                "SdxlUNet2DConditionModel",
                "Transformer2DModel",
                "BasicTransformerBlock",
                "CrossAttention",
            ),
        )
        resnet = Alias(
            "unet",
            "input_blocks.4.0.time_emb_proj",
            ancestor_module_classes=("SdxlUNet2DConditionModel", "ResnetBlock2D"),
        )
        outer_norm = Alias(
            "unet",
            "out.0",
            parameter_class="norm_weight",
            module_class="GroupNorm",
            ancestor_module_classes=("SdxlUNet2DConditionModel",),
        )
        base = Alias("unet", "time_embed.0")

        self.assertEqual(profile.classify_alias(transformer), "unet.transformer")
        self.assertEqual(profile.classify_alias(transformer_bias), "unet.transformer")
        self.assertEqual(profile.classify_alias(resnet), "unet.conv_resnet")
        self.assertEqual(profile.classify_alias(outer_norm), "unet.norm_bias_other")
        self.assertEqual(profile.classify_alias(base), "unet.base_other")
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                transformer,
                "unet.transformer",
            )[0]
        )
        self.assertFalse(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                transformer_bias,
                "unet.transformer",
            )[0]
        )

    def test_flux_full_respects_real_double_single_and_modulation_structure(self):
        profile = get_model_component_profile("flux-finetune")
        double = Alias("transformer", "double_blocks.0.img_attn.qkv")
        double_bias = Alias(
            "transformer",
            "double_blocks.0.img_attn.qkv",
            parameter_class="bias",
            parameter_role="bias",
        )
        single = Alias("transformer", "single_blocks.0.linear1")
        modulation = Alias(
            "transformer",
            "single_blocks.0.modulation.lin",
            ancestor_module_classes=("Flux", "SingleStreamBlock", "Modulation"),
        )
        input_projection = Alias("transformer", "img_in")
        final_projection = Alias("transformer", "final_layer.linear")

        self.assertEqual(profile.classify_alias(double), "transformer.double_stream")
        self.assertEqual(profile.classify_alias(double_bias), "transformer.double_stream")
        self.assertEqual(profile.classify_alias(single), "transformer.single_stream")
        self.assertEqual(
            profile.classify_alias(modulation),
            "transformer.modulation_norm_other",
        )
        self.assertEqual(
            profile.classify_alias(input_projection),
            "transformer.input_conditioning",
        )
        self.assertEqual(profile.classify_alias(final_projection), "transformer.final")
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                single,
                "transformer.single_stream",
            )[0]
        )
        self.assertFalse(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                double_bias,
                "transformer.double_stream",
            )[0]
        )
        self.assertFalse(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                input_projection,
                "transformer.input_conditioning",
            )[0]
        )

    def test_anima_modulation_wins_before_self_attention_name_fragment(self):
        profile = get_model_component_profile("anima-finetune")
        self_attn = Alias("dit", "blocks.0.self_attn.q_proj")
        cross_attn = Alias("dit", "blocks.0.cross_attn.q_proj")
        mlp = Alias("dit", "blocks.0.mlp.layer1")
        modulation = Alias("dit", "blocks.0.adaln_modulation_self_attn.1")
        llm_hidden = Alias("dit", "llm_adapter.blocks.0.cross_attn.q_proj")
        llm_output = Alias("dit", "llm_adapter.out_proj")
        qwen = Alias("qwen3", "model.layers.0.self_attn.q_proj")

        self.assertEqual(profile.classify_alias(self_attn), "dit.self_attention")
        self.assertEqual(profile.classify_alias(cross_attn), "dit.cross_attention")
        self.assertEqual(profile.classify_alias(mlp), "dit.mlp")
        self.assertEqual(profile.classify_alias(modulation), "dit.modulation")
        self.assertEqual(profile.classify_alias(llm_hidden), "dit.llm_adapter")
        self.assertEqual(profile.classify_alias(qwen), "qwen3")
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                llm_hidden,
                "dit.llm_adapter",
            )[0]
        )
        self.assertFalse(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                llm_output,
                "dit.llm_adapter",
            )[0]
        )
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                qwen,
                "qwen3",
            )[0]
        )

    def test_lora_classification_requires_explicit_original_target_metadata(self):
        profile = get_model_component_profile("flux-lora")
        self.assertIsNone(profile.classify_alias(Alias("network", "lora_down")))

        hidden = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "transformer",
                "double_blocks.0.img_attn.qkv",
            ),
        )
        modulation = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "transformer",
                "double_blocks.0.img_mod.lin",
            ),
        )
        self.assertEqual(
            profile.classify_alias(hidden),
            "transformer.double_stream.adapter",
        )
        self.assertEqual(
            profile.classify_alias(modulation),
            "transformer.double_stream.adapter",
        )
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                hidden,
                "transformer.double_stream.adapter",
            )[0]
        )
        self.assertFalse(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                modulation,
                "transformer.double_stream.adapter",
            )[0]
        )

    def test_chroma_never_defines_or_classifies_clip_l(self):
        profile = get_model_component_profile("chroma-lora")
        self.assertNotIn("clip_l.adapter", profile.components)
        clip_alias = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "clip_l",
                "text_model.encoder.layers.0.self_attn.q_proj",
            ),
        )
        self.assertIsNone(profile.classify_alias(clip_alias))

    def test_sd3_metadata_driven_components(self):
        profile = get_model_component_profile("sd3-lora")
        attention = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "mmdit",
                "joint_blocks.0.x_block.attn.qkv",
            ),
        )
        mlp = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "mmdit",
                "joint_blocks.0.x_block.mlp.fc1",
            ),
        )
        modulation = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "mmdit",
                "joint_blocks.0.x_block.adaLN_modulation.1",
            ),
        )
        t5 = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "t5xxl",
                "encoder.block.0.layer.0.SelfAttention.q",
            ),
        )

        self.assertEqual(
            profile.classify_alias(attention),
            "mmdit.attention.adapter",
        )
        self.assertEqual(profile.classify_alias(mlp), "mmdit.mlp.adapter")
        self.assertEqual(
            profile.classify_alias(modulation),
            "mmdit.modulation_norm.adapter",
        )
        self.assertEqual(profile.classify_alias(t5), "t5xxl.adapter")
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                attention,
                "mmdit.attention.adapter",
            )[0]
        )
        self.assertFalse(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                modulation,
                "mmdit.modulation_norm.adapter",
            )[0]
        )
        self.assertTrue(
            profile.eligibility_check(
                "model_hidden_2d_weight",
                t5,
                "t5xxl.adapter",
            )[0]
        )

    def test_sd_lora_target_availability_is_authoritative(self):
        unet = resolve_training_target_profile(
            "sd-lora",
            {"network_train_unet_only": True},
        )
        self.assertIn("unet.attention.adapter", unet.available_components)
        self.assertNotIn("text_encoder.adapter", unet.available_components)

        te = resolve_training_target_profile(
            "sd-lora",
            {"network_train_text_encoder_only": True},
        )
        self.assertEqual(
            te.available_components,
            frozenset({"text_encoder.adapter"}),
        )

        joint = resolve_training_target_profile("sd-lora", {})
        self.assertEqual(
            joint.available_components,
            frozenset(get_model_component_profile("sd-lora").components),
        )

        with self.assertRaises(ValueError):
            resolve_training_target_profile(
                "sd-lora",
                {
                    "network_train_unet_only": True,
                    "network_train_text_encoder_only": True,
                },
            )

    def test_flux_and_chroma_target_profiles_preserve_clip_and_t5_rules(self):
        flux = resolve_training_target_profile(
            "flux-lora",
            {
                "network_train_unet_only": False,
                "network_args": ["train_t5xxl=True"],
            },
        )
        self.assertIn("clip_l.adapter", flux.available_components)
        self.assertIn("t5xxl.adapter", flux.available_components)

        flux_lowercase_true = resolve_training_target_profile(
            "flux-lora",
            {
                "network_train_unet_only": False,
                "network_args": ["train_t5xxl=true"],
            },
        )
        self.assertIn("clip_l.adapter", flux_lowercase_true.available_components)
        self.assertNotIn("t5xxl.adapter", flux_lowercase_true.available_components)

        for inert_arg in ("TRAIN_T5XXL=True", "train_t5xxl =True"):
            with self.subTest(inert_arg=inert_arg):
                flux_inert = resolve_training_target_profile(
                    "flux-lora",
                    {
                        "network_train_unet_only": False,
                        "network_args": [inert_arg],
                    },
                )
                self.assertIn("clip_l.adapter", flux_inert.available_components)
                self.assertNotIn("t5xxl.adapter", flux_inert.available_components)

        flux_dit = resolve_training_target_profile(
            "flux-lora",
            {"network_train_unet_only": True},
        )
        self.assertNotIn("clip_l.adapter", flux_dit.available_components)
        self.assertNotIn("t5xxl.adapter", flux_dit.available_components)

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            resolve_training_target_profile(
                "flux-lora",
                {
                    "network_train_unet_only": True,
                    "network_args": ["train_t5xxl=True"],
                },
            )

        flux_te_only = resolve_training_target_profile(
            "flux-lora",
            {"network_train_text_encoder_only": True},
        )
        self.assertFalse(
            any(item.startswith("transformer.") for item in flux_te_only.available_components)
        )
        self.assertIn("clip_l.adapter", flux_te_only.available_components)

        with self.assertRaises(ValueError):
            resolve_training_target_profile(
                "flux-lora",
                {
                    "network_train_unet_only": True,
                    "network_train_text_encoder_only": True,
                },
            )

        chroma = resolve_training_target_profile(
            "chroma-lora",
            {
                "network_train_unet_only": False,
                "network_args": ["train_t5xxl=True"],
            },
        )
        self.assertIn("t5xxl.adapter", chroma.available_components)
        self.assertNotIn("clip_l.adapter", chroma.available_components)

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            resolve_training_target_profile(
                "chroma-lora",
                {
                    "network_train_unet_only": True,
                    "network_args": ["train_t5xxl=True"],
                },
            )

    def test_anima_full_and_lora_targets_preserve_existing_capabilities(self):
        full = resolve_training_target_profile(
            "anima-finetune",
            {"train_qwen3_text_encoder": False},
        )
        self.assertNotIn("qwen3", full.available_components)
        full_qwen = resolve_training_target_profile(
            "anima-finetune",
            {"train_qwen3_text_encoder": True},
        )
        self.assertIn("qwen3", full_qwen.available_components)

        dit = resolve_training_target_profile(
            "anima-lora",
            {
                "network_train_unet_only": True,
                "network_args": ["train_llm_adapter=True"],
            },
        )
        self.assertIn("llm_adapter.adapter", dit.available_components)
        self.assertNotIn("qwen3.adapter", dit.available_components)

        qwen = resolve_training_target_profile(
            "anima-lora",
            {
                "network_train_text_encoder_only": True,
                "network_args": ["train_llm_adapter=True"],
            },
        )
        self.assertEqual(
            qwen.available_components,
            frozenset({"qwen3.adapter"}),
        )

    def test_sdxl_dreambooth_and_sd3_higher_level_targets(self):
        sdxl = resolve_training_target_profile(
            "sdxl-finetune",
            {"train_text_encoder": False},
        )
        self.assertNotIn("text_encoder_1", sdxl.available_components)
        self.assertNotIn("text_encoder_2", sdxl.available_components)
        sdxl_te = resolve_training_target_profile(
            "sdxl-finetune",
            {"train_text_encoder": True},
        )
        self.assertIn("text_encoder_1", sdxl_te.available_components)
        self.assertIn("text_encoder_2", sdxl_te.available_components)

        dreambooth = resolve_training_target_profile(
            "sd-dreambooth",
            {"stop_text_encoder_training": -1},
        )
        self.assertNotIn("text_encoder", dreambooth.available_components)
        self.assertIn(
            "text_encoder",
            resolve_training_target_profile("sd-dreambooth", {}).available_components,
        )

        sd3 = resolve_training_target_profile(
            "sd3-lora",
            {
                "network_train_unet_only": False,
                "network_args": ["train_t5xxl=True"],
            },
        )
        self.assertIn("clip_l.adapter", sd3.available_components)
        self.assertIn("clip_g.adapter", sd3.available_components)
        self.assertIn("t5xxl.adapter", sd3.available_components)
        sd3_lowercase_true = resolve_training_target_profile(
            "sd3-lora",
            {
                "network_train_unet_only": False,
                "network_train_text_encoder_only": False,
                "network_args": ["train_t5xxl=true"],
            },
        )
        self.assertNotIn("t5xxl.adapter", sd3_lowercase_true.available_components)

        for inert_arg in ("TRAIN_T5XXL=True", "train_t5xxl =True"):
            with self.subTest(sd3_inert_arg=inert_arg):
                sd3_inert = resolve_training_target_profile(
                    "sd3-lora",
                    {
                        "network_train_unet_only": False,
                        "network_train_text_encoder_only": False,
                        "network_args": [inert_arg],
                    },
                )
                self.assertNotIn("t5xxl.adapter", sd3_inert.available_components)

        sd3_unet = resolve_training_target_profile(
            "sd3-lora",
            {"network_train_unet_only": True},
        )
        self.assertNotIn("clip_l.adapter", sd3_unet.available_components)
        self.assertNotIn("clip_g.adapter", sd3_unet.available_components)
        self.assertNotIn("t5xxl.adapter", sd3_unet.available_components)

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            resolve_training_target_profile(
                "sd3-lora",
                {
                    "network_train_unet_only": True,
                    "network_args": ["train_t5xxl=True"],
                },
            )

        sd3_te_only = resolve_training_target_profile(
            "sd3-lora",
            {
                "network_train_text_encoder_only": True,
                "network_args": ["train_t5xxl=True"],
            },
        )
        self.assertFalse(
            any(item.startswith("mmdit.") for item in sd3_te_only.available_components)
        )
        self.assertIn("clip_l.adapter", sd3_te_only.available_components)
        self.assertIn("clip_g.adapter", sd3_te_only.available_components)
        self.assertIn("t5xxl.adapter", sd3_te_only.available_components)

        with self.assertRaises(ValueError):
            resolve_training_target_profile(
                "sd3-lora",
                {
                    "network_train_unet_only": True,
                    "network_train_text_encoder_only": True,
                },
            )

    def test_unknown_parameter_eligibility_policy_fails_closed(self):
        profile = get_model_component_profile("flux-finetune")
        alias = Alias("transformer", "double_blocks.0.linear1")
        with self.assertRaisesRegex(ValueError, "does not support"):
            profile.eligibility_check(
                "future_unknown_policy",
                alias,
                "transformer.double_stream",
            )

    def test_target_profile_records_unavailable_reasons(self):
        target = resolve_training_target_profile(
            "sdxl-finetune",
            {"train_text_encoder": False},
        )
        self.assertIn("text_encoder_1", target.unavailable_reasons)
        self.assertIn("text_encoder_2", target.unavailable_reasons)
        self.assertTrue(target.is_available("unet.transformer"))
        self.assertFalse(target.is_available("text_encoder_1"))


    def test_anima_qwen_adapter_text_encoder_root_matches_component_root_contract(self):
        profile = get_model_component_profile("anima-lora")
        alias = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "text_encoder",
                "model.layers.0.self_attn.q_proj",
            ),
        )
        component_id = profile.classify_alias(alias)
        self.assertEqual(component_id, "qwen3.adapter")
        self.assertIn("text_encoder", profile.components[component_id].roots)

    def test_flux_does_not_accept_sd3_text_encoder_3_as_t5_root(self):
        profile = get_model_component_profile("flux-lora")
        alias = Alias(
            "network",
            "lora_down",
            adapter_target=Target(
                "text_encoder_3",
                "encoder.block.0.layer.0.SelfAttention.q",
            ),
        )
        self.assertIsNone(profile.classify_alias(alias))


if __name__ == "__main__":
    unittest.main()
