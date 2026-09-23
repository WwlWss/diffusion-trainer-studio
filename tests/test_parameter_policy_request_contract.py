import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy import (
    build_parameter_policy_sidecar,
    parameter_policy_runtime_blockers,
)
from mikazuki.parameter_policy_editor import (
    bootstrap_parameter_policy_editor,
    normalize_parameter_policy_editor_state,
)
from mikazuki.parameter_policy_matrix import PARAMETER_POLICY_RUNTIME_TRAIN_TYPES
from mikazuki.training_config import prepare_training_config


ROOT = Path(__file__).resolve().parents[1]
REQUEST = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")


def _load_prepare_request_config(**overrides):
    tree = ast.parse(REQUEST, filename="mikazuki/training_request.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_request_config"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "train_utils": SimpleNamespace(fix_config_types=lambda config: None),
        "normalize_parameter_policy_editor_state": lambda config: None,
        "build_parameter_policy_sidecar": lambda config, page_type: (None, {}, None),
        "build_multi_caption_sidecar": lambda config, page_type: (None, {}, None),
        "prepare_prompt_fields": lambda config, page_type: ({}, []),
        "prepare_training_config": None,
        "legacy_api": SimpleNamespace(resolve_training_backend=object()),
        "parameter_policy_runtime_blockers": lambda policy, **kwargs: [],
        "parameter_policy_gpu_selection_blockers": lambda gpu_ids: [],
        "PARAMETER_POLICY_RUNTIME_TRAIN_TYPES": frozenset({"sd-lora"}),
        "materialize_sidecars": lambda sidecars: None,
    }
    namespace.update(overrides)
    exec(compile(module, "mikazuki/training_request.py", "exec"), namespace)
    return namespace["prepare_request_config"]


def _component_gui_config(train_type, train_components, **extra):
    profile = get_model_component_profile(train_type)
    components = {
        component_id: {"train": False}
        for component_id in profile.components
    }
    for component_id in train_components:
        components[component_id] = {
            "train": True,
            "optimizer_profile": "main",
            "learning_rate": 1e-4,
        }
    config = {
        "optimization_mode": "component",
        "parameter_policy_profiles": {
            "main": {"type": "AdamW", "args": {}},
        },
        "parameter_policy_components": components,
        "optimizer_type": "AdamW",
        "learning_rate": 1e-4,
    }
    config.update(extra)
    return config


def _real_prepare_request():
    resolver = lambda config, requested: (requested, f"trainer/{requested}.py")
    return _load_prepare_request_config(
        normalize_parameter_policy_editor_state=normalize_parameter_policy_editor_state,
        build_parameter_policy_sidecar=build_parameter_policy_sidecar,
        prepare_training_config=prepare_training_config,
        legacy_api=SimpleNamespace(resolve_training_backend=resolver),
        parameter_policy_runtime_blockers=parameter_policy_runtime_blockers,
        parameter_policy_gpu_selection_blockers=lambda gpu_ids: [],
        PARAMETER_POLICY_RUNTIME_TRAIN_TYPES=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
    )


def _prepared(config=None, train_type="sd-lora"):
    return SimpleNamespace(
        train_type=train_type,
        config={} if config is None else dict(config),
        sidecars={},
        warnings=[],
        runtime_blockers=[],
        gpu_ids=None,
    )


class ParameterPolicyRealPipelineContractTests(unittest.TestCase):
    def test_sdxl_component_cache_follows_policy_on_request_pipeline(self):
        prepare = _real_prepare_request()
        frozen = prepare(
            _component_gui_config(
                "sdxl-lora",
                ["unet.attention.adapter"],
                lora_target="unet_text_encoder",
                cache_text_encoder_outputs=True,
            ),
            "sdxl-lora",
            launch=False,
        )
        self.assertEqual(frozen.runtime_blockers, [])

        trained = prepare(
            _component_gui_config(
                "sdxl-lora",
                ["text_encoder_1.adapter"],
                lora_target="unet_text_encoder",
                cache_text_encoder_outputs=True,
            ),
            "sdxl-lora",
            launch=False,
        )
        self.assertTrue(
            any(
                "text_encoder_1.adapter" in item
                and "cache_text_encoder_outputs" in item
                for item in trained.runtime_blockers
            ),
            trained.runtime_blockers,
        )

    def test_flux_chroma_sd3_component_cache_follows_policy_on_request_pipeline(self):
        prepare = _real_prepare_request()
        cases = (
            (
                "flux-lora",
                {"flux_lora_target": "dit_clip_l_t5xxl"},
                ["clip_l.adapter"],
                ["t5xxl.adapter"],
            ),
            (
                "chroma-lora",
                {"flux_lora_target": "dit_t5xxl"},
                ["transformer.double_stream.adapter"],
                ["t5xxl.adapter"],
            ),
            (
                "sd3-lora",
                {
                    "sd3_lora_target": "mmdit_text_encoder",
                    "train_t5xxl": True,
                },
                ["clip_l.adapter"],
                ["t5xxl.adapter"],
            ),
        )
        for train_type, target, frozen_train, t5_train in cases:
            with self.subTest(train_type=train_type, case="t5-frozen"):
                frozen = prepare(
                    _component_gui_config(
                        train_type,
                        frozen_train,
                        cache_text_encoder_outputs=True,
                        **target,
                    ),
                    train_type,
                    launch=False,
                )
                self.assertEqual(frozen.runtime_blockers, [])

            with self.subTest(train_type=train_type, case="t5-train"):
                trained = prepare(
                    _component_gui_config(
                        train_type,
                        t5_train,
                        cache_text_encoder_outputs=True,
                        **target,
                    ),
                    train_type,
                    launch=False,
                )
                self.assertTrue(
                    any(
                        "t5xxl.adapter" in item
                        and "cache_text_encoder_outputs" in item
                        for item in trained.runtime_blockers
                    ),
                    trained.runtime_blockers,
                )

    def test_fp8_survives_bootstrap_then_blocks_runtime(self):
        prepare = _real_prepare_request()
        resolver = lambda config, requested: (requested, f"trainer/{requested}.py")
        cases = (
            ("flux-lora", {"flux_lora_target": "dit"}),
            ("chroma-lora", {"flux_lora_target": "dit"}),
            (
                "sd3-lora",
                {"sd3_lora_target": "mmdit", "train_t5xxl": False},
            ),
        )
        for train_type, target in cases:
            with self.subTest(train_type=train_type):
                raw = {
                    "optimizer_type": "AdamW",
                    "learning_rate": 1e-4,
                    "fp8_base": True,
                    **target,
                }
                original = dict(raw)
                gui = bootstrap_parameter_policy_editor(
                    raw,
                    train_type,
                    resolve_backend=resolver,
                )
                self.assertEqual(raw, original)

                component_raw = dict(raw)
                component_raw.update(gui)
                prepared = prepare(
                    component_raw,
                    train_type,
                    launch=False,
                )
                self.assertTrue(prepared.config["fp8_base"])
                self.assertTrue(
                    any("fp8_base" in item for item in prepared.runtime_blockers),
                    prepared.runtime_blockers,
                )


class ParameterPolicyEditorRequestContractTests(unittest.TestCase):
    def test_editor_normalization_precedes_policy_sidecar_compile(self):
        normalize = REQUEST.index("normalize_parameter_policy_editor_state(config)")
        sidecar = REQUEST.index("build_parameter_policy_sidecar(config, page_type)")
        self.assertLess(normalize, sidecar)


class ParameterPolicyRequestContractTests(unittest.TestCase):
    def test_component_start_disables_launch_staging_and_fails_before_materialization(self):
        calls = []

        def prepare_training_config(config, **kwargs):
            calls.append(("prepare", kwargs["launch"]))
            return _prepared({"parameter_policy_config": "policy.json"})

        def materialize_sidecars(sidecars):
            calls.append(("materialize", dict(sidecars)))

        prepare = _load_prepare_request_config(
            build_parameter_policy_sidecar=lambda config, page_type: (
                "policy.json",
                {"policy.json": "{}"},
                {"version": 1},
            ),
            prepare_training_config=prepare_training_config,
            parameter_policy_runtime_blockers=lambda policy, **kwargs: ["component runtime blocked"],
            materialize_sidecars=materialize_sidecars,
        )

        with self.assertRaisesRegex(ValueError, "component runtime blocked"):
            prepare({}, "lora-master", launch=True, materialize=True)

        self.assertEqual(calls, [("prepare", False)])

    def test_standard_start_keeps_launch_semantics_and_materializes_when_requested(self):
        calls = []

        def prepare_training_config(config, **kwargs):
            calls.append(("prepare", kwargs["launch"]))
            return _prepared()

        def materialize_sidecars(sidecars):
            calls.append(("materialize", dict(sidecars)))

        prepare = _load_prepare_request_config(
            prepare_training_config=prepare_training_config,
            materialize_sidecars=materialize_sidecars,
        )
        result = prepare({}, "lora-master", launch=True, materialize=True)

        self.assertIsNotNone(result)
        self.assertEqual(calls, [("prepare", True), ("materialize", {})])

    def test_component_preview_exposes_blocker_without_launch_or_materialization(self):
        calls = []

        def prepare_training_config(config, **kwargs):
            calls.append(("prepare", kwargs["launch"]))
            return _prepared({"parameter_policy_config": "policy.json"})

        prepare = _load_prepare_request_config(
            build_parameter_policy_sidecar=lambda config, page_type: (
                "policy.json",
                {"policy.json": "{}"},
                {"version": 1},
            ),
            prepare_training_config=prepare_training_config,
            parameter_policy_runtime_blockers=lambda policy, **kwargs: ["component runtime blocked"],
            materialize_sidecars=lambda sidecars: calls.append(("materialize", dict(sidecars))),
        )
        result = prepare({}, "lora-master", launch=False, materialize=False)

        self.assertEqual(calls, [("prepare", False)])
        self.assertEqual(result.runtime_blockers, ["component runtime blocked"])
        self.assertEqual(result.sidecars, {"policy.json": "{}"})

    def test_step5_runtime_is_not_wired_into_request_launch_path(self):
        self.assertNotIn("parameter_policy_torch", REQUEST)
        self.assertNotIn("compile_parameter_policy_runtime_spec", REQUEST)
        self.assertNotIn("build_parameter_policy_optimizer", REQUEST)
        self.assertNotIn("build_parameter_policy_scheduler", REQUEST)

    def test_step6f_gate_passes_effective_backend_config_and_release_matrix(self):
        calls = []

        def blockers(policy, **kwargs):
            calls.append(kwargs)
            return ["component runtime blocked"]

        prepare = _load_prepare_request_config(
            build_parameter_policy_sidecar=lambda config, page_type: (
                "policy.json",
                {"policy.json": "{}"},
                {"version": 1},
            ),
            prepare_training_config=lambda config, **kwargs: _prepared(
                {"parameter_policy_config": "policy.json", "marker": 1},
                train_type="flux-lora",
            ),
            parameter_policy_runtime_blockers=blockers,
            PARAMETER_POLICY_RUNTIME_TRAIN_TYPES=frozenset({"sd-lora"}),
        )
        result = prepare({}, "flux-lora", launch=False)

        self.assertEqual(result.runtime_blockers, ["component runtime blocked"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["train_type"], "flux-lora")
        self.assertEqual(calls[0]["effective_config"]["marker"], 1)
        self.assertEqual(calls[0]["integrated_train_types"], frozenset({"sd-lora"}))


    def test_component_start_opens_when_backend_and_gpu_have_no_blockers(self):
        calls = []

        def prepare_training_config(config, **kwargs):
            calls.append(("prepare", kwargs["launch"]))
            prepared = _prepared(
                {"parameter_policy_config": "policy.json"},
                train_type="sd-lora",
            )
            prepared.gpu_ids = ["0"]
            return prepared

        prepare = _load_prepare_request_config(
            build_parameter_policy_sidecar=lambda config, page_type: (
                "policy.json",
                {"policy.json": "{}"},
                {"version": 1},
            ),
            prepare_training_config=prepare_training_config,
            parameter_policy_runtime_blockers=lambda policy, **kwargs: [],
            parameter_policy_gpu_selection_blockers=lambda gpu_ids: [],
            materialize_sidecars=lambda sidecars: calls.append(
                ("materialize", dict(sidecars))
            ),
        )
        result = prepare({}, "lora-master", launch=True, materialize=True)

        self.assertEqual(
            calls,
            [("prepare", False), ("materialize", {"policy.json": "{}"})],
        )
        self.assertEqual(result.runtime_blockers, [])

    def test_component_start_rejects_explicit_multi_gpu_before_materialization(self):
        calls = []

        def prepare_training_config(config, **kwargs):
            calls.append(("prepare", kwargs["launch"]))
            prepared = _prepared(
                {"parameter_policy_config": "policy.json"},
                train_type="sd-lora",
            )
            prepared.gpu_ids = ["0", "1"]
            return prepared

        prepare = _load_prepare_request_config(
            build_parameter_policy_sidecar=lambda config, page_type: (
                "policy.json",
                {"policy.json": "{}"},
                {"version": 1},
            ),
            prepare_training_config=prepare_training_config,
            parameter_policy_runtime_blockers=lambda policy, **kwargs: [],
            parameter_policy_gpu_selection_blockers=lambda gpu_ids: [
                "multi GPU blocked"
            ],
            materialize_sidecars=lambda sidecars: calls.append(
                ("materialize", dict(sidecars))
            ),
        )

        with self.assertRaisesRegex(ValueError, "multi GPU blocked"):
            prepare({}, "lora-master", launch=True, materialize=True)

        self.assertEqual(calls, [("prepare", False)])

    def test_parameter_policy_sidecar_is_host_owned(self):
        self.assertIn(
            "parameter_policy_config 是 DTS 托管字段，不能通过 ui_custom_params 手工注入",
            REQUEST,
        )
        self.assertIn(
            "parameter_policy_config 是 DTS 托管字段，不能通过 ui_custom_params 覆盖",
            REQUEST,
        )

    def test_policy_sidecars_are_merged_without_replacing_existing_sidecars(self):
        policy_index = REQUEST.index("prepared.sidecars.update(policy_sidecars)")
        multi_index = REQUEST.index("prepared.sidecars.update(multi_sidecars)")
        prompt_index = REQUEST.index("prepared.sidecars.update(sidecars)")
        self.assertLess(policy_index, multi_index)
        self.assertLess(multi_index, prompt_index)


if __name__ == "__main__":
    unittest.main()
