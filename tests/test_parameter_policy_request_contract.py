import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


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
        "build_parameter_policy_sidecar": lambda config, page_type: (None, {}, None),
        "build_multi_caption_sidecar": lambda config, page_type: (None, {}, None),
        "prepare_prompt_fields": lambda config, page_type: ({}, []),
        "prepare_training_config": None,
        "legacy_api": SimpleNamespace(resolve_training_backend=object()),
        "parameter_policy_runtime_blockers": lambda policy: [],
        "materialize_sidecars": lambda sidecars: None,
    }
    namespace.update(overrides)
    exec(compile(module, "mikazuki/training_request.py", "exec"), namespace)
    return namespace["prepare_request_config"]


def _prepared(config=None):
    return SimpleNamespace(
        config={} if config is None else dict(config),
        sidecars={},
        warnings=[],
        runtime_blockers=[],
    )


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
            parameter_policy_runtime_blockers=lambda policy: ["component runtime blocked"],
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
            parameter_policy_runtime_blockers=lambda policy: ["component runtime blocked"],
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
