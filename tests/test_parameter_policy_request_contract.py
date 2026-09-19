from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REQUEST = (ROOT / "mikazuki" / "training_request.py").read_text(encoding="utf-8")


class ParameterPolicyRequestContractTests(unittest.TestCase):
    def test_request_layer_builds_policy_before_effective_config(self):
        self.assertIn(
            "policy_path, policy_sidecars, policy = build_parameter_policy_sidecar(config, page_type)",
            REQUEST,
        )
        self.assertLess(
            REQUEST.index("build_parameter_policy_sidecar(config, page_type)"),
            REQUEST.index("prepare_training_config("),
        )

    def test_component_start_disables_launch_specific_staging(self):
        self.assertIn("effective_launch = launch and policy is None", REQUEST)
        self.assertIn("launch=effective_launch", REQUEST)

    def test_component_start_fails_closed_before_materialization(self):
        blocker_index = REQUEST.index("if launch:\n            raise ValueError(blockers[0])")
        materialize_index = REQUEST.index("if materialize:\n        materialize_sidecars")
        self.assertLess(blocker_index, materialize_index)

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

    def test_standard_path_has_no_policy_runtime_blocker(self):
        self.assertIn("if policy is not None:", REQUEST)
        self.assertIn("prepared.runtime_blockers.extend(blockers)", REQUEST)


if __name__ == "__main__":
    unittest.main()
