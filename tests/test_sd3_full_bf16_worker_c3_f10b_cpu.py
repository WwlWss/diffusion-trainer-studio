"""C3-F10-B tests-only RED: exact LoRA→real MMDiT target-path provenance.

The micro fixture uses six *real* BF16 LoRAModule objects, patched original
Linear.forward, real Optimizer children and Accelerate. It is not CUDA proof.
The reviewed full 263-path canonical map is mandatory only for real CUDA.
"""
from __future__ import annotations

import unittest

import torch

from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    R2, _lora_bundle, _run_real_lora_step, _assert_pre_backward_reject,
)
from tools import sd3_full_bf16_worker_observer as observer
from tools import sd3_full_bf16_evidence_support as host


class TinyRegisteredMmditRoot(torch.nn.Module):
    """Separate real Module tree whose canonical paths reach patched Linears."""
    def __init__(self, network):
        super().__init__()
        self.targets = torch.nn.ModuleDict({
            key: linear for key, linear in network.forward_targets.items()
        })


def _fixture():
    base = _lora_bundle()
    model = base[0]
    root = TinyRegisteredMmditRoot(model)
    paths = {
        lora.lora_name: "targets." + key
        for key, lora in model.loras.items()
    }
    for key, lora in model.loras.items():
        lora._dts_parameter_policy_target_v1 = (
            "mmdit", paths[lora.lora_name], "torch.nn.modules.linear.Linear",
        )
    return base, root, paths


def _bound_bundle(base, root, paths):
    old = base[5]
    args = dict(
        accelerator=base[1],
        composite_optimizer=base[4],
        composite_scheduler=old.scheduler,
        parameter_names=old.params,
        component_by_parameter=old.component_by_parameter,
        case_id=R2,
        run_id="run-c3-f10b-real-target-0001",
        phase_id="fresh-c3-f10b-real-target-0001",
        source_commit="a" * 40,
        phase="fresh",
        lora_network=base[0],
    )
    # F10-B GREEN: both root and independently supplied toy diagnostic paths
    # are mandatory, never silently skipped by a compatibility bridge.
    args["lora_mmdit_root"] = root
    args["expected_lora_target_paths"] = paths
    return (*base[:5], observer.PhysicalStepProbe(**args))


class F10BRealMMDiTTargetIdentity(unittest.TestCase):
    ERROR = "LoRA|target|path|MMDiT|provenance|identity"

    def test_wrong_but_existing_same_shape_path_rejected_at_probe_init(self):
        base, root, paths = _fixture()
        lora = base[0].loras["attention_0"]
        lora._dts_parameter_policy_target_v1 = (
            "mmdit", "targets.attention_1", "torch.nn.modules.linear.Linear"
        )
        with self.assertRaisesRegex(ValueError, self.ERROR):
            _bound_bundle(base, root, paths)

    def test_metadata_target_path_mutated_after_probe_init_rejected_pre_backward(self):
        base, root, paths = _fixture()
        bundle = _bound_bundle(base, root, paths)
        lora = bundle[0].loras["attention_0"]
        lora._dts_parameter_policy_target_v1 = (
            "mmdit", "targets.attention_1", "torch.nn.modules.linear.Linear"
        )
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_mmdit_target_relink_with_same_shape_rejected(self):
        base, root, paths = _fixture()
        bundle = _bound_bundle(base, root, paths)
        replacement = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        replacement.requires_grad_(False)
        root.targets["attention_0"] = replacement
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_mmdit_swapped_sameshape_targets_rejected(self):
        base, root, paths = _fixture()
        bundle = _bound_bundle(base, root, paths)
        first = root.targets["attention_0"]
        second = root.targets["attention_1"]
        root.targets["attention_0"] = second
        root.targets["attention_1"] = first
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_metadata_type_suffix_only_not_full_linear_type_rejected(self):
        base, root, paths = _fixture()
        lora = base[0].loras["attention_0"]
        lora._dts_parameter_policy_target_v1 = (
            "mmdit", paths[lora.lora_name], "forged.placeholder.Linear"
        )
        with self.assertRaisesRegex(ValueError, self.ERROR):
            _bound_bundle(base, root, paths)

    def test_all_263_canonical_paths_reach_real_meta_sd35_medium_mmdit_linears(self):
        # This is the genuine production MMDiT class on the Meta Device.
        # No 2.9B weights, no fake CUDA and no optimizer qualification.
        from library import sd3_models

        spec = host.build_sd35_medium_lora_target_spec()
        paths = observer._reference_sd3_mmdit_target_paths(spec)
        self.assertEqual(len(paths), 263)
        with torch.device("meta"):
            mmdit = sd3_models.MMDiT(
                depth=24, attn_mode="torch", qk_norm="rms",
                x_block_self_attn_layers=list(range(13)),
            )
        for name, target_path in paths.items():
            actual = mmdit.get_submodule(target_path)
            self.assertIs(type(actual), torch.nn.Linear, name)
            self.assertEqual(
                (actual.in_features, actual.out_features),
                (spec[name]["input_features"], spec[name]["output_features"]),
                name,
            )

    def test_target_path_changed_after_outer_check_before_projection_is_rejected(self):
        base, root, paths = _fixture()
        bundle = _bound_bundle(base, root, paths)
        model, accelerator, _, _, _, probe = bundle
        target = model.forward_targets["attention_0"]
        lora = model.loras["attention_0"]
        original_install = probe.install
        original_backward = accelerator.backward
        handles = []
        backward_calls = []

        def counted_backward(*args, **kwargs):
            backward_calls.append(True)
            return original_backward(*args, **kwargs)

        def mutate_target_path_after_outer_check(module, inputs):
            lora._dts_parameter_policy_target_v1 = (
                "mmdit", "targets.attention_1", "torch.nn.modules.linear.Linear"
            )

        def install_then_mutate_at_real_forward():
            original_install()
            # Runs after the Observer's outer target pre-hook but before
            # lora_down(x). The projection-level hook must catch the drift.
            handles.append(
                target.register_forward_pre_hook(mutate_target_path_after_outer_check)
            )

        probe.install = install_then_mutate_at_real_forward
        accelerator.backward = counted_backward
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                _run_real_lora_step(bundle)
            self.assertEqual(
                backward_calls, [],
                "Real Backward ran after late MMDiT target-path tampering",
            )
        finally:
            for handle in handles:
                handle.remove()
            if "install" in vars(probe):
                delattr(probe, "install")
            if "backward" in vars(accelerator):
                delattr(accelerator, "backward")

    def test_unmodified_real_target_root_training_two_steps_positive(self):
        base, root, paths = _fixture()
        bundle = _bound_bundle(base, root, paths)
        _run_real_lora_step(bundle)
        events = _run_real_lora_step(bundle)
        children = [
            event for event in events
            if event["logical_step"] == 2
            and event["event_type"] == "physical_child_stepped"
        ]
        self.assertEqual(len(children), 2)


if __name__ == "__main__":
    unittest.main()
