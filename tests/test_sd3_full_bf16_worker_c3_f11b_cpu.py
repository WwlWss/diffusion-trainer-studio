"""C3-F11-B Tests-only RED: LoRA projection registered Parameter identity.

Actual BF16 LoRAModule, real Muon/AdamW children and Accelerate.
Never treat actual mutable .weight or new bias as the frozen reference.
"""
from __future__ import annotations

import unittest
import torch

from tools import sd3_full_bf16_worker_observer as observer
from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    R2, _lora_bundle, _run_real_lora_step, _assert_pre_backward_reject,
)


def _frozen_bias(module):
    module.bias = torch.nn.Parameter(
        torch.full((module.out_features,), 0.5, dtype=torch.bfloat16),
        requires_grad=False,
    )


class F11BProjectionParameterOwnership(unittest.TestCase):
    ERROR = "LoRA|projection|Parameter|weight|bias|inventory|ownership"

    def test_down_frozen_bias_inserted_after_probe_init_rejected_before_backward(self):
        bundle = _lora_bundle()
        _frozen_bias(bundle[0].loras["attention_0"].lora_down)
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_up_frozen_bias_inserted_after_probe_init_rejected_before_backward(self):
        bundle = _lora_bundle()
        _frozen_bias(bundle[0].loras["attention_0"].lora_up)
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_up_weight_same_shape_new_parameter_rejected_before_backward(self):
        bundle = _lora_bundle()
        up = bundle[0].loras["attention_0"].lora_up
        up.weight = torch.nn.Parameter(
            up.weight.detach().clone(), requires_grad=True,
        )
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_down_up_wrong_shared_weight_alias_rejected_before_backward(self):
        bundle = _lora_bundle()
        lora = bundle[0].loras["attention_0"]
        # Same shape/dtype and finite: only the physical identity/owner changed.
        lora.lora_up.weight = lora.lora_down.weight
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_bias_inserted_inside_actual_forward_after_begin_batch_rejected(self):
        bundle = _lora_bundle()
        model = bundle[0]
        original = model.forward
        down = model.loras["attention_0"].lora_down
        injected = []

        def inject_after_begin_batch(x):
            if not injected:
                injected.append(True)
                _frozen_bias(down)
            return original(x)

        model.forward = inject_after_begin_batch
        _assert_pre_backward_reject(self, bundle, self.ERROR)
        self.assertEqual(injected, [True], "The real Forward mutation did not execute")

    def test_bias_inserted_between_prepared_and_new_probe_rejected_at_init(self):
        bundle = _lora_bundle()
        old = bundle[5]
        _frozen_bias(bundle[0].loras["attention_0"].lora_down)
        with self.assertRaisesRegex(ValueError, self.ERROR):
            observer.PhysicalStepProbe(
                accelerator=bundle[1],
                composite_optimizer=bundle[4],
                composite_scheduler=old.scheduler,
                parameter_names=old.params,
                component_by_parameter=old.component_by_parameter,
                lora_network=bundle[0],
                case_id=R2, phase="fresh",
                run_id="run-f11b-prepare-gap-0001",
                phase_id="fresh-f11b-prepare-gap-0001",
                source_commit="a" * 40,
            )

    def test_unmodified_real_bf16_lora_two_steps_are_positive(self):
        bundle = _lora_bundle()
        _run_real_lora_step(bundle)
        events = _run_real_lora_step(bundle)
        self.assertEqual(len([
            event for event in events
            if event["logical_step"] == 2
            and event["event_type"] == "physical_child_stepped"
        ]), 2)


if __name__ == "__main__":
    unittest.main()
