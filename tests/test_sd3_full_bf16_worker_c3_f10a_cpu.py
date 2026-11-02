"""C3-F10-A tests-only RED: real internal LoRA projection Forward semantics.

One real BF16 LoRAModule + real Muon/AdamW/Accelerate graph; the
negative assertions require rejection before original Backward.
No synthetic LoRAModule or altered optimization math.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    _lora_bundle, _run_real_lora_step, _assert_pre_backward_reject,
)


class F10AProjectionForwardAndHooks(unittest.TestCase):
    ERROR = "LoRA|projection|Forward|hook|semantic"

    def test_down_forward_override_must_reject_before_backward(self):
        bundle = _lora_bundle()
        down = bundle[0].loras["attention_0"].lora_down
        down.forward = lambda x: F.linear(x, down.weight) * 0.5
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_up_forward_override_must_reject_before_backward(self):
        bundle = _lora_bundle()
        up = bundle[0].loras["attention_0"].lora_up
        up.forward = lambda x: F.linear(x, up.weight) * 0.5
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_output_changing_forward_hook_must_reject(self):
        bundle = _lora_bundle()
        up = bundle[0].loras["attention_0"].lora_up
        handle = up.register_forward_hook(lambda module, args, output: output * 0.5)
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_input_changing_forward_pre_hook_must_reject(self):
        bundle = _lora_bundle()
        down = bundle[0].loras["attention_0"].lora_down
        handle = down.register_forward_pre_hook(
            lambda module, args: (args[0] * 0.5,)
        )
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_override_after_outer_target_hook_before_projection_must_reject(self):
        bundle = _lora_bundle()
        model = bundle[0]
        down = model.loras["attention_0"].lora_down
        target = model.forward_targets["attention_0"]
        probe = bundle[5]
        original_install = probe.install
        hooks = []
        from tests.test_sd3_full_bf16_worker_c3_f9a_cpu import _count_real_backward
        original_backward_calls = _count_real_backward(bundle)

        def after_outer_observer_hook(module, args):
            down.forward = lambda x: F.linear(x, down.weight) * 0.5

        def install_then_inject():
            original_install()
            # Executes AFTER the Observer's target-Linear pre-hook, but BEFORE
            # genuine LoRAModule.forward invokes lora_down(x).
            hooks.append(target.register_forward_pre_hook(after_outer_observer_hook))

        probe.install = install_then_inject
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                _run_real_lora_step(bundle)
            self.assertEqual(
                original_backward_calls, [],
                "Late projection mutation entered real Accelerator.backward",
            )
        finally:
            for handle in hooks:
                handle.remove()
            if "install" in vars(probe):
                delattr(probe, "install")

    def test_unmodified_real_bf16_lora_two_steps_stay_positive(self):
        bundle = _lora_bundle()
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
