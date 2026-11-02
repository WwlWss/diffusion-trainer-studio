"""C3-F11-C Tests-only RED: unreviewed real Backward Hook provenance.

Real BF16 LoRAModule, pinned CPU Accelerator, Muon and AdamW. Reject
unreviewed Module/global/Parameter hooks before true Backward where possible.
"""
from __future__ import annotations

import unittest
import torch

from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    _lora_bundle, _run_real_lora_step, _assert_pre_backward_reject,
)


def _scale_grad_input(module, grad_input, grad_output):
    return tuple(
        gradient * 0.5 if gradient is not None else None
        for gradient in grad_input
    )


def _scale_grad_output(module, grad_output):
    return tuple(
        gradient * 0.5 if gradient is not None else None
        for gradient in grad_output
    )


class F11CBackwardExecutionIntegrity(unittest.TestCase):
    ERROR = "LoRA|Backward|hook|gradient|Parameter|unreviewed"

    def test_up_full_backward_hook_rejected_before_original_backward(self):
        bundle = _lora_bundle()
        up = bundle[0].loras["attention_0"].lora_up
        handle = up.register_full_backward_hook(_scale_grad_input)
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_down_full_backward_pre_hook_rejected_before_original_backward(self):
        bundle = _lora_bundle()
        down = bundle[0].loras["attention_0"].lora_down
        handle = down.register_full_backward_pre_hook(_scale_grad_output)
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_outer_target_full_backward_hook_rejected(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        handle = target.register_full_backward_hook(_scale_grad_input)
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_global_module_full_backward_hook_rejected(self):
        bundle = _lora_bundle()
        up = bundle[0].loras["attention_0"].lora_up
        handle = torch.nn.modules.module.register_module_full_backward_hook(
            lambda module, gi, go:
                _scale_grad_input(module, gi, go) if module is up else None
        )
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_backward_hook_inserted_inside_real_forward_rejected(self):
        bundle = _lora_bundle()
        model = bundle[0]
        original = model.forward
        up = model.loras["attention_0"].lora_up
        handles = []

        def register_before_inner_forward(x):
            if not handles:
                handles.append(up.register_full_backward_hook(_scale_grad_input))
            return original(x)

        model.forward = register_before_inner_forward
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
            self.assertEqual(len(handles), 1)
        finally:
            for handle in handles:
                handle.remove()

    def test_backward_hook_added_after_forward_before_backward_rejected(self):
        bundle = _lora_bundle()
        model, accelerator, _, _, _, probe = bundle
        up = model.loras["attention_0"].lora_up
        original = accelerator.backward
        calls = []
        handle = None

        def counted(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        accelerator.backward = counted
        probe.install()
        try:
            with accelerator.accumulate(model):
                probe.begin_batch("dts_c2b_sample_0001")
                x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
                loss = model(x)
                # This may not alter the already-built graph; it still
                # invalidates the reviewed execution registry.
                handle = up.register_full_backward_hook(_scale_grad_input)
                with self.assertRaisesRegex(ValueError, self.ERROR):
                    accelerator.backward(loss)
                self.assertEqual(calls, [], "Unreviewed Hook reached genuine Backward")
        finally:
            probe.close()
            if handle is not None:
                handle.remove()
            delattr(accelerator, "backward")

    def test_parameter_grad_hook_rejected_before_backward(self):
        bundle = _lora_bundle()
        up = bundle[0].loras["attention_0"].lora_up
        handle = up.weight.register_hook(lambda grad: grad * 0.5)
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_parameter_post_accumulate_hook_rejected_before_backward(self):
        bundle = _lora_bundle()
        down = bundle[0].loras["attention_0"].lora_down
        def change_accumulated_gradient(parameter):
            parameter.grad.mul_(0.5)
            # PyTorch's post-accumulate callback MUST return None.
        handle = down.weight.register_post_accumulate_grad_hook(
            change_accumulated_gradient
        )
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_unmodified_real_bf16_two_physical_steps_remain_positive(self):
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
