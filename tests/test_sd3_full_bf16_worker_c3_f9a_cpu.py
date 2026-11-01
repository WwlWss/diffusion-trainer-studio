"""C3-F9-A real-Linear Forward hook and sampling-boundary qualification.

Independent real pinned BF16 CPU checks. Does NOT qualify physical CUDA/C4.
"""
from __future__ import annotations

import unittest

import torch

from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    _lora_bundle, _run_real_lora_step,
)


def _count_real_backward(bundle):
    accelerator = bundle[1]
    original = accelerator.backward
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    accelerator.backward = counted
    return calls


class F9ATrueForwardEntryAndHookLifetime(unittest.TestCase):
    def test_attribute_changed_after_begin_batch_before_target_forward_is_rejected(self):
        bundle = _lora_bundle()
        model = bundle[0]
        original_forward = model.forward
        calls = _count_real_backward(bundle)

        def mutate_on_live_forward(x):
            model.loras["attention_0"].scale = 0.5
            return original_forward(x)

        model.forward = mutate_on_live_forward
        with self.assertRaisesRegex(ValueError, "LoRA|Forward|scale"):
            _run_real_lora_step(bundle)
        self.assertEqual(calls, [], "Real Backward ran despite a real Forward-hook drift")

    def test_real_linear_forward_rebound_after_begin_batch_is_rejected(self):
        bundle = _lora_bundle()
        model = bundle[0]
        target = model.forward_targets["attention_0"]
        original_target_forward = target.forward
        original_model_forward = model.forward
        calls = _count_real_backward(bundle)

        def unreviewed_forward(x):
            return original_target_forward(x) * 0.75

        def mutate_on_live_forward(x):
            target.forward = unreviewed_forward
            return original_model_forward(x)

        model.forward = mutate_on_live_forward
        with self.assertRaisesRegex(ValueError, "LoRA|Forward|binding|rebound"):
            _run_real_lora_step(bundle)
        self.assertEqual(calls, [], "Real Backward ran after Linear.forward was rebound")

    def test_sampling_only_multiplier_zero_then_restore_one_is_allowed(self):
        bundle = _lora_bundle()
        model = bundle[0]
        _run_real_lora_step(bundle)
        model.set_multiplier(0.0)
        x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
        with torch.no_grad():
            sample = model(x)
        self.assertTrue(bool(torch.isfinite(sample).all()))
        model.set_multiplier(1.0)
        events = _run_real_lora_step(bundle)
        second_children = [
            event for event in events
            if event["logical_step"] == 2
            and event["event_type"] == "physical_child_stepped"
        ]
        self.assertEqual(len(second_children), 2)

    def test_real_forward_hooks_removed_on_close_and_second_install_works(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        original_hook_ids = tuple(target._forward_pre_hooks)
        probe = bundle[5]
        _run_real_lora_step(bundle)
        self.assertEqual(tuple(target._forward_pre_hooks), original_hook_ids)
        self.assertFalse(probe._lora_forward_handles)
        _run_real_lora_step(bundle)
        self.assertEqual(tuple(target._forward_pre_hooks), original_hook_ids)
        self.assertFalse(probe._lora_forward_handles)


if __name__ == "__main__":
    unittest.main()
