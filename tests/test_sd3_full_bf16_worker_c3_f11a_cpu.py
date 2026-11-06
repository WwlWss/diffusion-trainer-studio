"""C3-F11-A Tests-only RED: real outer LoRA target Linear Forward hooks.

Genuine BF16 patched Linear + LoRAModule + actual Accelerator, Muon and AdamW.
Negative cases must reject BEFORE original Accelerator.backward.
"""
from __future__ import annotations

import unittest
import torch

from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    _lora_bundle, _run_real_lora_step, _assert_pre_backward_reject,
)
from tests.test_sd3_full_bf16_worker_c3_f9a_cpu import _count_real_backward


class F11AOriginalTargetHookIntegrity(unittest.TestCase):
    ERROR = "LoRA|target|Forward|hook|execution"

    def test_target_output_hook_before_probe_rejected_pre_backward(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        handle = target.register_forward_hook(
            lambda module, args, output: output * 0.5
        )
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_target_input_pre_hook_before_probe_rejected_pre_backward(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        handle = target.register_forward_pre_hook(
            lambda module, args: (args[0] * 0.5,)
        )
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            handle.remove()

    def test_target_output_hook_added_after_probe_install_rejected(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        probe = bundle[5]
        original_install = probe.install
        handles = []

        def install_with_late_hook():
            original_install()
            handles.append(target.register_forward_hook(
                lambda module, args, output: output * 0.5
            ))

        probe.install = install_with_late_hook
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            for handle in handles:
                handle.remove()
            delattr(probe, "install")

    def test_output_hook_injected_after_probe_outer_pre_hook_rejected(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        probe = bundle[5]
        original_install = probe.install
        handles = []

        def inject_post_hook(module, args):
            if len(handles) == 1:
                handles.append(target.register_forward_hook(
                    lambda m, a, output: output * 0.5
                ))

        def install_then_inject():
            original_install()
            # This pre-hook executes after the Probe's own target pre-hook.
            handles.append(target.register_forward_pre_hook(inject_post_hook))

        probe.install = install_then_inject
        try:
            _assert_pre_backward_reject(self, bundle, self.ERROR)
        finally:
            for handle in reversed(handles):
                handle.remove()
            delattr(probe, "install")

    def test_unmodified_two_real_bf16_physical_steps_pass(self):
        bundle = _lora_bundle()
        _run_real_lora_step(bundle)
        events = _run_real_lora_step(bundle)
        self.assertEqual(len([
            ev for ev in events if ev["logical_step"] == 2
            and ev["event_type"] == "physical_child_stepped"
        ]), 2)

    def test_probe_owns_exact_one_readonly_target_hook_and_cleans_up(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        before = dict(target._forward_pre_hooks)
        probe = bundle[5]
        _run_real_lora_step(bundle)
        self.assertEqual(dict(target._forward_pre_hooks), before)
        self.assertEqual(len(probe._lora_forward_handles), 0)
        _run_real_lora_step(bundle)
        self.assertEqual(dict(target._forward_pre_hooks), before)
        self.assertEqual(len(probe._lora_forward_handles), 0)


if __name__ == "__main__":
    unittest.main()
