"""C3-F9-B pinned CPU: frozen RuntimeSpec order authority and late-stage drift.

Three focused controls complement F9's eight main ordered-Group tests.
Actual Muon/AdamW/Accelerate objects are used; not CUDA qualification.
"""
from __future__ import annotations

import unittest

import torch

from tests.test_sd3_full_bf16_worker_c3_strict_cpu import (
    three_component_cpu_probe,
)
from tests.test_sd3_full_bf16_worker_c3_f9_cpu import (
    _child, _ordered_groups, _new_r2_probe_using_audited_order,
)


class F9BFrozenOrderedIdentitySupplement(unittest.TestCase):
    def test_caller_mutates_prepared_order_container_after_probe_construction(self):
        bundle = three_component_cpu_probe()
        supplied = {
            profile: [list(group) for group in groups]
            for profile, groups in _ordered_groups(bundle[4]).items()
        }
        probe = _new_r2_probe_using_audited_order(bundle, supplied)
        frozen = probe._expected_group_parameter_order["muon_primary"]
        self.assertEqual(type(frozen), tuple)
        self.assertEqual(type(frozen[0]), tuple)
        supplied["muon_primary"][0].reverse()
        self.assertEqual(probe._expected_group_parameter_order["muon_primary"], frozen)
        muon = _child(bundle[4], "Muon")
        muon.param_groups[0]["params"].reverse()
        with self.assertRaisesRegex(ValueError, "Group|order|RuntimeSpec"):
            probe.install()

    def test_group_reordered_after_forward_before_backward_fails_closed(self):
        bundle = three_component_cpu_probe()
        model, accelerator, _prepared_opt, _prepared_scheduler, facade, probe = bundle
        original = accelerator.backward
        true_backward_calls = []

        def counted_original_backward(*args, **kwargs):
            true_backward_calls.append(True)
            return original(*args, **kwargs)

        accelerator.backward = counted_original_backward
        probe.install()
        try:
            x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
            with accelerator.accumulate(model):
                probe.begin_batch("dts_c2b_sample_0001")
                loss = sum(
                    model[key](x).square().float().sum() for key in model
                )
                _child(facade, "Muon").param_groups[0]["params"].reverse()
                with self.assertRaisesRegex(ValueError, "Group|order|RuntimeSpec"):
                    accelerator.backward(loss)
                self.assertEqual(
                    true_backward_calls, [],
                    "Misordered Group entered genuine Accelerator.backward",
                )
        finally:
            probe.close()
            delattr(accelerator, "backward")

    def test_cross_profile_swap_with_same_total_parameter_set_is_rejected(self):
        bundle = three_component_cpu_probe()
        actual = _ordered_groups(bundle[4])
        muon_first, muon_second = actual["muon_primary"][0]
        adamw_first = actual["adamw_fallback"][0][0]
        # Still three unique real Parameters, one Group per child, and
        # Group cardinalities unchanged; only the trusted route/order is wrong.
        forged = {
            "muon_primary": ((adamw_first, muon_second),),
            "adamw_fallback": ((muon_first,),),
        }
        with self.assertRaisesRegex(ValueError, "Group|order|RuntimeSpec"):
            _new_r2_probe_using_audited_order(bundle, forged)


if __name__ == "__main__":
    unittest.main()
