"""C3 Strict Review F1-F4 RED-first regressions, real pinned CPU only.

Tests exercise actual torch/Accelerate/Muon/AdamW; never grant CUDA status.
Negative controls deliberately fail until C3 Observer's physical checks exist.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from pytorch_optimizer import Muon

from mikazuki.parameter_policy_torch import (
    CompositeOptimizer, CompositeLRScheduler,
    OptimizerRuntimeEntry, SchedulerRuntimeEntry,
)
from tools import sd3_full_bf16_evidence_support as host
from tools import sd3_full_bf16_worker_observer as observer

R1 = "r1-adamw-accum1"
R2 = "r2-muon-adamw-fallback-accum2"
ATTN = "mmdit.attention.adapter"
MLP = "mmdit.mlp.adapter"
MOD = "mmdit.modulation_norm.adapter"


def reference_rows(case_id: str) -> dict:
    """Independently generate complete fixed target geometry/route metadata."""
    rows = {}
    for module, spec in host.build_sd35_medium_lora_target_spec().items():
        comp = spec["component_id"]
        fallback = case_id == R2 and comp == MOD
        for suffix, shape in (
            (".lora_down.weight", [4, spec["input_features"]]),
            (".lora_up.weight", [spec["output_features"], 4]),
        ):
            rows[module + suffix] = {
                "component_id": comp, "shape": shape, "dtype": "BF16",
                "route_kind": "fallback" if fallback else "primary",
                "optimizer_profile": (
                    "adamw_fallback" if fallback else "muon_primary"
                ) if case_id == R2 else "baseline",
                "parameter_class": "matrix_weight",
                "canonical_name": "network." + module + suffix,
                "requires_grad": True,
            }
    return rows


def three_component_cpu_probe(*, muon_lr=0.02, adamw_lr=0.01):
    """Physical real 3-component R2 model with independent Muon/AdamW."""
    torch.manual_seed(123456)
    model = torch.nn.ModuleDict({
        k: torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        for k in ("attention", "mlp", "modulation")
    })
    muon = Muon([{"params": [
        model["attention"].weight, model["mlp"].weight,
    ], "lr": muon_lr, "use_muon": True}])
    adamw = torch.optim.AdamW([model["modulation"].weight], lr=adamw_lr)
    facade = CompositeOptimizer(
        SimpleNamespace(topology_fingerprint="c3-strict-cpu-v1"),
        (
            OptimizerRuntimeEntry("muon_primary", "Muon", "c3-strict-muon", muon),
            OptimizerRuntimeEntry("adamw_fallback", "AdamW", "c3-strict-adamw", adamw),
        ),
    )
    m_sched = torch.optim.lr_scheduler.LambdaLR(muon, lambda _: 1.0)
    a_sched = torch.optim.lr_scheduler.LambdaLR(adamw, lambda _: 1.0)
    scheduler = CompositeLRScheduler(facade, (
        SchedulerRuntimeEntry(
            "muon_primary", "Muon", "c3-strict-muon",
            "external", muon, m_sched, "torch.optim.lr_scheduler.LambdaLR",
        ),
        SchedulerRuntimeEntry(
            "adamw_fallback", "AdamW", "c3-strict-adamw",
            "external", adamw, a_sched, "torch.optim.lr_scheduler.LambdaLR",
        ),
    ))
    accelerator = Accelerator(
        cpu=True, mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=2, adjust_scheduler=False,
        ),
        step_scheduler_with_optimizer=True,
    )
    model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
        model, facade, scheduler,
    )
    named = {"network." + key + ".weight": model[key].weight for key in model}
    components = {
        "network.attention.weight": ATTN,
        "network.mlp.weight": MLP,
        "network.modulation.weight": MOD,
    }
    probe = observer.PhysicalStepProbe(
        accelerator=accelerator, composite_optimizer=facade,
        composite_scheduler=scheduler, parameter_names=named,
        component_by_parameter=components,
        case_id=R2, run_id="run-c3-strict-0001",
        phase_id="fresh-c3-strict-0001",
        source_commit="a" * 40, phase="fresh",
    )
    return model, accelerator, prepared_optimizer, prepared_scheduler, facade, probe


def two_batches(bundle, *, first_zero=False, second_zero=False,
                clear_first_grad=False, zero_mlp_before_step=False):
    model, accelerator, prepared_opt, prepared_sched, facade, probe = bundle
    x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
    probe.install()
    try:
        for n in (1, 2):
            with accelerator.accumulate(model):
                probe.begin_batch(f"dts_c2b_sample_{n:04d}")
                loss = sum(model[key](x).square().float().sum() for key in model)
                if (n == 1 and first_zero) or (n == 2 and second_zero):
                    loss = loss * 0.0
                accelerator.backward(loss)
                if n == 2 and zero_mlp_before_step:
                    model["mlp"].weight.grad.zero_()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                prepared_opt.step()
                prepared_sched.step()
                prepared_opt.zero_grad(set_to_none=True)
            if n == 1 and clear_first_grad:
                model.zero_grad(set_to_none=True)
        probe.assert_step_complete()
        return probe.events
    finally:
        probe.close()


class C3FixF1OwnershipContract(unittest.TestCase):
    def test_reference_exact_shape_and_r2_430_96_route(self):
        for case in (R1, R2):
            with self.subTest(case=case):
                rows = reference_rows(case)
                summary = observer.validate_reference_ownership_rows(
                    case_id=case, rows=rows,
                )
                self.assertEqual(summary["component_tensor_counts"],
                                 {ATTN: 242, MLP: 188, MOD: 96})
                expected = (
                    {"baseline": 526} if case == R1 else
                    {"muon_primary": 430, "adamw_fallback": 96}
                )
                self.assertEqual(summary["child_tensor_counts"], expected)
                target = next(k for k,v in rows.items() if v["component_id"] == ATTN)
                tampered = copy.deepcopy(rows)
                tampered[target]["shape"][0] += 1
                with self.assertRaisesRegex(ValueError, "shape|reference|rank"):
                    observer.validate_reference_ownership_rows(case_id=case, rows=tampered)

    def test_internally_consistent_but_wrong_r2_route_refused(self):
        rows = reference_rows(R2)
        moved = next(k for k,v in rows.items() if v["component_id"] == MLP)
        rows[moved]["route_kind"] = "fallback"
        rows[moved]["optimizer_profile"] = "adamw_fallback"
        with self.assertRaisesRegex(ValueError, "route|Profile|component|ownership"):
            observer.validate_reference_ownership_rows(case_id=R2, rows=rows)

    def test_real_bf16_alpha_scalar_must_equal_four(self):
        target_spec = host.build_sd35_medium_lora_target_spec()
        buffers = {
            module + ".alpha": torch.tensor(4.0, dtype=torch.bfloat16)
            for module in target_spec
        }
        self.assertEqual(
            observer.validate_reference_alpha_buffers(
                buffers, target_spec, expected_device=torch.device("cpu"),
            ), 263,
        )
        bad = dict(buffers)
        first = next(iter(bad))
        bad[first] = torch.tensor(3.5, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "alpha|Alpha|four|4"):
            observer.validate_reference_alpha_buffers(
                bad, target_spec, expected_device=torch.device("cpu"),
            )
        bad[first] = torch.tensor([4.0], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "shape|scalar|Alpha|alpha"):
            observer.validate_reference_alpha_buffers(
                bad, target_spec, expected_device=torch.device("cpu"),
            )


class C3FixF2GradientContract(unittest.TestCase):
    def test_three_component_gradients_have_physical_coverage(self):
        events = two_batches(three_component_cpu_probe())
        backward = [e for e in events if e["event_type"] == "backward_completed"]
        self.assertEqual(len(backward), 2)
        for row in backward:
            components = row["component_gradients"]
            self.assertEqual(set(components), {ATTN, MLP, MOD})
            self.assertTrue(all(info["nonzero_tensor_count"] >= 1
                                for info in components.values()))

    def test_zero_mlp_with_live_attention_child_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Component|MLP|gradient"):
            two_batches(three_component_cpu_probe(), zero_mlp_before_step=True)

    def test_first_microbatch_zero_contribution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "microbatch|contribution|gradient|nonzero"):
            two_batches(three_component_cpu_probe(), first_zero=True)

    def test_first_microbatch_gradient_cleared_before_second_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "retained|gradient|accumulation"):
            two_batches(three_component_cpu_probe(), clear_first_grad=True)

    def test_second_microbatch_zero_contribution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contribution|gradient|accumulation"):
            two_batches(three_component_cpu_probe(), second_zero=True)


class C3FixF3PhysicalWeightContract(unittest.TestCase):
    def test_child_state_advance_but_zero_lr_weight_update_is_rejected(self):
        bundle = three_component_cpu_probe(adamw_lr=0.0)
        with self.assertRaisesRegex(ValueError, "weight|Parameter|unchanged|physical"):
            two_batches(bundle)

    def test_tiny_nonzero_lr_that_rounds_away_in_bf16_is_rejected(self):
        bundle = three_component_cpu_probe(adamw_lr=1e-12)
        with self.assertRaisesRegex(ValueError, "BF16|weight|changed"):
            two_batches(bundle)

    def test_changed_weight_counts_positive_for_both_children(self):
        events = two_batches(three_component_cpu_probe())
        changed = [
            e for e in events if e["event_type"] == "physical_child_stepped"
        ]
        self.assertEqual(len(changed), 2)
        self.assertTrue(all(
            e["changed_parameter_count"] > 0 and
            e["parameter_before"] != e["parameter_after"]
            for e in changed
        ))

    def test_nonfinite_weight_after_real_child_step_is_rejected(self):
        bundle = three_component_cpu_probe()
        model, facade = bundle[0], bundle[4]
        adamw = next(e.optimizer for e in facade.entries if e.optimizer_type == "AdamW")
        original = adamw.step

        def corrupt_after_step(*args, **kwargs):
            result = original(*args, **kwargs)
            with torch.no_grad():
                model["modulation"].weight.view(-1)[0] = float("nan")
            return result

        adamw.step = corrupt_after_step
        try:
            with self.assertRaisesRegex(ValueError, "finite|NaN|Inf|weight"):
                two_batches(bundle)
        finally:
            delattr(adamw, "step")


class C3FixF4MuonStateContract(unittest.TestCase):
    def test_actual_muon_and_adamw_state_advances_twice_without_extra_steps(self):
        bundle = three_component_cpu_probe()
        first_events = two_batches(bundle)
        self.assertEqual(len([e for e in first_events if e["event_type"] == "physical_child_stepped"]), 2)
        second_events = two_batches(bundle)
        self.assertEqual(len([e for e in second_events if e["event_type"] == "physical_child_stepped"]), 4)
        facade = bundle[4]
        muon = next(e.optimizer for e in facade.entries if e.optimizer_type == "Muon")
        adamw = next(e.optimizer for e in facade.entries if e.optimizer_type == "AdamW")
        self.assertEqual(muon.param_groups[0]["step"], 2)
        self.assertEqual(
            observer.validate_muon_parameter_states(muon, physical_step=2)["momentum_parameter_count"], 2,
        )
        self.assertTrue(all(int(state["step"].item()) == 2 for state in adamw.state.values()))
        self.assertTrue(all(
            event["changed_parameter_count"] > 0
            for event in second_events if event["event_type"] == "physical_child_stepped"
        ))

    def test_each_muon_parameter_requires_real_finite_momentum_buffer(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        facade = bundle[4]
        entry = next(e for e in facade.entries if e.optimizer_type == "Muon")
        self.assertEqual(
            observer.validate_muon_parameter_states(
                entry.optimizer, physical_step=1,
            )["momentum_parameter_count"], 2,
        )
        p1, p2 = entry.optimizer.param_groups[0]["params"]
        original = entry.optimizer.state[p2].pop("momentum_buffer")
        try:
            with self.assertRaisesRegex(ValueError, "momentum|Muon|buffer"):
                observer.validate_muon_parameter_states(entry.optimizer, physical_step=1)
        finally:
            entry.optimizer.state[p2]["momentum_buffer"] = original
        saved = original.clone()
        original.fill_(float("nan"))
        try:
            with self.assertRaisesRegex(ValueError, "finite|NaN|momentum|Muon"):
                observer.validate_muon_parameter_states(entry.optimizer, physical_step=1)
        finally:
            original.copy_(saved)
        entry.optimizer.state[p2]["momentum_buffer"] = torch.ones(
            (1,), dtype=p2.dtype, device=p2.device,
        )
        try:
            with self.assertRaisesRegex(ValueError, "shape|momentum|Muon"):
                observer.validate_muon_parameter_states(entry.optimizer, physical_step=1)
        finally:
            entry.optimizer.state[p2]["momentum_buffer"] = original
        entry.optimizer.state[p2]["momentum_buffer"] = torch.ones(
            p2.shape, dtype=p2.dtype, device="meta",
        )
        try:
            with self.assertRaisesRegex(ValueError, "device|momentum|Muon"):
                observer.validate_muon_parameter_states(entry.optimizer, physical_step=1)
        finally:
            entry.optimizer.state[p2]["momentum_buffer"] = original


if __name__ == "__main__":
    unittest.main()
