"""C3-F8 RED-first real CPU regressions for immutable semantics and Fresh continuity.

Uses pinned torch 2.7.0, pytorch-optimizer 3.10.0, Accelerate 1.6.0.
Neither tiny-CPU results nor this suite imply physical CUDA qualification.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

from mikazuki.parameter_policy_torch import (
    CompositeOptimizer, CompositeLRScheduler,
    OptimizerRuntimeEntry, SchedulerRuntimeEntry,
)
from tests.test_sd3_full_bf16_worker_c3_strict_cpu import (
    R1, R2, three_component_cpu_probe, two_batches,
)
from tests.test_sd3_full_bf16_worker_c3_f6_cpu import fake_reference_session
from tools import sd3_full_bf16_worker_observer as observer


def _child(facade, kind):
    matches = [e.optimizer for e in facade.entries if e.optimizer_type == kind]
    if len(matches) != 1:
        raise AssertionError("Missing or duplicate physical " + kind)
    return matches[0]


def _mutate_muon_coefficients(muon):
    """In-place list mutation, not Group assignment or replacement."""
    coefficients = muon.param_groups[0]["ns_coeffs"]
    assert isinstance(coefficients, list) and coefficients
    first = coefficients[0]
    assert isinstance(first, (tuple, list)) and len(first) == 3
    coefficients[0] = (float(first[0]) + 0.125, float(first[1]), float(first[2]))


def _counting_original(obj, method, counts):
    original = getattr(obj, method)

    def counted(*args, **kwargs):
        counts.append(True)
        return original(*args, **kwargs)

    setattr(obj, method, counted)


def _make_r1_cpu_probe():
    """Actual R1 AdamW+LambdaLR+Accelerate fixture for two physical steps."""
    torch.manual_seed(123456)
    model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    adamw = torch.optim.AdamW(model.parameters(), lr=0.01)
    facade = CompositeOptimizer(
        SimpleNamespace(topology_fingerprint="f8-r1-cpu"),
        (OptimizerRuntimeEntry("baseline", "AdamW", "f8-r1-adamw", adamw),),
    )
    scheduler = CompositeLRScheduler(facade, (
        SchedulerRuntimeEntry(
            "baseline", "AdamW", "f8-r1-adamw", "external",
            adamw, torch.optim.lr_scheduler.LambdaLR(adamw, lambda _: 1.0),
            "torch.optim.lr_scheduler.LambdaLR",
        ),
    ))
    accelerator = Accelerator(
        cpu=True, mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=1, adjust_scheduler=False,
        ),
    )
    model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
        model, facade, scheduler,
    )
    probe = observer.PhysicalStepProbe(
        accelerator=accelerator, composite_optimizer=facade,
        composite_scheduler=scheduler,
        parameter_names={"network.r1.weight": model.weight},
        component_by_parameter={
            "network.r1.weight": "mmdit.attention.adapter",
        },
        case_id=R1, run_id="run-c3-f8-r1-0001",
        phase_id="fresh-c3-f8-r1-0001",
        source_commit="a" * 40, phase="fresh",
    )
    return model, accelerator, prepared_optimizer, prepared_scheduler, facade, probe


def _r1_one_batch(bundle, number):
    model, accelerator, prepared_opt, prepared_sched, _, probe = bundle
    probe.install()
    try:
        with accelerator.accumulate(model):
            probe.begin_batch(f"dts_c2b_sample_{number:04d}")
            x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
            accelerator.backward(model(x).square().float().sum())
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            prepared_opt.step()
            prepared_sched.step()
            prepared_opt.zero_grad(set_to_none=True)
        probe.assert_step_complete()
    finally:
        probe.close()


class F8AImmutableProviderSemanticBaseline(unittest.TestCase):
    def test_prepared_reference_ns_coeffs_is_not_aliased_to_physical_group(self):
        session = fake_reference_session()
        snapshot = observer.validate_reference_optimizer_group_semantics(
            case_id=R2, session=session,
        )["muon_primary"]
        expected_sha = snapshot["semantic_sha256"]
        _mutate_muon_coefficients(_child(session.optimizer, "Muon"))
        actual_snapshot_sha = observer.fingerprint_state_tree({
            key: value for key, value in snapshot.items()
            if key != "semantic_sha256"
        })
        self.assertEqual(
            actual_snapshot_sha, expected_sha,
            "Prepared reference values must not track live ns_coeffs mutation",
        )

    def test_muon_in_place_coefficients_after_probe_init_rejected_before_step(self):
        bundle = three_component_cpu_probe()
        muon = _child(bundle[4], "Muon")
        calls = []
        _counting_original(muon, "step", calls)
        _mutate_muon_coefficients(muon)
        try:
            with self.assertRaisesRegex(ValueError, "semantic|drift|reference"):
                two_batches(bundle)
            self.assertEqual(calls, [], "Invalid semantics reached physical Muon.step")
        finally:
            delattr(muon, "step")

    def test_muon_in_place_mutation_inside_real_step_cannot_publish_success(self):
        bundle = three_component_cpu_probe()
        muon = _child(bundle[4], "Muon")
        original = muon.step
        calls = []

        def corrupt_after_real_step(*args, **kwargs):
            calls.append(True)
            output = original(*args, **kwargs)
            _mutate_muon_coefficients(muon)
            return output

        muon.step = corrupt_after_real_step
        try:
            with self.assertRaisesRegex(ValueError, "semantic|drift|reference"):
                two_batches(bundle)
            self.assertEqual(len(calls), 1, "Real Muon.step must execute once")
            self.assertFalse(any(
                event["event_type"] == "physical_child_stepped"
                and event.get("optimizer_profile") == "muon_primary"
                for event in bundle[5].events
            ), "Post-step semantic drift cannot publish a successful Child event")
        finally:
            delattr(muon, "step")

    def test_unmodified_r2_semantics_remain_valid_for_two_steps(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        events = two_batches(bundle)
        self.assertEqual(
            len([x for x in events if x["event_type"] == "physical_child_stepped"
                 and x["logical_step"] == 2]), 2,
        )


class F8BFreshMuonMomentumContinuity(unittest.TestCase):
    def test_missing_fresh_muon_baseline_rejected_before_real_step2(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        probe = bundle[5]
        self.assertIn("muon_primary", probe._last_muon_momentum_sha)
        probe._last_muon_momentum_sha.clear()
        muon = _child(bundle[4], "Muon")
        calls = []
        _counting_original(muon, "step", calls)
        try:
            with self.assertRaisesRegex(ValueError, "Muon|momentum|baseline"):
                two_batches(bundle)
            self.assertEqual(calls, [], "Missing baseline reached physical Muon.step")
        finally:
            delattr(muon, "step")

    def test_unmodified_fresh_muon_momentum_continuity_positive(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        first = bundle[5]._last_muon_momentum_sha["muon_primary"]
        two_batches(bundle)
        self.assertNotEqual(
            first, bundle[5]._last_muon_momentum_sha["muon_primary"],
        )

    def test_resume_step2_does_not_require_in_process_fresh_muon_sha(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        model, accelerator, prepared_opt, prepared_sched, facade, previous = bundle
        resumed = observer.PhysicalStepProbe(
            accelerator=accelerator,
            composite_optimizer=facade,
            composite_scheduler=previous.scheduler,
            parameter_names=previous.params,
            component_by_parameter=previous.component_by_parameter,
            case_id=R2, run_id="run-c3-f8-resume-0001",
            phase_id="resume-c3-f8-0001",
            source_commit="a" * 40, phase="resume", start_step=1,
        )
        self.assertEqual(resumed._last_muon_momentum_sha, {})
        events = two_batches((
            model, accelerator, prepared_opt, prepared_sched, facade, resumed,
        ))
        self.assertEqual(
            len([x for x in events if x["event_type"] == "physical_child_stepped"
                 and x["logical_step"] == 2]), 2,
        )


class F8CFreshAdapterParameterContinuity(unittest.TestCase):
    def _tamper_between_steps(self, kind):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        optimizer = _child(bundle[4], kind)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].add_(1.0)
        accelerator = bundle[1]
        calls = []
        _counting_original(accelerator, "backward", calls)
        try:
            with self.assertRaisesRegex(ValueError, "Adapter|parameter|continuity|Step"):
                two_batches(bundle)
            self.assertEqual(calls, [], "Tampered Adapter reached Step2 Backward")
        finally:
            delattr(accelerator, "backward")

    def test_finite_muon_owned_bf16_weight_mutated_between_steps_rejected(self):
        self._tamper_between_steps("Muon")

    def test_finite_adamw_owned_bf16_weight_mutated_between_steps_rejected(self):
        self._tamper_between_steps("AdamW")

    def test_missing_fresh_adapter_sha_rejected_before_step2_backward(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        probe = bundle[5]
        probe._last_completed_adapter_sha256 = None
        accelerator = bundle[1]
        calls = []
        _counting_original(accelerator, "backward", calls)
        try:
            with self.assertRaisesRegex(ValueError, "Adapter|parameter|baseline"):
                two_batches(bundle)
            self.assertEqual(calls, [], "Missing Adapter baseline reached Backward")
        finally:
            delattr(accelerator, "backward")

    def test_unmodified_r1_two_complete_adamw_steps_remain_valid(self):
        bundle = _make_r1_cpu_probe()
        _r1_one_batch(bundle, 1)
        _r1_one_batch(bundle, 2)
        probe = bundle[5]
        self.assertEqual(
            sorted(probe._step_children),
            [1, 2],
        )
        self.assertEqual(probe._step_scheduler, {1: 1, 2: 1})

    def test_unmodified_r2_accumulation_and_resume_first_step_remain_valid(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        events = two_batches(bundle)
        self.assertEqual(len([
            x for x in events if x["event_type"] == "physical_scheduler_stepped"
            and x["logical_step"] == 2
        ]), 1)


if __name__ == "__main__":
    unittest.main()
