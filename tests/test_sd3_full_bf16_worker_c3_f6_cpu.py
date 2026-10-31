"""C3-F6 independent RED negatives: Backward provenance, fixed LR, AdamW state.

All optimizer tests use real pinned CPU PyTorch/Accelerate; these are NOT
physical CUDA qualification. The named tests must initially be RED and later
GREEN without any production-trainer or qualification-registry modification.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from pytorch_optimizer import Muon

from mikazuki.parameter_policy_torch import (
    CompositeOptimizer, OptimizerRuntimeEntry,
)
from tests.test_sd3_full_bf16_worker_c3_strict_cpu import (
    R1, R2, ATTN, MOD, three_component_cpu_probe, two_batches,
)
from tools import sd3_full_bf16_worker_observer as observer


def run_faulty_accumulation(*, stale_first=False, mutate_before_second=False):
    """Real two-microbatch Accelerate path, with explicit deterministic fault."""
    model, accelerator, prepared_opt, prepared_sched, _facade, probe = (
        three_component_cpu_probe()
    )
    data = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
    probe.install()
    try:
        for micro in (1, 2):
            with accelerator.accumulate(model):
                probe.begin_batch(f"dts_c2b_sample_{micro:04d}")
                loss = sum(model[name](data).square().float().sum() for name in model)
                if micro == 1 and stale_first:
                    # Existing nonzero gradients are not this microbatch's contribution.
                    for param in model.parameters():
                        param.grad = torch.full_like(param, 0.125)
                    loss = loss * 0
                if micro == 2 and mutate_before_second:
                    # begin_batch() already audited the retained first gradients;
                    # process_batch/hook corruption before actual Backward is simulated.
                    for param in model.parameters():
                        param.grad.add_(torch.full_like(param.grad, 0.5))
                    loss = loss * 0
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                prepared_opt.step()
                prepared_sched.step()
                prepared_opt.zero_grad(set_to_none=True)
        probe.assert_step_complete()
        return probe.events
    finally:
        probe.close()


def fake_reference_session(*, lr_muon=0.0001, lr_adamw=0.0001):
    """Small real Optimizer objects, independently reviewed case reference.

    Deliberately not a CUDA/526-Parameter ownership fixture: validation must
    reject *reference LR* drift even when RuntimeSpec and Child LR agree.
    """
    attn = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    mod = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    muon = Muon([{"params": [attn], "lr": lr_muon, "use_muon": True}])
    adamw = torch.optim.AdamW([mod], lr=lr_adamw)
    facade = CompositeOptimizer(
        SimpleNamespace(topology_fingerprint="f6-lr-cpu"),
        (
            OptimizerRuntimeEntry("muon_primary", "Muon", "f6-muon", muon),
            OptimizerRuntimeEntry("adamw_fallback", "AdamW", "f6-adamw", adamw),
        ),
    )
    assignments = [
        SimpleNamespace(
            canonical_name="network.attention.weight",
            component_id=ATTN, optimizer_profile="muon_primary",
            route_kind="primary", learning_rate=lr_muon,
        ),
        SimpleNamespace(
            canonical_name="network.modulation.weight",
            component_id=MOD, optimizer_profile="adamw_fallback",
            route_kind="fallback", learning_rate=lr_adamw,
        ),
    ]
    group_specs = []
    for name, parameter, lr, profile in (
        ("network.attention.weight", attn, lr_muon, "muon_primary"),
        ("network.modulation.weight", mod, lr_adamw, "adamw_fallback"),
    ):
        group_specs.append(
            SimpleNamespace(
                profile_name=profile,
                groups=(SimpleNamespace(
                    learning_rate=lr,
                    parameters=(SimpleNamespace(
                        canonical_name=name, parameter_id=id(parameter),
                        parameter=parameter,
                    ),),
                ),),
            )
        )
    return SimpleNamespace(
        optimizer=facade,
        routing_plan=SimpleNamespace(assignments=assignments),
        runtime_spec=SimpleNamespace(optimizers=tuple(group_specs)),
    )



def fake_r1_reference_session(*, learning_rate=0.0001):
    """Independent R1 single real AdamW profile at the frozen reference LR."""
    left = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    right = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([left, right], lr=learning_rate)
    facade = CompositeOptimizer(
        SimpleNamespace(topology_fingerprint="f6-r1-lr-cpu"),
        (OptimizerRuntimeEntry("baseline", "AdamW", "f6-r1-adamw", optimizer),),
    )
    assignments = [
        SimpleNamespace(
            canonical_name="network.attention.weight", component_id=ATTN,
            optimizer_profile="baseline", route_kind="primary",
            learning_rate=learning_rate,
        ),
        SimpleNamespace(
            canonical_name="network.modulation.weight", component_id=MOD,
            optimizer_profile="baseline", route_kind="primary",
            learning_rate=learning_rate,
        ),
    ]
    params = tuple(SimpleNamespace(
        canonical_name=name, parameter=parameter,
        parameter_id=id(parameter),
    ) for name, parameter in (
        ("network.attention.weight", left), ("network.modulation.weight", right),
    ))
    group = SimpleNamespace(learning_rate=learning_rate, parameters=params)
    return SimpleNamespace(
        optimizer=facade,
        routing_plan=SimpleNamespace(assignments=assignments),
        runtime_spec=SimpleNamespace(optimizers=(
            SimpleNamespace(profile_name="baseline", groups=(group,)),
        )),
    )


def physical_adamw_state():
    parameter = torch.nn.Parameter(torch.full(
        (4, 4), 0.75, dtype=torch.bfloat16
    ))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.full_like(parameter, 0.1)
    optimizer.step()
    return optimizer, parameter


class F6ABackwardProvenanceContract(unittest.TestCase):
    def test_stale_first_microbatch_nonzero_gradient_not_attributed_to_zero_loss(self):
        with self.assertRaisesRegex(ValueError, "Backward|pre-existing|stale|contribution|gradient"):
            run_faulty_accumulation(stale_first=True)

    def test_mutation_between_begin_batch_and_backward_cannot_fake_second_contribution(self):
        with self.assertRaisesRegex(ValueError, "Backward|pre-existing|retained|gradient"):
            run_faulty_accumulation(mutate_before_second=True)

    def test_unmodified_real_accumulation_remains_accepted(self):
        events = two_batches(three_component_cpu_probe())
        self.assertEqual(
            len([item for item in events if item["event_type"] == "backward_completed"]),
            2,
        )


class F6BReferenceLearningRateContract(unittest.TestCase):
    def test_real_r1_reference_single_adamw_group_lr(self):
        self.assertEqual(
            observer.validate_reference_group_learning_rates(
                case_id=R1, session=fake_r1_reference_session(),
            ),
            {"baseline": [0.0001]},
        )
        with self.assertRaisesRegex(ValueError, "learning|reference|rate"):
            observer.validate_reference_group_learning_rates(
                case_id=R1, session=fake_r1_reference_session(learning_rate=0.001),
            )

    def test_true_reference_primary_and_fallback_learning_rates(self):
        session = fake_reference_session()
        contract = observer.validate_reference_group_learning_rates(
            case_id=R2, session=session,
        )
        self.assertEqual(contract, {
            "muon_primary": [0.0001], "adamw_fallback": [0.0001],
        })

    def test_self_consistent_wrong_runtime_and_physical_primary_lr_rejected(self):
        session = fake_reference_session(lr_muon=0.001)
        with self.assertRaisesRegex(ValueError, "reference|learning|LR|rate"):
            observer.validate_reference_group_learning_rates(
                case_id=R2, session=session,
            )

    def test_self_consistent_wrong_fallback_lr_rejected(self):
        session = fake_reference_session(lr_adamw=0.001)
        with self.assertRaisesRegex(ValueError, "reference|learning|LR|rate"):
            observer.validate_reference_group_learning_rates(
                case_id=R2, session=session,
            )

    def test_lr_mutation_after_probe_initialization_rejected_before_child_step(self):
        bundle = three_component_cpu_probe()
        facade = bundle[4]
        adamw = next(e.optimizer for e in facade.entries if e.optimizer_type == "AdamW")
        adamw.param_groups[0]["lr"] = 0.2
        with self.assertRaisesRegex(ValueError, "learning|LR|rate|drift"):
            two_batches(bundle)

    def test_constant_scheduler_cannot_mutate_physical_group_lr(self):
        bundle = three_component_cpu_probe()
        scheduler = bundle[5].scheduler
        facade = bundle[4]
        adamw = next(e.optimizer for e in facade.entries if e.optimizer_type == "AdamW")
        previous_step = scheduler.step

        def drift_after_scheduler(*args, **kwargs):
            returned = previous_step(*args, **kwargs)
            adamw.param_groups[0]["lr"] = 0.15
            return returned

        scheduler.step = drift_after_scheduler
        try:
            with self.assertRaisesRegex(ValueError, "learning|rate|drift|scheduler"):
                two_batches(bundle)
        finally:
            delattr(scheduler, "step")

    def test_unmodified_cpu_diagnostic_lr_is_stable(self):
        # CPU uses larger LR only to make BF16 physical updates visible;
        # a passing CPU diagnostic never replaces the GPU Policy LR proof.
        events = two_batches(three_component_cpu_probe())
        self.assertEqual(len([
            item for item in events if item["event_type"] == "physical_child_stepped"
        ]), 2)


class F6CAdamWStateContract(unittest.TestCase):
    def test_complete_real_pinned_adamw_state_accepted(self):
        optimizer, param = physical_adamw_state()
        result = observer.validate_adamw_parameter_states(
            optimizer, physical_step=1,
            canonical_names_by_id={id(param): "network.modulation.weight"},
        )
        self.assertEqual(result["moment_parameter_count"], 1)

    def test_missing_or_corrupt_adamw_moments_rejected(self):
        for corruption in (
            "missing_avg", "missing_sq", "nan", "inf", "wrong_shape",
            "wrong_dtype", "meta", "wrong_step", "extra_field",
        ):
            with self.subTest(corruption=corruption):
                optimizer, param = physical_adamw_state()
                state = optimizer.state[param]
                if corruption == "missing_avg":
                    del state["exp_avg"]
                elif corruption == "missing_sq":
                    del state["exp_avg_sq"]
                elif corruption == "nan":
                    state["exp_avg"].fill_(float("nan"))
                elif corruption == "inf":
                    state["exp_avg_sq"].fill_(float("inf"))
                elif corruption == "wrong_shape":
                    state["exp_avg"] = torch.ones((1,), dtype=param.dtype)
                elif corruption == "wrong_dtype":
                    state["exp_avg"] = torch.ones(
                        param.shape, dtype=torch.float32,
                    )
                elif corruption == "meta":
                    state["exp_avg_sq"] = torch.ones(
                        param.shape, dtype=param.dtype, device="meta",
                    )
                elif corruption == "wrong_step":
                    state["step"].fill_(2)
                else:
                    state["unreviewed"] = torch.zeros(())
                with self.assertRaisesRegex(
                    ValueError, "AdamW|moment|exp_avg|finite|device|shape",
                ):
                    observer.validate_adamw_parameter_states(
                        optimizer, physical_step=1,
                        canonical_names_by_id={id(param): "network.modulation.weight"},
                    )

    def test_corrupted_adamw_momentum_after_real_step_cannot_record_success(self):
        bundle = three_component_cpu_probe()
        facade = bundle[4]
        adamw = next(e.optimizer for e in facade.entries if e.optimizer_type == "AdamW")
        original_step = adamw.step

        def corrupt_step(*args, **kwargs):
            returned = original_step(*args, **kwargs)
            for state in adamw.state.values():
                state["exp_avg"].fill_(float("nan"))
            return returned

        adamw.step = corrupt_step
        try:
            with self.assertRaisesRegex(ValueError, "AdamW|moment|finite"):
                two_batches(bundle)
            # The instance-local Child wrapper has been removed in the
            # inherited training/probe finally path; the intentional
            # corruption injector (preexisting instance attribute) remains.
            self.assertIs(adamw.step, corrupt_step)
            self.assertFalse(bundle[5]._installed)
            self.assertFalse(any(
                event["event_type"] == "physical_child_stepped" and
                event.get("optimizer_profile") == "adamw_fallback"
                for event in bundle[5].events
            ))
        finally:
            delattr(adamw, "step")


if __name__ == "__main__":
    unittest.main()
