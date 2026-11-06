"""C3-F9 RED-to-GREEN: actual LoRA Forward + ordered optimizer ownership.

Pinned real CPU: torch==2.7.0, pytorch-optimizer==3.10.0,
accelerate==1.6.0. These tiny physical diagnostics NEVER qualify CUDA/C4.

Negative tests deliberately expect ValueError at an audited boundary.
Do not replace real LoRAModule, BF16 training, or optimizer steps with mocks.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from pytorch_optimizer import Muon

from mikazuki.parameter_policy_torch import (
    CompositeOptimizer, CompositeLRScheduler,
    OptimizerRuntimeEntry, SchedulerRuntimeEntry,
)
from tests.test_sd3_full_bf16_worker_c3_strict_cpu import (
    R2, ATTN, MLP, MOD, three_component_cpu_probe, two_batches,
)
from tests.test_sd3_full_bf16_worker_c3_f8_cpu import (
    _make_r1_cpu_probe, _r1_one_batch,
)
from tools import sd3_full_bf16_worker_observer as observer

# Import the exact production LoRAModule class, not a synthetic substitute.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from networks.lora_flux import LoRAModule  # noqa: E402


class TinyRealLoraNetwork(torch.nn.Module):
    """Small BF16 LoRA adapter executing genuine patched Linear.forward calls."""

    def __init__(self):
        super().__init__()
        self.multiplier = 1.0
        self.split_qkv = False
        self.train_t5xxl = False
        self.loras = torch.nn.ModuleDict()
        # Deliberately not registered twice: these are the frozen base linears.
        self.forward_targets = {}
        self.component_for_key = {}
        for component, label in (
            (ATTN, "attention"), (MLP, "mlp"), (MOD, "modulation"),
        ):
            for index in range(2):
                key = f"{label}_{index}"
                original = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
                original.weight.requires_grad_(False)
                lora = LoRAModule(
                    f"lora_unet_f9_{key}", original,
                    multiplier=1.0, lora_dim=4, alpha=4.0,
                )
                lora.apply_to()
                lora.to(dtype=torch.bfloat16)
                self.loras[key] = lora
                self.forward_targets[key] = original
                self.component_for_key[key] = component

    @property
    def unet_loras(self):
        return tuple(self.loras.values())

    @property
    def text_encoder_loras(self):
        return ()

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.loras.values():
            lora.multiplier = multiplier

    def forward(self, x):
        # Calls patched original Linear, whose forward is the genuine LoRAModule.
        return sum(
            target(x).square().float().sum()
            for target in self.forward_targets.values()
        )


def _child(facade, optimizer_type):
    found = [entry.optimizer for entry in facade.entries
             if entry.optimizer_type == optimizer_type]
    if len(found) != 1:
        raise AssertionError("Unexpected physical optimizer profile inventory")
    return found[0]


def _lora_bundle():
    """Real CPU Accelerator + two independent real Children + six real LoRAs."""
    torch.manual_seed(123456)
    model = TinyRealLoraNetwork()
    muon_params, adamw_params = [], []
    for key, module in model.loras.items():
        target = adamw_params if key.startswith("modulation") else muon_params
        target.extend((module.lora_down.weight, module.lora_up.weight))
    muon = Muon([{"params": muon_params, "lr": 0.02, "use_muon": True}])
    adamw = torch.optim.AdamW(adamw_params, lr=0.01)
    facade = CompositeOptimizer(
        SimpleNamespace(topology_fingerprint="c3-f9-real-lora-cpu"),
        (
            OptimizerRuntimeEntry("muon_primary", "Muon", "f9-muon", muon),
            OptimizerRuntimeEntry("adamw_fallback", "AdamW", "f9-adamw", adamw),
        ),
    )
    scheduler = CompositeLRScheduler(facade, (
        SchedulerRuntimeEntry(
            "muon_primary", "Muon", "f9-muon", "external", muon,
            torch.optim.lr_scheduler.LambdaLR(muon, lambda _: 1.0),
            "torch.optim.lr_scheduler.LambdaLR",
        ),
        SchedulerRuntimeEntry(
            "adamw_fallback", "AdamW", "f9-adamw", "external", adamw,
            torch.optim.lr_scheduler.LambdaLR(adamw, lambda _: 1.0),
            "torch.optim.lr_scheduler.LambdaLR",
        ),
    ))
    accelerator = Accelerator(
        cpu=True, mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=2, adjust_scheduler=False,
        ),
        step_scheduler_with_optimizer=True,
    )
    model, prepared_opt, prepared_scheduler = accelerator.prepare(
        model, facade, scheduler,
    )
    params = {"network." + key: value for key, value in model.named_parameters()}
    components = {
        name: model.component_for_key[name.split(".")[2]]
        for name in params
    }
    probe_kwargs = {
        "accelerator": accelerator,
        "composite_optimizer": facade,
        "composite_scheduler": scheduler,
        "parameter_names": params,
        "component_by_parameter": components,
        "case_id": R2,
        "run_id": "run-c3-f9-real-lora-0001",
        "phase_id": "fresh-c3-f9-real-lora-0001",
        "source_commit": "a" * 40,
        "phase": "fresh",
        # F9-A now REQUIRES the real LoRA Network binding; no optional bridge.
        "lora_network": model,
    }
    probe = observer.PhysicalStepProbe(**probe_kwargs)
    return model, accelerator, prepared_opt, prepared_scheduler, facade, probe


def _run_real_lora_step(bundle):
    model, accelerator, prepared_opt, prepared_sched, _, probe = bundle
    x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
    probe.install()
    try:
        for n in (1, 2):
            with accelerator.accumulate(model):
                probe.begin_batch(f"dts_c2b_sample_{n:04d}")
                accelerator.backward(model(x))
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                prepared_opt.step()
                prepared_sched.step()
                prepared_opt.zero_grad(set_to_none=True)
        probe.assert_step_complete()
        return probe.events
    finally:
        probe.close()


def _assert_pre_backward_reject(test_case, bundle, error_regex):
    accelerator = bundle[1]
    original = accelerator.backward
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    accelerator.backward = counted
    try:
        with test_case.assertRaisesRegex(ValueError, error_regex):
            _run_real_lora_step(bundle)
        test_case.assertEqual(
            calls, [], "Unreviewed LoRA/Group drift entered real Backward",
        )
    finally:
        delattr(accelerator, "backward")



def _assert_strict_cpu_pre_backward_reject(test_case, bundle, error_regex):
    """Existing three-component Linear fixture, not the real-LoRA fixture."""
    accelerator = bundle[1]
    original = accelerator.backward
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    accelerator.backward = counted
    try:
        with test_case.assertRaisesRegex(ValueError, error_regex):
            two_batches(bundle)
        test_case.assertEqual(calls, [], "Group drift reached real Backward")
    finally:
        delattr(accelerator, "backward")


def _ordered_groups(facade):
    return {
        entry.profile_name: tuple(
            tuple(id(p) for p in group["params"])
            for group in entry.optimizer.param_groups
        )
        for entry in facade.entries
    }


def _new_r2_probe_using_audited_order(bundle, audited):
    """Re-create after a simulated Prepared/Probe gap, with trusted IDs."""
    original = bundle[5]
    kwargs = dict(
        accelerator=bundle[1], composite_optimizer=bundle[4],
        composite_scheduler=original.scheduler,
        parameter_names=original.params,
        component_by_parameter=original.component_by_parameter,
        case_id=R2, run_id="run-c3-f9-order-gap-0001",
        phase_id="fresh-c3-f9-order-gap-0001",
        source_commit="a" * 40, phase="fresh",
    )
    # F9-B requires an independently frozen, pre-mutation Group order.
    # The real CUDA path derives this from audited RuntimeSpec canonical IDs;
    # this tiny CPU fixture captures it before the deliberate tampering.
    kwargs["expected_group_parameter_order"] = audited
    return observer.PhysicalStepProbe(**kwargs)


class F9ALiveLoraForwardSemantics(unittest.TestCase):
    ERROR = "LoRA|Forward|semantic|alpha|scale|multiplier|identity"

    def test_real_lora_scale_mutation_before_forward_must_reject(self):
        bundle = _lora_bundle()
        self.assertIsInstance(bundle[0].loras["attention_0"], LoRAModule)
        bundle[0].loras["attention_0"].scale = 0.5
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_lora_multiplier_mutation_before_forward_must_reject(self):
        bundle = _lora_bundle()
        bundle[0].loras["attention_0"].multiplier = 0.75
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_lora_alpha_buffer_finite_edit_must_reject(self):
        bundle = _lora_bundle()
        with torch.no_grad():
            bundle[0].loras["attention_0"].alpha.fill_(3.5)
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_lora_dropout_mutation_before_forward_must_reject(self):
        bundle = _lora_bundle()
        bundle[0].loras["attention_0"].dropout = 0.15
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_lora_rank_dropout_mutation_before_forward_must_reject(self):
        bundle = _lora_bundle()
        bundle[0].loras["attention_0"].rank_dropout = 0.2
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_root_set_multiplier_mutation_must_reject(self):
        bundle = _lora_bundle()
        bundle[0].set_multiplier(0.75)
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_lora_canonical_name_drift_must_reject(self):
        bundle = _lora_bundle()
        bundle[0].loras["attention_0"].lora_name = "lora_unet_f9_wrong"
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_real_target_linear_forward_rebound_must_reject(self):
        bundle = _lora_bundle()
        target = bundle[0].forward_targets["attention_0"]
        original = target.forward

        def unreviewed_but_differentiable(x):
            return original(x) * 0.5

        target.forward = unreviewed_but_differentiable
        _assert_pre_backward_reject(self, bundle, self.ERROR)

    def test_mutation_inside_real_muon_step_cannot_publish_child_success(self):
        bundle = _lora_bundle()
        muon = _child(bundle[4], "Muon")
        original = muon.step
        calls = []

        def change_scale_after_original(*args, **kwargs):
            calls.append(True)
            result = original(*args, **kwargs)
            bundle[0].loras["attention_0"].scale = 0.75
            return result

        muon.step = change_scale_after_original
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                _run_real_lora_step(bundle)
            self.assertEqual(len(calls), 1, "Original physical Muon.step did not run")
            self.assertFalse(any(
                item["event_type"] == "physical_child_stepped"
                and item.get("optimizer_profile") == "muon_primary"
                for item in bundle[5].events
            ), "Invalid post-Child Forward semantics received success evidence")
        finally:
            delattr(muon, "step")

    def test_unmodified_real_lora_two_physical_r2_steps_positive(self):
        bundle = _lora_bundle()
        self.assertTrue(all(
            isinstance(module, LoRAModule)
            for module in bundle[0].loras.values()
        ))
        for _ in range(2):
            events = _run_real_lora_step(bundle)
        self.assertEqual(len([
            item for item in events
            if item["event_type"] == "physical_child_stepped"
        ]), 4)


class F9BPhysicalGroupOrderedIdentity(unittest.TestCase):
    ERROR = "Group|order|Parameter|ownership|RuntimeSpec|prepared"

    def test_muon_group_reverse_after_probe_init_rejected_before_step(self):
        bundle = three_component_cpu_probe()
        muon = _child(bundle[4], "Muon")
        muon.param_groups[0]["params"].reverse()
        _assert_strict_cpu_pre_backward_reject(self, bundle, self.ERROR)

    def test_muon_group_reverse_between_accepted_steps_rejected(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        muon = _child(bundle[4], "Muon")
        muon.param_groups[0]["params"].reverse()
        accelerator = bundle[1]
        calls, original = [], accelerator.backward

        def counted(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        accelerator.backward = counted
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                two_batches(bundle)
            self.assertEqual(calls, [], "Group drift entered next Backward")
        finally:
            delattr(accelerator, "backward")

    def test_group_reverse_after_prepared_before_probe_construction_rejected(self):
        bundle = three_component_cpu_probe()
        audited = _ordered_groups(bundle[4])
        muon = _child(bundle[4], "Muon")
        muon.param_groups[0]["params"].reverse()
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                _new_r2_probe_using_audited_order(bundle, audited)
        finally:
            muon.param_groups[0]["params"].reverse()

    def test_group_reversed_inside_scheduler_cannot_publish_success(self):
        bundle = three_component_cpu_probe()
        facade = bundle[4]
        scheduler = bundle[5].scheduler
        original = scheduler.step

        def change_order_after_real_scheduler(*args, **kwargs):
            result = original(*args, **kwargs)
            _child(facade, "Muon").param_groups[0]["params"].reverse()
            return result

        scheduler.step = change_order_after_real_scheduler
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                two_batches(bundle)
            self.assertFalse(any(
                item["event_type"] == "physical_scheduler_stepped"
                for item in bundle[5].events
            ), "Invalid Group order published scheduler success")
        finally:
            delattr(scheduler, "step")

    def test_duplicate_group_parameter_after_probe_init_must_reject(self):
        bundle = three_component_cpu_probe()
        muon = _child(bundle[4], "Muon")
        muon.param_groups[0]["params"].append(muon.param_groups[0]["params"][0])
        _assert_strict_cpu_pre_backward_reject(self, bundle, self.ERROR)

    def test_original_step_reordering_detection_still_rejects(self):
        bundle = three_component_cpu_probe()
        muon = _child(bundle[4], "Muon")
        original = muon.step

        def reverse_after_real_step(*args, **kwargs):
            result = original(*args, **kwargs)
            muon.param_groups[0]["params"].reverse()
            return result

        muon.step = reverse_after_real_step
        try:
            with self.assertRaisesRegex(ValueError, self.ERROR):
                two_batches(bundle)
        finally:
            delattr(muon, "step")

    def test_unmodified_real_muon_adamw_two_steps_positive(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        events = two_batches(bundle)
        self.assertEqual(len([
            event for event in events
            if event["logical_step"] == 2
            and event["event_type"] == "physical_child_stepped"
        ]), 2)

    def test_unmodified_real_r1_adamw_two_steps_positive(self):
        bundle = _make_r1_cpu_probe()
        _r1_one_batch(bundle, 1)
        _r1_one_batch(bundle, 2)
        self.assertEqual(bundle[5]._step_scheduler, {1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()
