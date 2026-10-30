"""C3 RED first: real Accelerate CPU microbatch/child-step observer contracts.

These tests are intentionally RED before C3 implementation. They use the
actual torch 2.7.0, Accelerate 1.6.0 and pytorch-optimizer 3.10.0; CPU evidence
must never be misreported as candidate CUDA qualification.
"""
from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import accelerate
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
import torch
from pytorch_optimizer import Muon

from mikazuki.parameter_policy_torch import (
    CompositeOptimizer, CompositeLRScheduler,
    OptimizerRuntimeEntry, SchedulerRuntimeEntry,
)


R2 = "r2-muon-adamw-fallback-accum2"


def observer():
    return importlib.import_module("tools.sd3_full_bf16_worker_observer")


def build_two_child_cpu(*, production_default_accumulator: bool = False):
    """Real BF16 two-Child facade with explicit or production-default accumulation."""
    torch.manual_seed(123456)
    model = torch.nn.ModuleDict({
        "left": torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16),
        "right": torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16),
    })
    muon = Muon([{
        "params": [model["left"].weight], "lr": 1e-4, "use_muon": True,
    }])
    adamw = torch.optim.AdamW(
        [model["right"].weight], lr=1e-4, weight_decay=0.01,
    )
    topology = SimpleNamespace(topology_fingerprint="c3-physical-cpu-v1")
    optimizer = CompositeOptimizer(topology, (
        OptimizerRuntimeEntry(
            profile_name="muon_primary", optimizer_type="Muon",
            topology_fingerprint="c3-muon", optimizer=muon,
        ),
        OptimizerRuntimeEntry(
            profile_name="adamw_fallback", optimizer_type="AdamW",
            topology_fingerprint="c3-adamw", optimizer=adamw,
        ),
    ))
    muon_scheduler = torch.optim.lr_scheduler.LambdaLR(muon, lambda _: 1.0)
    adamw_scheduler = torch.optim.lr_scheduler.LambdaLR(adamw, lambda _: 1.0)
    scheduler = CompositeLRScheduler(optimizer, (
        SchedulerRuntimeEntry(
            profile_name="muon_primary", optimizer_type="Muon",
            topology_fingerprint="c3-muon", mode="external",
            optimizer=muon, scheduler=muon_scheduler,
            scheduler_type="torch.optim.lr_scheduler.LambdaLR",
        ),
        SchedulerRuntimeEntry(
            profile_name="adamw_fallback", optimizer_type="AdamW",
            topology_fingerprint="c3-adamw", mode="external",
            optimizer=adamw, scheduler=adamw_scheduler,
            scheduler_type="torch.optim.lr_scheduler.LambdaLR",
        ),
    ))
    if production_default_accumulator:
        # Mirrors production train_util.prepare_accelerator(args): it passes
        # gradient_accumulation_steps, not a custom GradientAccumulationPlugin.
        acc = Accelerator(
            cpu=True, mixed_precision="no",
            gradient_accumulation_steps=2,
            step_scheduler_with_optimizer=True,
        )
    else:
        acc = Accelerator(
            cpu=True, mixed_precision="no",
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=2, adjust_scheduler=False,
            ),
            step_scheduler_with_optimizer=True,
        )
    model, prepared_optimizer, prepared_scheduler = acc.prepare(
        model, optimizer, scheduler,
    )
    return acc, model, optimizer, scheduler, prepared_optimizer, prepared_scheduler


def make_probe():
    acc, model, optimizer, scheduler, prepared_optimizer, prepared_scheduler = build_two_child_cpu()
    probe = observer().PhysicalStepProbe(
        accelerator=acc, composite_optimizer=optimizer,
        composite_scheduler=scheduler,
        parameter_names={
            "network.left.weight": model["left"].weight,
            "network.right.weight": model["right"].weight,
        },
        case_id=R2, run_id="run-c3cpu-0001",
        phase_id="fresh-c3cpu-0001",
        source_commit="a" * 40, phase="fresh",
    )
    return acc, model, optimizer, scheduler, prepared_optimizer, prepared_scheduler, probe


def actual_accumulated_two_batches(
    acc, model, optimizer, scheduler, prepared_optimizer, prepared_scheduler, probe,
    *, mutate_after_first: bool = False,
):
    probe.install()
    try:
        for i in (1, 2):
            x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
            with acc.accumulate(model):
                probe.begin_batch(f"dts_c2b_sample_{i:04d}")
                loss = (
                    model["left"](x).square().float().sum()
                    + model["right"](x).square().float().sum()
                )
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                prepared_optimizer.step()
                prepared_scheduler.step()
                prepared_optimizer.zero_grad(set_to_none=True)
            if i == 1 and mutate_after_first:
                with torch.no_grad():
                    model["left"].weight.add_(torch.ones_like(model["left"].weight))
        probe.assert_step_complete()
        return probe.events
    finally:
        probe.close()


class C3PhysicalObserverCPU(unittest.TestCase):
    def test_exact_tensor_state_and_rng_fingerprints(self):
        api = observer()
        x = torch.tensor([1.0, 1.5], dtype=torch.bfloat16)
        a = api.fingerprint_tensor(x)
        self.assertEqual(a["dtype"], "torch.bfloat16")
        self.assertEqual(a["shape"], [2])
        self.assertEqual(a, api.fingerprint_tensor(x.clone()))
        self.assertNotEqual(a["sha256"], api.fingerprint_tensor(x.float())["sha256"])
        p = {1: {"momentum_buffer": x}, "flags": (1, 2)}
        q = {"flags": (1, 2), 1: {"momentum_buffer": x.clone()}}
        self.assertEqual(api.fingerprint_state_tree(p), api.fingerprint_state_tree(q))
        q[1]["momentum_buffer"][0] = 0
        self.assertNotEqual(api.fingerprint_state_tree(p), api.fingerprint_state_tree(q))
        with self.assertRaises(ValueError):
            api.fingerprint_state_tree({"broken": float("nan")})
        first = api.snapshot_rng_fingerprint()
        second = api.snapshot_rng_fingerprint()
        self.assertEqual(first, second)
        self.assertIn("numpy", first)
        self.assertIn("torch_cpu", first)

    def test_physical_muon_and_adamw_only_step_on_second_microbatch(self):
        acc, model, optim, sched, po, ps, probe = make_probe()
        self.assertFalse(acc.gradient_state.adjust_scheduler)
        self.assertIs(acc._optimizers[0].optimizer, optim)
        self.assertIs(acc._schedulers[0].scheduler, sched)
        self.assertTrue(optim.entries[0].optimizer.param_groups[0]["use_muon"])
        events = actual_accumulated_two_batches(
            acc, model, optim, sched, po, ps, probe,
        )
        types = [row["event_type"] for row in events]
        self.assertEqual(types.count("batch_started"), 2)
        self.assertEqual(types.count("backward_completed"), 2)
        self.assertEqual(types.count("physical_child_stepped"), 2)
        self.assertEqual(types.count("physical_scheduler_stepped"), 1)
        self.assertEqual(
            [row["sync_gradients"] for row in events if row["event_type"] == "backward_completed"],
            [False, True],
        )
        self.assertEqual(
            {row["optimizer_profile"] for row in events if row["event_type"] == "physical_child_stepped"},
            {"muon_primary", "adamw_fallback"},
        )
        self.assertEqual(sched.state_dict()["step_count"], 2)
        self.assertEqual(sched.state_dict()["last_epoch"], 1)
        self.assertEqual(int(optim.entries[0].optimizer.param_groups[0]["step"]), 1)
        self.assertTrue(bool(optim.entries[0].optimizer.state))
        self.assertTrue(bool(optim.entries[1].optimizer.state))
        self.assertEqual(probe.sync_sequence, (False, True))
        self.assertTrue(all(
            len(entry["child_gradients"]) == 2
            and all(len(x["gradient_sha256"]) == 64
                    for x in entry["child_gradients"].values())
            for entry in events
            if entry["event_type"] == "backward_completed"
        ))
        self.assertTrue(all(
            event["physical_step"] == 1
            and event["optimizer_state_before"] != event["optimizer_state_after"]
            for event in events
            if event["event_type"] == "physical_child_stepped"
        ))
        self.assertFalse("step" in vars(optim), "method wrapper leaked")
        self.assertFalse("backward" in vars(acc), "Accelerator patch leaked")

    def test_unmodified_production_accelerator_effective_default_is_safe(self):
        """Pin effective v1.6.0 production behavior, not dataclass defaults.

        KwargsHandler.to_kwargs() drops the plugin's default True value;
        GradientState.adjust_scheduler reads absent keys as False.
        This surprising behavior must be retested on future upgrades.
        """
        acc, model, optimizer, scheduler, prepared_opt, prepared_sched = (
            build_two_child_cpu(production_default_accumulator=True)
        )
        self.assertEqual(acc.gradient_accumulation_steps, 2)
        self.assertFalse(
            acc.gradient_state.adjust_scheduler,
            "Production-style GradientState effective scheduler flag changed",
        )
        self.assertNotIn(
            "adjust_scheduler", acc.gradient_state.plugin_kwargs,
            "Unexpected explicit adjust_scheduler override in stock production path",
        )
        state_before = observer().fingerprint_state_tree(
            {entry.profile_name: entry.optimizer.state_dict()
             for entry in optimizer.entries}
        )
        scheduler_before = scheduler.state_dict()
        x = torch.tensor([[0.25, -0.5, 1., -1.5]], dtype=torch.bfloat16)
        with acc.accumulate(model):
            self.assertFalse(acc.sync_gradients)
            loss = (
                model["left"](x).square().float().sum()
                + model["right"](x).square().float().sum()
            )
            acc.backward(loss)
            prepared_opt.step()
            prepared_sched.step()
            prepared_opt.zero_grad(set_to_none=True)
        state_after = observer().fingerprint_state_tree(
            {entry.profile_name: entry.optimizer.state_dict()
             for entry in optimizer.entries}
        )
        self.assertEqual(state_before, state_after)
        scheduler_after = scheduler.state_dict()
        self.assertEqual(scheduler_after["last_epoch"], scheduler_before["last_epoch"])
        self.assertEqual(
            scheduler_after["step_count"], scheduler_before["step_count"],
            "Stock production-style unsynced Scheduler mutated internal state",
        )
        verified = observer().verify_prepared_identity(
            session=SimpleNamespace(optimizer=optimizer),
            accelerator=acc, scheduler=scheduler,
        )
        self.assertEqual(verified["optimizer_children"], 2)
        self.assertEqual(verified["scheduler_children"], 2)

    def test_non_sync_parameter_mutation_is_rejected(self):
        acc, model, optim, sched, po, ps, probe = make_probe()
        with self.assertRaisesRegex(ValueError, "non-sync|accumulation|premature"):
            actual_accumulated_two_batches(
                acc, model, optim, sched, po, ps, probe,
                mutate_after_first=True,
            )
        self.assertFalse("step" in vars(optim))
        self.assertFalse("backward" in vars(acc))

    def test_prepared_identity_wrong_session_optimizer_rejected(self):
        acc, model, optim, sched, po, ps = build_two_child_cpu()
        wrong = SimpleNamespace(optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4))
        with self.assertRaises((ValueError, RuntimeError)):
            observer().verify_prepared_identity(
                session=wrong, accelerator=acc, scheduler=sched,
            )
        actual = SimpleNamespace(optimizer=optim)
        res = observer().verify_prepared_identity(
            session=actual, accelerator=acc, scheduler=sched,
        )
        self.assertEqual(res["optimizer_children"], 2)
        self.assertEqual(res["scheduler_children"], 2)
        self.assertEqual(res["device_type"], "cpu")

    def test_observe_prepared_ownership_cannot_certify_cpu(self):
        acc, model, optim, sched, po, ps = build_two_child_cpu()
        with self.assertRaises((ValueError, RuntimeError)):
            observer().observe_prepared_ownership(
                session=SimpleNamespace(optimizer=optim),
                accelerator=acc, network=model,
                train_dataloader=[1, 2], case_id=R2,
            )


    def test_observed_trainer_is_actual_production_subclass_and_fail_closed_c4(self):
        cls = observer().ObservedSd3NetworkTrainer
        import sd3_train_network
        self.assertTrue(issubclass(cls, sd3_train_network.Sd3NetworkTrainer))
        trainer = cls(
            invocation=SimpleNamespace(case_id=R2),
            cuda_environment={"cuda_available": False},
            qualification_lease={"backend_lease_only": True},
        )
        self.assertIsNone(trainer.observer)
        self.assertIsNone(trainer._c3_session)
        base = sd3_train_network.Sd3NetworkTrainer

        # No production Trainer replacement: the existing method is called
        # exactly once before the C3 Scheduler instance capture.
        acc, network, facade, scheduler, _, _ = build_two_child_cpu()
        session = SimpleNamespace(build_scheduler=lambda factory: scheduler)
        original_build = session.build_scheduler
        with patch.object(base, "configure_parameter_policy_training",
                          return_value=(True, False)) as parent:
            result = trainer.configure_parameter_policy_training(
                SimpleNamespace(), session, [],
            )
            parent.assert_called_once()
        self.assertEqual(result, (True, False))
        self.assertIs(session.build_scheduler(None), scheduler)
        self.assertIs(trainer._c3_scheduler, scheduler)
        with self.assertRaises(ValueError):
            session.build_scheduler(None)

        # Abort before real training; the original failure must propagate
        # unmodified and Session's method must be restored in finally.
        with patch.object(base, "train",
                          side_effect=RuntimeError("intentional production failure")) as parent:
            with self.assertRaisesRegex(RuntimeError, "intentional production failure"):
                trainer.train(SimpleNamespace())
            parent.assert_called_once()
        self.assertIs(session.build_scheduler, original_build)

    def test_original_sd3_hook_and_step_start_delegate_first(self):
        cls = observer().ObservedSd3NetworkTrainer
        import sd3_train_network
        trainer = cls(
            invocation=SimpleNamespace(case_id=R2),
            cuda_environment={}, qualification_lease={},
        )
        base = sd3_train_network.Sd3NetworkTrainer
        with patch.object(
            base, "register_additional_state_hooks",
            side_effect=RuntimeError("real sd3 state hook invoked"),
        ) as parent:
            with self.assertRaisesRegex(RuntimeError, "real sd3 state hook invoked"):
                trainer.register_additional_state_hooks(
                    accelerator=None, args=None, unet=None,
                    train_dataloader=None, current_epoch=None, current_step=None,
                )
            parent.assert_called_once()
        with patch.object(
            base, "on_step_start",
            side_effect=RuntimeError("real sd3 step-start invoked"),
        ) as parent:
            with self.assertRaisesRegex(RuntimeError, "real sd3 step-start invoked"):
                trainer.on_step_start(
                    None,None,None,None,None,{},torch.bfloat16,True,
                )
            parent.assert_called_once()

    def test_fingerprint_rejects_meta_and_sparse_tensors(self):
        api = observer()
        with self.assertRaises(ValueError):
            api.fingerprint_tensor(torch.empty(3, device="meta"))
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0],[0]]), torch.tensor([1.0]), (2,2),
        )
        with self.assertRaises(ValueError):
            api.fingerprint_tensor(sparse)


    def test_r1_real_adamw_one_child_sync_each_batch(self):
        torch.manual_seed(123456)
        net = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        adamw = torch.optim.AdamW(net.parameters(), lr=0.01)
        facade = CompositeOptimizer(
            SimpleNamespace(topology_fingerprint="r1-c3-cpu"),
            (OptimizerRuntimeEntry(
                profile_name="baseline", optimizer_type="AdamW",
                topology_fingerprint="r1-c3-adamw", optimizer=adamw,
            ),),
        )
        child_sched = torch.optim.lr_scheduler.LambdaLR(adamw, lambda _: 1.0)
        sched = CompositeLRScheduler(
            facade, (SchedulerRuntimeEntry(
                profile_name="baseline", optimizer_type="AdamW",
                topology_fingerprint="r1-c3-adamw",
                mode="external", optimizer=adamw,
                scheduler=child_sched,
                scheduler_type="torch.optim.lr_scheduler.LambdaLR",
            ),),
        )
        acc = Accelerator(
            cpu=True, mixed_precision="no",
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=1, adjust_scheduler=False,
            ),
        )
        net, prepared_opt, prepared_sch = acc.prepare(net, facade, sched)
        probe = observer().PhysicalStepProbe(
            accelerator=acc, composite_optimizer=facade,
            composite_scheduler=sched,
            parameter_names={"network.adamw.weight":net.weight},
            case_id="r1-adamw-accum1", run_id="run-c3r1-0001",
            phase_id="fresh-c3r1-0001", phase="fresh",
            source_commit="a"*40,
        )
        original_adamw_step = adamw.step  # LambdaLR already installed its own instance wrapper
        original_composite_step = facade.step
        probe.install()
        try:
            with acc.accumulate(net):
                probe.begin_batch("dts_c2b_sample_0001")
                x = torch.tensor([[1.0,-2.0,3.0,-4.0]], dtype=torch.bfloat16)
                acc.backward(net(x).square().float().sum())
                self.assertTrue(acc.sync_gradients)
                acc.clip_grad_norm_(net.parameters(),1.0)
                prepared_opt.step()
                prepared_sch.step()
                prepared_opt.zero_grad(set_to_none=True)
            probe.assert_step_complete()
            kinds = [e["event_type"] for e in probe.events]
            self.assertEqual(kinds, [
                "batch_started","backward_completed","physical_child_stepped",
                "physical_scheduler_stepped",
            ])
            self.assertEqual(probe.sync_sequence,(True,))
            self.assertEqual(int(adamw.state[net.weight]["step"].item()),1)
        finally:
            probe.close()
        self.assertFalse("backward" in vars(acc))
        self.assertIs(adamw.step, original_adamw_step)
        self.assertIs(facade.step.__func__, original_composite_step.__func__)
        self.assertIs(facade.step.__self__, facade)
        self.assertNotIn("step", vars(facade), "C3 leaked instance-level CompositeOptimizer wrapper")

    def test_probe_rejects_unknown_optimizer_parameter(self):
        acc, model, optim, sched, po, ps = build_two_child_cpu()
        with self.assertRaisesRegex(ValueError, "unregistered|misses|unknown"):
            observer().PhysicalStepProbe(
                accelerator=acc, composite_optimizer=optim,
                composite_scheduler=sched,
                parameter_names={"network.left.weight":model["left"].weight},
                case_id=R2, run_id="run-c3cpu-0002",
                phase_id="fresh-c3cpu-0002",
                phase="fresh", source_commit="a"*40,
            )


    def test_post_dataset_real_sd3_effective_args_are_checked_without_mutation(self):
        """Original SD3 assert_extra_args runs, then Worker read-only scope gate."""
        from tools import sd3_full_bf16_evidence_worker as worker
        from tests.test_sd3_full_bf16_gpu_evidence import _generate, R1, R2
        from tools import sd3_full_bf16_evidence_support as support

        cls = observer().ObservedSd3NetworkTrainer
        dataset = SimpleNamespace(
            is_text_encoder_output_cacheable=lambda: True,
            verify_bucket_reso_steps=lambda steps: None,
            get_resolutions=lambda: [(512, 512)],
        )
        for case_id in (R1, R2):
            with self.subTest(case=case_id), tempfile.TemporaryDirectory() as td:
                record = _generate(Path(td), case_id)
                manifest = support._read_json(Path(record["manifest"]))
                args = worker._production_namespace(SimpleNamespace(
                    case_id=case_id, trainer_argv=manifest["fresh_command"],
                ))
                trainer = cls(
                    invocation=SimpleNamespace(case_id=case_id),
                    cuda_environment={}, qualification_lease={},
                )
                before = vars(args).copy()
                result = trainer.assert_extra_args(args, dataset, None)
                self.assertIsNone(result)
                self.assertEqual(vars(args), before)
                self.assertEqual(trainer.resolutions, [(512, 512)])
                args.max_train_steps = 3
                with self.assertRaisesRegex(ValueError, "max_train_steps"):
                    trainer.assert_extra_args(args, dataset, None)
                args.max_train_steps = 2
                args.resolution = (768, 768)
                with self.assertRaisesRegex(ValueError, "resolution"):
                    trainer.assert_extra_args(args, dataset, None)

    def test_post_dataset_parent_sd3_hook_runs_first_even_on_error(self):
        cls = observer().ObservedSd3NetworkTrainer
        import sd3_train_network
        trainer = cls(
            invocation=SimpleNamespace(case_id=R2),
            cuda_environment={}, qualification_lease={},
        )
        with patch.object(
            sd3_train_network.Sd3NetworkTrainer,
            "assert_extra_args",
            side_effect=RuntimeError("real SD3 parent invoked first"),
        ) as parent:
            with self.assertRaisesRegex(RuntimeError, "real SD3 parent invoked first"):
                trainer.assert_extra_args(SimpleNamespace(), object(), None)
            parent.assert_called_once()

    def test_real_runtime_spec_group_and_route_ownership_binding(self):
        from mikazuki.parameter_policy_runtime import (
            RuntimeParameterSpec, OptimizerGroupSpec,
            OptimizerInstanceSpec, ParameterPolicyRuntimeSpec,
        )
        from mikazuki.parameter_routing import RoutingAssignment
        acc, model, optimizer, scheduler, _, _ = build_two_child_cpu()
        param_by_profile = {
            "muon_primary": model["left"].weight,
            "adamw_fallback": model["right"].weight,
        }
        names = {
            "muon_primary": "network.left.weight",
            "adamw_fallback": "network.right.weight",
        }
        component = {
            "muon_primary": "mmdit.attention.adapter",
            "adamw_fallback": "mmdit.modulation_norm.adapter",
        }
        route_kind = {"muon_primary": "primary", "adamw_fallback": "fallback"}
        specs, assignments = [], {}
        for child in optimizer.entries:
            profile = child.profile_name
            param = param_by_profile[profile]
            row = RuntimeParameterSpec(
                parameter=param, parameter_id=id(param),
                canonical_name=names[profile], shape=tuple(param.shape),
                numel=param.numel(), component_id=component[profile],
                route_kind=route_kind[profile], parameter_class="matrix_weight",
            )
            specs.append(OptimizerInstanceSpec(
                profile_name=profile, optimizer_type=child.optimizer_type,
                optimizer_arguments_json="{}", supports_group_lr=True,
                uses_external_scheduler=True, lr_semantics="fixed",
                groups=(OptimizerGroupSpec(
                    learning_rate=child.optimizer.param_groups[0]["lr"],
                    parameters=(row,),
                ),),
                topology_fingerprint=child.topology_fingerprint,
            ))
            assignments[names[profile]] = RoutingAssignment(
                parameter=param, parameter_id=id(param),
                canonical_name=names[profile], parameter_class="matrix_weight",
                component_id=component[profile], route_kind=route_kind[profile],
                optimizer_profile=profile, learning_rate=1e-4,
                reason="real CPU identity fixture",
            )
        session = SimpleNamespace(
            optimizer=optimizer,
            runtime_spec=ParameterPolicyRuntimeSpec(
                version=1, optimizers=tuple(specs),
                topology_fingerprint="c3-physical-cpu-v1",
            ),
        )
        by_id = {id(p): names[profile] for profile,p in param_by_profile.items()}
        auditor = observer().audit_physical_optimizer_groups
        children = auditor(session=session, canonical_names_by_id=by_id,
                           routing_assignments=assignments)
        self.assertEqual(children, {
            "muon_primary": ["network.left.weight"],
            "adamw_fallback": ["network.right.weight"],
        })
        bad_route = dict(assignments)
        bad_route["network.right.weight"] = replace(
            assignments["network.right.weight"],
            component_id="clip_l.adapter",
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            auditor(session=session, canonical_names_by_id=by_id,
                    routing_assignments=bad_route)
        adamw = optimizer.entries[1].optimizer
        saved_lr = adamw.param_groups[0]["lr"]
        try:
            adamw.param_groups[0]["lr"] = 0.025
            with self.assertRaisesRegex(ValueError, "learning-rate"):
                auditor(session=session, canonical_names_by_id=by_id,
                        routing_assignments=assignments)
        finally:
            adamw.param_groups[0]["lr"] = saved_lr
        alien = torch.nn.Parameter(torch.ones(4, 4))
        adamw.param_groups[0]["params"].append(alien)
        try:
            with self.assertRaisesRegex(ValueError, "order|Parameter"):
                auditor(session=session, canonical_names_by_id=by_id,
                        routing_assignments=assignments)
        finally:
            adamw.param_groups[0]["params"].pop()


if __name__ == "__main__":
    unittest.main()
