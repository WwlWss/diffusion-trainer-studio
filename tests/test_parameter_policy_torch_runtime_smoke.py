from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import tempfile
import unittest


_REQUIRED_MODULES = ("torch", "accelerate", "schedulefree", "pytorch_optimizer")
_RUNTIME_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in _REQUIRED_MODULES)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator
    import pytorch_optimizer

    from mikazuki.parameter_policy_runtime import compile_parameter_policy_runtime_spec
    from mikazuki.parameter_policy_torch import (
        CompositeLRScheduler,
        CompositeOptimizer,
        ParameterPolicyTorchRuntimeError,
        build_parameter_policy_optimizer,
        build_parameter_policy_scheduler,
    )
    from mikazuki.parameter_routing import (
        ParameterStat,
        RoutingAssignment,
        RoutingPlan,
        RoutingStats,
        build_parameter_routing_plan,
        scan_parameter_roots,
    )
    from tools.parameter_policy_execution_gpu_runtime import state_fingerprint


def _stats():
    zero = ParameterStat(0, 0)
    return RoutingStats(
        total=zero,
        assigned=zero,
        unassigned=zero,
        conflicts=zero,
        unroutable=zero,
        by_component={},
        by_route={},
        by_parameter_class={},
    )


def _assignment(
    parameter,
    *,
    name,
    component,
    profile,
    lr,
    route="primary",
    parameter_class="matrix_weight",
):
    return RoutingAssignment(
        parameter=parameter,
        parameter_id=id(parameter),
        canonical_name=name,
        parameter_class=parameter_class,
        component_id=component,
        route_kind=route,
        optimizer_profile=profile,
        learning_rate=lr,
        reason="runtime-smoke",
    )


def _train(profile, lr):
    return {
        "train": True,
        "optimizer_profile": profile,
        "learning_rate": lr,
    }


def _compile(assignments, *, profiles, components):
    policy = {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": components,
    }
    plan = RoutingPlan(tuple(assignments), (), _stats())
    return compile_parameter_policy_runtime_spec(policy, plan)


def _constant_scheduler(_spec, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )


def _make_split_model(seed=1234):
    torch.manual_seed(seed)

    class SplitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adam = torch.nn.Linear(4, 4, bias=False)
            self.schedulefree = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.schedulefree(torch.tanh(self.adam(x)))

    return SplitModel()


def _compile_mixed_model(model):
    adam_lr = 1e-2
    sf_lr = 5e-3
    return _compile(
        (
            _assignment(
                model.adam.weight,
                name="model.adam.weight",
                component="adam",
                profile="adam",
                lr=adam_lr,
            ),
            _assignment(
                model.schedulefree.weight,
                name="model.schedulefree.weight",
                component="schedulefree",
                profile="schedulefree",
                lr=sf_lr,
            ),
        ),
        profiles={
            "adam": {
                "type": "AdamW",
                "args": {"weight_decay": 0.0},
            },
            "schedulefree": {
                "type": "AdamWScheduleFree",
                "args": {"weight_decay": 0.0},
            },
        },
        components={
            "adam": _train("adam", adam_lr),
            "schedulefree": _train("schedulefree", sf_lr),
        },
    )


def _build_mixed_runtime(model):
    spec = _compile_mixed_model(model)
    optimizer = build_parameter_policy_optimizer(spec)
    scheduler = build_parameter_policy_scheduler(optimizer, _constant_scheduler)
    return spec, optimizer, scheduler


def _direct_step(model, optimizer, scheduler, x):
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


@unittest.skipUnless(
    _RUNTIME_DEPS_AVAILABLE,
    "Step 5D runtime dependencies are installed only in the dedicated smoke job.",
)
class ParameterPolicyTorchRuntimeSmokeTests(unittest.TestCase):
    def test_runtime_job_uses_exact_pinned_versions(self):
        self.assertTrue(torch.__version__.split("+", 1)[0].startswith("2.7.0"))
        self.assertEqual(importlib.metadata.version("accelerate"), "1.6.0")
        self.assertEqual(importlib.metadata.version("schedulefree"), "1.4")
        self.assertEqual(importlib.metadata.version("pytorch-optimizer"), "3.10.0")

    def test_all_supported_schedulefree_optimizers_run_without_external_scheduler(self):
        cases = (
            ("RAdamScheduleFree", {"weight_decay": 0.0}),
            ("AdamWScheduleFree", {"weight_decay": 0.0}),
            ("SGDScheduleFree", {"momentum": 0.9, "weight_decay": 0.0}),
        )
        for optimizer_type, optimizer_args in cases:
            with self.subTest(optimizer_type=optimizer_type):
                parameter = torch.nn.Parameter(torch.full((4, 4), 0.25))
                lr = 1e-2
                spec = _compile(
                    (
                        _assignment(
                            parameter,
                            name="model.weight",
                            component="main",
                            profile="schedulefree",
                            lr=lr,
                        ),
                    ),
                    profiles={
                        "schedulefree": {
                            "type": optimizer_type,
                            "args": optimizer_args,
                        }
                    },
                    components={"main": _train("schedulefree", lr)},
                )

                optimizer = build_parameter_policy_optimizer(spec)
                scheduler = build_parameter_policy_scheduler(optimizer, None)

                self.assertIsInstance(optimizer, CompositeOptimizer)
                self.assertIsInstance(scheduler, CompositeLRScheduler)
                self.assertEqual(optimizer.lifecycle_mode, "train")
                self.assertEqual(len(scheduler.entries), 1)
                self.assertEqual(scheduler.entries[0].mode, "optimizer_managed")
                self.assertIsNone(scheduler.entries[0].scheduler)

                before = parameter.detach().clone()
                # RAdamScheduleFree defaults silent_sgd_phase=True, so its early
                # unrectified RAdam phase intentionally advances optimizer
                # statistics without changing model weights. Run enough steps
                # to validate real update behavior for every supported
                # ScheduleFree implementation instead of assuming step 1 must
                # move parameters.
                for _ in range(8):
                    parameter.grad = torch.ones_like(parameter)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                self.assertEqual(
                    optimizer.entries[0].optimizer.param_groups[0]["k"],
                    8,
                )
                self.assertFalse(torch.equal(before, parameter.detach()))
                self.assertEqual(
                    scheduler.get_last_lr(),
                    [optimizer.param_groups[0]["lr"]],
                )
                state = scheduler.state_dict()
                self.assertNotIn(
                    "state_dict",
                    state["children"]["schedulefree"],
                )

    def test_mixed_runtime_steps_and_scheduler_groups_share_live_children(self):
        model = _make_split_model()
        _, optimizer, scheduler = _build_mixed_runtime(model)

        self.assertEqual(
            [entry.profile_name for entry in optimizer.entries],
            ["adam", "schedulefree"],
        )
        self.assertEqual(
            [entry.mode for entry in scheduler.entries],
            ["external", "optimizer_managed"],
        )
        self.assertIs(
            scheduler.entries[0].scheduler.optimizer,
            optimizer.entries[0].optimizer,
        )

        adam_group = optimizer.entries[0].optimizer.param_groups[0]
        sf_group = optimizer.entries[1].optimizer.param_groups[0]
        self.assertIs(optimizer.param_groups[0], adam_group)
        self.assertIs(optimizer.param_groups[1], sf_group)

        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8.0
        before_adam = model.adam.weight.detach().clone()
        before_sf = model.schedulefree.weight.detach().clone()
        initial_adam_lr = adam_group["lr"]

        _direct_step(model, optimizer, scheduler, x)

        self.assertFalse(torch.equal(before_adam, model.adam.weight.detach()))
        self.assertFalse(torch.equal(before_sf, model.schedulefree.weight.detach()))
        self.assertAlmostEqual(adam_group["lr"], initial_adam_lr * 0.5)
        self.assertEqual(
            scheduler.entries[0].scheduler._step_count,
            scheduler._step_count,
        )

    def test_schedulefree_eval_checkpoint_resumes_in_train_mode_and_matches_reference(self):
        model = _make_split_model()
        _, optimizer, scheduler = _build_mixed_runtime(model)
        x1 = torch.tensor(
            [[0.2, -0.4, 0.1, 0.8], [0.3, 0.5, -0.2, -0.7]],
            dtype=torch.float32,
        )
        x2 = torch.tensor(
            [[-0.1, 0.6, 0.9, -0.2], [0.4, -0.3, 0.7, 0.2]],
            dtype=torch.float32,
        )

        _direct_step(model, optimizer, scheduler, x1)

        optimizer.eval()
        self.assertEqual(optimizer.lifecycle_mode, "eval")
        self.assertFalse(optimizer.entries[1].optimizer.param_groups[0]["train_mode"])

        checkpoint_model = copy.deepcopy(model.state_dict())
        checkpoint_optimizer = copy.deepcopy(optimizer.state_dict())
        checkpoint_scheduler = copy.deepcopy(scheduler.state_dict())

        optimizer.train()
        _direct_step(model, optimizer, scheduler, x2)
        reference_model = copy.deepcopy(model.state_dict())
        reference_lr = scheduler.get_last_lr()

        resumed = _make_split_model(seed=9999)
        resumed.load_state_dict(checkpoint_model)
        _, resumed_optimizer, resumed_scheduler = _build_mixed_runtime(resumed)

        self.assertEqual(resumed_optimizer.lifecycle_mode, "train")
        resumed_optimizer.load_state_dict(checkpoint_optimizer)
        resumed_scheduler.load_state_dict(checkpoint_scheduler)

        self.assertEqual(resumed_optimizer.lifecycle_mode, "train")
        self.assertTrue(
            resumed_optimizer.entries[1].optimizer.param_groups[0]["train_mode"]
        )

        _direct_step(resumed, resumed_optimizer, resumed_scheduler, x2)

        for name, tensor in reference_model.items():
            self.assertTrue(
                torch.allclose(tensor, resumed.state_dict()[name], atol=1e-6, rtol=1e-5),
                msg=f"resume mismatch for {name}",
            )
        self.assertEqual(reference_lr, resumed_scheduler.get_last_lr())

    def test_optimizer_and_scheduler_metadata_mismatch_fail_closed(self):
        model = _make_split_model()
        _, optimizer, scheduler = _build_mixed_runtime(model)
        x = torch.ones(2, 4)
        _direct_step(model, optimizer, scheduler, x)

        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())

        bad_topology = copy.deepcopy(optimizer_state)
        bad_topology["topology_fingerprint"] = "wrong"
        with self.assertRaisesRegex(
            ParameterPolicyTorchRuntimeError,
            "topology fingerprint",
        ):
            optimizer.load_state_dict(bad_topology)

        bad_child = copy.deepcopy(optimizer_state)
        bad_child["children"]["adam"]["optimizer_type"] = "Muon"
        with self.assertRaisesRegex(
            ParameterPolicyTorchRuntimeError,
            "optimizer type",
        ):
            optimizer.load_state_dict(bad_child)

        bad_scheduler = copy.deepcopy(scheduler_state)
        bad_scheduler["children"]["adam"]["scheduler_type"] = "wrong.Scheduler"
        with self.assertRaisesRegex(
            ParameterPolicyTorchRuntimeError,
            "scheduler type",
        ):
            scheduler.load_state_dict(bad_scheduler)

    def test_real_pinned_muon_uses_only_muon_groups_and_steps_on_cpu(self):
        parameter = torch.nn.Parameter(
            torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64.0
        )
        lr = 2e-2
        spec = _compile(
            (
                _assignment(
                    parameter,
                    name="model.hidden.weight",
                    component="hidden",
                    profile="muon",
                    lr=lr,
                ),
            ),
            profiles={
                "muon": {
                    "type": "Muon",
                    "args": {"ns_steps": 2, "weight_decay": 0.0},
                }
            },
            components={"hidden": _train("muon", lr)},
        )

        optimizer = build_parameter_policy_optimizer(spec)
        child = optimizer.entries[0].optimizer

        self.assertIsInstance(child, pytorch_optimizer.Muon)
        self.assertTrue(all(group["use_muon"] is True for group in child.param_groups))

        before = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertIn("momentum_buffer", child.state[parameter])
        self.assertNotIn("exp_avg", child.state[parameter])

    def test_muon_adamw_explicit_fallback_uses_real_production_routing(self):
        model = torch.nn.Module()
        model.double_blocks = torch.nn.ModuleList(
            [
                torch.nn.Linear(8, 12, bias=True),
                torch.nn.Linear(12, 8, bias=True),
            ]
        )
        policy = {
            "version": 1,
            "optimizer_profiles": {
                "muon": {
                    "type": "Muon",
                    "args": {
                        "ns_steps": 2,
                        "ns_coeffs": "original",
                        "weight_decay": 0.0,
                    },
                },
                "adamw_fallback": {
                    "type": "AdamW",
                    "args": {},
                },
            },
            "components": {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": 2e-2,
                    "fallback_optimizer_profile": "adamw_fallback",
                    "fallback_learning_rate": 1e-2,
                }
            },
        }
        descriptors = scan_parameter_roots({"transformer": model})
        plan = build_parameter_routing_plan(
            policy,
            train_type="flux-finetune",
            effective_config={},
            descriptors=descriptors,
        )
        self.assertTrue(plan.is_valid)
        rows = {
            assignment.canonical_name: (
                assignment.route_kind,
                assignment.optimizer_profile,
                assignment.parameter_class,
                tuple(assignment.parameter.shape),
            )
            for assignment in plan.assignments
            if assignment.route_kind in {"primary", "fallback"}
        }
        self.assertEqual(
            rows,
            {
                "transformer.double_blocks.0.weight": (
                    "primary",
                    "muon",
                    "matrix_weight",
                    (12, 8),
                ),
                "transformer.double_blocks.0.bias": (
                    "fallback",
                    "adamw_fallback",
                    "bias",
                    (12,),
                ),
                "transformer.double_blocks.1.weight": (
                    "primary",
                    "muon",
                    "matrix_weight",
                    (8, 12),
                ),
                "transformer.double_blocks.1.bias": (
                    "fallback",
                    "adamw_fallback",
                    "bias",
                    (8,),
                ),
            },
        )

    def test_muon_adamw_explicit_fallback_accumulation_on_cpu(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 12, bias=True),
            torch.nn.Linear(12, 8, bias=True),
        )
        muon_lr = 2e-2
        adamw_lr = 1e-2
        assignments = (
            _assignment(
                model[0].weight,
                name="model.hidden.expand.weight",
                component="hidden",
                profile="muon",
                lr=muon_lr,
            ),
            _assignment(
                model[0].bias,
                name="model.hidden.expand.bias",
                component="hidden",
                profile="adamw_fallback",
                lr=adamw_lr,
                route="fallback",
                parameter_class="bias",
            ),
            _assignment(
                model[1].weight,
                name="model.hidden.contract.weight",
                component="hidden",
                profile="muon",
                lr=muon_lr,
            ),
            _assignment(
                model[1].bias,
                name="model.hidden.contract.bias",
                component="hidden",
                profile="adamw_fallback",
                lr=adamw_lr,
                route="fallback",
                parameter_class="bias",
            ),
        )
        spec = _compile(
            assignments,
            profiles={
                "muon": {
                    "type": "Muon",
                    "args": {
                        "ns_steps": 2,
                        "ns_coeffs": "original",
                        "weight_decay": 0.0,
                    },
                },
                "adamw_fallback": {
                    "type": "AdamW",
                    "args": {},
                },
            },
            components={
                "hidden": {
                    "train": True,
                    "optimizer_profile": "muon",
                    "learning_rate": muon_lr,
                    "fallback_optimizer_profile": "adamw_fallback",
                    "fallback_learning_rate": adamw_lr,
                }
            },
        )
        optimizer = build_parameter_policy_optimizer(spec)
        scheduler = build_parameter_policy_scheduler(optimizer, _constant_scheduler)
        children = {entry.profile_name: entry.optimizer for entry in optimizer.entries}
        self.assertEqual(set(children), {"muon", "adamw_fallback"})

        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
        raw_scheduler = prepared_scheduler.scheduler
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        initial_scheduler_step = raw_scheduler._step_count

        sync_pattern = []
        for microstep in range(2):
            with accelerator.accumulate(model):
                value = torch.eye(8, dtype=model[0].weight.dtype) * (
                    0.5 + 0.125 * microstep
                )
                loss = model(value).square().mean()
                accelerator.backward(loss)
                sync_pattern.append(bool(accelerator.sync_gradients))
                prepared_optimizer.step()
                prepared_scheduler.step()
                prepared_optimizer.zero_grad(set_to_none=True)

            if microstep == 0:
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.equal(before[name], parameter.detach()), msg=name)
                self.assertEqual(len(children["muon"].state), 0)
                self.assertEqual(len(children["adamw_fallback"].state), 0)
                self.assertEqual(raw_scheduler._step_count, initial_scheduler_step)

        self.assertEqual(sync_pattern, [False, True])
        for name, parameter in model.named_parameters():
            self.assertFalse(torch.equal(before[name], parameter.detach()), msg=name)

        muon = children["muon"]
        adamw = children["adamw_fallback"]
        self.assertEqual(len(muon.state), 2)
        self.assertEqual(len(adamw.state), 2)
        self.assertTrue(all(group["use_muon"] is True for group in muon.param_groups))
        self.assertTrue(all(group.get("step") == 1 for group in muon.param_groups))
        for state in muon.state.values():
            self.assertIn("momentum_buffer", state)
            self.assertNotIn("exp_avg", state)
            self.assertNotIn("exp_avg_sq", state)
        for state in adamw.state.values():
            self.assertIn("step", state)
            self.assertIn("exp_avg", state)
            self.assertIn("exp_avg_sq", state)
            step = state["step"]
            if isinstance(step, torch.Tensor):
                step = step.item()
            self.assertEqual(int(step), 1)

        self.assertEqual(raw_scheduler._step_count, initial_scheduler_step + 1)
        self.assertEqual(
            set(raw_scheduler.get_last_lr_by_profile()),
            {"muon", "adamw_fallback"},
        )
        accelerator.end_training()

    def test_muon_adamw_explicit_fallback_state_round_trip(self):
        def build():
            weights = (
                torch.nn.Parameter(
                    torch.arange(96, dtype=torch.float32).reshape(12, 8) / 96.0
                ),
                torch.nn.Parameter(
                    torch.arange(96, dtype=torch.float32).reshape(8, 12) / 128.0
                ),
            )
            biases = (
                torch.nn.Parameter(torch.zeros(12, dtype=torch.float32)),
                torch.nn.Parameter(torch.zeros(8, dtype=torch.float32)),
            )
            muon_lr = 2e-2
            adamw_lr = 1e-2
            spec = _compile(
                (
                    _assignment(
                        weights[0],
                        name="model.hidden.expand.weight",
                        component="hidden",
                        profile="muon",
                        lr=muon_lr,
                    ),
                    _assignment(
                        biases[0],
                        name="model.hidden.expand.bias",
                        component="hidden",
                        profile="adamw_fallback",
                        lr=adamw_lr,
                        route="fallback",
                        parameter_class="bias",
                    ),
                    _assignment(
                        weights[1],
                        name="model.hidden.contract.weight",
                        component="hidden",
                        profile="muon",
                        lr=muon_lr,
                    ),
                    _assignment(
                        biases[1],
                        name="model.hidden.contract.bias",
                        component="hidden",
                        profile="adamw_fallback",
                        lr=adamw_lr,
                        route="fallback",
                        parameter_class="bias",
                    ),
                ),
                profiles={
                    "muon": {
                        "type": "Muon",
                        "args": {
                            "ns_steps": 2,
                            "ns_coeffs": "original",
                            "weight_decay": 0.0,
                        },
                    },
                    "adamw_fallback": {"type": "AdamW", "args": {}},
                },
                components={
                    "hidden": {
                        "train": True,
                        "optimizer_profile": "muon",
                        "learning_rate": muon_lr,
                        "fallback_optimizer_profile": "adamw_fallback",
                        "fallback_learning_rate": adamw_lr,
                    }
                },
            )
            optimizer = build_parameter_policy_optimizer(spec)
            scheduler = build_parameter_policy_scheduler(optimizer, _constant_scheduler)
            return weights, biases, optimizer, scheduler

        weights, biases, optimizer, scheduler = build()
        for parameter in (*weights, *biases):
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scheduler = copy.deepcopy(scheduler.state_dict())

        fresh_weights, fresh_biases, fresh_optimizer, fresh_scheduler = build()
        self.assertTrue(
            all(len(entry.optimizer.state) == 0 for entry in fresh_optimizer.entries)
        )
        fresh_optimizer.load_state_dict(saved_optimizer)
        fresh_scheduler.load_state_dict(saved_scheduler)

        self.assertEqual(
            state_fingerprint(fresh_optimizer.state_dict()),
            state_fingerprint(saved_optimizer),
        )
        self.assertEqual(
            state_fingerprint(fresh_scheduler.state_dict()),
            state_fingerprint(saved_scheduler),
        )

        fresh_children = {
            entry.profile_name: entry.optimizer for entry in fresh_optimizer.entries
        }
        self.assertEqual(len(fresh_children["muon"].state), 2)
        self.assertEqual(len(fresh_children["adamw_fallback"].state), 2)
        self.assertEqual(
            fresh_scheduler.state_dict()["step_count"],
            saved_scheduler["step_count"],
        )
        self.assertEqual(
            set(fresh_scheduler.get_last_lr_by_profile()),
            {"muon", "adamw_fallback"},
        )

        for parameter in (*fresh_weights, *fresh_biases):
            parameter.grad = torch.ones_like(parameter)
        fresh_optimizer.step()
        fresh_scheduler.step()

        self.assertTrue(
            all(group.get("step") == 2 for group in fresh_children["muon"].param_groups)
        )
        adamw_steps = []
        for state in fresh_children["adamw_fallback"].state.values():
            step = state["step"]
            if isinstance(step, torch.Tensor):
                step = step.item()
            adamw_steps.append(int(step))
        self.assertEqual(sorted(adamw_steps), [2, 2])

    def test_muon_accumulation_suppresses_unsynced_step_on_cpu(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 12, bias=False),
            torch.nn.Linear(12, 8, bias=False),
        )
        lr = 2e-2
        spec = _compile(
            (
                _assignment(
                    model[0].weight,
                    name="model.hidden.expand.weight",
                    component="hidden",
                    profile="muon",
                    lr=lr,
                ),
                _assignment(
                    model[1].weight,
                    name="model.hidden.contract.weight",
                    component="hidden",
                    profile="muon",
                    lr=lr,
                ),
            ),
            profiles={
                "muon": {
                    "type": "Muon",
                    "args": {
                        "ns_steps": 2,
                        "ns_coeffs": "original",
                        "weight_decay": 0.0,
                    },
                }
            },
            components={"hidden": _train("muon", lr)},
        )
        optimizer = build_parameter_policy_optimizer(spec)
        scheduler = build_parameter_policy_scheduler(optimizer, _constant_scheduler)
        child = optimizer.entries[0].optimizer

        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
        raw_scheduler = prepared_scheduler.scheduler
        initial_weights = [
            model[0].weight.detach().clone(),
            model[1].weight.detach().clone(),
        ]
        initial_scheduler_step = raw_scheduler._step_count
        sync_pattern = []

        for index in range(2):
            with accelerator.accumulate(model):
                value = torch.eye(8, dtype=model[0].weight.dtype) * (
                    0.5 + 0.125 * index
                )
                loss = model(value).square().mean()
                accelerator.backward(loss)
                sync_pattern.append(bool(accelerator.sync_gradients))
                prepared_optimizer.step()
                prepared_scheduler.step()
                prepared_optimizer.zero_grad(set_to_none=True)

            if index == 0:
                self.assertTrue(
                    torch.equal(model[0].weight.detach(), initial_weights[0])
                )
                self.assertTrue(
                    torch.equal(model[1].weight.detach(), initial_weights[1])
                )
                self.assertEqual(len(child.state), 0)
                self.assertTrue(
                    all(int(group.get("step", 0)) == 0 for group in child.param_groups)
                )
                self.assertEqual(raw_scheduler._step_count, initial_scheduler_step)

        self.assertEqual(sync_pattern, [False, True])
        self.assertFalse(torch.equal(model[0].weight.detach(), initial_weights[0]))
        self.assertFalse(torch.equal(model[1].weight.detach(), initial_weights[1]))
        self.assertTrue(all(group["use_muon"] is True for group in child.param_groups))
        self.assertTrue(all(group.get("step") == 1 for group in child.param_groups))
        self.assertEqual(raw_scheduler._step_count, initial_scheduler_step + 1)
        self.assertEqual(len(child.state), 2)
        for state in child.state.values():
            self.assertIn("momentum_buffer", state)
            self.assertNotIn("exp_avg", state)
            self.assertNotIn("exp_avg_sq", state)
        accelerator.end_training()

    def test_muon_state_round_trip_restores_group_step_and_coefficients(self):
        def build():
            expand = torch.nn.Parameter(
                torch.arange(96, dtype=torch.float32).reshape(12, 8) / 96.0
            )
            contract = torch.nn.Parameter(
                torch.arange(96, dtype=torch.float32).reshape(8, 12) / 128.0
            )
            lr = 2e-2
            spec = _compile(
                (
                    _assignment(
                        expand,
                        name="model.hidden.expand.weight",
                        component="hidden",
                        profile="muon",
                        lr=lr,
                    ),
                    _assignment(
                        contract,
                        name="model.hidden.contract.weight",
                        component="hidden",
                        profile="muon",
                        lr=lr,
                    ),
                ),
                profiles={
                    "muon": {
                        "type": "Muon",
                        "args": {
                            "ns_steps": 2,
                            "ns_coeffs": "original",
                            "weight_decay": 0.0,
                        },
                    }
                },
                components={"hidden": _train("muon", lr)},
            )
            optimizer = build_parameter_policy_optimizer(spec)
            return (
                (expand, contract),
                optimizer,
                optimizer.entries[0].optimizer,
            )

        parameters, optimizer, child = build()
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        saved_coeffs = copy.deepcopy(child.param_groups[0]["ns_coeffs"])
        saved_momenta = [
            child.state[parameter]["momentum_buffer"].detach().clone()
            for parameter in parameters
        ]

        fresh_parameters, fresh_optimizer, fresh_child = build()
        self.assertEqual(len(fresh_child.state), 0)
        self.assertEqual(int(fresh_child.param_groups[0].get("step", 0)), 0)
        fresh_optimizer.load_state_dict(saved)

        self.assertEqual(fresh_child.param_groups[0]["step"], 1)
        self.assertEqual(fresh_child.param_groups[0]["ns_coeffs"], saved_coeffs)
        self.assertEqual(len(fresh_child.state), 2)
        for index, fresh_parameter in enumerate(fresh_parameters):
            self.assertIn("momentum_buffer", fresh_child.state[fresh_parameter])
            self.assertNotIn("exp_avg", fresh_child.state[fresh_parameter])
            self.assertNotIn("exp_avg_sq", fresh_child.state[fresh_parameter])
            self.assertTrue(
                torch.equal(
                    fresh_child.state[fresh_parameter]["momentum_buffer"],
                    saved_momenta[index],
                )
            )

        for fresh_parameter in fresh_parameters:
            fresh_parameter.grad = torch.ones_like(fresh_parameter)
        fresh_optimizer.step()
        self.assertEqual(fresh_child.param_groups[0]["step"], 2)

    def test_accelerate_prepare_accumulation_and_state_round_trip(self):
        model = _make_split_model()
        _, optimizer, scheduler = _build_mixed_runtime(model)

        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )

        self.assertEqual(len(accelerator._optimizers), 1)
        self.assertEqual(len(accelerator._schedulers), 1)
        self.assertIs(prepared_optimizer.optimizer, optimizer)
        self.assertIs(prepared_scheduler.scheduler, scheduler)

        raw_scheduler = prepared_scheduler.scheduler
        external_scheduler = raw_scheduler.entries[0].scheduler
        initial_step_count = raw_scheduler._step_count
        initial_last_epoch = raw_scheduler.last_epoch
        initial_weights = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        inputs = (
            torch.tensor(
                [[0.1, 0.2, -0.3, 0.4], [0.5, -0.1, 0.2, -0.4]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[-0.2, 0.6, 0.1, 0.3], [0.7, 0.2, -0.5, 0.1]],
                dtype=torch.float32,
            ),
        )

        sync_pattern = []
        after_first = None
        for index, x in enumerate(inputs):
            with accelerator.accumulate(model):
                loss = model(x).square().mean()
                accelerator.backward(loss)
                sync_pattern.append(accelerator.sync_gradients)
                prepared_optimizer.step()
                prepared_scheduler.step()
                prepared_optimizer.zero_grad(set_to_none=True)
            if index == 0:
                after_first = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }

        self.assertEqual(sync_pattern, [False, True])
        self.assertIsNotNone(after_first)
        for name, tensor in initial_weights.items():
            self.assertTrue(torch.equal(tensor, after_first[name]), msg=name)
        self.assertTrue(
            any(
                not torch.equal(tensor, model.state_dict()[name])
                for name, tensor in initial_weights.items()
            )
        )

        # Accelerate 1.6.0's GradientAccumulationPlugin.to_kwargs() omits
        # default-valued adjust_scheduler=True, while GradientState falls back
        # to False when that key is absent. Lock the actual pinned behavior:
        # only the synchronized optimizer update advances the scheduler.
        self.assertFalse(prepared_scheduler.gradient_state.adjust_scheduler)
        self.assertEqual(raw_scheduler._step_count, initial_step_count + 1)
        self.assertEqual(external_scheduler._step_count, raw_scheduler._step_count)
        self.assertEqual(raw_scheduler.last_epoch, initial_last_epoch + 1)

        # The Composite property still supports Accelerate's direct counter
        # adjustment path if a caller/runtime explicitly uses it.
        before_direct_adjust = external_scheduler._step_count
        raw_scheduler._step_count += 2
        self.assertEqual(
            external_scheduler._step_count,
            before_direct_adjust + 2,
        )
        raw_scheduler._step_count -= 2
        self.assertEqual(
            external_scheduler._step_count,
            raw_scheduler._step_count,
        )
        self.assertEqual(
            optimizer.entries[1].optimizer.param_groups[0]["k"],
            1,
        )

        prepared_optimizer.eval()
        self.assertEqual(optimizer.lifecycle_mode, "eval")
        prepared_optimizer.train()
        self.assertEqual(optimizer.lifecycle_mode, "train")

        saved_weights = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        saved_step_count = raw_scheduler._step_count
        saved_last_epoch = raw_scheduler.last_epoch

        with tempfile.TemporaryDirectory() as output_dir:
            accelerator.save_state(output_dir)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(7.0)
            raw_scheduler._step_count += 3
            raw_scheduler.last_epoch += 2

            accelerator.load_state(output_dir)

        for name, tensor in saved_weights.items():
            self.assertTrue(
                torch.equal(tensor, model.state_dict()[name]),
                msg=f"Accelerate model restore mismatch for {name}",
            )
        self.assertEqual(raw_scheduler._step_count, saved_step_count)
        self.assertEqual(raw_scheduler.last_epoch, saved_last_epoch)
        self.assertEqual(external_scheduler._step_count, raw_scheduler._step_count)

        accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
