from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


_REQUIRED_MODULES = ("torch", "accelerate")
_RUNTIME_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in _REQUIRED_MODULES)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator

    from mikazuki.parameter_policy_trainer import (
        PARAMETER_POLICY_CHECKPOINT_MANIFEST,
        ParameterPolicyTrainerRuntimeError,
        create_parameter_policy_session,
        make_legacy_scheduler_factory,
    )


def _policy(optimizer_type: str = "AdamW"):
    return {
        "version": 1,
        "optimizer_profiles": {
            "main": {"type": optimizer_type, "args": {}},
        },
        "components": {
            "transformer.double_stream": {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 1e-3,
            },
        },
    }


def _scheduler_args(policy_path: Path, **overrides):
    values = {
        "parameter_policy_config": str(policy_path),
        "optimizer_type": "DAdaptation",
        "lr_scheduler": "constant",
        "lr_scheduler_type": "",
        "lr_scheduler_args": None,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "lr_scheduler_power": 1.0,
        "lr_scheduler_timescale": None,
        "lr_scheduler_min_lr_ratio": None,
        "max_train_steps": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


if _RUNTIME_DEPS_AVAILABLE:
    class TinyFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def forward(self, value):
            return self.double_blocks[0](value)
else:
    TinyFlux = object


@unittest.skipUnless(_RUNTIME_DEPS_AVAILABLE, "runtime dependencies are not installed")
class ParameterPolicyTrainerRuntimeSmokeTests(unittest.TestCase):
    @staticmethod
    def _scheduler_factory(args, observed=None):
        def get_scheduler_fix(child_args, optimizer, num_processes):
            if observed is not None:
                observed.append((child_args.optimizer_type, num_processes))
            # Deliberately return the same scheduler class for every config.
            # Resume safety must therefore come from the audited scheduler
            # identity, not from scheduler_type alone.
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda _step: 1.0,
            )

        return make_legacy_scheduler_factory(
            args=args,
            get_scheduler_fix=get_scheduler_fix,
            num_processes=1,
        )

    def _build_runtime(self, policy_path, *, scheduler_overrides=None):
        args = _scheduler_args(policy_path, **(scheduler_overrides or {}))
        model = TinyFlux()
        structural_bias = model.double_blocks[0].bias
        session = create_parameter_policy_session(
            args=args,
            train_type="flux-finetune",
            roots={"transformer": model},
            structural_frozen_parameters=(structural_bias,),
        )
        observed = []
        scheduler = session.build_scheduler(
            self._scheduler_factory(args, observed=observed)
        )
        self.assertEqual(observed, [("AdamW", 1)])
        return args, model, structural_bias, session, scheduler

    def test_session_owns_freeze_scheduler_prepare_device_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(
                json.dumps(_policy(), sort_keys=True),
                encoding="utf-8",
            )
            _args, model, structural_bias, session, scheduler = self._build_runtime(
                policy_path
            )

            self.assertTrue(session.trains_component("transformer.double_stream"))
            self.assertTrue(model.double_blocks[0].weight.requires_grad)
            self.assertFalse(structural_bias.requires_grad)
            self.assertEqual(
                session.trainable_parameter_ids,
                frozenset({id(model.double_blocks[0].weight)}),
            )

            accelerator = Accelerator(cpu=True)
            session.register_checkpoint_manifest(accelerator, scheduler=scheduler)
            model, prepared_optimizer, prepared_scheduler = accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            session.audit_after_prepare(
                accelerator=accelerator,
                optimizer=prepared_optimizer,
            )

            loss = model(torch.ones(2, 4)).sum()
            accelerator.backward(loss)
            prepared_optimizer.step()
            prepared_scheduler.step()
            prepared_optimizer.zero_grad()

            state_dir = Path(temp_dir) / "state"
            accelerator.save_state(state_dir)
            manifest_path = state_dir / PARAMETER_POLICY_CHECKPOINT_MANIFEST
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["runtime_topology_fingerprint"],
                session.runtime_spec.topology_fingerprint,
            )
            self.assertEqual(
                manifest["scheduler_signature"],
                session.scheduler_signature,
            )
            self.assertEqual(
                manifest["scheduler_identity"],
                session.scheduler_identity,
            )
            accelerator.end_training()

    def test_accelerator_load_state_accepts_same_scheduler_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, _bias, session, scheduler = self._build_runtime(policy_path)

            save_accelerator = Accelerator(cpu=True)
            session.register_checkpoint_manifest(save_accelerator, scheduler=scheduler)
            model, optimizer, scheduler = save_accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            loss = model(torch.ones(2, 4)).sum()
            save_accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            state_dir = Path(temp_dir) / "state"
            save_accelerator.save_state(state_dir)
            save_accelerator.end_training()

            _args2, model2, _bias2, session2, scheduler2 = self._build_runtime(policy_path)
            load_accelerator = Accelerator(cpu=True)
            session2.register_checkpoint_manifest(load_accelerator, scheduler=scheduler2)
            model2, optimizer2, scheduler2 = load_accelerator.prepare(
                model2,
                session2.optimizer,
                scheduler2,
            )
            self.assertEqual(session2.optimizer.state_dict()["children"]["main"]["state_dict"]["state"], {})
            load_accelerator.load_state(state_dir)
            self.assertTrue(
                session2.optimizer.state_dict()["children"]["main"]["state_dict"]["state"]
            )
            load_accelerator.end_training()

    def test_scheduler_mismatch_fails_before_optimizer_or_scheduler_state_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, _bias, session, scheduler = self._build_runtime(policy_path)

            save_accelerator = Accelerator(cpu=True)
            session.register_checkpoint_manifest(save_accelerator, scheduler=scheduler)
            model, optimizer, scheduler = save_accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            loss = model(torch.ones(2, 4)).sum()
            save_accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            state_dir = Path(temp_dir) / "state"
            save_accelerator.save_state(state_dir)
            save_accelerator.end_training()

            _args2, model2, _bias2, session2, scheduler2 = self._build_runtime(
                policy_path,
                scheduler_overrides={"lr_warmup_steps": 0.25},
            )
            self.assertNotEqual(session.scheduler_signature, session2.scheduler_signature)

            load_accelerator = Accelerator(cpu=True)
            session2.register_checkpoint_manifest(load_accelerator, scheduler=scheduler2)
            model2, optimizer2, scheduler2 = load_accelerator.prepare(
                model2,
                session2.optimizer,
                scheduler2,
            )
            optimizer_before = copy.deepcopy(session2.optimizer.state_dict())
            scheduler_before = copy.deepcopy(scheduler2.state_dict())

            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "scheduler configuration",
            ):
                load_accelerator.load_state(state_dir)

            self.assertEqual(session2.optimizer.state_dict(), optimizer_before)
            self.assertEqual(scheduler2.state_dict(), scheduler_before)
            load_accelerator.end_training()

    def test_optimizer_managed_scheduler_ignores_unused_global_scheduler_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(
                json.dumps(_policy("AdamWScheduleFree")),
                encoding="utf-8",
            )
            args = _scheduler_args(
                policy_path,
                lr_scheduler="definitely-unused",
                lr_scheduler_args=["not-even-key-value"],
            )
            model = TinyFlux()
            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
            )

            def must_not_be_called(*_args, **_kwargs):
                raise AssertionError("external scheduler factory must not run")

            factory = make_legacy_scheduler_factory(
                args=args,
                get_scheduler_fix=must_not_be_called,
                num_processes=1,
            )
            scheduler = session.build_scheduler(factory)

            self.assertTrue(all(entry.mode == "optimizer_managed" for entry in scheduler.entries))
            self.assertEqual(
                session.scheduler_identity["mode"],
                "optimizer_managed",
            )
            self.assertIsNone(factory.scheduler_identity)
            self.assertIsNone(factory.scheduler_signature)


if __name__ == "__main__":
    unittest.main()
