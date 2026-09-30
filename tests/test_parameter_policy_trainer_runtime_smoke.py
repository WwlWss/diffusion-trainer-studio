from __future__ import annotations

import copy
import importlib.util
import gc
import json
import tempfile
import unittest
import weakref
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


_REQUIRED_MODULES = ("torch", "accelerate")
_RUNTIME_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in _REQUIRED_MODULES)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator

    from mikazuki.parameter_policy_trainer import (
        PARAMETER_POLICY_CHECKPOINT_MANIFEST,
        PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION,
        ParameterPolicyTrainerRuntimeError,
        _same_runtime_device,
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

    def test_baseline_session_has_no_execution_contract_or_root_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, _model, _bias, session, _scheduler = self._build_runtime(policy_path)
            self.assertIsNone(session.execution_contract)
            self.assertEqual(session.execution_root_refs, ())

    def test_mock_qualified_full_bf16_session_captures_contract_and_weak_roots(self):
        from mikazuki import parameter_policy_execution as execution

        released_backend = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only backend release",
            "test:backend",
        )
        released_optimizer = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only optimizer release",
            "test:adamw",
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"flux-finetune": released_backend},
        ), mock.patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": released_optimizer},
        ):
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            args = _scheduler_args(
                policy_path,
                full_bf16=True,
                mixed_precision="bf16",
            )
            model = TinyFlux()
            model_ref = weakref.ref(model)
            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
            )
            self.assertIsNotNone(session.execution_contract)
            self.assertEqual(session.execution_contract.features, ("full_bf16",))
            self.assertEqual(session.execution_contract.mixed_precision, "bf16")
            self.assertEqual(len(session.execution_root_refs), 1)
            self.assertEqual(session.execution_root_refs[0].scan_key, "transformer")
            self.assertIs(session.execution_root_refs[0].reference(), model)

            del model
            gc.collect()
            self.assertIsNone(model_ref())
            self.assertIsNone(session.execution_root_refs[0].reference())

    def test_dead_execution_root_fails_closed_without_strong_module_reference(self):
        from mikazuki import parameter_policy_execution as execution

        released_backend = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only backend release",
            "test:backend",
        )
        released_optimizer = execution.ExecutionFeatureQualification(
            "qualified",
            "test-only optimizer release",
            "test:adamw",
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {"flux-finetune": released_backend},
        ), mock.patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {"AdamW": released_optimizer},
        ):
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            args = _scheduler_args(
                policy_path,
                full_bf16=True,
                mixed_precision="bf16",
            )
            model = TinyFlux()
            model_ref = weakref.ref(model)
            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
            )

            del model
            gc.collect()
            self.assertIsNone(model_ref())
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "execution root 'transformer' is no longer alive",
            ):
                session.assert_live_root_identity_contract()

    def _build_mock_qualified_full_bf16_session(self, policy_path):
        from mikazuki import parameter_policy_execution as execution

        backend_patch = mock.patch.dict(
            execution.FULL_BF16_BACKEND_QUALIFICATIONS,
            {
                "flux-finetune": execution.ExecutionFeatureQualification(
                    "qualified",
                    "test-only backend release",
                    "test:backend",
                )
            },
        )
        optimizer_patch = mock.patch.dict(
            execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
            {
                "AdamW": execution.ExecutionFeatureQualification(
                    "qualified",
                    "test-only optimizer release",
                    "test:adamw",
                )
            },
        )
        backend_patch.start()
        optimizer_patch.start()
        self.addCleanup(backend_patch.stop)
        self.addCleanup(optimizer_patch.stop)

        args = _scheduler_args(
            policy_path,
            full_bf16=True,
            mixed_precision="bf16",
        )
        model = TinyFlux()
        structural_bias = model.double_blocks[0].bias
        session = create_parameter_policy_session(
            args=args,
            train_type="flux-finetune",
            roots={"transformer": model},
            structural_frozen_parameters=(structural_bias,),
        )
        return args, model, session

    def test_full_bf16_live_root_identity_accepts_unchanged_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            session.assert_live_root_identity_contract()
            self.assertIs(
                session.execution_root_refs[0].reference(),
                model,
            )

    def test_full_bf16_real_module_cast_prepare_and_finalize_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            original_weight = model.double_blocks[0].weight
            original_weight_id = id(original_weight)

            model.to(torch.bfloat16)
            self.assertEqual(
                id(model.double_blocks[0].weight),
                original_weight_id,
            )
            self.assertEqual(
                model.double_blocks[0].weight.dtype,
                torch.bfloat16,
            )

            accelerator = Accelerator(cpu=True)
            model, optimizer = accelerator.prepare(model, session.optimizer)
            self.assertEqual(
                id(accelerator.unwrap_model(model).double_blocks[0].weight),
                original_weight_id,
            )
            self.assertEqual(
                accelerator.unwrap_model(model).double_blocks[0].weight.dtype,
                torch.bfloat16,
            )

            session.finalize_after_prepare(
                accelerator=accelerator,
                optimizer=optimizer,
            )
            accelerator.end_training()

    def test_full_bf16_live_root_identity_rejects_in_place_shape_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            parameter = model.double_blocks[0].weight
            original_parameter_id = id(parameter)
            original_shape = tuple(parameter.shape)

            parameter.data = parameter.data.reshape(-1)
            self.assertEqual(id(parameter), original_parameter_id)
            self.assertNotEqual(tuple(parameter.shape), original_shape)

            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "live-root identity audit failed.*reshaped=1",
            ):
                session.assert_live_root_identity_contract()

    def test_full_bf16_live_root_identity_rejects_parameter_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            old_weight = model.double_blocks[0].weight
            replacement = torch.nn.Parameter(
                old_weight.detach().clone(),
                requires_grad=old_weight.requires_grad,
            )
            model.double_blocks[0].weight = replacement

            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "live-root identity audit failed.*replaced=1",
            ):
                session.assert_live_root_identity_contract()

    def test_full_bf16_live_root_identity_rejects_alias_topology_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            model.double_blocks[0].register_parameter(
                "shadow_weight",
                model.double_blocks[0].weight,
            )

            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "live-root identity audit failed.*unexpected=1",
            ):
                session.assert_live_root_identity_contract()

    def test_full_bf16_post_resume_rechecks_live_root_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            model.double_blocks[0].weight.data = (
                model.double_blocks[0].weight.data.to(torch.bfloat16)
            )
            accelerator = Accelerator(cpu=True)
            model, optimizer = accelerator.prepare(model, session.optimizer)
            session.assert_runtime_contract(
                phase="post_prepare",
                accelerator=accelerator,
                optimizer=optimizer,
            )

            current = accelerator.unwrap_model(model).double_blocks[0].weight
            accelerator.unwrap_model(model).double_blocks[0].weight = torch.nn.Parameter(
                current.detach().clone(),
                requires_grad=current.requires_grad,
            )
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "post_resume.*live-root identity audit failed",
            ):
                session.assert_runtime_contract(
                    phase="post_resume",
                    accelerator=accelerator,
                    optimizer=optimizer,
                )
            accelerator.end_training()

    def test_full_bf16_epoch_start_does_not_rescan_live_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            model.double_blocks[0].weight.data = (
                model.double_blocks[0].weight.data.to(torch.bfloat16)
            )
            accelerator = Accelerator(cpu=True)
            model, optimizer = accelerator.prepare(model, session.optimizer)
            with mock.patch(
                "mikazuki.parameter_policy_trainer.scan_parameter_roots",
                side_effect=AssertionError("epoch_start must not rescan roots"),
            ):
                session.assert_runtime_contract(
                    phase="epoch_start",
                    accelerator=accelerator,
                    optimizer=optimizer,
                )
            accelerator.end_training()

    def test_full_bf16_dtype_contract_accepts_trainable_bf16_and_frozen_fp32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            trainable_weight = model.double_blocks[0].weight
            frozen_bias = model.double_blocks[0].bias
            trainable_weight.data = trainable_weight.data.to(torch.bfloat16)

            self.assertEqual(trainable_weight.dtype, torch.bfloat16)
            self.assertEqual(frozen_bias.dtype, torch.float32)
            session.assert_execution_dtype_contract()

    def test_full_bf16_dtype_contract_rejects_trainable_fp32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            self.assertEqual(model.double_blocks[0].weight.dtype, torch.float32)
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "execution dtype audit failed.*bfloat16",
            ):
                session.assert_execution_dtype_contract()

    def test_full_bf16_runtime_contract_checks_dtype_without_optimizer_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, session = self._build_mock_qualified_full_bf16_session(
                policy_path
            )
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "dtype_only.*execution dtype audit failed",
            ):
                session.assert_runtime_contract(phase="dtype_only")

    def test_full_bf16_direct_trainer_fails_before_parameter_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            args = _scheduler_args(
                policy_path,
                full_bf16=True,
                mixed_precision="bf16",
            )
            model = TinyFlux()

            with mock.patch(
                "mikazuki.parameter_policy_trainer.scan_parameter_roots"
            ) as scan:
                with self.assertRaisesRegex(
                    ParameterPolicyTrainerRuntimeError,
                    "full_bf16",
                ):
                    create_parameter_policy_session(
                        args=args,
                        train_type="flux-finetune",
                        roots={"transformer": model},
                    )
                scan.assert_not_called()

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

    def test_manifest_v2_metadata_diagnostics_and_finalize_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, _bias, session, scheduler = self._build_runtime(policy_path)

            accelerator = Accelerator(cpu=True)
            model, optimizer, scheduler = accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            session.finalize_after_prepare(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            manifest = session.checkpoint_manifest(scheduler)
            self.assertNotIn("execution_identity", manifest)
            self.assertNotIn("execution_signature", manifest)
            self.assertEqual(
                manifest["version"],
                PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION,
            )
            self.assertEqual(PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION, 2)
            self.assertEqual(
                manifest["trainable_components"],
                ["transformer.double_stream"],
            )
            self.assertEqual(manifest["optimizer_profiles"], {"main": "AdamW"})
            self.assertEqual(manifest["trainable_parameter_tensors"], 1)
            self.assertEqual(
                manifest["trainable_parameter_elements"],
                model.double_blocks[0].weight.numel(),
            )

            metadata = session.model_metadata()
            self.assertNotIn("ss_dts_parameter_policy_execution_identity", metadata)
            self.assertNotIn("ss_dts_parameter_policy_execution_signature", metadata)
            self.assertEqual(
                metadata["ss_dts_parameter_policy_train_type"],
                "flux-finetune",
            )
            self.assertEqual(
                metadata["ss_dts_parameter_policy_manifest_version"],
                "2",
            )
            self.assertIn(
                "transformer.double_stream",
                metadata["ss_dts_parameter_policy_trainable_components"],
            )
            diagnostics = session.startup_diagnostics()
            self.assertEqual(diagnostics["optimizer_profiles"], {"main": "AdamW"})
            self.assertEqual(diagnostics["trainable_parameter_tensors"], 1)
            accelerator.end_training()

    def test_runtime_contract_detects_requires_grad_mutation_with_phase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, structural_bias, session, scheduler = self._build_runtime(policy_path)

            accelerator = Accelerator(cpu=True)
            model, optimizer, scheduler = accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            session.finalize_after_prepare(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            structural_bias.requires_grad_(True)
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "epoch_start.*requires-grad contract",
            ):
                session.assert_runtime_contract(
                    phase="epoch_start",
                    accelerator=accelerator,
                    optimizer=optimizer,
                )
            accelerator.end_training()

    def test_runtime_contract_detects_frozen_optimizer_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, structural_bias, session, scheduler = self._build_runtime(policy_path)

            accelerator = Accelerator(cpu=True)
            model, optimizer, scheduler = accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            composite = optimizer.optimizer
            composite.param_groups[0]["params"].append(structural_bias)
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "Frozen Parameter Policy parameters leaked",
            ):
                session.assert_runtime_contract(
                    phase="test_optimizer_leak",
                    accelerator=accelerator,
                    optimizer=optimizer,
                )
            accelerator.end_training()

    def test_runtime_device_comparison_resolves_only_implicit_cuda_alias(self):
        cases = (
            ("cuda", "cuda:0", 0, True),
            ("cuda:0", "cuda", 0, True),
            ("cuda", "cuda:1", 1, True),
            ("cuda:1", "cuda", 1, True),
            ("cuda", "cuda:1", 0, False),
            ("cuda", "cuda:0", 1, False),
            ("cuda:0", "cuda:1", 0, False),
            ("cuda:1", "cuda:0", 1, False),
            ("cpu", "cpu", 0, True),
            ("cpu", "meta", 0, False),
            ("cpu", "cuda", 0, False),
        )
        for actual, expected, current_index, matches in cases:
            with self.subTest(
                actual=actual,
                expected=expected,
                current_index=current_index,
            ):
                self.assertEqual(
                    _same_runtime_device(
                        torch.device(actual),
                        torch.device(expected),
                        current_cuda_index=current_index,
                    ),
                    matches,
                )

    def test_explicit_or_non_cuda_device_comparison_does_not_query_current_cuda(self):
        with mock.patch(
            "mikazuki.parameter_policy_trainer.torch.cuda.current_device",
            side_effect=AssertionError("current_device must not be queried"),
        ):
            self.assertTrue(
                _same_runtime_device(
                    torch.device("cuda:0"),
                    torch.device("cuda:0"),
                )
            )
            self.assertTrue(
                _same_runtime_device(
                    torch.device("cpu"),
                    torch.device("cpu"),
                )
            )
            self.assertFalse(
                _same_runtime_device(
                    torch.device("cpu"),
                    torch.device("meta"),
                )
            )

    def test_implicit_cuda_device_uses_current_cuda_device(self):
        with mock.patch(
            "mikazuki.parameter_policy_trainer.torch.cuda.current_device",
            return_value=0,
        ) as current_device:
            self.assertTrue(
                _same_runtime_device(
                    torch.device("cuda:0"),
                    torch.device("cuda"),
                )
            )
            self.assertFalse(
                _same_runtime_device(
                    torch.device("cuda:1"),
                    torch.device("cuda"),
                )
            )
            self.assertEqual(current_device.call_count, 2)

    def test_runtime_contract_detects_trainable_device_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, _model, _bias, session, _scheduler = self._build_runtime(policy_path)

            fake_accelerator = SimpleNamespace(device=torch.device("meta"))
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "device audit failed",
            ):
                session.assert_runtime_contract(
                    phase="device_test",
                    accelerator=fake_accelerator,
                    optimizer=session.optimizer,
                )

    def test_checkpoint_save_hook_rejects_mutated_runtime_before_manifest_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            _args, model, _bias, session, scheduler = self._build_runtime(policy_path)

            accelerator = Accelerator(cpu=True)
            model, optimizer, scheduler = accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            session.finalize_after_prepare(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            model.double_blocks[0].weight.requires_grad_(False)
            state_dir = Path(temp_dir) / "bad-state"
            with self.assertRaisesRegex(
                ParameterPolicyTrainerRuntimeError,
                "checkpoint_save.*requires-grad contract",
            ):
                accelerator.save_state(state_dir)
            self.assertFalse(
                (state_dir / PARAMETER_POLICY_CHECKPOINT_MANIFEST).is_file()
            )
            accelerator.end_training()

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
