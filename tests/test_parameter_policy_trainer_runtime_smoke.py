from __future__ import annotations

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
        create_parameter_policy_session,
        make_legacy_scheduler_factory,
    )


@unittest.skipUnless(_RUNTIME_DEPS_AVAILABLE, "runtime dependencies are not installed")
class ParameterPolicyTrainerRuntimeSmokeTests(unittest.TestCase):
    def test_session_owns_freeze_scheduler_prepare_device_and_manifest(self):
        class TinyFlux(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

            def forward(self, value):
                return self.double_blocks[0](value)

        policy = {
            "version": 1,
            "optimizer_profiles": {
                "adam": {"type": "AdamW", "args": {}},
            },
            "components": {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "adam",
                    "learning_rate": 1e-3,
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(
                json.dumps(policy, sort_keys=True),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                parameter_policy_config=str(policy_path),
                optimizer_type="DAdaptation",
            )
            model = TinyFlux()
            structural_bias = model.double_blocks[0].bias

            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
                structural_frozen_parameters=(structural_bias,),
            )

            self.assertTrue(session.trains_component("transformer.double_stream"))
            self.assertTrue(model.double_blocks[0].weight.requires_grad)
            self.assertFalse(structural_bias.requires_grad)
            self.assertEqual(
                session.trainable_parameter_ids,
                frozenset({id(model.double_blocks[0].weight)}),
            )

            observed_optimizer_types = []

            def get_scheduler_fix(child_args, optimizer, num_processes):
                observed_optimizer_types.append((child_args.optimizer_type, num_processes))
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

            scheduler_factory = make_legacy_scheduler_factory(
                args=args,
                get_scheduler_fix=get_scheduler_fix,
                num_processes=1,
            )
            scheduler = session.build_scheduler(scheduler_factory)
            self.assertEqual(observed_optimizer_types, [("AdamW", 1)])

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

            state_dir = Path(temp_dir) / "state"
            accelerator.save_state(state_dir)
            manifest_path = state_dir / PARAMETER_POLICY_CHECKPOINT_MANIFEST
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["runtime_topology_fingerprint"],
                session.runtime_spec.topology_fingerprint,
            )
            session.validate_checkpoint_manifest(state_dir, scheduler=scheduler)

            accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
