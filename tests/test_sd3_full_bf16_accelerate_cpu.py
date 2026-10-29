"""Pinned CPU-only Accelerate 1.6.0 checkpoint semantics for C2-B0.

This verifies the *real* package's prepared save/load layout. It does not
claim that the SD3 GPU training or its Optimizer Child state has passed.
Run separately from dependency-light Host tests.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch
import accelerate
from accelerate import Accelerator


class Accelerate16CheckpointCPU(unittest.TestCase):
    def test_pinned_single_process_model_optimizer_scheduler_rng_save_load(self):
        self.assertEqual(accelerate.__version__, "1.6.0")
        self.assertEqual(torch.__version__.split("+")[0], "2.7.0")
        acc = Accelerator(cpu=True, mixed_precision="no")
        self.assertEqual(acc.num_processes, 1)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        self.assertEqual(scheduler._step_count, 1)
        self.assertEqual(scheduler.last_epoch, 0)
        model, optimizer, scheduler = acc.prepare(model, optimizer, scheduler)
        optimizer.zero_grad(set_to_none=True)
        acc.backward(model(torch.ones((1, 4))).square().sum())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        # one real update plus the constructor's implicit Scheduler step
        actual_scheduler = getattr(scheduler, "scheduler", scheduler)
        self.assertEqual(actual_scheduler._step_count, 2)
        self.assertEqual(actual_scheduler.last_epoch, 1)
        expected_weights = {
            k: v.detach().cpu().clone()
            for k, v in acc.unwrap_model(model).state_dict().items()
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            acc.save_state(str(root), safe_serialization=True)
            self.assertEqual(
                {x.name for x in root.iterdir()},
                {
                    "model.safetensors", "optimizer.bin",
                    "scheduler.bin", "random_states_0.pkl",
                },
            )
            # Mutate prepared Network state and verify the real load_state
            # restores it from model.safetensors rather than a toy JSON stub.
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1000)
            self.assertFalse(torch.equal(
                acc.unwrap_model(model).weight.detach().cpu(),
                expected_weights["weight"],
            ))
            acc.load_state(str(root))
            for k, value in expected_weights.items():
                self.assertTrue(
                    torch.equal(
                        acc.unwrap_model(model).state_dict()[k].detach().cpu(),
                        value,
                    ),
                    f"Accelerate load_state failed to restore {k}",
                )

    def test_host_reference_scheduler_identity_matches_production_factory(self):
        from mikazuki.parameter_policy_trainer import _scheduler_identity_from_args
        from tools import sd3_full_bf16_evidence_support as host

        # Actual pinned train_util argparse defaults (unprovided in B0 CLI)
        # plus the two mandatory R1/R2 scheduler config fields.
        args = {
            "lr_scheduler": "constant",
            "lr_scheduler_type": "",
            "lr_scheduler_args": None,
            "lr_warmup_steps": 0,
            "lr_decay_steps": 0,
            "lr_scheduler_num_cycles": 1,
            "lr_scheduler_power": 1.0,
            "lr_scheduler_timescale": None,
            "lr_scheduler_min_lr_ratio": None,
            "max_train_steps": 2,
        }
        identity, signature = _scheduler_identity_from_args(args, num_processes=1)
        self.assertEqual(identity, host._expected_reference_scheduler_identity())
        self.assertEqual(signature, host._json_sha256(identity))


if __name__ == "__main__":
    unittest.main()
