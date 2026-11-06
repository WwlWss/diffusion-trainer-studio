"""Fix C C1 RED: pinned real Torch/Accelerate observer interfaces.

Do not convert missing observer functions to skips or invent CUDA success.
"""
from __future__ import annotations

import importlib
import unittest

import torch
import accelerate


def observer():
    return importlib.import_module("tools.sd3_full_bf16_worker_observer")


class WorkerPhysicalObserverRED(unittest.TestCase):
    def test_pinned_real_cpu_packages(self):
        self.assertEqual(torch.__version__.split("+")[0], "2.7.0")
        self.assertEqual(accelerate.__version__, "1.6.0")

    def test_exact_bf16_tensor_fingerprint_uses_original_bytes(self):
        api = observer()
        bf16 = torch.tensor([1.0, 1.5], dtype=torch.bfloat16)
        fp = api.fingerprint_tensor(bf16)
        self.assertEqual(fp["dtype"], "torch.bfloat16")
        self.assertEqual(fp["shape"], [2])
        self.assertNotEqual(fp["sha256"], api.fingerprint_tensor(bf16.float())["sha256"])
        self.assertEqual(fp, api.fingerprint_tensor(bf16.clone()))

    def test_nested_optimizer_state_fingerprint_stable_across_object_instances(self):
        api = observer()
        a = {"step": 1, "a": torch.tensor([1, 2], dtype=torch.int64)}
        b = {"a": torch.tensor([1, 2], dtype=torch.int64), "step": 1}
        c = {"a": torch.tensor([1, 3], dtype=torch.int64), "step": 1}
        self.assertEqual(api.fingerprint_state_tree(a), api.fingerprint_state_tree(b))
        self.assertNotEqual(api.fingerprint_state_tree(a), api.fingerprint_state_tree(c))

    def test_actual_accelerate_optimizer_and_scheduler_identity(self):
        from accelerate import Accelerator
        from mikazuki.parameter_policy_torch import (
            CompositeOptimizer, CompositeLRScheduler,
            OptimizerRuntimeEntry, SchedulerRuntimeEntry,
        )
        from types import SimpleNamespace

        api = observer()
        model = torch.nn.Linear(4, 3)
        child = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(child, lambda _: 1.0)
        facade = CompositeOptimizer(
            SimpleNamespace(topology_fingerprint="c1-real-cpu"),
            (OptimizerRuntimeEntry(
                profile_name="adamw_main", optimizer_type="AdamW",
                topology_fingerprint="c1-child", optimizer=child,
            ),),
        )
        scheduler_facade = CompositeLRScheduler(
            facade, (SchedulerRuntimeEntry(
                profile_name="adamw_main", optimizer_type="AdamW",
                topology_fingerprint="c1-child", mode="external",
                optimizer=child, scheduler=scheduler,
                scheduler_type="torch.optim.lr_scheduler.LambdaLR",
            ),),
        )
        acc = Accelerator(cpu=True, mixed_precision="no")
        model, prepared_optimizer, prepared_scheduler = acc.prepare(model, facade, scheduler_facade)
        self.assertEqual(len(acc._optimizers), 1)
        self.assertEqual(len(acc._schedulers), 1)
        self.assertIs(prepared_optimizer.optimizer, facade)
        self.assertIs(prepared_scheduler.scheduler, scheduler_facade)
        self.assertTrue(callable(api.PhysicalStepProbe))

    def test_resume_rng_snapshot_interface(self):
        api = observer()
        snapshot = api.snapshot_rng_fingerprint()
        self.assertIn("python", snapshot)
        self.assertIn("numpy", snapshot)
        self.assertIn("torch_cpu", snapshot)


if __name__ == "__main__":
    unittest.main()
