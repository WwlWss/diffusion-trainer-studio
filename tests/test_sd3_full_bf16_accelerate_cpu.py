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

    def test_one_composite_optimizer_and_scheduler_have_one_accelerate_state_each(self):
        from types import SimpleNamespace
        from mikazuki.parameter_policy_torch import (
            CompositeOptimizer, CompositeLRScheduler,
            OptimizerRuntimeEntry, SchedulerRuntimeEntry,
        )

        model = torch.nn.Linear(3, 2)
        child = torch.optim.AdamW(model.parameters(), lr=0.0001)
        scheduler_child = torch.optim.lr_scheduler.LambdaLR(child, lambda _: 1.0)
        topology = SimpleNamespace(topology_fingerprint="host-accelerate-composite-fixture")
        optimizer = CompositeOptimizer(
            topology,
            (OptimizerRuntimeEntry(
                profile_name="adamw_main", optimizer_type="AdamW",
                topology_fingerprint="cpu-child", optimizer=child,
            ),),
        )
        scheduler = CompositeLRScheduler(
            optimizer,
            (SchedulerRuntimeEntry(
                profile_name="adamw_main", optimizer_type="AdamW",
                topology_fingerprint="cpu-child", mode="external",
                optimizer=child, scheduler=scheduler_child,
                scheduler_type="torch.optim.lr_scheduler.LambdaLR",
            ),),
        )
        self.assertEqual(scheduler.state_dict()["step_count"], 1)
        self.assertEqual(scheduler.state_dict()["last_epoch"], 0)
        acc = Accelerator(cpu=True, mixed_precision="no")
        model, optimizer, scheduler = acc.prepare(model, optimizer, scheduler)
        optimizer.zero_grad(set_to_none=True)
        acc.backward(model(torch.ones((2, 3))).square().sum())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        actual = getattr(scheduler, "scheduler", scheduler)
        progress = actual.state_dict()
        self.assertEqual(progress["step_count"], 2)
        self.assertEqual(progress["last_epoch"], 1)
        self.assertEqual(set(progress["children"]), {"adamw_main"})
        saved = {k: v.detach().clone() for k, v in acc.unwrap_model(model).state_dict().items()}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            acc.save_state(str(root), safe_serialization=True)
            self.assertEqual(
                set(p.name for p in root.iterdir()),
                {"model.safetensors", "optimizer.bin",
                 "scheduler.bin", "random_states_0.pkl"},
            )
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(100)
            acc.load_state(str(root))
            self.assertEqual(actual.state_dict()["step_count"], 2)
            self.assertEqual(actual.state_dict()["last_epoch"], 1)
            for name, tensor in saved.items():
                self.assertTrue(torch.equal(
                    acc.unwrap_model(model).state_dict()[name], tensor,
                ))

    def test_production_style_hook_discards_unet_and_retains_only_adapter_model(self):
        # Production NetworkTrainer.prepare_unet_with_accelerator prepares
        # the MMDiT as an extra registered model, then its save/load pre-hook
        # removes all model weights except the LoRA Network.
        acc = Accelerator(cpu=True, mixed_precision="no")
        base = torch.nn.Sequential(torch.nn.Linear(3, 3))
        network = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        base = acc.prepare(base)
        network, optimizer, scheduler = acc.prepare(network, optimizer, scheduler)
        self.assertEqual(len(acc._models), 2)

        def save_hook(models, weights, output_dir):
            to_remove = [
                i for i, model in enumerate(models)
                if not isinstance(model, type(acc.unwrap_model(network)))
            ]
            for i in reversed(to_remove):
                if len(weights) > i:
                    weights.pop(i)

        def load_hook(models, input_dir):
            for i in reversed([
                i for i, model in enumerate(models)
                if not isinstance(model, type(acc.unwrap_model(network)))
            ]):
                models.pop(i)

        acc.register_save_state_pre_hook(save_hook)
        acc.register_load_state_pre_hook(load_hook)
        with tempfile.TemporaryDirectory() as td:
            acc.save_state(td, safe_serialization=True)
            folder = Path(td)
            self.assertEqual(
                {p.name for p in folder.iterdir()},
                {"model.safetensors", "optimizer.bin",
                 "scheduler.bin", "random_states_0.pkl"},
            )
            # The one remaining model.safetensors contains the Network,
            # not the separately registered MMDiT.
            from safetensors import safe_open
            with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as f:
                self.assertEqual(set(f.keys()), {"weight", "bias"})
                saved = {k: f.get_tensor(k).clone() for k in f.keys()}
            with torch.no_grad():
                network.weight.add_(33)
                base[0].weight.add_(55)
            base_after = base[0].weight.detach().clone()
            acc.load_state(td)
            self.assertTrue(torch.equal(network.weight.detach().cpu(), saved["weight"]))
            self.assertTrue(torch.equal(base[0].weight.detach(), base_after))

    def test_real_lora_module_alpha_cast_and_safetensors_save_roundtrip(self):
        # Import the real DTS LoRAModule; do not replace it with a toy class.
        import sys
        from safetensors.torch import load_file, save_file
        from tools import sd3_full_bf16_evidence_support as host

        sys.path.insert(0, str(host.ROOT / "scripts" / "dev"))
        from networks.lora_flux import LoRAModule
        lora = LoRAModule(
            "lora_unet_reference_test", torch.nn.Linear(8, 16),
            lora_dim=4, alpha=4.0,
        )
        self.assertEqual(lora.alpha.dtype, torch.float32)
        lora.to(torch.bfloat16)
        self.assertEqual(lora.alpha.dtype, torch.bfloat16)
        self.assertEqual(tuple(lora.lora_down.weight.shape), (4, 8))
        self.assertEqual(tuple(lora.lora_up.weight.shape), (16, 4))
        self.assertEqual(lora.alpha.item(), 4.0)
        state = {k: t.detach().cpu().contiguous()
                 for k, t in lora.state_dict().items()}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lora.safetensors"
            save_file(state, str(path))
            saved = load_file(str(path), device="cpu")
            self.assertEqual(set(saved), set(state))
            for key in state:
                self.assertTrue(torch.equal(saved[key], state[key]), key)
            self.assertEqual(saved["alpha"].dtype, torch.bfloat16)
            import json
            import struct
            file_data = path.read_bytes()
            header_size = struct.unpack("<Q", file_data[:8])[0]
            header = json.loads(file_data[8:8 + header_size])
            begin, end = header["alpha"]["data_offsets"]
            self.assertEqual(file_data[8 + header_size + begin:8 + header_size + end],
                             b"\x80\x40")

    def test_real_sd35_medium_lora_target_enumeration_matches_host_spec(self):
        """Use production MMDiT and LoRANetwork; no checkpoint weights/GPU."""
        import sys
        from tools import sd3_full_bf16_evidence_support as host

        sys.path.insert(0, str(host.ROOT / "scripts" / "dev"))
        from library.sd3_models import MMDiT
        from networks.lora_sd3 import LoRANetwork
        expected = host.build_sd35_medium_lora_target_spec()
        with torch.device("meta"):
            mmdit = MMDiT(
                depth=24, qk_norm="rms",
                x_block_self_attn_layers=list(range(13)),
                model_type="5-medium",
            )
            network = LoRANetwork(
                [], mmdit, lora_dim=4, alpha=4.0,
                split_qkv=False, train_t5xxl=False,
            )
        from mikazuki.model_component_profiles import get_model_component_profile
        component_profile = get_model_component_profile("sd3-lora")
        actual = {}
        actual_eligibility = {}
        for lora in network.unet_loras:
            name = lora.lora_name
            self.assertEqual(lora.split_dims, None)
            self.assertIsInstance(lora.org_module, torch.nn.Linear)
            actual[name] = {
                "input_features": lora.org_module.in_features,
                "output_features": lora.org_module.out_features,
                "rank": lora.lora_dim,
            }
            target_root, target_path, target_type = getattr(
                lora, "_dts_parameter_policy_target_v1"
            )
            alias = {
                "adapter_target": {
                    "root": target_root, "module_path": target_path,
                    "module_type": target_type,
                },
                "parameter_class": "matrix_weight",
            }
            component = component_profile.classify_alias(alias)
            self.assertEqual(component, expected[name]["component_id"], name)
            eligible, _reason = component_profile.eligibility_check(
                "model_hidden_2d_weight", alias, component
            )
            actual_eligibility[name] = bool(eligible)
        self.assertEqual(len(actual), 263)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(
            sum(actual_eligibility.values()), 215,
            "R2 must route 121 Attention + 94 MLP modules to Muon",
        )
        self.assertEqual(len(actual_eligibility) - sum(actual_eligibility.values()), 48)
        for name, real in actual.items():
            ref = expected[name]
            self.assertEqual(real["input_features"], ref["input_features"], name)
            self.assertEqual(real["output_features"], ref["output_features"], name)
            self.assertEqual(real["rank"], ref["rank"], name)
        network.apply_to([], mmdit, apply_text_encoder=False, apply_unet=True)
        network.to(torch.bfloat16)
        tensors = network.state_dict()
        self.assertEqual(len(tensors), 3 * 263)
        self.assertEqual(set(tensors), {
            name + suffix for name in expected
            for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")
        })
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in tensors.values()),
                        [(key, str(value.dtype)) for key, value in tensors.items()
                         if value.dtype != torch.bfloat16][:8])

    def test_r2_two_real_optimizer_children_have_disjoint_cpu_state(self):
        """Actual Muon + AdamW child topology; no BF16 CUDA assertion."""
        from types import SimpleNamespace
        from pytorch_optimizer import Muon
        from mikazuki.parameter_policy_torch import (
            CompositeOptimizer, CompositeLRScheduler,
            OptimizerRuntimeEntry, SchedulerRuntimeEntry,
        )

        muon_param = torch.nn.Parameter(torch.randn(8, 8))
        adamw_param = torch.nn.Parameter(torch.randn(8, 8))
        muon = Muon([{"params": [muon_param], "use_muon": True}], lr=0.0001)
        adamw = torch.optim.AdamW([adamw_param], lr=0.0001)
        outer = CompositeOptimizer(
            SimpleNamespace(topology_fingerprint="cpu-r2-two-child"),
            (
                OptimizerRuntimeEntry(
                    profile_name="muon_primary", optimizer_type="Muon",
                    topology_fingerprint="muon-child-cpu", optimizer=muon,
                ),
                OptimizerRuntimeEntry(
                    profile_name="adamw_fallback", optimizer_type="AdamW",
                    topology_fingerprint="adamw-child-cpu", optimizer=adamw,
                ),
            ),
        )
        self.assertEqual(len(outer.child_optimizers), 2)
        self.assertNotEqual(id(muon_param), id(adamw_param))
        muon_scheduler = torch.optim.lr_scheduler.LambdaLR(muon, lambda _: 1.0)
        adamw_scheduler = torch.optim.lr_scheduler.LambdaLR(adamw, lambda _: 1.0)
        scheduler = CompositeLRScheduler(
            outer,
            (
                SchedulerRuntimeEntry(
                    profile_name="muon_primary", optimizer_type="Muon",
                    topology_fingerprint="muon-child-cpu", mode="external",
                    optimizer=muon, scheduler=muon_scheduler,
                    scheduler_type="torch.optim.lr_scheduler.LambdaLR",
                ),
                SchedulerRuntimeEntry(
                    profile_name="adamw_fallback", optimizer_type="AdamW",
                    topology_fingerprint="adamw-child-cpu", mode="external",
                    optimizer=adamw, scheduler=adamw_scheduler,
                    scheduler_type="torch.optim.lr_scheduler.LambdaLR",
                ),
            ),
        )
        self.assertEqual(scheduler.state_dict()["step_count"], 1)
        # Two actual Backwards, but exactly one optimizer/scheduler update.
        outer.zero_grad(set_to_none=True)
        for _microbatch in range(2):
            (muon_param.square().sum() + adamw_param.square().sum()).backward()
        self.assertIsNotNone(muon_param.grad)
        self.assertIsNotNone(adamw_param.grad)
        self.assertFalse(muon.state)
        self.assertFalse(adamw.state)
        outer.step()
        scheduler.step()
        outer.zero_grad(set_to_none=True)
        self.assertEqual(scheduler.state_dict()["step_count"], 2)
        self.assertEqual(scheduler.state_dict()["last_epoch"], 1)
        state = outer.state_dict()
        scheduler_state = scheduler.state_dict()
        self.assertEqual(set(state["children"]), {"muon_primary", "adamw_fallback"})
        self.assertEqual(
            state["children"]["adamw_fallback"]["optimizer_type"], "AdamW",
        )
        # Validate exact Profile and topology roundtrip of the two actual
        # child optimizers; not a GPU or full-BF16 qualification claim.
        outer.load_state_dict(state)
        scheduler.load_state_dict(scheduler_state)
        self.assertEqual(set(outer.state_dict()["children"]), set(state["children"]))
        self.assertEqual(set(scheduler.state_dict()["children"]), {"muon_primary", "adamw_fallback"})
        self.assertEqual(scheduler.state_dict()["step_count"], 2)

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
