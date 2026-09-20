from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


_RUNTIME_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "accelerate")
)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator
    from mikazuki.parameter_policy_trainer import create_parameter_policy_session


def _args(policy_path: Path, **overrides):
    values = {
        "parameter_policy_config": str(policy_path),
        "optimizer_type": "AdamW",
        "lr_scheduler": "constant",
        "lr_scheduler_type": "",
        "lr_scheduler_args": None,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "lr_scheduler_power": 1.0,
        "lr_scheduler_timescale": None,
        "lr_scheduler_min_lr_ratio": None,
        "max_train_steps": 4,
        "train_qwen3_text_encoder": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


if _RUNTIME_DEPS_AVAILABLE:
    class TinyAnima(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.self_attn = torch.nn.Linear(4, 4)
            self.block.cross_attn = torch.nn.Linear(4, 4)
            self.block.mlp = torch.nn.Linear(4, 4)
            self.adaln_modulation = torch.nn.Linear(4, 4)
            self.final = torch.nn.Linear(4, 4)

    class TinyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "self_attn": torch.nn.Linear(4, 4),
                    "mlp": torch.nn.Linear(4, 4),
                })
            ])
else:
    TinyAnima = TinyQwen = object


@unittest.skipUnless(_RUNTIME_DEPS_AVAILABLE, "runtime dependencies are not installed")
class ParameterPolicyAnimaRuntimeSmokeTests(unittest.TestCase):
    def _write(self, directory: str, policy: dict) -> Path:
        path = Path(directory) / "policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path

    def test_anima_full_dit_and_qwen_ownership(self):
        policy = {
            "version": 1,
            "optimizer_profiles": {
                "dit_opt": {"type": "AdamW", "args": {}},
                "qwen_opt": {"type": "AdamW", "args": {}},
            },
            "components": {
                "dit.self_attention": {"train": True, "optimizer_profile": "dit_opt", "learning_rate": 1e-4},
                "dit.cross_attention": {"train": False},
                "dit.mlp": {"train": True, "optimizer_profile": "dit_opt", "learning_rate": 8e-5},
                "dit.modulation": {"train": False},
                "dit.llm_adapter": {"train": False},
                "dit.base_other": {"train": False},
                "qwen3": {"train": True, "optimizer_profile": "qwen_opt", "learning_rate": 5e-7},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, policy)
            dit = TinyAnima()
            qwen = TinyQwen()
            session = create_parameter_policy_session(
                args=_args(path),
                train_type="anima-finetune",
                roots={"dit": dit, "qwen3": qwen},
            )
            self.assertTrue(session.trains_component("dit.self_attention"))
            self.assertFalse(session.trains_component("dit.cross_attention"))
            self.assertTrue(session.trains_component("qwen3"))
            self.assertTrue(all(p.requires_grad for p in dit.block.self_attn.parameters()))
            self.assertTrue(all(not p.requires_grad for p in dit.block.cross_attn.parameters()))
            self.assertTrue(all(p.requires_grad for p in qwen.parameters()))

            accelerator = Accelerator(cpu=True)
            dit, qwen, optimizer = accelerator.prepare(dit, qwen, session.optimizer)
            session.audit_after_prepare(accelerator=accelerator, optimizer=optimizer)
            accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
