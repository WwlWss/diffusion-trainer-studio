"""CPU-only fail-closed tests for SD3 extra-attention checkpoint semantics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "dev" / "library"


def _load_contract():
    spec = importlib.util.spec_from_file_location("sd3_forward_contract", LIB / "sd3_forward_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAccelerator:
    is_main_process = True

    def __init__(self):
        self.save_hooks = []
        self.load_hooks = []

    def register_save_state_pre_hook(self, callback):
        self.save_hooks.append(callback)

    def register_load_state_pre_hook(self, callback):
        self.load_hooks.append(callback)


def _model(*, extra=True, hidden=1536, heads=24):
    layers = []
    for i in range(24):
        active = extra and i < 13
        attn2 = SimpleNamespace(head_dim=hidden // heads, num_heads=heads) if active else None
        layers.append(SimpleNamespace(x_block=SimpleNamespace(
            x_block_self_attn=active, attn2=attn2,
        )))
    return SimpleNamespace(
        joint_blocks=layers, hidden_size=hidden,
        num_heads=heads, model_type="sd35m",
    )


class SD3ForwardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract()

    def test_real_sd35_medium_architecture_contract(self):
        result = self.contract.forward_semantics_for_mmdit(_model())
        self.assertEqual(result["head_dim"], 64)
        self.assertEqual(result["num_heads"], 24)
        self.assertEqual(result["extra_attention_layers"], list(range(13)))
        self.assertEqual(result["attention_contract"], "extra-attention-head-dim-v1")

    def test_no_extra_attention_remains_unchanged(self):
        self.assertIsNone(self.contract.forward_semantics_for_mmdit(_model(extra=False)))
        acc = FakeAccelerator()
        self.assertFalse(self.contract.register_sd3_forward_semantics_hooks(acc, _model(extra=False)))
        self.assertFalse(acc.save_hooks)
        self.assertFalse(acc.load_hooks)

    def test_rejects_inconsistent_live_extra_attention(self):
        model = _model()
        model.joint_blocks[12].x_block.attn2.head_dim = 24
        with self.assertRaisesRegex(self.contract.SD3ForwardSemanticsError, "head_dim"):
            self.contract.forward_semantics_for_mmdit(model)

    def test_save_and_resume_round_trip(self):
        acc = FakeAccelerator()
        self.assertTrue(self.contract.register_sd3_forward_semantics_hooks(acc, _model()))
        with tempfile.TemporaryDirectory() as td:
            acc.save_hooks[0]([], [], td)
            path = Path(td) / self.contract.SD3_FORWARD_SEMANTICS_FILE
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["head_dim"], 64)
            acc.load_hooks[0]([], td)

    def test_legacy_missing_sidecar_rejected_before_optimizer_load(self):
        acc = FakeAccelerator()
        self.contract.register_sd3_forward_semantics_hooks(acc, _model())
        restored_optimizer = []
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(self.contract.SD3ForwardSemanticsError, "lacks a forward-semantics"):
                for hook in acc.load_hooks:
                    hook([], td)
                restored_optimizer.append(True)
        self.assertEqual(restored_optimizer, [])

    def test_corrupt_mismatched_and_old_algorithm_rejected(self):
        acc = FakeAccelerator()
        self.contract.register_sd3_forward_semantics_hooks(acc, _model())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / self.contract.SD3_FORWARD_SEMANTICS_FILE
            for text in ('{broken', '{"version": 0}', '{"attention_contract": "legacy-wrong-heads"}'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(self.contract.SD3ForwardSemanticsError):
                        acc.load_hooks[0]([], td)

    def test_scoped_to_sd3_before_resume_in_both_trainer_paths(self):
        base = (ROOT / "scripts" / "dev" / "train_network.py").read_text(encoding="utf-8")
        sd3 = (ROOT / "scripts" / "dev" / "sd3_train_network.py").read_text(encoding="utf-8")
        full = (ROOT / "scripts" / "dev" / "sd3_train.py").read_text(encoding="utf-8")
        self.assertIn("self.register_additional_state_hooks(", base)
        self.assertIn("def register_additional_state_hooks(", sd3)
        self.assertIn("register_sd3_forward_semantics_hooks(", full)
        self.assertLess(
            full.index("register_sd3_forward_semantics_hooks("),
            full.index("    train_util.resume_from_local_or_hf_if_specified(accelerator, args)"),
        )


if __name__ == "__main__":
    unittest.main()
