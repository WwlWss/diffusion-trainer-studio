"""Independent CPU reference checks for the production SD3.5 extra-attention path.

Execute the real source AST nodes instead of importing the entire trainer stack:
CI needs CPU Torch, not transformers/diffusers/xformers or real SD3 weights.
The oracle deliberately uses torch SDPA directly, never sd3_models.attention().
"""
from __future__ import annotations

import ast
from functools import partial
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "dev" / "library" / "sd3_models.py"
NODES = frozenset({
    "modulate", "rmsnorm", "RMSNorm", "MLP", "SwiGLUFeedForward",
    "AttentionLinears", "vanilla_attention", "attention",
    "SingleDiTBlock", "MMDiTBlock",
})


def _production_classes():
    """Only strip heavyweight module imports; execute the actual model methods."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NODES:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "MEMORY_LAYOUTS" for target in node.targets):
                selected.append(node)
    missing = NODES - {node.name for node in selected if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if missing:
        raise AssertionError(f"Missing SD3 math nodes: {sorted(missing)}")
    ns = {
        "torch": torch, "nn": nn, "F": F, "partial": partial,
        "checkpoint": checkpoint, "Optional": __import__("typing").Optional,
        "memory_efficient_attention": None,
        "math": __import__("math"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def _reference_sdpa(q, k, v, num_heads):
    """Independent oracle: [B,L,H] -> [B,heads,L,head_dim]."""
    batch, length, hidden = q.shape
    assert hidden % num_heads == 0
    head_dim = hidden // num_heads
    def split(t):
        return t.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    value = F.scaled_dot_product_attention(split(q), split(k), split(v))
    return value.transpose(1, 2).reshape(batch, length, hidden)


def _reference_joint_block(block, context, x, c):
    ctx_qkv, ctx_parts = block.context_block.pre_attention(context, c)
    if block.x_block.x_block_self_attn:
        x_qkv, extra_qkv, x_parts = block.x_block.pre_attention_x(x, c)
    else:
        x_qkv, x_parts = block.x_block.pre_attention(x, c)
    ctx_len = ctx_qkv[0].shape[1]
    joint = [
        torch.cat([ctx_qkv[i], x_qkv[i]], dim=1)
        for i in range(3)
    ]
    joint_result = _reference_sdpa(*joint, num_heads=block.x_block.attn.num_heads)
    ctx_out, x_out = joint_result[:, :ctx_len], joint_result[:, ctx_len:]
    if block.x_block.x_block_self_attn:
        extra = _reference_sdpa(
            *extra_qkv, num_heads=block.x_block.attn2.num_heads,
        )
        output = block.x_block.post_attention_x(x_out, extra, *x_parts)
    else:
        output = block.x_block.post_attention(x_out, *x_parts)
    if not block.context_block.pre_only:
        context = block.context_block.post_attention(ctx_out, *ctx_parts)
    else:
        context = None
    return context, output


class SD3AttentionMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = _production_classes()

    def _block(self, *, extra):
        torch.manual_seed(1729)
        block = self.production["MMDiTBlock"](
            192, 3, attn_mode="torch", pre_only=False,
            qkv_bias=True, qk_norm="rms", x_block_self_attn=extra,
        )
        with torch.no_grad():
            for module in block.modules():
                if isinstance(module, self.production["RMSNorm"]) and module.weight is not None:
                    module.weight.fill_(1)
        return block

    @staticmethod
    def _inputs():
        torch.manual_seed(71)
        context = torch.randn(2, 3, 192, requires_grad=True)
        x = torch.randn(2, 4, 192, requires_grad=True)
        c = torch.randn(2, 192, requires_grad=True)
        return context, x, c

    def test_independent_extra_attention_value_and_gradient(self):
        block = self._block(extra=True)
        self.assertEqual(block.x_block.attn2.head_dim, 64)
        self.assertEqual(block.x_block.attn2.num_heads, 3)
        context, x, c = self._inputs()
        got_context, got_x = block(context, x, c)
        ref_context, ref_x = _reference_joint_block(block, context, x, c)
        torch.testing.assert_close(got_context, ref_context, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got_x, ref_x, atol=1e-5, rtol=1e-5)
        trainables = (x, context, c, block.x_block.attn2.qkv.weight)
        got_grad = torch.autograd.grad((got_x.sum() + got_context.sum()), trainables, retain_graph=True)
        ref_grad = torch.autograd.grad((ref_x.sum() + ref_context.sum()), trainables)
        for got, ref in zip(got_grad, ref_grad):
            torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-5)

    def test_wrong_heads_are_not_an_equivalent_layout(self):
        torch.manual_seed(19)
        q, k, v = [torch.randn(1, 5, 192) for _ in range(3)]
        correct = _reference_sdpa(q, k, v, num_heads=3)
        wrong = self.production["attention"](q, k, v, head_dim=3, mode="torch")
        self.assertGreater(float((correct - wrong).abs().max()), 1e-3)

    def test_without_extra_attention_still_matches_joint_reference(self):
        block = self._block(extra=False)
        context, x, c = self._inputs()
        got = block(context, x, c)
        expected = _reference_joint_block(block, context, x, c)
        for actual, oracle in zip(got, expected):
            torch.testing.assert_close(actual, oracle, atol=1e-5, rtol=1e-5)

    def test_checkpointing_extra_attention_matches_reference(self):
        block = self._block(extra=True)
        block.enable_gradient_checkpointing()
        block.train()
        context, x, c = self._inputs()
        got_context, got_x = block(context, x, c)
        ref_context, ref_x = _reference_joint_block(block, context, x, c)
        torch.testing.assert_close(got_context, ref_context, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got_x, ref_x, atol=1e-5, rtol=1e-5)
        gradient = torch.autograd.grad(got_x.sum(), block.x_block.attn2.qkv.weight)[0]
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_actual_sd35_medium_dimensions_not_exchanged(self):
        self.assertEqual(1536 // 24, 64)
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("head_dim=self.x_block.attn2.head_dim", source)
        self.assertNotIn("x_q2, x_k2, x_v2, self.x_block.attn2.num_heads", source)


if __name__ == "__main__":
    unittest.main()
