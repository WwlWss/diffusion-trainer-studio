"""CPU-only, torch-free contracts for read-only SD3 structure & LoRA audits."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mikazuki.sd3_architecture_contract import (
    SD3ArchitectureError,
    inspect_sd3_architecture,
    normalize_sd3_shape_index,
)
from mikazuki.sd3_component_audit import (
    SD3_DIAGNOSTIC_COMPONENTS,
    SD3AdapterTensor,
    SD3ComponentAuditError,
    audit_sd3_adapter_tensors,
    audit_sd3_parameter_descriptors,
    classify_sd3_adapter_target,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_sd3_shapes(variant: str, *, prefixed: bool = False) -> dict:
    """Shape metadata only. No huge tensors or GPU resources are required."""
    specs = {
        "sd3-medium": (24, 192, False, ()),
        "sd35-medium": (24, 384, True, tuple(range(13))),
        "sd35-large-topology": (38, 192, True, ()),
    }
    depth, grid, qknorm, extras = specs[variant]
    hidden = depth * 64
    idx = {
        "x_embedder.proj.weight": [hidden, 16, 2, 2],
        "context_embedder.weight": [hidden, 4096],
        "pos_embed": [1, grid * grid, hidden],
    }
    for i in range(depth):
        for stream in ("x_block", "context_block"):
            root = f"joint_blocks.{i}.{stream}"
            idx[f"{root}.attn.qkv.weight"] = [3 * hidden, hidden]
            idx[f"{root}.adaLN_modulation.1.weight"] = [
                hidden * (2 if stream == "context_block" and i == depth-1
                          else 9 if stream == "x_block" and i in extras else 6),
                hidden,
            ]
            if not (stream == "context_block" and i == depth-1):
                idx[f"{root}.attn.proj.weight"] = [hidden, hidden]
                idx[f"{root}.mlp.fc1.weight"] = [4 * hidden, hidden]
            if qknorm:
                idx[f"{root}.attn.ln_q.weight"] = [64]
                idx[f"{root}.attn.ln_k.weight"] = [64]
        if i in extras:
            root = f"joint_blocks.{i}.x_block.attn2"
            idx[f"{root}.qkv.weight"] = [3 * hidden, hidden]
            idx[f"{root}.proj.weight"] = [hidden, hidden]
            idx[f"{root}.ln_q.weight"] = [64]
            idx[f"{root}.ln_k.weight"] = [64]
    if prefixed:
        return {"model.diffusion_model." + k: v for k, v in idx.items()} | {
            "text_encoders.clip_l.fake.weight": [1]
        }
    return idx


class ArchitectureContractTests(unittest.TestCase):
    def test_all_known_variants_and_topology_report(self):
        for variant, depth, hidden, grid, extras, kind in (
            ("sd3-medium", 24, 1536, 192, (), "joint_attention"),
            ("sd35-medium", 24, 1536, 384, tuple(range(13)), "dual_attention"),
            ("sd35-large-topology", 38, 2432, 192, (), "joint_attention"),
        ):
            with self.subTest(variant=variant):
                model = inspect_sd3_architecture(synthetic_sd3_shapes(variant, prefixed=True))
                self.assertEqual((model.variant, model.depth, model.hidden_size),
                                 (variant, depth, hidden))
                self.assertEqual(model.pos_embed_grid, grid)
                self.assertEqual(model.extra_attention_layers, extras)
                self.assertEqual(model.block_kind(0), kind)
                self.assertEqual(model.block_kind(depth-1), "context_pre_only")
                self.assertEqual(model.num_heads, depth)
                self.assertEqual(model.head_dim, 64)
                report = model.report()
                self.assertFalse(report["release_identity_verified"])
                self.assertEqual(len(report["shape_index_sha256"]), 64)
                self.assertEqual(len(report["architecture_signature_sha256"]), 64)
                self.assertEqual(model, inspect_sd3_architecture(
                    synthetic_sd3_shapes(variant, prefixed=True)
                ))

    def test_bare_and_prefixed_have_identical_structure_identity(self):
        a = inspect_sd3_architecture(synthetic_sd3_shapes("sd35-medium"))
        b = inspect_sd3_architecture(synthetic_sd3_shapes("sd35-medium", prefixed=True))
        self.assertEqual(a.identity(), b.identity())
        self.assertEqual(a.shape_index_sha256, b.shape_index_sha256)

    def test_mixed_roots_and_missing_root_rejected(self):
        prefixed = synthetic_sd3_shapes("sd35-medium", prefixed=True)
        prefixed["x_embedder.proj.weight"] = [1536, 16, 2, 2]
        with self.assertRaises(SD3ArchitectureError):
            inspect_sd3_architecture(prefixed)
        with self.assertRaises(SD3ArchitectureError):
            normalize_sd3_shape_index({"unrelated": [1]})

    def test_missing_noncontiguous_block_and_shape_mismatch_rejected(self):
        key = "joint_blocks.7.context_block.attn.qkv.weight"
        m = synthetic_sd3_shapes("sd35-medium")
        del m[key]
        with self.assertRaisesRegex(SD3ArchitectureError, "qkv"):
            inspect_sd3_architecture(m)
        m = synthetic_sd3_shapes("sd35-medium")
        m["joint_blocks.24.x_block.attn.qkv.weight"] = [4608, 1536]
        with self.assertRaisesRegex(SD3ArchitectureError, "contiguous"):
            inspect_sd3_architecture(m)
        m = synthetic_sd3_shapes("sd35-medium")
        m["joint_blocks.6.x_block.mlp.fc1.weight"] = [42, 1536]
        with self.assertRaisesRegex(SD3ArchitectureError, "mlp"):
            inspect_sd3_architecture(m)

    def test_additional_attention_architecture_is_exact(self):
        m = synthetic_sd3_shapes("sd35-medium")
        for suffix, shape in (
            ("qkv.weight", [4608, 1536]),
            ("proj.weight", [1536, 1536]),
            ("ln_q.weight", [64]),
            ("ln_k.weight", [64]),
        ):
            m[f"joint_blocks.13.x_block.attn2.{suffix}"] = shape
        with self.assertRaisesRegex(SD3ArchitectureError, "Unknown SD3 architecture"):
            inspect_sd3_architecture(m)
        m = synthetic_sd3_shapes("sd35-medium")
        del m["joint_blocks.11.x_block.attn2.ln_k.weight"]
        with self.assertRaisesRegex(SD3ArchitectureError, "attn2"):
            inspect_sd3_architecture(m)

    def test_qknorm_cannot_be_partially_missing_or_added(self):
        m = synthetic_sd3_shapes("sd35-medium")
        del m["joint_blocks.5.context_block.attn.ln_q.weight"]
        with self.assertRaisesRegex(SD3ArchitectureError, "ln_q"):
            inspect_sd3_architecture(m)
        m = synthetic_sd3_shapes("sd3-medium")
        m["joint_blocks.0.context_block.attn.ln_k.weight"] = [64]
        with self.assertRaises(SD3ArchitectureError):
            inspect_sd3_architecture(m)

    def test_pre_only_last_context_block_rejected_when_violated(self):
        m = synthetic_sd3_shapes("sd35-medium")
        m["joint_blocks.23.context_block.attn.proj.weight"] = [1536, 1536]
        with self.assertRaisesRegex(SD3ArchitectureError, "pre-only"):
            inspect_sd3_architecture(m)
        m = synthetic_sd3_shapes("sd35-medium")
        m["joint_blocks.23.context_block.mlp.fc2.weight"] = [1536, 6144]
        with self.assertRaisesRegex(SD3ArchitectureError, "pre-only"):
            inspect_sd3_architecture(m)

    def test_no_guessing_unrecognized_dtype_or_structure(self):
        m = synthetic_sd3_shapes("sd35-medium")
        m["pos_embed"] = [1, 192 * 192, 1536]
        with self.assertRaisesRegex(SD3ArchitectureError, "Unknown SD3 architecture"):
            inspect_sd3_architecture(m)
        m = synthetic_sd3_shapes("sd35-medium")
        m["x_embedder.proj.weight"] = [1536, 16, 4, 4]
        with self.assertRaisesRegex(SD3ArchitectureError, "patch size"):
            inspect_sd3_architecture(m)
        for malformed in ([True, 2], [-1, 2], [0, 3], [1.3, 3], "1,3"):
            with self.subTest(malformed=malformed):
                m = synthetic_sd3_shapes("sd35-medium")
                m["pos_embed"] = malformed
                with self.assertRaises(SD3ArchitectureError):
                    inspect_sd3_architecture(m)
        a = inspect_sd3_architecture(synthetic_sd3_shapes("sd35-medium"))
        with self.assertRaises(SD3ArchitectureError):
            a.block_kind(24)
        with self.assertRaises(SD3ArchitectureError):
            a.block_kind(True)


class ComponentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.medium = inspect_sd3_architecture(synthetic_sd3_shapes("sd35-medium"))
        cls.sd3 = inspect_sd3_architecture(synthetic_sd3_shapes("sd3-medium"))
        cls.large = inspect_sd3_architecture(synthetic_sd3_shapes("sd35-large-topology"))

    def test_all_eleven_groups_classify_by_original_targets(self):
        fixtures = [
            ("mmdit", "joint_blocks.0.x_block.attn.qkv", "mmdit.x_attention.adapter"),
            ("mmdit", "joint_blocks.0.context_block.attn.proj", "mmdit.context_attention.adapter"),
            ("mmdit", "joint_blocks.12.x_block.attn2.qkv", "mmdit.x_extra_attention.adapter"),
            ("mmdit", "joint_blocks.0.x_block.mlp.fc1", "mmdit.x_mlp.adapter"),
            ("mmdit", "joint_blocks.3.context_block.mlp.fc2", "mmdit.context_mlp.adapter"),
            ("mmdit", "joint_blocks.0.x_block.adaLN_modulation.1", "mmdit.x_modulation.adapter"),
            ("mmdit", "joint_blocks.23.context_block.adaLN_modulation.1", "mmdit.context_modulation.adapter"),
            ("mmdit", "final_layer.linear", "mmdit.input_output.adapter"),
            ("clip_l", "text_model.encoder.layers.0.self_attn.q_proj", "clip_l.adapter"),
            ("clip_g", "text_model.encoder.layers.0.self_attn.k_proj", "clip_g.adapter"),
            ("t5xxl", "encoder.block.0.layer.0.SelfAttention.q", "t5xxl.adapter"),
        ]
        actual = {
            classify_sd3_adapter_target(self.medium, root, path, "torch.nn.Linear").component_id
            for root, path, component in fixtures
        }
        self.assertEqual(actual, set(SD3_DIAGNOSTIC_COMPONENTS))
        for root, path, component in fixtures:
            self.assertEqual(
                classify_sd3_adapter_target(self.medium, root, path, "torch.nn.Linear").component_id,
                component,
            )

    def test_report_counts_tensors_distinct_original_targets_and_layer_bands(self):
        records = [
            SD3AdapterTensor("mmdit", "joint_blocks.0.x_block.attn.qkv", "torch.nn.Linear", (4, 8)),
            SD3AdapterTensor("mmdit", "joint_blocks.0.x_block.attn.qkv", "torch.nn.Linear", (8, 4)),
            SD3AdapterTensor("mmdit", "joint_blocks.12.x_block.attn2.proj", "torch.nn.Linear", (5, 5)),
            SD3AdapterTensor("clip_l", "text_model.encoder.layers.1.self_attn.q_proj", "Linear", (3, 4)),
        ]
        report = audit_sd3_adapter_tensors(self.medium, records)
        self.assertTrue(report["read_only"])
        self.assertFalse(report["changes_parameter_policy"])
        self.assertEqual(len(report["components"]), 11)
        self.assertEqual(report["total_adapter_tensors"], 4)
        self.assertEqual(report["components"]["mmdit.x_attention.adapter"]["tensor_count"], 2)
        self.assertEqual(report["components"]["mmdit.x_attention.adapter"]["target_module_count"], 1)
        self.assertEqual(report["components"]["mmdit.x_attention.adapter"]["parameter_elements"], 64)
        self.assertEqual(report["blocks"][0]["block_kind"], "dual_attention")
        self.assertEqual(report["blocks"][0]["tensor_count"], 2)
        self.assertEqual(report["blocks"][12]["tensor_count"], 1)
        self.assertEqual(report["blocks"][23]["block_kind"], "context_pre_only")
        self.assertEqual(report["blocks"][23]["tensor_count"], 0)
        self.assertEqual(report, audit_sd3_adapter_tensors(self.medium, records))

    def test_medium_extra_attention_boundaries(self):
        for layer in (0, 12):
            self.assertEqual(
                classify_sd3_adapter_target(self.medium, "mmdit", f"joint_blocks.{layer}.x_block.attn2.qkv", "Linear").component_id,
                "mmdit.x_extra_attention.adapter",
            )
        for arch, layer in ((self.medium, 13), (self.sd3, 0), (self.large, 0)):
            with self.subTest(variant=arch.variant, layer=layer):
                with self.assertRaises(SD3ComponentAuditError):
                    classify_sd3_adapter_target(arch, "mmdit", f"joint_blocks.{layer}.x_block.attn2.qkv", "Linear")

    def test_last_context_preonly_and_no_late_attn_proj(self):
        for arch in (self.medium, self.sd3, self.large):
            last = arch.depth - 1
            self.assertEqual(
                classify_sd3_adapter_target(arch, "transformer",
                    f"joint_blocks.{last}.context_block.attn.qkv", "Linear").component_id,
                "mmdit.context_attention.adapter",
            )
            for name in ("mlp.fc1", "attn.proj"):
                with self.subTest(variant=arch.variant, name=name):
                    with self.assertRaisesRegex(SD3ComponentAuditError, "pre-only"):
                        classify_sd3_adapter_target(
                            arch, "mmdit", f"joint_blocks.{last}.context_block.{name}", "Linear"
                        )

    def test_explicit_input_output_and_reject_unknown(self):
        self.assertEqual(
            classify_sd3_adapter_target(self.medium, "mmdit", "x_embedder.proj", "Conv2d").component_id,
            "mmdit.input_output.adapter",
        )
        for path in ("context_embedder", "final_layer.linear", "t_embedder.mlp.0",
                     "y_embedder.mlp.2"):
            with self.subTest(path=path):
                self.assertEqual(
                    classify_sd3_adapter_target(self.medium, "mmdit", path, "Linear").component_id,
                    "mmdit.input_output.adapter",
                )
        for root, path, ty in (
            ("mmdit", "x_embedder.proj", "Linear"),
            ("mmdit", "final_layer.linear", "Conv2d"),
            ("mmdit", "joint_blocks.3.context_block.attn2.qkv", "Linear"),
            ("mmdit", "joint_blocks.3.x_block.attn.ln_k", "Linear"),
            ("mmdit", "joint_blocks.3.x_block.mlp.fc1", "Conv2d"),
            ("mmdit", "joint_blocks.3.x_block.attn.qkv", "RMSNorm"),
            ("mmdit", "joint_blocks.38.x_block.attn.qkv", "Linear"),
            ("mmdit", "joint_blocks.9.x_block.not_real", "Linear"),
            ("other", "joint_blocks.2.x_block.attn.qkv", "Linear"),
        ):
            with self.subTest(target=(root, path, ty)):
                with self.assertRaises(SD3ComponentAuditError):
                    classify_sd3_adapter_target(self.medium, root, path, ty)

    def test_schema_rejects_malformed_records(self):
        for item in (
            {"target_root": "mmdit", "target_path": "joint_blocks.0.x_block.attn.qkv", "target_type": "Linear"},
            {"target_root": "mmdit", "target_path": "joint_blocks.0.x_block.attn.qkv", "target_type": "Linear", "parameter_shape": [True, 4]},
            {"target_root": "mmdit", "target_path": "joint_blocks.0.x_block.attn.qkv", "target_type": "Linear", "parameter_shape": [0, 4]},
            {"target_root": "mmdit", "target_path": "joint_blocks.0.x_block.attn.qkv", "target_type": "Linear", "parameter_shape": [4, 4], "unused": 2},
        ):
            with self.subTest(record=item):
                with self.assertRaises(SD3ComponentAuditError):
                    audit_sd3_adapter_tensors(self.medium, [item])

    def test_scan_descriptors_are_read_only_and_alias_conflicts_fail(self):
        target = SimpleNamespace(
            root="mmdit", module_path="joint_blocks.12.x_block.attn2.qkv",
            module_type="torch.nn.Linear"
        )
        alias = SimpleNamespace(adapter_target=target)
        p = SimpleNamespace(parameter_id=42, shape=(8, 8), aliases=(alias, alias))
        result = audit_sd3_parameter_descriptors(self.medium, [p])
        self.assertEqual(result["components"]["mmdit.x_extra_attention.adapter"]["tensor_count"], 1)
        p2 = SimpleNamespace(parameter_id=42, shape=(8, 8), aliases=(alias,))
        with self.assertRaisesRegex(SD3ComponentAuditError, "Duplicate"):
            audit_sd3_parameter_descriptors(self.medium, [p, p2])
        conflict = SimpleNamespace(
            root="clip_l", module_path="other", module_type="torch.nn.Linear"
        )
        with self.assertRaisesRegex(SD3ComponentAuditError, "disagree"):
            audit_sd3_parameter_descriptors(self.medium, [
                SimpleNamespace(parameter_id=11, shape=(4, 4),
                    aliases=(alias, SimpleNamespace(adapter_target=conflict)))
            ])
        with self.assertRaisesRegex(SD3ComponentAuditError, "lacks Adapter"):
            audit_sd3_parameter_descriptors(self.medium, [
                SimpleNamespace(parameter_id=12, shape=(4, 4),
                    aliases=(SimpleNamespace(adapter_target=None),))
            ])


class CLIAuditTests(unittest.TestCase):
    def test_cli_shape_index_and_adapter_target_report(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            shapes = d / "shapes.json"
            targets = d / "targets.json"
            result = d / "audit.json"
            shapes.write_text(
                json.dumps({"shapes": synthetic_sd3_shapes("sd35-medium")}),
                encoding="utf-8"
            )
            targets.write_text(json.dumps([{
                "target_root": "mmdit",
                "target_path": "joint_blocks.0.x_block.attn.qkv",
                "target_type": "Linear",
                "parameter_shape": [8, 4],
            }]), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "audit_sd3_architecture.py"),
                 "--shape-index", str(shapes), "--adapter-targets", str(targets),
                 "--output", str(result)],
                cwd=ROOT, text=True, capture_output=True
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertFalse(payload["qualifies_full_bf16"])
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["architecture"]["variant"], "sd35-medium")
            self.assertEqual(payload["adapter_audit"]["total_adapter_tensors"], 1)
            self.assertEqual(
                payload["adapter_audit"]["components"]["mmdit.x_attention.adapter"]["tensor_count"], 1
            )

    def test_cli_rejects_unknown_architecture_without_writing_output(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            shapes = d / "shapes.json"
            result = d / "audit.json"
            shapes.write_text(json.dumps({"unknown": [1]}), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "audit_sd3_architecture.py"),
                 "--shape-index", str(shapes), "--output", str(result)],
                cwd=ROOT, text=True, capture_output=True
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("rejected input", proc.stderr)
            self.assertFalse(result.exists())

    def test_safetensors_header_reader_does_not_materialize_weights(self):
        from tools.audit_sd3_architecture import read_safetensors_shapes

        class Slice:
            def get_shape(self):
                return [1536, 16, 2, 2]

        class FakeSafeOpen:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def keys(self):
                return ["model.diffusion_model.x_embedder.proj.weight"]

            def get_slice(self, _key):
                return Slice()

            def get_tensor(self, _key):
                raise AssertionError("audit must not read tensor values")

        with patch.dict(sys.modules, {
            "safetensors": SimpleNamespace(safe_open=lambda *a, **kw: FakeSafeOpen())
        }):
            result = read_safetensors_shapes(Path("anything.safetensors"))
        self.assertEqual(
            result["model.diffusion_model.x_embedder.proj.weight"], (1536, 16, 2, 2)
        )


if __name__ == "__main__":
    unittest.main()
