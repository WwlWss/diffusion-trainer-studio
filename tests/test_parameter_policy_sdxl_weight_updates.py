"""Host-only SDXL real adapter-weight evidence tests (no torch installation needed)."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.parameter_policy_backend_feature_gpu_support import (
    BackendFeatureGpuMatrixError,
    compare_sdxl_lora_weight_updates,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeTensor:
    def __init__(self, value: object, shape: tuple[int, ...] = (4, 4),
                 dtype: str = "bfloat16", finite: bool = True):
        self.value = value
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.finite = finite


class _FakeScalar:
    def __init__(self, value: bool):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _FakeSafeOpen:
    def __init__(self, contents: dict[str, _FakeTensor]):
        self.contents = contents

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def keys(self):
        return tuple(self.contents)

    def get_tensor(self, key):
        return self.contents[key]


def _fixtures():
    # Real stock-LoRA safetensors use lora_te1_, lora_te2_ and lora_unet_.
    keys = {
        "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_up.weight":
            (4, 4),
        "lora_unet_input_blocks_4_0_conv1.lora_down.weight":
            (4, 320, 3, 3),
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight":
            (4, 4),
        "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight":
            (4, 4),
    }
    before = {key: _FakeTensor(0, shape) for key, shape in keys.items()}
    after = {key: _FakeTensor(1, shape) for key, shape in keys.items()}
    return before, after


class SdXlPhysicalAdapterAdvancementTests(unittest.TestCase):
    def _compare(self, *, before=None, after=None, require_conv=True):
        if before is None or after is None:
            before, after = _fixtures()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = root / "sdxl-step00000001-state"
            resume = root / "sdxl-step00000002-state"
            left = fresh.with_name("sdxl-step00000001.safetensors")
            right = resume.with_name("sdxl-step00000002.safetensors")
            left.write_bytes(b"fake-step-one")
            right.write_bytes(b"fake-step-two")
            objects = {
                str(left): before,
                str(right): after,
            }
            fake_torch = types.ModuleType("torch")
            fake_torch.equal = lambda a, b: (
                a.value == b.value and a.shape == b.shape and a.dtype == b.dtype
            )
            fake_torch.isfinite = lambda tensor: _FakeScalar(tensor.finite)
            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = lambda name, **kw: _FakeSafeOpen(
                objects[str(name)]
            )
            with patch.dict(sys.modules, {
                "torch": fake_torch,
                "safetensors": fake_safetensors,
            }):
                return compare_sdxl_lora_weight_updates(
                    fresh, resume, require_conv_fallback=require_conv
                )

    def test_all_three_roots_and_conv_fallback_have_real_weight_advancement(self):
        evidence = self._compare()
        self.assertTrue(evidence["weight_keys_match"])
        self.assertEqual(evidence["model_family"], "sdxl-base")
        self.assertEqual(
            evidence["required_groups"], ["unet", "te1", "te2", "conv3x3"]
        )
        self.assertTrue(
            all(evidence["changed_tensor_counts"][component] > 0
                for component in evidence["required_groups"])
        )
        self.assertEqual(len(evidence["fresh_weights_sha256"]), 64)
        self.assertEqual(len(evidence["resume_weights_sha256"]), 64)

    def test_missing_any_required_component_update_fails_closed(self):
        for target in ("lora_unet_", "lora_te1_", "lora_te2_", "conv1"):
            with self.subTest(target=target):
                before, after = _fixtures()
                for key in after:
                    if target in key:
                        after[key].value = before[key].value
                if target == "lora_unet_":
                    # Also zero the Conv change, because both are U-Net.
                    pass
                with self.assertRaisesRegex(
                    BackendFeatureGpuMatrixError, "no adapter parameter advancement"
                ):
                    self._compare(before=before, after=after)

    def test_different_keys_dtypes_or_nonfinite_values_fail_closed(self):
        before, after = _fixtures()
        after.pop(next(key for key in after if key.startswith("lora_te2_")))
        with self.assertRaisesRegex(BackendFeatureGpuMatrixError, "key set mismatch"):
            self._compare(before=before, after=after)

        before, after = _fixtures()
        after[next(key for key in after if key.startswith("lora_te1_"))].dtype = "float32"
        with self.assertRaisesRegex(BackendFeatureGpuMatrixError, "dtype/finite"):
            self._compare(before=before, after=after)

        before, after = _fixtures()
        after[next(key for key in after if key.startswith("lora_te2_"))].finite = False
        with self.assertRaisesRegex(BackendFeatureGpuMatrixError, "dtype/finite"):
            self._compare(before=before, after=after)

    def test_conv_only_required_in_muon_fallback_case(self):
        before, after = _fixtures()
        conv = next(key for key in after if "conv1" in key)
        after[conv].value = before[conv].value
        valid = self._compare(before=before, after=after, require_conv=False)
        self.assertEqual(valid["changed_tensor_counts"]["conv3x3"], 0)
        with self.assertRaisesRegex(BackendFeatureGpuMatrixError, "conv3x3"):
            self._compare(before=before, after=after, require_conv=True)


class SdXlLoRaConstructionSourceTests(unittest.TestCase):
    def test_conv_lora_target_list_is_not_mutated_across_network_builds(self):
        path = ROOT / "scripts" / "stable" / "networks" / "lora.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn(
            "target_modules = list(LoRANetwork.UNET_TARGET_REPLACE_MODULE)",
            source,
        )
        self.assertNotIn(
            "target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE\n",
            source,
        )
        self.assertTrue(tree.body)


if __name__ == "__main__":
    unittest.main()
