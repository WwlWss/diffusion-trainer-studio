from __future__ import annotations

import importlib.util
import unittest


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if _TORCH_AVAILABLE:
    import torch
    from tools.parameter_policy_execution_gpu_runtime import (
        gradient_evidence,
        state_fingerprint,
        tensor_fingerprint,
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is not installed")
class ExecutionGpuRuntimeEvidenceTests(unittest.TestCase):
    def test_bfloat16_tensor_fingerprint_is_stable_and_sensitive(self):
        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        first = tensor_fingerprint(tensor)
        second = tensor_fingerprint(tensor.clone())
        self.assertEqual(first, second)
        self.assertEqual(first["dtype"], "torch.bfloat16")
        self.assertTrue(first["finite"])
        changed = tensor.clone()
        changed[0, 0] = 3.0
        self.assertNotEqual(first["sha256"], tensor_fingerprint(changed)["sha256"])

    def test_state_fingerprint_is_mapping_order_independent(self):
        left = state_fingerprint({"b": 2, "a": torch.tensor([1.0])})
        right = state_fingerprint({"a": torch.tensor([1.0]), "b": 2})
        self.assertEqual(left["sha256"], right["sha256"])

    def test_gradient_evidence_records_missing_and_present_gradients(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        first.grad = torch.tensor([0.5])
        evidence = gradient_evidence([first, second])
        self.assertIsNotNone(evidence["0"])
        self.assertIsNone(evidence["1"])


if __name__ == "__main__":
    unittest.main()
