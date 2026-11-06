"""Fix C2 pinned real CPU checks: parser interoperability and CUDA fail-closed.

Tests never train or promote qualifications, and use real production argparse.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import accelerate

from tests.test_sd3_full_bf16_gpu_evidence import _generate, R1, R2
from tools import sd3_full_bf16_evidence_support as host
from tools import sd3_full_bf16_evidence_worker as worker


class C2RealCpuContract(unittest.TestCase):
    def test_pinned_environment_does_not_permit_cpu_worker(self):
        self.assertEqual(torch.__version__.split("+", 1)[0], "2.7.0")
        self.assertEqual(accelerate.__version__, "1.6.0")
        self.assertFalse(torch.cuda.is_available())
        for case_id in (R1, R2):
            with self.subTest(case=case_id):
                with self.assertRaisesRegex(Exception, "real CUDA device"):
                    worker.validate_runtime_environment(case_id)

    def test_real_sd3_parser_accepts_only_reviewed_reference_namespace(self):
        for case_id in (R1, R2):
            with self.subTest(case=case_id), tempfile.TemporaryDirectory() as td:
                fixture = _generate(Path(td), case_id)
                manifest = host._read_json(Path(fixture["manifest"]))
                args = worker._production_namespace(SimpleNamespace(
                    case_id=case_id, trainer_argv=manifest["fresh_command"],
                ))
                spec = host.get_case_spec(case_id)
                self.assertIs(args.full_bf16, True)
                self.assertEqual(args.mixed_precision, "bf16")
                self.assertEqual(args.max_train_steps, 2)
                self.assertEqual(args.gradient_accumulation_steps,
                                 spec["gradient_accumulation_steps"])
                self.assertIsNone(args.save_precision)
                self.assertIsNone(args.scale_weight_norms)
                self.assertIsNone(args.config_file)

    def test_extra_unreviewed_trainer_option_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = _generate(Path(td), R1)
            manifest = host._read_json(Path(fixture["manifest"]))
            command = list(manifest["fresh_command"])
            command.extend(["--save_precision", "float"])
            with self.assertRaisesRegex(ValueError, "Unknown|unreviewed"):
                worker._production_namespace(
                    SimpleNamespace(case_id=R1, trainer_argv=command),
                )


if __name__ == "__main__":
    unittest.main()
