"""C4-B-FSR-G3: actual LoRA writer Epoch-to-saved-Progress coupling.

Three pinned real CPU LoRANetwork tests only. No fake original serializer,
no CUDA representation, and no production checkpoint receipt. The negative
control must fail without a writer/progress binding contract.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.test_sd3_full_bf16_worker_c4b_cpu import _production_one_lora
from tools.sd3_full_bf16_worker_save_verifier import (
    C4BLoRAWriteObserver, verify_lora_step_metadata,
)
from tools import sd3_full_bf16_worker_save_verifier as verifier


class C4BFSRG3OriginalEpochCpu(unittest.TestCase):
    def _real_writer(self, root, *, epoch):
        network = _production_one_lora()
        writer = C4BLoRAWriteObserver(
            network=network, output_dir=root, output_name="verified",
            step_provider=lambda: 1,
        )
        writer.install()
        self.addCleanup(writer.close)
        file = root / "verified-step00000001.safetensors"
        network.save_weights(
            str(file), None, {"ss_steps": "1", "ss_epoch": str(epoch)}
        )
        return file, writer.completed_step_write(logical_step=1)

    def test_g301_correct_bytes_wrong_epoch_must_not_match_saved_progress(self):
        with tempfile.TemporaryDirectory() as dirname:
            _, obs = self._real_writer(Path(dirname), epoch=9)
            verifier_fn = getattr(verifier, "verify_lora_writer_progress", None)
            self.assertTrue(
                callable(verifier_fn),
                "C4-B-FSR-G3 CONTRACT ABSENT: source-bound saved-Progress "
                "Epoch validator missing",
            )
            try:
                verifier_fn(obs, expected_epoch=1)
            except ValueError as exc:
                self.assertRegex(str(exc), "epoch|Epoch|progress|Step")
            else:
                self.fail(
                    "C4-B-FSR-G3 CONTRACT ABSENT: genuine Writer Epoch 9 "
                    "was falsely accepted as saved Progress Epoch 1",
                )

    def test_g302_actual_original_writer_and_saved_epoch_match(self):
        with tempfile.TemporaryDirectory() as dirname:
            _, obs = self._real_writer(Path(dirname), epoch=1)
            verifier_fn = getattr(verifier, "verify_lora_writer_progress", None)
            self.assertTrue(
                callable(verifier_fn),
                "C4-B-FSR-G3 CONTRACT ABSENT: real positive Writer/Progress "
                "Epoch validator missing",
            )
            result = verifier_fn(obs, expected_epoch=1)
            self.assertEqual(result["ss_steps"], 1)
            self.assertEqual(result["ss_epoch"], 1)
            self.assertEqual(result["file_sha256"], obs.file_sha256)

    def test_g303_existing_metadata_parser_rejects_foreign_epoch(self):
        with tempfile.TemporaryDirectory() as dirname:
            path, _obs = self._real_writer(Path(dirname), epoch=9)
            with self.assertRaisesRegex(ValueError, "epoch|Epoch"):
                verify_lora_step_metadata(
                    path, expected_step=1, expected_epoch=1,
                )


if __name__ == "__main__":
    unittest.main()
