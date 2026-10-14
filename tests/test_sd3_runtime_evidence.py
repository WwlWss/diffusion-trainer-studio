"""Host-only contract tests for the exact-head SD3B GPU evidence runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "run_sd3_runtime_gpu_evidence.py"


def _load():
    spec = importlib.util.spec_from_file_location("sd3_runtime_gpu_evidence", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SD3RuntimeEvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load()

    def _manifest(self, temp):
        root = Path(temp)
        base = root / "sd3.5_medium.safetensors"
        base.write_bytes(b"fake model stub for host contract testing")
        fresh = root / "fresh"
        resume = root / "resume"
        common = [
            "scripts/dev/sd3_train_network.py",
            "--pretrained_model_name_or_path", str(base),
            "--max_train_steps", "2",
            "--save_every_n_steps", "1",
            "--save_state",
            "--mixed_precision", "bf16",
            "--optimizer_type", "AdamW",
            "--network_module", "networks.lora_sd3",
            "--network_train_unet_only",
            "--gradient_accumulation_steps", "2",
        ]
        return {
            "schema": self.tool.SCHEMA, "version": 1,
            "fresh_command": [*common, "--output_dir", str(fresh)],
            "resume_command": [
                *common, "--resume", str(fresh / "t-step00000001-state"),
                "--output_dir", str(resume),
            ],
            "fresh_checkpoint_dir": str(fresh / "t-step00000001-state"),
            "resume_checkpoint_dir": str(resume / "t-step00000002-state"),
            "fresh_adapter_path": str(fresh / "t-step00000001.safetensors"),
            "resume_adapter_path": str(resume / "t-step00000002.safetensors"),
        }

    def test_reference_manifest_shape(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._manifest(td)
            result = self.tool.validate_manifest(data)
            self.assertEqual(result["fresh"]["--gradient_accumulation_steps"], "2")

    def test_rejects_unqualified_features_and_wrong_step_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            for key, value in (("--full_bf16", True), ("--max_train_epochs", "2"), ("--fp8_base", True)):
                with self.subTest(feature=key):
                    data = self._manifest(td)
                    for side in ("fresh_command", "resume_command"):
                        data[side] += [key] if value is True else [key, value]
                    with self.assertRaises(self.tool.SD3RuntimeEvidenceError):
                        self.tool.validate_manifest(data)
            data = self._manifest(td)
            data["fresh_command"][data["fresh_command"].index("2")]="5"
            with self.assertRaises(self.tool.SD3RuntimeEvidenceError):
                self.tool.validate_manifest(data)

    def test_rejects_fake_resume_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._manifest(td)
            idx = data["resume_command"].index("--resume") + 1
            data["resume_command"][idx] = str(Path(td) / "something-else")
            with self.assertRaisesRegex(self.tool.SD3RuntimeEvidenceError, "exact Step-1"):
                self.tool.validate_manifest(data)
            data = self._manifest(td)
            (Path(td) / "fresh").mkdir()
            with self.assertRaisesRegex(self.tool.SD3RuntimeEvidenceError, "already exists"):
                self.tool.validate_manifest(data)

    def test_requires_absolute_paths_outside_repository(self):
        with self.assertRaisesRegex(self.tool.SD3RuntimeEvidenceError, "absolute"):
            self.tool._outside_repo("relative-output.json", name="Evidence JSON")
        with self.assertRaisesRegex(self.tool.SD3RuntimeEvidenceError, "outside"):
            self.tool._outside_repo(str(ROOT / "inside.json"), name="Evidence JSON")

    def test_forward_and_step_evidence_not_qualification(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('"qualifies_full_bf16": False', source)
        self.assertIn('"production_qualification_mutated": False', source)
        self.assertIn('_compare_adapter_weights', source)
        self.assertIn('optimizer_state_sha256', source)


if __name__ == "__main__":
    unittest.main()
