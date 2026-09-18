import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mikazuki import gpu_runtime


ROOT = Path(__file__).resolve().parents[1]


class _FakeCuda:
    def __init__(self, available=True, names=None):
        self._available = available
        self._names = names or []

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._names) if self._available else 0

    def get_device_name(self, index):
        return self._names[index]


class TrainingGpuRuntimeTests(unittest.TestCase):
    def _torch(self, available=True, names=None):
        return SimpleNamespace(cuda=_FakeCuda(available=available, names=names))

    def test_cuda_preflight_rejects_cpu_only_runtime(self):
        with patch.dict(sys.modules, {"torch": self._torch(False, [])}):
            with self.assertRaisesRegex(RuntimeError, "will not fall back to CPU training"):
                gpu_runtime.validate_cuda_training_runtime()

    def test_cuda_preflight_accepts_gpu_and_normalizes_selection(self):
        with patch.dict(sys.modules, {"torch": self._torch(True, ["GPU 0", "GPU 1"])}):
            selected, names = gpu_runtime.validate_cuda_training_runtime(["1"])
        self.assertEqual(selected, ["1"])
        self.assertEqual(names, ["GPU 1"])

    def test_cuda_preflight_rejects_out_of_range_gpu_id(self):
        with patch.dict(sys.modules, {"torch": self._torch(True, ["GPU 0"])}):
            with self.assertRaisesRegex(RuntimeError, "outside the visible CUDA device range"):
                gpu_runtime.validate_cuda_training_runtime(["1"])

    def test_repo_accelerate_config_forbids_cpu(self):
        config = (ROOT / "config" / "accelerate-gpu.yaml").read_text(encoding="utf-8")
        self.assertIn("use_cpu: false", config)
        self.assertIn("gpu_ids: all", config)

    def test_common_launcher_always_uses_repo_gpu_config(self):
        launcher = (ROOT / "mikazuki" / "training_launcher.py").read_text(encoding="utf-8")
        runtime = (ROOT / "mikazuki" / "gpu_runtime.py").read_text(encoding="utf-8")
        self.assertIn("validate_cuda_training_runtime(gpu_ids)", launcher)
        self.assertIn('"--config_file",\n        str(ACCELERATE_GPU_CONFIG)', launcher)
        self.assertIn("Training was not started; DTS will not fall back to CPU training", runtime)

    def test_every_webui_training_backend_uses_the_common_launch_path(self):
        api_source = (ROOT / "mikazuki" / "app" / "api.py").read_text(encoding="utf-8")
        process_source = (ROOT / "mikazuki" / "process.py").read_text(encoding="utf-8")
        training_api_source = (ROOT / "mikazuki" / "app" / "training_api.py").read_text(encoding="utf-8")

        expected = {
            "sd-lora",
            "sdxl-lora",
            "sd-dreambooth",
            "sdxl-finetune",
            "sd3-lora",
            "flux-lora",
            "chroma-lora",
            "flux-finetune",
            "anima-lora",
            "anima-finetune",
        }
        for backend in expected:
            self.assertIn(f'"{backend}"', api_source, backend)

        self.assertIn("run_prepared_train", process_source)
        self.assertIn("run_prepared_train(", training_api_source)


if __name__ == "__main__":
    unittest.main()
