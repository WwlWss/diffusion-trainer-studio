import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mikazuki import launch_utils


ROOT = Path(__file__).resolve().parents[1]


class OnnxRuntimeBootstrapTests(unittest.TestCase):
    def _clean_ort_env(self):
        return patch.dict(
            os.environ,
            {
                "ONNXRUNTIME_PACKAGE": "",
                "ONNXRUNTIME_VERSION": "",
                "ONNXRUNTIME_INDEX_URL": "",
            },
            clear=False,
        )

    def test_cuda12_defaults_to_single_pinned_gpu_runtime(self):
        with self._clean_ort_env(), \
                patch.object(launch_utils.sys, "platform", "win32"), \
                patch.object(launch_utils, "_torch_cuda_major", return_value=12):
            package, version, index = launch_utils._onnxruntime_target()

        self.assertEqual(package, "onnxruntime-gpu")
        self.assertEqual(version, launch_utils.DEFAULT_ONNXRUNTIME_VERSION)
        self.assertIsNone(index or None)

    def test_cuda11_defaults_to_cpu_runtime_instead_of_incompatible_gpu_wheel(self):
        with self._clean_ort_env(), \
                patch.object(launch_utils.sys, "platform", "win32"), \
                patch.object(launch_utils, "_torch_cuda_major", return_value=11):
            package, version, _ = launch_utils._onnxruntime_target()

        self.assertEqual(package, "onnxruntime")
        self.assertEqual(version, launch_utils.DEFAULT_ONNXRUNTIME_VERSION)

    def test_expert_override_can_select_package_version_and_index(self):
        with patch.dict(
            os.environ,
            {
                "ONNXRUNTIME_PACKAGE": "onnxruntime-gpu",
                "ONNXRUNTIME_VERSION": "9.9.9",
                "ONNXRUNTIME_INDEX_URL": "https://example.invalid/simple",
            },
            clear=False,
        ), patch.object(launch_utils.sys, "platform", "win32"):
            self.assertEqual(
                launch_utils._onnxruntime_target(),
                ("onnxruntime-gpu", "9.9.9", "https://example.invalid/simple"),
            )

    def test_mixed_cpu_gpu_install_is_removed_before_one_target_is_installed(self):
        versions = {"onnxruntime": "1.30.0", "onnxruntime-gpu": "1.30.0"}
        with patch.object(
            launch_utils,
            "_onnxruntime_target",
            return_value=("onnxruntime-gpu", "1.24.1", None),
        ), patch.object(
            launch_utils,
            "_installed_version",
            side_effect=lambda package: versions.get(package),
        ), patch.object(launch_utils, "run_pip") as run_pip, patch.object(
            launch_utils, "pip_install"
        ) as pip_install:
            launch_utils.setup_onnxruntime()

        run_pip.assert_called_once_with(
            "uninstall -y onnxruntime onnxruntime-gpu",
            "old ONNX Runtime packages",
            live=True,
        )
        pip_install.assert_called_once_with(
            "onnxruntime-gpu", "1.24.1", index_url=None, live=True
        )

    def test_correct_single_gpu_install_is_left_untouched(self):
        versions = {"onnxruntime": None, "onnxruntime-gpu": "1.24.1"}
        with patch.object(
            launch_utils,
            "_onnxruntime_target",
            return_value=("onnxruntime-gpu", "1.24.1", None),
        ), patch.object(
            launch_utils,
            "_installed_version",
            side_effect=lambda package: versions.get(package),
        ), patch.object(launch_utils, "run_pip") as run_pip, patch.object(
            launch_utils, "pip_install"
        ) as pip_install:
            launch_utils.setup_onnxruntime()

        run_pip.assert_not_called()
        pip_install.assert_not_called()


class RuntimeDependencyFileContractTests(unittest.TestCase):
    def test_requirements_pin_tensorboard_and_shared_protobuf_intersection(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        active = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]

        self.assertIn("tensorboard==2.20.0", active)
        self.assertIn("protobuf==3.20.3", active)
        self.assertIn("open-clip-torch==2.20.0", active)
        self.assertIn("wandb==0.16.2", active)
        self.assertFalse(any(line.startswith("onnxruntime") for line in active))
        self.assertEqual(active.count("prodigy-plus-schedule-free==1.9.2"), 1)

    def test_gui_honors_skip_prepare_onnxruntime(self):
        source = (ROOT / "gui.py").read_text(encoding="utf-8")
        self.assertIn(
            "prepare_onnxruntime=not args.skip_prepare_onnxruntime",
            source,
        )

    def test_windows_launchers_keep_cache_local_without_overriding_global_hf_auth(self):
        for filename in ("run_gui.ps1", "install.ps1", "install-cn.ps1"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("HF_HUB_CACHE", source, filename)
            self.assertNotIn('$Env:HF_HOME = "huggingface"', source, filename)

    def test_installers_check_dependency_consistency(self):
        for filename in ("install.ps1", "install-cn.ps1", "install.bash"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("pip check", source, filename)

    def test_linux_installer_does_not_advertise_obsolete_torch_112_branches(self):
        source = (ROOT / "install.bash").read_text(encoding="utf-8")
        self.assertNotIn("torch==1.12.1", source)
        self.assertIn("cuda_major_version == 11 && cuda_minor_version >= 8", source)

    def test_animetimm_is_cache_first_and_uses_verified_cuda_helper(self):
        source = (
            ROOT / "mikazuki/tagger/interrogators/animetimm.py"
        ).read_text(encoding="utf-8")
        self.assertIn("local_files_only=True", source)
        self.assertIn("GatedRepoError", source)
        self.assertIn("create_cuda_onnx_session", source)
        self.assertNotIn('add_session_config_entry("session.disable_cpu_ep_fallback"', source)

    def test_all_local_onnx_taggers_use_verified_cuda_session_helper(self):
        helper = (ROOT / "mikazuki/tagger/interrogators/onnx_gpu.py").read_text(encoding="utf-8")
        self.assertIn('session.record_ep_graph_assignment_info", "1"', helper)
        self.assertIn('providers=["CUDAExecutionProvider", "CPUExecutionProvider"]', helper)
        self.assertIn("get_provider_graph_assignment_info", helper)
        self.assertIn("cuda_compute_nodes", helper)
        self.assertIn("assigned no substantial tagger compute to CUDA", helper)
        self.assertNotIn('add_session_config_entry("session.disable_cpu_ep_fallback"', helper)

        for filename in ("animetimm.py", "wd14.py", "cl.py", "pixai.py", "danbooru_query.py"):
            source = (ROOT / "mikazuki/tagger/interrogators" / filename).read_text(encoding="utf-8")
            self.assertIn("create_cuda_onnx_session", source, filename)
            self.assertNotIn('add_session_config_entry("session.disable_cpu_ep_fallback"', source, filename)


if __name__ == "__main__":
    unittest.main()
