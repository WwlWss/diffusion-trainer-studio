from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.parameter_policy_gpu_qualification_environment import (
    QUALIFICATION_MUON_PROVIDER_VERSION,
    QUALIFICATION_TORCH_BASE_VERSION,
    QUALIFICATION_TORCHVISION_VERSION,
    QualificationEnvironmentError,
    qualification_environment_snapshot,
    validate_qualification_environment_snapshot,
)


def _pinned_packages() -> dict[str, str]:
    return {
        "accelerate": "1.6.0",
        "diffusers": "0.32.1",
        "pytorch-optimizer": "3.10.0",
        "transformers": "4.54.1",
    }


def _snapshot() -> dict:
    pins = _pinned_packages()
    return {
        "python_major_minor": [3, 11],
        "torch_full_version": "2.7.0+cu128",
        "torch_base_version": "2.7.0",
        "torchvision_version": "0.22.0",
        "pytorch_optimizer_version": "3.10.0",
        "requirements_sha256": "abc123",
        "requirements_exact_pins": dict(pins),
        "installed_requirement_versions": dict(pins),
        "cuda_available": True,
        "bf16_supported": True,
    }


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_bf16_supported():
        return True


class _FakeTorch:
    __version__ = "2.7.0+cu128"
    cuda = _FakeCuda()


class QualificationEnvironmentTests(unittest.TestCase):
    def test_pinned_environment_passes(self):
        contract = validate_qualification_environment_snapshot(
            _snapshot(),
            require_muon=True,
        )
        self.assertEqual(contract["status"], "pass")
        self.assertEqual(contract["python_major_minor"], "3.11")
        self.assertEqual(
            contract["torch_base_version"],
            QUALIFICATION_TORCH_BASE_VERSION,
        )
        self.assertEqual(
            contract["torchvision_version"],
            QUALIFICATION_TORCHVISION_VERSION,
        )
        self.assertEqual(
            contract["pytorch_optimizer_version"],
            QUALIFICATION_MUON_PROVIDER_VERSION,
        )
        self.assertEqual(
            contract["requirements_exact_pins"],
            _pinned_packages(),
        )
        self.assertEqual(
            contract["installed_requirement_versions"],
            _pinned_packages(),
        )

    def test_python_314_fails_closed(self):
        snapshot = _snapshot()
        snapshot["python_major_minor"] = [3, 14]
        with self.assertRaisesRegex(
            QualificationEnvironmentError,
            "Python 3.11 is required",
        ):
            validate_qualification_environment_snapshot(
                snapshot,
                require_muon=True,
            )

    def test_wrong_torch_or_torchvision_fails_closed(self):
        for field, value, message in (
            ("torch_base_version", "2.8.0", "torch base version 2.7.0"),
            ("torchvision_version", "0.23.0", "torchvision 0.22.0"),
        ):
            with self.subTest(field=field):
                snapshot = _snapshot()
                snapshot[field] = value
                with self.assertRaisesRegex(
                    QualificationEnvironmentError,
                    message,
                ):
                    validate_qualification_environment_snapshot(
                        snapshot,
                        require_muon=True,
                    )

    def test_wrong_muon_provider_fails_closed(self):
        snapshot = _snapshot()
        snapshot["pytorch_optimizer_version"] = "3.11.0"
        with self.assertRaisesRegex(
            QualificationEnvironmentError,
            "pytorch-optimizer 3.10.0",
        ):
            validate_qualification_environment_snapshot(
                snapshot,
                require_muon=True,
            )

    def test_wrong_pinned_production_package_fails_closed(self):
        snapshot = _snapshot()
        snapshot["installed_requirement_versions"]["accelerate"] = "1.7.0"
        with self.assertRaisesRegex(
            QualificationEnvironmentError,
            "accelerate 1.6.0 is required",
        ):
            validate_qualification_environment_snapshot(
                snapshot,
                require_muon=True,
            )

    def test_missing_pinned_production_package_fails_closed(self):
        snapshot = _snapshot()
        snapshot["installed_requirement_versions"]["diffusers"] = None
        with self.assertRaisesRegex(
            QualificationEnvironmentError,
            "diffusers 0.32.1 is required",
        ):
            validate_qualification_environment_snapshot(
                snapshot,
                require_muon=True,
            )

    def test_requirements_exact_pins_and_extras_are_discovered_from_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requirements.txt").write_text(
                "\n".join(
                    [
                        "accelerate==1.6.0",
                        "diffusers[torch]==0.32.1",
                        "requests",
                        "# comment",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            versions = {
                "accelerate": "1.6.0",
                "diffusers": "0.32.1",
                "torchvision": "0.22.0",
                "pytorch-optimizer": "3.10.0",
            }
            with patch(
                "tools.parameter_policy_gpu_qualification_environment._package_version",
                side_effect=lambda name: versions.get(name),
            ):
                snapshot = qualification_environment_snapshot(
                    repo_root=root,
                    torch_module=_FakeTorch,
                    require_muon=True,
                )

        self.assertEqual(
            snapshot["requirements_exact_pins"],
            {
                "accelerate": "1.6.0",
                "diffusers": "0.32.1",
            },
        )
        self.assertEqual(
            snapshot["installed_requirement_versions"],
            {
                "accelerate": "1.6.0",
                "diffusers": "0.32.1",
            },
        )

    def test_cuda_and_bf16_are_required(self):
        for field in ("cuda_available", "bf16_supported"):
            with self.subTest(field=field):
                snapshot = _snapshot()
                snapshot[field] = False
                with self.assertRaises(QualificationEnvironmentError):
                    validate_qualification_environment_snapshot(
                        snapshot,
                        require_muon=True,
                    )

    def test_requirements_hash_is_required(self):
        snapshot = _snapshot()
        snapshot["requirements_sha256"] = ""
        with self.assertRaisesRegex(
            QualificationEnvironmentError,
            "requirements.txt SHA256",
        ):
            validate_qualification_environment_snapshot(
                snapshot,
                require_muon=True,
            )


if __name__ == "__main__":
    unittest.main()
