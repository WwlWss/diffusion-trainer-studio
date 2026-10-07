from __future__ import annotations

import unittest

from tools.parameter_policy_gpu_qualification_environment import (
    QUALIFICATION_MUON_PROVIDER_VERSION,
    QUALIFICATION_TORCH_BASE_VERSION,
    QUALIFICATION_TORCHVISION_VERSION,
    QualificationEnvironmentError,
    validate_qualification_environment_snapshot,
)


def _snapshot() -> dict:
    return {
        "python_major_minor": [3, 11],
        "torch_full_version": "2.7.0+cu128",
        "torch_base_version": "2.7.0",
        "torchvision_version": "0.22.0",
        "pytorch_optimizer_version": "3.10.0",
        "requirements_sha256": "abc123",
        "cuda_available": True,
        "bf16_supported": True,
    }


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
