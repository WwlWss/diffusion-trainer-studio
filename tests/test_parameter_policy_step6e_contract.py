from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ParameterPolicyStep6EContractTests(unittest.TestCase):
    def test_integrated_repository_trainers_use_hardened_lifecycle(self):
        paths = (
            "scripts/stable/train_network.py",
            "scripts/dev/train_network.py",
            "scripts/stable/train_db.py",
            "scripts/stable/sdxl_train.py",
            "scripts/dev/flux_train.py",
        )
        for relpath in paths:
            with self.subTest(path=relpath):
                source = (ROOT / relpath).read_text(encoding="utf-8")
                self.assertIn(
                    "parameter_policy_session.finalize_after_prepare(",
                    source,
                )
                self.assertIn('phase="post_resume"', source)
                self.assertIn('phase="epoch_start"', source)
                self.assertNotIn(
                    "parameter_policy_session.audit_after_prepare(\n",
                    source,
                )
                self.assertNotIn(
                    "parameter_policy_session.register_checkpoint_manifest(\n",
                    source,
                )

    def test_network_trainers_use_session_owned_model_metadata(self):
        for relpath in (
            "scripts/stable/train_network.py",
            "scripts/dev/train_network.py",
        ):
            with self.subTest(path=relpath):
                source = (ROOT / relpath).read_text(encoding="utf-8")
                self.assertIn(
                    "metadata.update(parameter_policy_session.model_metadata())",
                    source,
                )
                self.assertNotIn(
                    'metadata["ss_dts_parameter_policy_hash"] =',
                    source,
                )

    def test_full_trainers_expose_unified_model_metadata_to_savers(self):
        trainer_paths = (
            "scripts/stable/train_db.py",
            "scripts/stable/sdxl_train.py",
            "scripts/dev/flux_train.py",
        )
        for relpath in trainer_paths:
            with self.subTest(path=relpath):
                source = (ROOT / relpath).read_text(encoding="utf-8")
                self.assertIn(
                    "args._dts_parameter_policy_model_metadata =",
                    source,
                )
                self.assertIn(
                    "parameter_policy_session.model_metadata()",
                    source,
                )

        saver_paths = (
            "scripts/stable/library/train_util.py",
            "scripts/stable/library/sdxl_train_util.py",
            "scripts/dev/library/flux_train_utils.py",
        )
        for relpath in saver_paths:
            with self.subTest(path=relpath):
                source = (ROOT / relpath).read_text(encoding="utf-8")
                self.assertIn(
                    '_dts_parameter_policy_model_metadata',
                    source,
                )

    def test_staged_anima_patch_uses_same_hardening_api(self):
        source = (
            ROOT / "tools" / "apply_anima_parameter_policy_runtime_patch.py"
        ).read_text(encoding="utf-8")
        self.assertIn("finalize_after_prepare", source)
        self.assertIn('phase=\\\"post_resume\\\"', source)
        self.assertIn('phase=\\\"epoch_start\\\"', source)
        self.assertIn("model_metadata()", source)

    def test_manifest_v2_and_step6f_release_matrix_remain_explicit(self):
        trainer = (
            ROOT / "mikazuki" / "parameter_policy_trainer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION = 2",
            trainer,
        )
        request = (
            ROOT / "mikazuki" / "training_request.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PARAMETER_POLICY_RUNTIME_TRAIN_TYPES",
            request,
        )
        matrix = (
            ROOT / "mikazuki" / "parameter_policy_matrix.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PARAMETER_POLICY_BACKEND_MATRIX_VERSION",
            matrix,
        )


if __name__ == "__main__":
    unittest.main()
