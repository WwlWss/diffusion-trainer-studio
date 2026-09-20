from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_runtime import materialize_anima_runtime_tree
from tools import apply_anima_parameter_policy_metadata_patch as metadata_patch


class ParameterPolicyAnimaStagingTests(unittest.TestCase):
    def _source_clean(self, source: Path) -> str:
        return subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_runtime_feature_materializes_isolated_tree(self):
        source = Path("sd-scripts").resolve()
        before = self._source_clean(source)
        self.assertEqual(before, "")

        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("parameter_policy_runtime",),
                source_dir=source,
                cache_root=temp_dir,
            )
            self.assertNotEqual(tree.resolve(), source)
            self.assertTrue((tree / "library/dts_parameter_policy_bridge.py").is_file())
            self.assertIn(
                'return "anima-lora"',
                (tree / "anima_train_network.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                '"anima-finetune"',
                (tree / "anima_train.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "register_checkpoint_manifest",
                (tree / "train_network.py").read_text(encoding="utf-8"),
            )
            second = materialize_anima_runtime_tree(
                ("parameter_policy_runtime",),
                source_dir=source,
                cache_root=temp_dir,
            )
            self.assertEqual(tree, second)

        self.assertEqual(self._source_clean(source), "")

    def test_lora_runtime_composes_with_metadata_and_multi_caption(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("multi_caption", "parameter_policy_metadata", "parameter_policy_runtime"),
                source_dir=source,
                cache_root=temp_dir,
            )
            self.assertIn(
                metadata_patch.MARKER_ATTR,
                (tree / "networks/lora_anima.py").read_text(encoding="utf-8"),
            )
            self.assertTrue((tree / "library/multi_caption.py").is_file())
            payload = json.loads((tree / ".mikazuki-anima-runtime").read_text(encoding="utf-8"))
            self.assertEqual(
                payload["features"],
                ["multi_caption", "parameter_policy_metadata", "parameter_policy_runtime"],
            )

    def test_full_runtime_composes_after_qwen_patch(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("qwen3_joint", "parameter_policy_runtime"),
                source_dir=source,
                cache_root=temp_dir,
            )
            trainer = (tree / "anima_train.py").read_text(encoding="utf-8")
            self.assertIn("anima_qwen3_training.json", trainer)
            self.assertIn("parameter_policy_session", trainer)
            self.assertIn("register_checkpoint_manifest", trainer)
            self.assertLess(
                trainer.index("register_checkpoint_manifest"),
                trainer.index("register_load_state_pre_hook(load_qwen3_mode_hook)"),
            )

    def test_all_features_compose(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                (
                    "multi_caption",
                    "parameter_policy_metadata",
                    "parameter_policy_runtime",
                    "qwen3_joint",
                ),
                source_dir=source,
                cache_root=temp_dir,
            )
            self.assertTrue((tree / "library/multi_caption.py").is_file())
            self.assertTrue((tree / "library/dts_parameter_policy_bridge.py").is_file())
            self.assertIn(
                metadata_patch.MARKER_ATTR,
                (tree / "networks/lora_anima.py").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
