import subprocess
import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_runtime import (
    materialize_anima_runtime_tree,
    requested_runtime_features,
)


class Prepared:
    def __init__(self, train_type, config):
        self.train_type = train_type
        self.config = config
        self.trainer_file = "unchanged.py"


class AnimaComposableRuntimeTests(unittest.TestCase):
    def test_feature_detection_is_opt_in(self):
        self.assertEqual(requested_runtime_features(Prepared("anima-lora", {})), ())
        self.assertEqual(
            requested_runtime_features(
                Prepared("anima-lora", {"multi_caption_config": "policy.json"})
            ),
            ("multi_caption",),
        )
        self.assertEqual(
            requested_runtime_features(
                Prepared(
                    "anima-finetune",
                    {
                        "train_qwen3_text_encoder": True,
                        "multi_caption_config": "policy.json",
                    },
                )
            ),
            ("multi_caption", "qwen3_joint"),
        )
        self.assertEqual(
            requested_runtime_features(
                Prepared("flux-lora", {"multi_caption_config": "policy.json"})
            ),
            (),
        )

    def test_multi_only_stages_without_mutating_submodule(self):
        source = Path("sd-scripts").resolve()
        before = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(before.strip(), "")

        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("multi_caption",), source_dir=source, cache_root=temp_dir
            )
            self.assertNotEqual(tree.resolve(), source)
            self.assertTrue((tree / "library" / "multi_caption.py").is_file())
            self.assertIn(
                "--multi_caption_config",
                (tree / "library" / "args.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "configure_multi_caption_dataset_groups",
                (tree / "anima_train.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "configure_multi_caption_dataset_groups",
                (tree / "train_network.py").read_text(encoding="utf-8"),
            )

        after = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(after.strip(), "")

    def test_qwen_and_multi_compose_in_one_tree(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("qwen3_joint", "multi_caption"),
                source_dir=source,
                cache_root=temp_dir,
            )
            anima_train = (tree / "anima_train.py").read_text(encoding="utf-8")
            self.assertIn("--train_qwen3_text_encoder", (tree / "library" / "anima_train_utils.py").read_text(encoding="utf-8"))
            self.assertIn("blocks_to_swap requires AdaFactor", anima_train)
            self.assertIn("configure_multi_caption_dataset_groups", anima_train)
            self.assertTrue((tree / "library" / "multi_caption.py").is_file())


    def test_staging_ignores_untracked_worktree_files(self):
        source = Path("sd-scripts").resolve()
        probe = source / "__dts_untracked_probe__.txt"
        probe.write_text("must not be staged", encoding="utf-8")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                tree = materialize_anima_runtime_tree(
                    ("multi_caption",), source_dir=source, cache_root=temp_dir
                )
                self.assertFalse((tree / probe.name).exists())
                self.assertLess(len(tree.name), 32)
        finally:
            probe.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
