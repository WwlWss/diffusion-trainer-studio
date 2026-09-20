from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_runtime import (
    materialize_anima_runtime_tree,
    requested_runtime_features,
)
from tools import apply_anima_parameter_policy_metadata_patch as metadata_patch
from tools import apply_anima_qwen3_sd_scripts_patch as qwen_patch


class Prepared:
    def __init__(self, train_type, config):
        self.train_type = train_type
        self.config = config
        self.trainer_file = "unchanged.py"


class AnimaParameterPolicyMetadataRuntimeTests(unittest.TestCase):
    def test_feature_detection_is_component_lora_only(self):
        self.assertEqual(
            requested_runtime_features(Prepared("anima-lora", {})),
            (),
        )
        self.assertEqual(
            requested_runtime_features(
                Prepared(
                    "anima-lora",
                    {"parameter_policy_config": "parameter-policy.json"},
                )
            ),
            ("parameter_policy_metadata", "parameter_policy_runtime"),
        )
        self.assertEqual(
            requested_runtime_features(
                Prepared(
                    "anima-lora",
                    {
                        "parameter_policy_config": "parameter-policy.json",
                        "multi_caption_config": "captions.json",
                    },
                )
            ),
            ("multi_caption", "parameter_policy_metadata", "parameter_policy_runtime"),
        )
        self.assertEqual(
            requested_runtime_features(
                Prepared(
                    "anima-finetune",
                    {"parameter_policy_config": "parameter-policy.json"},
                )
            ),
            ("parameter_policy_runtime",),
        )
        self.assertEqual(
            requested_runtime_features(
                Prepared(
                    "flux-lora",
                    {"parameter_policy_config": "parameter-policy.json"},
                )
            ),
            (),
        )

    def test_metadata_patch_is_pinned_to_same_sd_scripts_head(self):
        self.assertEqual(
            metadata_patch.EXPECTED_SD_SCRIPTS_HEAD,
            qwen_patch.EXPECTED_SD_SCRIPTS_HEAD,
        )

    def test_metadata_patch_validates_pinned_source_without_writing(self):
        source = Path("sd-scripts").resolve()
        before = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(before.strip(), "")

        metadata_patch.verify_head(source)
        metadata_patch.validate_patch(source)

        after = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(after.strip(), "")

    def test_metadata_patch_keeps_original_name_and_uses_real_target_path(self):
        source = Path("sd-scripts").resolve()
        patched = metadata_patch.patch_files(source)
        path = source / "networks/lora_anima.py"
        text = patched[path]

        self.assertIn(
            'target_root = "dit" if is_unet else ("qwen3" if text_encoder_idx == 0 else None)',
            text,
        )
        self.assertIn("lora.original_name = original_name", text)
        self.assertIn(
            "_attach_dts_parameter_policy_target(\n"
            "                                lora,\n"
            "                                target_root,\n"
            "                                original_name,\n"
            "                                child_module,\n",
            text,
        )
        self.assertNotIn("mikazuki.parameter_routing", text)
        self.assertNotIn("AdapterTargetMetadata", text)

    def test_standard_anima_multi_caption_runtime_does_not_gain_metadata(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("multi_caption",),
                source_dir=source,
                cache_root=temp_dir,
            )
            lora_text = (tree / "networks/lora_anima.py").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(metadata_patch.MARKER_ATTR, lora_text)
            payload = json.loads(
                (tree / ".mikazuki-anima-runtime").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["features"], ["multi_caption"])

    def test_metadata_only_runtime_is_isolated_and_capability_checked(self):
        source = Path("sd-scripts").resolve()
        before = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(before.strip(), "")

        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                ("parameter_policy_metadata",),
                source_dir=source,
                cache_root=temp_dir,
            )
            self.assertNotEqual(tree.resolve(), source)
            lora_text = (tree / "networks/lora_anima.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(metadata_patch.MARKER_ATTR, lora_text)
            self.assertIn('"dit"', lora_text)
            self.assertIn('"qwen3"', lora_text)

            payload = json.loads(
                (tree / ".mikazuki-anima-runtime").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["features"],
                ["parameter_policy_metadata"],
            )

        after = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(after.strip(), "")

    def test_metadata_composes_with_qwen_and_multi_caption(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = materialize_anima_runtime_tree(
                (
                    "parameter_policy_metadata",
                    "parameter_policy_runtime",
                    "qwen3_joint",
                    "multi_caption",
                ),
                source_dir=source,
                cache_root=temp_dir,
            )
            self.assertIn(
                metadata_patch.MARKER_ATTR,
                (tree / "networks/lora_anima.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "--train_qwen3_text_encoder",
                (tree / "library/anima_train_utils.py").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "configure_multi_caption_dataset_groups",
                (tree / "anima_train.py").read_text(encoding="utf-8"),
            )
            self.assertTrue((tree / "library/multi_caption.py").is_file())

            payload = json.loads(
                (tree / ".mikazuki-anima-runtime").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["features"],
                [
                    "multi_caption",
                    "parameter_policy_metadata",
                    "parameter_policy_runtime",
                    "qwen3_joint",
                ],
            )

    def test_unknown_metadata_feature_name_still_fails_closed(self):
        source = Path("sd-scripts").resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "Unknown Anima runtime features"):
                materialize_anima_runtime_tree(
                    ("parameter-policy-metadata",),
                    source_dir=source,
                    cache_root=temp_dir,
                )


if __name__ == "__main__":
    unittest.main()
