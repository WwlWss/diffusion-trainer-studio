import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stable.library.multi_caption import CaptionProcessingOptions, MultiCaptionResolver
from scripts.stable.library.train_util import BaseDataset


class Info:
    def __init__(self, path, is_reg=False):
        self.absolute_path = str(path)
        self.image_key = str(path)
        self.is_reg = is_reg


def write_policy(root, policy):
    path = Path(root) / "policy.json"
    path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    return path


class StableMultiCaptionResolverTests(unittest.TestCase):
    def test_files_missing_group_is_excluded_before_weighted_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            (Path(tmp) / "001.txt").write_text("tags", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "files"},
                "groups": {
                    "missing": {"enabled": True, "weight": 100, "source": {"extension": ".nl.txt"}, "processing": {}},
                    "tags": {"enabled": True, "weight": 3, "source": {"extension": ".txt"}, "processing": {}},
                },
            }
            resolver = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            with patch("scripts.stable.library.multi_caption.random.choices") as choices:
                choices.return_value = [(resolver.groups[1], "tags")]
                resolved = resolver.choose(image_path=str(image))
                args, kwargs = choices.call_args
                self.assertEqual(kwargs["weights"], [3.0])
                self.assertEqual(resolved.caption, "tags")

    def test_multiline_uses_physical_one_based_lines_and_blank_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            (Path(tmp) / "001.txt").write_text("first\n\nthird\n", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "multiline", "extension": ".txt"},
                "groups": {
                    "blank": {"enabled": True, "weight": 50, "source": {"line": 2}, "processing": {}},
                    "third": {"enabled": True, "weight": 1, "source": {"line": 3}, "processing": {}},
                },
            }
            resolver = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            with patch("scripts.stable.library.multi_caption.random.choices") as choices:
                choices.return_value = [(resolver.groups[1], "third")]
                resolved = resolver.choose(image_path=str(image))
                self.assertEqual(choices.call_args.kwargs["weights"], [1.0])
                self.assertEqual(resolved.caption, "third")

    def test_json_filename_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a" / "001.png"
            b = root / "b" / "001.png"
            a.parent.mkdir()
            b.parent.mkdir()
            a.write_bytes(b"x")
            b.write_bytes(b"x")
            source = root / "captions.json"
            source.write_text("{}", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "json", "path": str(source), "image_key_mode": "filename"},
                "groups": {"tags": {"enabled": True, "weight": 1, "source": {"key": "tags"}, "processing": {}}},
            }
            with self.assertRaisesRegex(ValueError, "collision"):
                MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(a), Info(b)])

    def test_jsonl_duplicate_image_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "001.png"
            image.write_bytes(b"x")
            source = root / "captions.jsonl"
            source.write_text(
                '{"image":"001.png","tags":"a"}\n{"image":"001.png","tags":"b"}\n',
                encoding="utf-8",
            )
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "jsonl", "path": str(source), "image_key_mode": "filename", "image_key_field": "image"},
                "groups": {"tags": {"enabled": True, "weight": 1, "source": {"key": "tags"}, "processing": {}}},
            }
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])

    def test_group_processing_is_returned_with_selected_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            (Path(tmp) / "001.txt").write_text("a, b, c", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "files"},
                "groups": {
                    "tags": {
                        "enabled": True,
                        "weight": 1,
                        "source": {"extension": ".txt"},
                        "processing": {"shuffle_caption": True, "keep_tokens": 1},
                    }
                },
            }
            resolver = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            resolved = resolver.choose(image_path=str(image))
            self.assertTrue(resolved.processing.shuffle_caption)
            self.assertEqual(resolved.processing.keep_tokens, 1)


    def test_all_groups_missing_fails_closed_instead_of_using_standard_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "files"},
                "groups": {
                    "tags": {
                        "enabled": True,
                        "weight": 1,
                        "source": {"extension": ".txt"},
                        "processing": {},
                    }
                },
            }
            resolver = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            with self.assertRaisesRegex(ValueError, "no enabled non-empty caption group"):
                resolver.choose(image_path=str(image))

    def test_separate_files_preserve_multiline_for_wildcard_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            (Path(tmp) / "001.txt").write_text("first\nsecond\n", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "files"},
                "groups": {
                    "tags": {
                        "enabled": True,
                        "weight": 1,
                        "source": {"extension": ".txt"},
                        "processing": {"enable_wildcard": True},
                    }
                },
            }
            resolver = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            resolved = resolver.choose(image_path=str(image))
            self.assertEqual(resolved.caption, "first\nsecond")

    def test_processing_adapter_is_mutable_like_sd_scripts_subset(self):
        dataset = object.__new__(BaseDataset)
        dataset.current_epoch = 1
        dataset.current_step = 1
        dataset.max_train_steps = 100
        dataset.replacements = {}
        processing = CaptionProcessingOptions(
            shuffle_caption=True,
            token_warmup_step=0.5,
        )
        with patch("scripts.stable.library.train_util.random.shuffle", lambda values: None):
            caption = BaseDataset.process_caption(dataset, processing, "a, b, c")
        self.assertEqual(processing.token_warmup_step, 50)
        self.assertIsInstance(caption, str)

    def test_resolver_is_spawn_pickle_safe_after_opening_sqlite_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            (Path(tmp) / "001.txt").write_text("tags", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "files"},
                "groups": {
                    "tags": {
                        "enabled": True,
                        "weight": 1,
                        "source": {"extension": ".txt"},
                        "processing": {},
                    }
                },
            }
            resolver = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            self.assertEqual(resolver.choose(image_path=str(image)).caption, "tags")
            restored = pickle.loads(pickle.dumps(resolver))
            self.assertEqual(restored.choose(image_path=str(image)).caption, "tags")

    def test_validation_clone_is_deterministic_per_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"x")
            (Path(tmp) / "001.a.txt").write_text("a", encoding="utf-8")
            (Path(tmp) / "001.b.txt").write_text("b", encoding="utf-8")
            policy = {
                "version": 1,
                "selection": {"mode": "weighted_one"},
                "storage": {"mode": "files"},
                "groups": {
                    "a": {"enabled": True, "weight": 1, "source": {"extension": ".a.txt"}, "processing": {}},
                    "b": {"enabled": True, "weight": 1, "source": {"extension": ".b.txt"}, "processing": {}},
                },
            }
            train = MultiCaptionResolver.from_file(str(write_policy(tmp, policy)), [Info(image)])
            val = train.clone(deterministic=True)
            chosen = {val.choose(image_path=str(image)).group_name for _ in range(10)}
            self.assertEqual(len(chosen), 1)


if __name__ == "__main__":
    unittest.main()
