import json
import unittest

from mikazuki.multi_caption_config import (
    build_multi_caption_sidecar,
    canonicalize_multi_caption_policy,
    extract_multi_caption_gui_state,
    rehydrate_multi_caption_policy,
    serialize_multi_caption_policy,
)


class MultiCaptionConfigTests(unittest.TestCase):
    def test_missing_caption_mode_is_standard_and_stale_fields_are_removed(self):
        config = {
            "learning_rate": 1e-4,
            "multi_caption_storage": "json",
            "multi_caption_json_path": "stale.json",
            "multi_caption_json_groups": {"old": {"weight": 1, "key": "caption"}},
        }
        path, sidecars, policy = build_multi_caption_sidecar(config, "flux-lora")
        self.assertIsNone(path)
        self.assertEqual(sidecars, {})
        self.assertIsNone(policy)
        self.assertEqual(config, {"learning_rate": 1e-4})

    def test_files_policy_is_canonical_and_group_order_does_not_change_hash(self):
        a = {
            "caption_mode": "multi",
            "multi_caption_storage": "files",
            "multi_caption_file_groups": {
                "tags": {"enabled": True, "weight": "5", "extension": "txt", "processing": {"shuffle_caption": True}},
                "natural": {"enabled": True, "weight": 3, "extension": ".nl.txt", "processing": {"shuffle_caption": False}},
            },
        }
        b = {
            "caption_mode": "multi",
            "multi_caption_storage": "files",
            "multi_caption_file_groups": {
                "natural": {"enabled": True, "weight": 3, "extension": ".nl.txt", "processing": {"shuffle_caption": False}},
                "tags": {"enabled": True, "weight": 5.0, "extension": ".txt", "processing": {"shuffle_caption": True}},
            },
        }
        p1, s1, policy1 = build_multi_caption_sidecar(a, "flux-lora")
        p2, s2, policy2 = build_multi_caption_sidecar(b, "flux-lora")
        self.assertEqual(policy1, policy2)
        self.assertEqual(p1, p2)
        self.assertEqual(s1, s2)
        self.assertEqual(a, {"multi_caption_config": p1})
        self.assertEqual(b, {"multi_caption_config": p2})

    def test_multiline_physical_line_number_is_one_based(self):
        policy = canonicalize_multi_caption_policy({
            "storage_mode": "multiline",
            "profile": "shared",
            "extension": ".txt",
            "groups": {
                "a": {"line": 1, "weight": 1},
                "b": {"line": 3, "weight": 1},
            },
        })
        self.assertEqual(policy["groups"]["a"]["source"]["line"], 1)
        self.assertEqual(policy["groups"]["b"]["source"]["line"], 3)
        with self.assertRaisesRegex(ValueError, "从 1 开始"):
            canonicalize_multi_caption_policy({
                "storage_mode": "multiline",
                "profile": "shared",
                "extension": ".txt",
                "groups": {"a": {"line": 0, "weight": 1}},
            })

    def test_jsonl_requires_source_and_has_explicit_image_key_field(self):
        policy = canonicalize_multi_caption_policy({
            "storage_mode": "jsonl",
            "profile": "shared",
            "path": "captions.jsonl",
            "image_key_mode": "relative_path",
            "jsonl_image_key_field": "image",
            "groups": {"tags": {"key": "tags", "weight": 1}},
        })
        self.assertEqual(policy["storage"]["image_key_field"], "image")
        with self.assertRaisesRegex(ValueError, "dedicated caption"):
            canonicalize_multi_caption_policy({
                "storage_mode": "json",
                "profile": "shared",
                "groups": {"tags": {"key": "tags", "weight": 1}},
            })

    def test_requires_positive_enabled_weight(self):
        with self.assertRaisesRegex(ValueError, "weight > 0"):
            canonicalize_multi_caption_policy({
                "storage_mode": "files",
                "profile": "shared",
                "groups": {"tags": {"extension": ".txt", "weight": 0}},
            })

    def test_basic_profile_rejects_shared_or_anima_only_processing(self):
        with self.assertRaisesRegex(ValueError, "不支持字段"):
            canonicalize_multi_caption_policy({
                "storage_mode": "files",
                "profile": "basic",
                "groups": {
                    "tags": {
                        "extension": ".txt",
                        "weight": 1,
                        "processing": {"caption_tag_dropout_rate": 0.1},
                    }
                },
            })

    def test_weighted_captions_and_token_length_cannot_be_group_processing(self):
        for key in ("weighted_captions", "max_token_length", "qwen3_max_token_length"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "全局 tokenizer/encoding"):
                    canonicalize_multi_caption_policy({
                        "storage_mode": "files",
                        "profile": "anima",
                        "groups": {
                            "tags": {
                                "extension": ".txt",
                                "weight": 1,
                                "processing": {key: True},
                            }
                        },
                    })

    def test_unicode_group_names_and_round_trip(self):
        config = {
            "caption_mode": "multi",
            "multi_caption_storage": "files",
            "multi_caption_file_groups": {
                "自然语言": {"extension": ".nl.txt", "weight": 2, "processing": {"shuffle_caption": False}},
            },
        }
        _, _, policy = build_multi_caption_sidecar(config, "anima-lora")
        state = rehydrate_multi_caption_policy(policy, "anima-lora")
        self.assertEqual(state["multi_caption_file_groups"]["自然语言"]["extension"], ".nl.txt")

    def test_serialization_is_content_addressed(self):
        policy = canonicalize_multi_caption_policy({
            "storage_mode": "files",
            "profile": "shared",
            "groups": {"tags": {"extension": ".txt", "weight": 1}},
        })
        path, content = serialize_multi_caption_policy(policy)
        self.assertTrue(path.startswith("config/autosave/multi-caption/"))
        self.assertTrue(path.endswith(".json"))
        self.assertEqual(json.loads(content), policy)

    def test_standard_extract_never_leaks_multi_fields(self):
        config = {
            "caption_mode": "standard",
            "multi_caption_file_groups": {"x": {"extension": ".txt"}},
            "train_batch_size": 1,
        }
        state = extract_multi_caption_gui_state(config, "sdxl-finetune")
        self.assertIsNone(state)
        self.assertEqual(config, {"train_batch_size": 1})


if __name__ == "__main__":
    unittest.main()
