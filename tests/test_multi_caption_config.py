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

    def test_switching_to_multi_without_groups_bootstraps_editable_files_group(self):
        config = {
            "caption_mode": "multi",
            "multi_caption_storage": "files",
        }
        state = extract_multi_caption_gui_state(config, "anima-finetune")
        self.assertEqual(
            state["groups"],
            {
                "caption": {
                    "enabled": True,
                    "weight": 1.0,
                    "extension": ".txt",
                    "processing": {},
                }
            },
        )
        policy = canonicalize_multi_caption_policy(state)
        self.assertEqual(policy["groups"]["caption"]["source"]["extension"], ".txt")

    def test_multi_caption_malformed_groups_still_fail_closed(self):
        for value in ("bad", [], 1):
            with self.subTest(value=value):
                config = {
                    "caption_mode": "multi",
                    "multi_caption_storage": "files",
                    "multi_caption_file_groups": value,
                }
                with self.assertRaisesRegex(
                    ValueError,
                    "Caption Groups 必须是 object/dict",
                ):
                    extract_multi_caption_gui_state(config, "anima-finetune")

    def test_multi_caption_explicit_empty_groups_are_not_silently_defaulted(self):
        config = {
            "caption_mode": "multi",
            "multi_caption_storage": "files",
            "multi_caption_file_groups": {},
        }
        state = extract_multi_caption_gui_state(config, "anima-finetune")
        self.assertEqual(state["groups"], {})
        with self.assertRaisesRegex(ValueError, "至少需要一个 Caption Group"):
            canonicalize_multi_caption_policy(state)

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

    def test_all_pages_share_full_group_processing_contract(self):
        for profile in ("basic", "shared", "anima"):
            with self.subTest(profile=profile):
                policy = canonicalize_multi_caption_policy({
                    "storage_mode": "files",
                    "profile": profile,
                    "groups": {
                        "tags": {
                            "extension": ".txt",
                            "weight": 1,
                            "processing": {
                                "caption_separator": "|",
                                "enable_wildcard": True,
                                "caption_prefix": "prefix",
                                "token_warmup_step": 0.5,
                                "caption_tag_dropout_rate": 0.1,
                            },
                        }
                    },
                })
                processing = policy["groups"]["tags"]["processing"]
                self.assertEqual(processing["caption_separator"], "|")
                self.assertTrue(processing["enable_wildcard"])
                self.assertEqual(processing["token_warmup_step"], 0.5)

    def test_numeric_fields_fail_closed_instead_of_truncating_or_accepting_nan(self):
        bad_values = [
            ("weight", float("nan")),
            ("line", 1.5),
        ]
        for field, value in bad_values:
            with self.subTest(field=field):
                group = {"weight": 1, "line": 1}
                group[field] = value
                state = {
                    "storage_mode": "multiline",
                    "profile": "shared",
                    "extension": ".txt",
                    "groups": {"tags": group},
                }
                with self.assertRaises(ValueError):
                    canonicalize_multi_caption_policy(state)

    def test_group_names_cannot_collide_after_normalization(self):
        with self.assertRaisesRegex(ValueError, "规范化后重复"):
            canonicalize_multi_caption_policy({
                "storage_mode": "files",
                "profile": "shared",
                "groups": {
                    "tags": {"extension": ".txt", "weight": 1},
                    " tags ": {"extension": ".nl.txt", "weight": 1},
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
