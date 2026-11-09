"""QG-2-B3-A: strict torch-free Policy loader + immutable snapshot RED."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mikazuki.direct_trainer_admission_intent import (
    DirectTrainerAdmissionIntent,
    extract_direct_trainer_admission_intent,
)
from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy import (
    PARAMETER_POLICY_DIR,
    serialize_parameter_policy,
    validate_parameter_policy,
)
from mikazuki.parameter_policy_admission import assess_component_training_admission
from mikazuki.parameter_policy_execution import (
    ExecutionFeatureQualification,
    FULL_BF16_BACKEND_QUALIFICATIONS,
)

ROOT = Path(__file__).resolve().parents[1]
MARKER = "QG-2-B3-A POLICY SNAPSHOT CONTRACT ABSENT"
ACK = "experimental_full_bf16_ack"


def _policy(train_type="sd-lora", optimizer="AdamW"):
    profile = get_model_component_profile(train_type)
    components = {name: {"train": False} for name in profile.components}
    selected = (
        "mmdit.attention.adapter"
        if train_type == "sd3-lora"
        else "unet.attention.adapter"
    )
    if selected not in components:
        selected = next(iter(components))
    components[selected] = {
        "train": True,
        "optimizer_profile": "main",
        "learning_rate": 1e-4,
    }
    return {
        "version": 1,
        "optimizer_profiles": {
            "main": {"type": optimizer, "args": {}},
        },
        "components": components,
    }


def _config(**overrides):
    return {
        "full_bf16": True,
        "mixed_precision": "bf16",
        "network_train_unet_only": False,
        "network_train_text_encoder_only": False,
        **overrides,
    }


def _pending():
    return ExecutionFeatureQualification(
        "pending",
        "QG-2-B3-A test-only missing release evidence",
        None,
    )


def _write_policy(path: Path, policy=None, *, indent=None, sort_keys=True):
    raw = json.dumps(
        _policy() if policy is None else policy,
        ensure_ascii=False,
        sort_keys=sort_keys,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        allow_nan=False,
    )
    path.write_text(raw, encoding="utf-8")
    return raw


def _module(name):
    return importlib.import_module(name)


def _top_level_imports(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        else:
            if node.module:
                names.add(node.module.split(".", 1)[0])
    return names


class QG2B3APolicySnapshotTargetedRed(unittest.TestCase):
    def _need(self, value, detail):
        if not value:
            self.fail(f"{MARKER}: {detail}")

    def _loader(self):
        policy_module = _module("mikazuki.parameter_policy")
        loader = getattr(policy_module, "load_parameter_policy_sidecar", None)
        self._need(callable(loader), "torch-free canonical Policy sidecar loader is absent")
        return loader

    def _fingerprint(self):
        module = _module("mikazuki.parameter_policy_admission")
        function = getattr(module, "fingerprint_effective_training_config", None)
        self._need(
            callable(function),
            "canonical effective-config fingerprint function is absent",
        )
        return function

    def _snapshot_type(self):
        module = _module("mikazuki.parameter_policy_admission")
        kind = getattr(module, "DirectTrainerAdmissionSnapshot", None)
        self._need(
            kind is not None,
            "immutable DirectTrainerAdmissionSnapshot is absent",
        )
        return kind

    def _prepare(self):
        module = _module("mikazuki.parameter_policy_admission")
        function = getattr(module, "prepare_direct_trainer_admission", None)
        self._need(
            callable(function),
            "first-party Direct Trainer Admission snapshot factory is absent",
        )
        return function

    def test_b3a201_public_policy_loader_is_in_existing_torch_free_host_module(self):
        loader = self._loader()
        imported = _top_level_imports("mikazuki/parameter_policy.py")
        self.assertNotIn("torch", imported)
        self.assertNotIn("accelerate", imported)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            canonical, digest = loader(path)
        self.assertEqual(canonical, validate_parameter_policy(_policy()))
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_b3a202_loader_rejects_ambiguous_or_invalid_json_and_paths(self):
        loader = self._loader()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_payloads = (
                '{"version":1,"version":1,"optimizer_profiles":{},"components":{}}',
                '{"version":1,"optimizer_profiles":{"main":{"type":"AdamW","args":{"x":NaN}}},"components":{}}',
                '{"version":1,"optimizer_profiles":{"main":{"type":"AdamW","args":{"x":Infinity}}},"components":{}}',
                "not-json",
                "[]",
                '{"version":2,"optimizer_profiles":{"main":{"type":"AdamW","args":{}}},"components":{"unet.attention.adapter":{"train":true,"optimizer_profile":"main","learning_rate":0.0001}}}',
                '{"version":1,"unknown":true,"optimizer_profiles":{},"components":{}}',
            )
            for index, raw in enumerate(invalid_payloads):
                path = root / f"invalid-{index}.json"
                path.write_text(raw, encoding="utf-8")
                with self.subTest(index=index):
                    with self.assertRaises((ValueError, OSError)):
                        loader(path)
            with self.assertRaises((ValueError, OSError)):
                loader(root / "missing.json")
            with self.assertRaises((ValueError, OSError)):
                loader(root)
            # A path that is not a regular file must be refused *before*
            # attempting to open it (e.g. a FIFO would otherwise block).
            valid_path = root / "regular.json"
            _write_policy(valid_path)
            with patch("mikazuki.parameter_policy.Path.is_file", return_value=False):
                with patch("builtins.open", side_effect=AssertionError("unexpected open")):
                    with self.assertRaises((ValueError, OSError)):
                        loader(valid_path)

    def test_b3a203_policy_hash_uses_full_canonical_serialized_content(self):
        loader = self._loader()
        policy = _policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            changed = root / "changed.json"
            _write_policy(first, policy, indent=2, sort_keys=True)
            reversed_policy = {
                "components": dict(reversed(list(policy["components"].items()))),
                "optimizer_profiles": {
                    "main": {"args": {}, "type": "AdamW"},
                },
                "version": 1,
            }
            _write_policy(second, reversed_policy, indent=None, sort_keys=False)
            changed_policy = json.loads(json.dumps(policy))
            changed_policy["components"]["unet.attention.adapter"]["learning_rate"] = 2e-4
            _write_policy(changed, changed_policy)

            canonical, digest_a = loader(first)
            _canonical_b, digest_b = loader(second)
            _canonical_changed, digest_changed = loader(changed)

        _path, canonical_text = serialize_parameter_policy(canonical)
        expected = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        self.assertEqual(digest_a, expected)
        self.assertEqual(digest_a, digest_b)
        self.assertNotEqual(digest_a, digest_changed)
        self.assertEqual(len(digest_a), 64)
        self.assertEqual(len(Path(_path).stem), 24)
        self.assertEqual(digest_a[:24], Path(_path).stem)

    def test_b3a204_effective_config_fingerprint_is_canonical_strict_and_detached(self):
        fingerprint = self._fingerprint()
        first = {
            "full_bf16": True,
            "mixed_precision": "bf16",
            "nested": {"b": [1, 2], "a": 3},
        }
        reordered = {
            "nested": {"a": 3, "b": (1, 2)},
            "mixed_precision": "bf16",
            "full_bf16": True,
        }
        digest = fingerprint(first)
        self.assertEqual(digest, fingerprint(reordered))
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            digest,
            fingerprint({**first, "full_bf16": False}),
        )

        invalid = (
            {1: "non-string key"},
            {"bad": float("nan")},
            {"bad": float("inf")},
            {"bad": {1, 2}},
            {"bad": b"bytes"},
            {"bad": Path("model.safetensors")},
            {"nested": {ACK: True}},
            {ACK: False},
        )
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    fingerprint(value)

        before = json.dumps(first, sort_keys=True)
        fingerprint(first)
        self.assertEqual(json.dumps(first, sort_keys=True), before)

    def test_b3a205_snapshot_is_frozen_slotted_and_has_only_identity_fields(self):
        kind = self._snapshot_type()
        admission = assess_component_training_admission(
            _policy(),
            train_type="sd-lora",
            effective_config=_config(),
            purpose="direct_trainer",
            experimental_ack=False,
        )
        snapshot = kind(
            policy_hash="a" * 64,
            effective_config_fingerprint="b" * 64,
            train_type="sd-lora",
            experimental_ack=False,
            admission=admission,
        )
        self.assertTrue(is_dataclass(kind))
        self.assertTrue(kind.__dataclass_params__.frozen)
        self.assertTrue(hasattr(kind, "__slots__"))
        self.assertEqual(
            [item.name for item in fields(kind)],
            [
                "policy_hash",
                "effective_config_fingerprint",
                "train_type",
                "experimental_ack",
                "admission",
            ],
        )
        with self.assertRaises(FrozenInstanceError):
            snapshot.policy_hash = "c" * 64
        for invalid in (
            {"policy_hash": "short"},
            {"effective_config_fingerprint": "not-a-hash"},
            {"experimental_ack": 1},
        ):
            values = {
                "policy_hash": "a" * 64,
                "effective_config_fingerprint": "b" * 64,
                "train_type": "sd-lora",
                "experimental_ack": False,
                "admission": admission,
            }
            values.update(invalid)
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    kind(**values)

    def test_b3a206_factory_uses_typed_intent_and_first_party_direct_admission(self):
        prepare = self._prepare()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            with self.assertRaises((TypeError, ValueError)):
                prepare(
                    path,
                    train_type="sd-lora",
                    effective_config=_config(),
                    admission_intent=True,
                )
            legacy = prepare(
                path,
                train_type="sd-lora",
                effective_config=_config(),
                admission_intent=DirectTrainerAdmissionIntent(False),
            )
            self.assertEqual(legacy.admission.status, "legacy_approved")
            self.assertEqual(legacy.admission.purpose, "direct_trainer")

            with patch.dict(
                FULL_BF16_BACKEND_QUALIFICATIONS,
                {"sd-lora": _pending()},
            ):
                no_ack = prepare(
                    path,
                    train_type="sd-lora",
                    effective_config=_config(),
                    admission_intent=DirectTrainerAdmissionIntent(False),
                )
                with_ack = prepare(
                    path,
                    train_type="sd-lora",
                    effective_config=_config(),
                    admission_intent=DirectTrainerAdmissionIntent(True),
                )
            self.assertEqual(no_ack.admission.status, "ack_required")
            self.assertEqual(with_ack.admission.status, "experimental_host_candidate")
            self.assertEqual(with_ack.admission.readiness.evidence_tier, "unverified")

            sd3_path = Path(directory) / "sd3.json"
            _write_policy(sd3_path, _policy("sd3-lora"))
            unsupported = prepare(
                sd3_path,
                train_type="sd3-lora",
                effective_config=_config(),
                admission_intent=DirectTrainerAdmissionIntent(True),
            )
            self.assertEqual(unsupported.admission.status, "blocked")
            self.assertFalse(unsupported.admission.host_admissible)

    def test_b3a207_snapshot_assert_matches_rejects_each_identity_change(self):
        prepare = self._prepare()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            snapshot = prepare(
                path,
                train_type="sd-lora",
                effective_config=_config(),
                admission_intent=DirectTrainerAdmissionIntent(False),
            )
        self.assertTrue(callable(getattr(snapshot, "assert_matches", None)))
        expected = {
            "policy_hash": snapshot.policy_hash,
            "effective_config_fingerprint": snapshot.effective_config_fingerprint,
            "train_type": snapshot.train_type,
            "experimental_ack": snapshot.experimental_ack,
        }
        snapshot.assert_matches(**expected)
        variants = (
            {**expected, "policy_hash": "f" * 64},
            {**expected, "effective_config_fingerprint": "e" * 64},
            {**expected, "train_type": "sdxl-lora"},
            {**expected, "experimental_ack": True},
        )
        for changed in variants:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    snapshot.assert_matches(**changed)

    def test_b3a208_factory_is_read_only_and_snapshot_retains_no_mutable_inputs(self):
        loader = self._loader()
        prepare = self._prepare()
        fingerprint = self._fingerprint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "policy.json"
            original_policy = _policy()
            original_config = _config(nested={"items": [1, 2]})
            raw_before = _write_policy(path, original_policy)
            config_before = json.dumps(original_config, sort_keys=True)
            stat_before = path.stat()
            entries_before = sorted(item.name for item in root.iterdir())

            snapshot = prepare(
                path,
                train_type="sd-lora",
                effective_config=original_config,
                admission_intent=DirectTrainerAdmissionIntent(False),
            )

            stat_after = path.stat()
            entries_after = sorted(item.name for item in root.iterdir())
            self.assertEqual(path.read_text(encoding="utf-8"), raw_before)
            self.assertEqual(stat_before.st_mtime_ns, stat_after.st_mtime_ns)
            self.assertEqual(entries_before, entries_after)
            self.assertEqual(json.dumps(original_config, sort_keys=True), config_before)
            self.assertEqual(
                snapshot.policy_hash,
                loader(path)[1],
            )
            self.assertEqual(
                snapshot.effective_config_fingerprint,
                fingerprint(original_config),
            )
            self.assertFalse(
                any(
                    hasattr(snapshot, field)
                    for field in ("policy", "canonical_policy", "effective_config", "policy_path")
                )
            )
            self.assertIsNotNone(snapshot.admission)
            self.assertEqual(snapshot.admission.purpose, "direct_trainer")


class QG2B3APolicySnapshotBaselines(unittest.TestCase):
    def test_b3a901_existing_policy_validation_and_serialization_abi(self):
        policy = validate_parameter_policy(_policy())
        path, content = serialize_parameter_policy(policy)
        self.assertEqual(path, (PARAMETER_POLICY_DIR / f"{hashlib.sha256(content.encode('utf-8')).hexdigest()[:24]}.json").as_posix())
        self.assertTrue(content.endswith("\n"))
        self.assertEqual(json.loads(content), policy)
        self.assertEqual(len(hashlib.sha256(content.encode("utf-8")).hexdigest()), 64)

    def test_b3a902_d1_legacy_and_unsupported_backend_classification_unchanged(self):
        legacy = assess_component_training_admission(
            _policy(),
            train_type="sd-lora",
            effective_config=_config(),
            purpose="direct_trainer",
            experimental_ack=False,
        )
        self.assertEqual(legacy.status, "legacy_approved")
        unsupported = assess_component_training_admission(
            _policy("sd3-lora"),
            train_type="sd3-lora",
            effective_config=_config(),
            purpose="direct_trainer",
            experimental_ack=True,
        )
        self.assertEqual(unsupported.status, "blocked")
        self.assertFalse(unsupported.host_admissible)

    def test_b3a903_host_modules_keep_standard_mode_torch_lazy(self):
        for path in (
            "mikazuki/parameter_policy.py",
            "mikazuki/parameter_policy_admission.py",
            "mikazuki/direct_trainer_admission_intent.py",
        ):
            source = (ROOT / path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=path)
            imports = [
                node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            names = set()
            for node in imports:
                if isinstance(node, ast.Import):
                    names.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif node.module:
                    names.add(node.module.split(".", 1)[0])
            self.assertNotIn("torch", names, path)
            self.assertNotIn("accelerate", names, path)
            self.assertNotIn("parameter_policy_trainer", names, path)

    def test_b3a904_existing_direct_cli_intent_remains_strict_and_one_shot(self):
        args = type("Args", (), {})()
        args.experimental_full_bf16_ack = True
        intent = extract_direct_trainer_admission_intent(args)
        self.assertIs(intent.experimental_full_bf16_ack, True)
        self.assertFalse(hasattr(args, ACK))
        for invalid in ("true", 1, None):
            bad = type("Args", (), {})()
            bad.experimental_full_bf16_ack = invalid
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):
                    extract_direct_trainer_admission_intent(bad)


if __name__ == "__main__":
    unittest.main()
