"""D1-SD3C2-B0 Fix B / RED-1: Host-only executable evidence contracts.

INTENTIONALLY RED until the dedicated B0 Host support and case generator
exist. These tests never invoke a trainer, use model weights, or claim CUDA
qualification. Keep any real SD3/GPU tests in later B1/B2 phases.

Frozen prospective APIs:
  tools.sd3_full_bf16_evidence_support:
    FullBF16EvidenceError, get_case_spec, build_reference_policy,
    validate_case_manifest, decoded_rgb_pixel_sha256,
    validate_qualification_preconditions, validate_qualification_lease_records,
    issue_phase_ticket, claim_phase_ticket, publish_phase_result,
    verify_adapter_file, validate_adapter_checkpoint_binding
  tools.prepare_sd3_full_bf16_cases:
    generate(case_id=..., model=..., data_dir=..., work_dir=...,
             external_models=...)

RED is not a passing acceptance gate. No missing B0 module is skipped or
replaced by a test stub: a missing module, missing API, or failed assertion
must fail the test, never count as a simulated CUDA PASS.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import struct
import tempfile
import unittest


R1 = "r1-adamw-accum1"
R2 = "r2-muon-adamw-fallback-accum2"
D0_ID = "phase-d0:shared-adamw-muon-full-bf16:v1"
SHA = "a" * 40


def _support():
    return importlib.import_module("tools.sd3_full_bf16_evidence_support")


def _generator():
    return importlib.import_module("tools.prepare_sd3_full_bf16_cases")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(status: str, evidence: str | None = D0_ID) -> dict:
    return {"status": status, "reason": "Host fixture only", "evidence_case_id": evidence}


def _qualifications(case_id: str) -> tuple[dict, dict[str, dict]]:
    optimizers = {"AdamW": _record("qualified")}
    if case_id == R2:
        optimizers["Muon"] = _record("qualified")
    return _record("pending", None), optimizers


def _lease_records(case_id: str) -> list[dict]:
    names = ("AdamW",) if case_id == R1 else ("Muon", "AdamW")
    return [
        {
            "kind": "backend", "name": "sd3-lora",
            "before_status": "pending", "before_evidence_case_id": None,
            "lease_applied": True,
        },
        *[
            {
                "kind": "optimizer", "name": name,
                "before_status": "qualified",
                "before_evidence_case_id": D0_ID,
                "lease_applied": False,
            }
            for name in names
        ],
    ]


def _source_dataset(root: Path, *, identical_pixels: bool = False, count: int = 2) -> Path:
    """Create real 512x512 RGB PNGs with captions, no model/network imports."""
    from PIL import Image, PngImagePlugin

    source = root / "source"
    source.mkdir()
    for n in range(1, count + 1):
        rgb = (17, 29, 41) if identical_pixels else (n * 17, n * 29, n * 41)
        image = Image.new("RGB", (512, 512), rgb)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("independent-file-metadata", str(n))
        image.save(source / f"original_{n:04d}.png", pnginfo=metadata)
        (source / f"original_{n:04d}.caption").write_text(
            f"an isolated test illustration, subject {n}\n", encoding="utf-8"
        )
    return source


def _generate(root: Path, case_id: str, *, identical_pixels: bool = False) -> dict:
    src = _source_dataset(root, identical_pixels=identical_pixels, count=1 if case_id == R1 else 2)
    model = root / "model.safetensors"
    # Host fixture is deliberately NOT a real SD3 checkpoint. A separate
    # asset preflight must reject its header; generation only binds bytes.
    model.write_bytes(b"host-test-only-model-not-valid-safetensors")
    external = {}
    for name in ("--clip_l", "--clip_g", "--t5xxl", "--vae"):
        p = root / (name[2:] + ".safetensors")
        p.write_bytes(("fixture " + name).encode("ascii"))
        external[name] = p
    # The Source gate is strict in production. Host fixtures can run against
    # a development tree and explicitly inject a synthetic source identity.
    from unittest.mock import patch
    fake_submodules = {
        name: {"gitlink_sha": SHA, "checkout": "uninitialized"}
        for name in ("frontend", "mikazuki/dataset-tag-editor", "sd-scripts")
    }
    with (patch.object(_generator(), "_source_commit", return_value=SHA),
          patch.object(_generator(), "_source_submodule_inventory",
                       return_value=fake_submodules)):
        return _generator().generate(
            case_id=case_id,
            model=model,
            data_dir=src,
            work_dir=root / "work",
            external_models=external,
        )


def _write_tiny_bf16_safetensors(path: Path, *, payload: bytes) -> tuple[str, dict]:
    """Small genuine safetensors byte layout; no Torch is required."""
    if len(payload) != 8:
        raise ValueError("Two-by-two BF16 requires eight data bytes")
    name = "lora_unet_joint_blocks_0_attn_proj_lora_up.weight"
    header = {
        name: {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [0, len(payload)],
        }
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    return _digest(path.read_bytes()), {
        name: {"dtype": "BF16", "shape": [2, 2], "sha256": _digest(payload)}
    }



def _write_reference_adapter(
    path: Path, *, alpha_dtype: str = "F32", alpha_bytes: bytes | None = None,
    weight_dtype: str = "BF16", include_alpha: bool = True, unknown: bool = False,
) -> tuple[str, dict]:
    """Real SD3 normalized LoRA state_dict triple, including scalar alpha.

    Pure CPU Safetensors layout, avoiding a heavy Torch dependency in Host CI.
    Split QKV is already normalized by LoRANetwork.state_dict() in production.
    """
    prefix = "lora_unet_joint_blocks_0_attn_proj"
    down = prefix + ".lora_down.weight"
    up = prefix + ".lora_up.weight"
    alpha = prefix + ".alpha"
    down_bytes = b"\x00\x3f\x00\x40\x80\x40\x00\x41"
    up_bytes = (b"\x00\x3f\x00\x40\x80\x40\x00\x41"
                if weight_dtype == "BF16" else struct.pack("<4f", 1, 2, 3, 4))
    if alpha_bytes is None:
        alpha_bytes = struct.pack("<f", 4.0) if alpha_dtype == "F32" else b"\x80\x40"
    entries = [
        (down, "BF16", [2, 2], down_bytes, "trainable_weight"),
        (up, weight_dtype, [2, 2], up_bytes, "trainable_weight"),
    ]
    if include_alpha:
        entries.append((alpha, alpha_dtype, [], alpha_bytes, "alpha_buffer"))
    if unknown:
        entries.append((prefix + ".unreviewed", "BF16", [2], b"\x00\x40\x00\x41", "unreviewed"))
    header, payload, expected = {}, b"", {}
    for key, dtype, shape, raw, role in entries:
        offset = len(payload)
        payload += raw
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [offset, offset + len(raw)]}
        expected[key] = {"dtype": dtype, "shape": shape, "role": role,
                         "sha256": _digest(raw)}
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    return _digest(path.read_bytes()), expected

def _write_three_component_reference_adapter(path: Path) -> tuple[str, dict, dict]:
    """Pure Host serialization of 3 real SD3 LoRA triplets, NOT live CUDA."""
    modules = (
        ("lora_unet_joint_blocks_0_attn_proj", "mmdit.attention.adapter"),
        ("lora_unet_joint_blocks_0_mlp_fc1", "mmdit.mlp.adapter"),
        ("lora_unet_joint_blocks_0_adaLN_modulation", "mmdit.modulation_norm.adapter"),
    )
    arrays = {}
    inventory = {}
    components = {}
    payload = b""
    for prefix, component in modules:
        for suffix, dtype, shape, raw, role in (
            (".lora_down.weight", "BF16", [2, 2],
             b"\x00\x3f\x00\x40\x80\x40\x00\x41", "trainable_weight"),
            (".lora_up.weight", "BF16", [2, 2],
             b"\x00\x40\x00\x3f\x00\x41\x80\x40", "trainable_weight"),
            (".alpha", "F32", [], struct.pack("<f", 4.0), "alpha_buffer"),
        ):
            key = prefix + suffix
            start = len(payload)
            payload += raw
            arrays[key] = {"dtype": dtype, "shape": shape,
                           "data_offsets": [start, len(payload)]}
            inventory[key] = {"dtype": dtype, "shape": shape, "role": role,
                              "sha256": _digest(raw)}
            if key.endswith(".weight"):
                components[key] = component
    header = json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return _digest(path.read_bytes()), inventory, components


def _host_parameter_ownership(case: dict, inventory: dict, components: dict,
                              *, topology: str) -> dict:
    """Explicitly synthetic routing evidence for pure Host shape tests."""
    policy = _support().build_reference_policy(case["case_id"])
    rows = {}
    children = {name: [] for name in policy["optimizer_profiles"]}
    for key in sorted(components):
        component = components[key]
        route = policy["components"][component]
        mode = ("fallback" if case["case_id"] == R2
                and component == "mmdit.modulation_norm.adapter" else "primary")
        profile = (route["fallback_optimizer_profile"] if mode == "fallback"
                   else route["optimizer_profile"])
        canonical = "network." + key
        rows[key] = {
            "canonical_name": canonical, "component_id": component,
            "optimizer_profile": profile, "route_kind": mode,
            "requires_grad": True, "parameter_class": "matrix_weight",
            "dtype": inventory[key]["dtype"], "shape": inventory[key]["shape"],
        }
        children[profile].append(canonical)
    return {
        "source": "prepared-runtime-ownership-v1",
        "network_split_qkv": False,
        "runtime_topology_fingerprint": topology,
        "parameter_rows": rows,
        "optimizer_child_parameter_names": {k: sorted(v) for k, v in children.items()},
    }


def _ticket(root: Path) -> dict:
    return {
        "schema": "dts.sd3c2b.phase-ticket", "version": 1,
        "run_id": "run-host-0001", "phase_id": "fresh-host-0001",
        "phase": "fresh", "case_id": R1,
        "source_commit": SHA,
        "manifest_sha256": "b" * 64,
        "input_identity_sha256": "c" * 64,
        "effective_argv_sha256": "d" * 64,
        "expected_resume_checkpoint_id": None,
        "worker_entrypoint": "tools/sd3_full_bf16_evidence_worker.py",
        "output_dir": str(root / "fresh"),
        "nonce": "e" * 64,
    }


class B0CaseAndManifestRED(unittest.TestCase):
    def test_reference_case_specs_pin_geometry_and_profiles(self):
        api = _support()
        r1, r2 = api.get_case_spec(R1), api.get_case_spec(R2)
        self.assertEqual(r1["gradient_accumulation_steps"], 1)
        self.assertEqual(r1["effective_batches_per_epoch"], 1)
        self.assertEqual(r1["optimizer_types"], ["AdamW"])
        self.assertEqual(r2["gradient_accumulation_steps"], 2)
        self.assertEqual(r2["effective_batches_per_epoch"], 2)
        self.assertEqual(set(r2["optimizer_types"]), {"Muon", "AdamW"})
        with self.assertRaises(api.FullBF16EvidenceError):
            api.get_case_spec("sd3-all-full-bf16")

    def test_canonical_r1_and_r2_policy_match_c2a_independently(self):
        api = _support()
        from tools.prepare_sd3_component_baseline import make_policy
        from tests.test_sd3_full_bf16_scope import r2_policy

        self.assertEqual(api.build_reference_policy(R1), make_policy())
        self.assertEqual(api.build_reference_policy(R2), r2_policy())

    def test_manifest_and_effective_args_reject_unreviewed_switches(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            result = _generate(Path(td), R2)
            original = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertIsNotNone(api.validate_case_manifest(original))

            for args in (
                ["--scale_weight_norms", "0"],
                ["--save_precision", "bf16"],
                ["--fp8_base"],
                ["--config_file", "unreviewed.toml"],
                ["--gradient_accumulation_steps", "1"],
            ):
                with self.subTest(options=args):
                    bad = copy.deepcopy(original)
                    for phase in ("fresh_command", "resume_command"):
                        bad[phase].extend(args)
                    with self.assertRaises(api.FullBF16EvidenceError):
                        api.validate_case_manifest(bad)

    def test_manifest_rejects_schema_case_and_missing_full_bf16(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            original = json.loads(
                Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8")
            )
            for field, value in (("version", True), ("version", 9),
                                 ("schema", "dts.sd3-component-baseline-manifest"),
                                 ("case_id", "r2-looks-like-muon")):
                with self.subTest(field=field, value=value):
                    bad = copy.deepcopy(original)
                    bad[field] = value
                    with self.assertRaises(api.FullBF16EvidenceError):
                        api.validate_case_manifest(bad)
            bad = copy.deepcopy(original)
            bad["fresh_command"].remove("--full_bf16")
            with self.assertRaises(api.FullBF16EvidenceError):
                api.validate_case_manifest(bad)


class B0GeneratorAndInputIdentityRED(unittest.TestCase):
    def test_generator_copies_isolated_dreambooth_and_never_precreates_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = _generate(root, R2)
            work = root / "work"
            subset = work / "data" / "1_dts_sd3c2b"
            self.assertFalse((work / "fresh").exists())
            self.assertFalse((work / "resume").exists())
            self.assertEqual(sorted(p.name for p in subset.glob("*.png")),
                             ["sample_0001.png", "sample_0002.png"])
            self.assertEqual(sorted(p.name for p in subset.glob("*.caption")),
                             ["sample_0001.caption", "sample_0002.caption"])
            self.assertEqual(len(list((root / "source").glob("*.png"))), 2)
            self.assertNotEqual(
                (subset / "sample_0001.caption").read_text(encoding="utf-8"),
                (subset / "sample_0002.caption").read_text(encoding="utf-8"),
            )
            self.assertEqual(Path(result["manifest"]).resolve(), (work / "manifest.json").resolve())
            payload = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["reference"]["effective_batches_per_epoch"], 2)
            self.assertEqual(payload["reference"]["gradient_accumulation_steps"], 2)
            self.assertEqual(payload["reference"]["max_train_steps"], 2)

    def test_two_distinct_png_file_hashes_do_not_prove_distinct_rgb_pixels(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _source_dataset(root, identical_pixels=True)
            a, b = sorted(src.glob("*.png"))
            self.assertNotEqual(_digest(a.read_bytes()), _digest(b.read_bytes()))
            self.assertEqual(api.decoded_rgb_pixel_sha256(a),
                             api.decoded_rgb_pixel_sha256(b))

    def test_r2_generator_rejects_different_png_metadata_with_same_pixels(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(api.FullBF16EvidenceError):
                _generate(Path(td), R2, identical_pixels=True)

    def test_generator_fails_closed_on_existing_root(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _generate(root, R1)
            with self.assertRaises(api.FullBF16EvidenceError):
                _generator().generate(
                    case_id=R1, model=root / "model.safetensors",
                    data_dir=root / "source", work_dir=root / "work",
                    external_models={
                        name: root / (name[2:] + ".safetensors")
                        for name in ("--clip_l", "--clip_g", "--t5xxl", "--vae")
                    },
                )


    def test_generator_rejects_missing_source_caption(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _source_dataset(root)
            (src / "original_0002.caption").unlink()
            model = root / "model.safetensors"
            model.write_bytes(b"host-only-header-fixture")
            assets = {}
            for key in ("--clip_l", "--clip_g", "--t5xxl", "--vae"):
                asset = root / (key[2:] + ".safetensors")
                asset.write_bytes(b"host-only-asset")
                assets[key] = asset
            with self.assertRaises(api.FullBF16EvidenceError):
                _generator().generate(
                    case_id=R2,
                    model=model,
                    data_dir=src,
                    work_dir=root / "work",
                    external_models=assets,
                )



class B0CrossPhaseInputIdentityRED(unittest.TestCase):
    def _fresh_step_artifacts(self, manifest: dict) -> None:
        ckpt = Path(manifest["fresh_checkpoint_dir"])
        ckpt.mkdir(parents=True, exist_ok=False)
        Path(manifest["fresh_adapter_path"]).write_bytes(b"checkpoint-file-fixture")

    def test_resume_preflight_accepts_finished_fresh_but_not_stale_resume(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            manifest = json.loads(
                Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8")
            )
            self._fresh_step_artifacts(manifest)
            with self.assertRaises(api.FullBF16EvidenceError):
                api.validate_case_manifest(manifest)
            self.assertEqual(
                api.validate_case_manifest(manifest, phase="resume")["phase"],
                "resume",
            )
            self.assertEqual(
                api.verify_case_inputs(manifest, phase="resume")["status"],
                "identity_pass",
            )
            resume = Path(manifest["resume_checkpoint_dir"])
            resume.mkdir(parents=True, exist_ok=False)
            Path(manifest["resume_adapter_path"]).write_bytes(
                b"checkpoint-file-fixture"
            )
            self.assertEqual(
                api.validate_case_manifest(manifest, phase="complete")["phase"],
                "complete",
            )

    def test_host_fixture_identity_pass_is_not_real_model_preflight_pass(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            manifest = json.loads(
                Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                api.verify_case_inputs(manifest)["status"], "identity_pass"
            )
            # The generator intentionally accepts fake bytes as a Host
            # fixture; the real SD3 asset/architecture gate must reject them.
            with self.assertRaises(api.FullBF16EvidenceError):
                api.validate_case_assets(manifest)

    def test_resume_rehash_rejects_mutated_caption_after_fresh(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = json.loads(
                Path(_generate(root, R1)["manifest"]).read_text(encoding="utf-8")
            )
            self._fresh_step_artifacts(manifest)
            api.verify_case_inputs(manifest, phase="resume")
            caption = root / "work" / "data" / "1_dts_sd3c2b" / "sample_0001.caption"
            caption.write_text("tampered input after Fresh\n", encoding="utf-8")
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_case_inputs(manifest, phase="resume")

    def test_resume_rehash_rejects_unrecorded_dataset_file(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = json.loads(
                Path(_generate(root, R1)["manifest"]).read_text(encoding="utf-8")
            )
            self._fresh_step_artifacts(manifest)
            (root / "work" / "data" / "1_dts_sd3c2b" / "surprise.npz").write_bytes(
                b"unapproved cache"
            )
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_case_inputs(manifest, phase="resume")

    def test_manifest_rejects_boolean_used_as_integer_reference(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = json.loads(
                Path(_generate(root, R1)["manifest"]).read_text(encoding="utf-8")
            )
            for field in ("train_batch_size", "gradient_accumulation_steps",
                          "effective_batches_per_epoch"):
                bad = copy.deepcopy(manifest)
                bad["reference"][field] = True
                with self.subTest(field=field), self.assertRaises(
                    api.FullBF16EvidenceError
                ):
                    api.validate_case_manifest(bad)


class B0OptimizerQualificationRED(unittest.TestCase):
    def test_only_pending_backend_may_receive_qualification_lease(self):
        api = _support()
        for case in (R1, R2):
            with self.subTest(case=case):
                backend, optimizers = _qualifications(case)
                self.assertIsNotNone(
                    api.validate_qualification_preconditions(case, backend, optimizers)
                )
                before = copy.deepcopy(optimizers)
                after = copy.deepcopy(optimizers)
                self.assertIsNotNone(api.validate_qualification_lease_records(
                    case, _lease_records(case), before, after
                ))

    def test_pending_optimizer_or_wrong_evidence_id_cannot_be_leased(self):
        api = _support()
        for case, name in ((R1, "AdamW"), (R2, "Muon"), (R2, "AdamW")):
            for status, evidence in (("pending", None),
                                     ("qualified", None),
                                     ("qualified", "not-d0")):
                with self.subTest(case=case, optimizer=name, status=status,
                                  evidence=evidence):
                    backend, optimizers = _qualifications(case)
                    optimizers[name] = _record(status, evidence)
                    with self.assertRaises(api.FullBF16EvidenceError):
                        api.validate_qualification_preconditions(
                            case, backend, optimizers
                        )

    def test_backend_already_qualified_or_unsupported_is_rejected(self):
        api = _support()
        for status in ("qualified", "unsupported"):
            backend, optimizers = _qualifications(R1)
            backend["status"] = status
            with self.assertRaises(api.FullBF16EvidenceError):
                api.validate_qualification_preconditions(R1, backend, optimizers)

    def test_lease_record_mutation_or_optimizer_lease_is_rejected(self):
        api = _support()
        before_backend, original = _qualifications(R2)
        _ = before_backend
        for mutation in ("optimizer_lease", "optimizer_drift",
                         "missing_backend", "extra_backend"):
            with self.subTest(mutation=mutation):
                records = _lease_records(R2)
                before, after = copy.deepcopy(original), copy.deepcopy(original)
                if mutation == "optimizer_lease":
                    records[1]["lease_applied"] = True
                elif mutation == "optimizer_drift":
                    after["Muon"]["evidence_case_id"] = "different"
                elif mutation == "missing_backend":
                    records.pop(0)
                else:
                    records.append(copy.deepcopy(records[0]))
                with self.assertRaises(api.FullBF16EvidenceError):
                    api.validate_qualification_lease_records(
                        R2, records, before, after
                    )


class B0OneShotTicketRED(unittest.TestCase):
    def test_ticket_claim_is_persistent_and_replay_fails(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, payload = root / "ticket.json", _ticket(root)
            api.issue_phase_ticket(path, payload)
            self.assertTrue(path.is_file())
            self.assertIsNotNone(api.claim_phase_ticket(path, expected=payload))
            self.assertTrue((root / "ticket.json.consumed").is_file())
            with self.assertRaises(api.FullBF16EvidenceError):
                api.claim_phase_ticket(path, expected=payload)
            with self.assertRaises(api.FullBF16EvidenceError):
                api.issue_phase_ticket(path, payload)

    def test_ticket_rejects_invalid_nonce_and_path_like_identifiers(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            for field, value in (
                ("nonce", "short"),
                ("nonce", "G" * 64),
                ("run_id", "../../outside"),
                ("phase_id", "x"),
            ):
                with self.subTest(field=field):
                    payload = _ticket(Path(td))
                    payload[field] = value
                    with self.assertRaises(api.FullBF16EvidenceError):
                        api.issue_phase_ticket(
                            Path(td) / f"invalid_{field}.json", payload,
                        )

    def test_ticket_tamper_fails_before_consumption(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, payload = root / "ticket.json", _ticket(root)
            api.issue_phase_ticket(path, payload)
            tampered = copy.deepcopy(payload)
            tampered["manifest_sha256"] = "f" * 64
            with self.assertRaises(api.FullBF16EvidenceError):
                api.claim_phase_ticket(path, expected=tampered)
            self.assertFalse((root / "ticket.json.consumed").exists())

    def test_concurrent_claim_has_exactly_one_winner(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, payload = root / "ticket.json", _ticket(root)
            api.issue_phase_ticket(path, payload)

            def claim(_):
                try:
                    api.claim_phase_ticket(path, expected=payload)
                    return True
                except api.FullBF16EvidenceError:
                    return False

            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(claim, range(8)))
            self.assertEqual(outcomes.count(True), 1)

    def test_atomic_result_must_not_overwrite_prior_proof(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            result = Path(td) / "result.json"
            payload = {"schema": "dts.sd3c2b.full-bf16-worker-phase",
                       "version": 1, "status": "incomplete"}
            api.publish_phase_result(result, payload)
            self.assertEqual(json.loads(result.read_text(encoding="utf-8")),
                             payload)
            with self.assertRaises(api.FullBF16EvidenceError):
                api.publish_phase_result(result, {"status": "pass"})


class B0AdapterArtifactBindingRED(unittest.TestCase):
    def test_real_bf16_file_tensors_match_in_memory_expected_inventory(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dts-step00000001.safetensors"
            sha, inventory = _write_reference_adapter(path)
            self.assertIsNotNone(api.verify_adapter_file(
                path, expected_file_sha256=sha, expected_tensors=inventory
            ))

    def test_swapped_or_rewritten_adapter_fails_even_when_filename_matches(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dts-step00000001.safetensors"
            sha, inventory = _write_tiny_bf16_safetensors(
                path, payload=b"\x00\x3f\x00\x40\x80\x40\x00\x41"
            )
            _write_reference_adapter(path, alpha_bytes=struct.pack("<f", 8.0))
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_adapter_file(
                    path, expected_file_sha256=sha, expected_tensors=inventory
                )

    def test_checkpoint_binding_rejects_wrong_phase_step_or_checkpoint_id(self):
        api = _support()
        binding = {
            "case_id": R1, "phase_id": "fresh-unique", "logical_step": 1,
            "checkpoint_id": "c" * 32,
            "adapter_path": str(Path(tempfile.gettempdir()) / "dts-step00000001.safetensors"),
            "adapter_file_sha256": "a" * 64,
            "adapter_tensor_inventory_sha256": "b" * 64,
        }
        progress = {"schema": "dts.parameter-policy.checkpoint-progress",
                    "version": 1, "checkpoint_id": "c" * 32}
        self.assertIsNotNone(api.validate_adapter_checkpoint_metadata(
            binding, progress, expected_case_id=R1,
            expected_phase_id="fresh-unique", expected_step=1,
            expected_adapter_path=binding["adapter_path"],
        ))
        for field, value in (("logical_step", 2),
                             ("phase_id", "resume-unique"),
                             ("checkpoint_id", "d" * 32)):
            with self.subTest(field=field):
                wrong = dict(binding)
                wrong[field] = value
                with self.assertRaises(api.FullBF16EvidenceError):
                    api.validate_adapter_checkpoint_metadata(
                        wrong, progress, expected_case_id=R1,
                        expected_phase_id="fresh-unique", expected_step=1,
                        expected_adapter_path=binding["adapter_path"],
                    )




def _write_host_checkpoint(
    case: dict, *, phase: str = "fresh",
    phase_id: str = "host-phase-1", parent_id: str | None = None,
) -> tuple[dict, dict]:
    """Real file layout + synthetic CPU values, never a CUDA qualification."""
    api = _support()
    step = 1 if phase == "fresh" else 2
    checkpoint = Path(case["fresh_checkpoint_dir" if phase == "fresh" else "resume_checkpoint_dir"])
    adapter = Path(case["fresh_adapter_path" if phase == "fresh" else "resume_adapter_path"])
    checkpoint.mkdir(parents=True, exist_ok=False)
    adapter.parent.mkdir(parents=True, exist_ok=True)
    file_sha, live_tensors, components = _write_three_component_reference_adapter(adapter)
    # The pinned Accelerate save_state produces a separately serialized
    # Network state, not merely the externally saved Step Adapter.
    (checkpoint / "model.safetensors").write_bytes(adapter.read_bytes())
    (checkpoint / "random_states_0.pkl").write_bytes(b"cpu-host-placeholder-rng-UNPICKLED")
    policy = api.build_reference_policy(case["case_id"])
    profiles = {name: info["type"] for name, info in policy["optimizer_profiles"].items()}
    execution = {
        "schema": "dts.parameter-policy.execution-identity", "version": 1,
        "train_type": "sd3-lora",
        "features": {"full_bf16": {
            "mixed_precision": "bf16", "trainable_parameter_dtype": "bfloat16",
        }},
    }
    sched = api._expected_reference_scheduler_identity()
    topo = "host-reference-topology-v1"
    pmanifest = {
        "version": 2, "train_type": "sd3-lora",
        "policy_hash": case["input_identity"]["canonical_policy_hash"],
        "runtime_topology_fingerprint": topo,
        "trainable_components": sorted(k for k, v in policy["components"].items() if v["train"]),
        "frozen_components": sorted(k for k, v in policy["components"].items() if not v["train"]),
        "trainable_parameter_tensors": len(components),
        "trainable_parameter_elements": len(components) * 4,
        "optimizer_profiles": profiles,
        "optimizers": [{"profile_name": k, "optimizer_type": v,
                        "topology_fingerprint": "host-child"}
                       for k, v in profiles.items()],
        "scheduler_identity": sched, "scheduler_signature": api._json_sha256(sched),
        "schedulers": [{"profile_name": k, "mode": "external",
                        "scheduler_type": "torch.optim.lr_scheduler.LambdaLR",
                        "topology_fingerprint": "host-scheduler"}
                       for k in profiles],
        "execution_identity": execution,
        "execution_signature": api._json_sha256(execution),
    }
    checkpoint_id = ("a" if step == 1 else "b") * 32
    progress = {
        "schema": "dts.parameter-policy.checkpoint-progress", "version": 1,
        "train_type": "sd3-lora",
        "policy_hash": pmanifest["policy_hash"],
        "runtime_topology_fingerprint": topo,
        "execution_signature": pmanifest["execution_signature"],
        "checkpoint_id": checkpoint_id,
        "resume_source_checkpoint_id": parent_id,
        "optimizer_profiles": {
            name: {"optimizer_type": typ, "step_values": [step]}
            for name, typ in profiles.items()
        },
        "scheduler": {"step_count": step + 1, "last_epoch": step},
    }
    geometry = case["reference"]
    sd3 = {
        "schema": "dts.sd3.resume-progress", "version": 1,
        "completed_optimizer_steps": step, "current_epoch": step,
        "dataloader_batches": geometry["effective_batches_per_epoch"],
        "gradient_accumulation_steps": geometry["gradient_accumulation_steps"],
        "num_processes": 1,
    }
    sidecars = {
        "dts_parameter_policy_manifest.json": pmanifest,
        "dts_parameter_policy_progress.json": progress,
        "dts_sd3_resume_progress.json": sd3,
        "train_state.json": {"current_step": step, "current_epoch": step},
        "dts_sd3_forward_semantics.json": {
            "schema": "dts.sd3.forward-semantics", "version": 1,
            "attention_contract": "extra-attention-head-dim-v1",
            "model_type_hint": "5-medium",
            "depth": 24, "hidden_size": 1536, "num_heads": 24, "head_dim": 64,
            "extra_attention_layers": list(range(13)),
        },
    }
    for name, payload in sidecars.items():
        (checkpoint / name).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    (checkpoint / "optimizer.bin").write_bytes(b"host-optimizer-file-" + bytes([step]))
    (checkpoint / "scheduler.bin").write_bytes(b"host-scheduler-file-" + bytes([step]))
    worker = {
        "case_id": case["case_id"], "phase_id": phase_id,
        "logical_step": step, "checkpoint_id": checkpoint_id,
        "expected_parent_checkpoint_id": parent_id,
        "source_commit": case["source_commit"],
        "input_identity_sha256": api._json_sha256(case["input_identity"]),
        "runtime_topology_fingerprint": topo,
        "scheduler_identity": sched,
        "scheduler_signature": api._json_sha256(sched),
        "forward_semantics": sidecars["dts_sd3_forward_semantics.json"],
        "parameter_ownership": _host_parameter_ownership(
            case, live_tensors, components, topology=topo,
        ),
        "adapter_tensors": live_tensors,
        "state_file_sha256": {
            p.name: _digest(p.read_bytes()) for p in checkpoint.iterdir()
        },
    }
    binding = {
        "case_id": case["case_id"], "phase_id": phase_id,
        "logical_step": step, "checkpoint_id": checkpoint_id,
        "adapter_path": str(adapter),
        "adapter_file_sha256": file_sha,
        "adapter_tensor_inventory_sha256": api._json_sha256(live_tensors),
    }
    return worker, binding

class B0StrictReviewRegressionRED(unittest.TestCase):
    def test_real_format_scalar_alpha_f32_and_bf16_are_full_hash_verified(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            for dtype in ("F32", "BF16"):
                with self.subTest(alpha=dtype):
                    path = Path(td) / f"{dtype}-step00000001.safetensors"
                    sha, tensors = _write_reference_adapter(path, alpha_dtype=dtype)
                    evidence = api.verify_adapter_file(
                        path, expected_file_sha256=sha, expected_tensors=tensors
                    )
                    self.assertEqual(evidence["tensor_count"], 3)

    def test_scalar_alpha_missing_or_tampered_or_wrong_role_rejected(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.safetensors"
            sha, expected = _write_reference_adapter(path)
            for mutation in ("missing_alpha", "wrong_buffer_dtype", "wrong_buffer_hash",
                             "wrong_buffer_shape", "unexpected_key", "fp32_trainable"):
                with self.subTest(mutation=mutation):
                    if mutation == "missing_alpha":
                        alt_sha, _ = _write_reference_adapter(path, include_alpha=False)
                        inventory = expected
                    elif mutation == "unexpected_key":
                        alt_sha, inventory = _write_reference_adapter(path, unknown=True)
                    elif mutation == "fp32_trainable":
                        alt_sha, inventory = _write_reference_adapter(path, weight_dtype="F32")
                    else:
                        alt_sha, inventory = _write_reference_adapter(path)
                        inventory = copy.deepcopy(inventory)
                        alpha = next(k for k in inventory if k.endswith(".alpha"))
                        if mutation == "wrong_buffer_dtype":
                            inventory[alpha]["dtype"] = "BF16"
                        elif mutation == "wrong_buffer_shape":
                            inventory[alpha]["shape"] = [1]
                        else:
                            inventory[alpha]["sha256"] = "f" * 64
                    with self.assertRaises(api.FullBF16EvidenceError):
                        api.verify_adapter_file(
                            path, expected_file_sha256=alt_sha,
                            expected_tensors=inventory,
                        )

    def test_extra_dreambooth_sibling_subset_rejected_before_fresh_and_resume(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = json.loads(Path(_generate(root, R1)["manifest"]).read_text(encoding="utf-8"))
            sibling = root / "work" / "data" / "1_extra"
            sibling.mkdir()
            from PIL import Image
            Image.new("RGB", (512, 512), (9, 17, 22)).save(sibling / "unexpected.png")
            (sibling / "unexpected.caption").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_case_inputs(manifest, phase="fresh")
            ckpt = Path(manifest["fresh_checkpoint_dir"])
            ckpt.mkdir(parents=True)
            Path(manifest["fresh_adapter_path"]).write_bytes(b"fixture")
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_case_inputs(manifest, phase="resume")

    def test_input_parent_rejects_extra_file_and_nested_unapproved_subtree(self):
        api = _support()
        for suffix in ("extra-file", "extra-directory"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                manifest = json.loads(Path(_generate(root, R1)["manifest"]).read_text(encoding="utf-8"))
                extra = root / "work" / "data" / "stray"
                if suffix == "extra-directory":
                    extra.mkdir()
                    (extra / "junk").write_bytes(b"data")
                else:
                    extra.write_bytes(b"data")
                with self.subTest(extra=suffix), self.assertRaises(api.FullBF16EvidenceError):
                    api.verify_case_inputs(manifest)

    def test_metadata_only_does_not_claim_physical_checkpoint_pass(self):
        api = _support()
        path = Path(tempfile.gettempdir()) / "uncreated-step00000001.safetensors"
        data = {"case_id": R1, "phase_id": "fresh-unique", "logical_step": 1,
                "checkpoint_id": "a"*32, "adapter_path": str(path),
                "adapter_file_sha256": "b"*64, "adapter_tensor_inventory_sha256": "c"*64}
        progress = {"schema":"dts.parameter-policy.checkpoint-progress",
                    "version":1,"checkpoint_id":"a"*32}
        metadata = api.validate_adapter_checkpoint_metadata(
            data, progress, expected_case_id=R1, expected_phase_id="fresh-unique",
            expected_step=1, expected_adapter_path=str(path))
        self.assertEqual(metadata["status"], "metadata_valid")
        with self.assertRaises(api.FullBF16EvidenceError):
            api.verify_adapter_checkpoint_binding(
                case_manifest={"schema":"untrusted"}, phase="fresh",
                worker_snapshot={}, binding=data
            )
        wrong = dict(data, case_id=R2)
        with self.assertRaises(api.FullBF16EvidenceError):
            api.validate_adapter_checkpoint_metadata(
                wrong, progress, expected_case_id=R1, expected_phase_id="fresh-unique",
                expected_step=1, expected_adapter_path=str(path))

    def test_result_writer_refuses_final_cuda_pass_without_host_verification(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            for status in ("pass", "qualified"):
                with self.subTest(status=status), self.assertRaises(api.FullBF16EvidenceError):
                    api.publish_phase_result(
                        Path(td) / (status + ".json"),
                        {"schema":"dts.sd3c2b.full-bf16-worker-phase",
                         "version":1,"status":status}
                    )
            self.assertIsNone(next(Path(td).glob("*.json"), None))

    def test_generator_source_is_not_accepted_when_git_status_is_dirty(self):
        gen = _generator()
        from unittest.mock import patch
        from types import SimpleNamespace
        def fake_run(cmd, **kwargs):
            if cmd[1] == "rev-parse":
                return SimpleNamespace(stdout=SHA + "\n")
            if "status" in cmd:
                return SimpleNamespace(stdout=" M tools/prepare_sd3_full_bf16_cases.py\n")
            return SimpleNamespace(stdout="")
        with patch.object(gen.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(_support().FullBF16EvidenceError):
                gen._source_commit()


    def test_physical_host_binding_checks_real_state_file_layout_not_cuda(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = json.loads(Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8"))
            snapshot, binding = _write_host_checkpoint(case)
            result = api.verify_adapter_checkpoint_binding(
                case_manifest=case, phase="fresh", worker_snapshot=snapshot,
                binding=binding, expected_phase_id="host-phase-1",
            )
            self.assertEqual(result["status"], "physical_binding_valid")
            self.assertEqual(result["checkpoint_id"], "a" * 32)
            self.assertNotEqual(result["status"], "pass")

    def test_r2_fresh_then_resumed_step2_bound_to_exact_parent_checkpoint(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = json.loads(Path(_generate(Path(td), R2)["manifest"]).read_text(encoding="utf-8"))
            fresh_snapshot, fresh_binding = _write_host_checkpoint(
                case, phase="fresh", phase_id="host-r2-fresh",
            )
            first = api.verify_adapter_checkpoint_binding(
                case_manifest=case, phase="fresh", worker_snapshot=fresh_snapshot,
                binding=fresh_binding, expected_phase_id="host-r2-fresh",
            )
            self.assertEqual(first["status"], "physical_binding_valid")
            resumed_snapshot, resumed_binding = _write_host_checkpoint(
                case, phase="resume", phase_id="host-r2-resume",
                parent_id=first["checkpoint_id"],
            )
            second = api.verify_adapter_checkpoint_binding(
                case_manifest=case, phase="resume", worker_snapshot=resumed_snapshot,
                binding=resumed_binding, expected_phase_id="host-r2-resume",
                accepted_fresh_evidence=first,
            )
            self.assertEqual(second["status"], "physical_binding_valid")
            self.assertNotEqual(second["checkpoint_id"], first["checkpoint_id"])
            resumed_snapshot["expected_parent_checkpoint_id"] = "wrong-parent"
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_adapter_checkpoint_binding(
                    case_manifest=case, phase="resume", worker_snapshot=resumed_snapshot,
                    binding=resumed_binding, expected_phase_id="host-r2-resume",
                    accepted_fresh_evidence=first,
                )

    def test_complete_saved_module_allows_frozen_f32_weights_with_live_role(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "frozen.safetensors"
            sha, saved = _write_reference_adapter(path, weight_dtype="F32")
            # This fixture models frozen Adapter weights in the same file.
            # Full-BF16 is mandatory for trainable parameters, not frozen ones.
            for key, value in saved.items():
                if key.endswith(".weight"):
                    value["role"] = "frozen_weight"
            # The fake generator's down Matrix remains BF16; both dtypes
            # must be checked against the exact Worker live descriptors.
            self.assertEqual(api.verify_adapter_file(
                path, expected_file_sha256=sha, expected_tensors=saved,
            )["status"], "file_verified")

    def test_real_state_binding_rejects_missing_swapped_and_mismatched_evidence(self):
        api = _support()
        changes = ("wrong_case", "missing_adapter", "swapped_adapter",
                   "wrong_state_dir", "altered_optimizer", "progress_mismatch",
                   "wrong_phase_id", "wrong_scheduler", "wrong_parent",
                   "wrong_child_optimizer_type")
        for mutation in changes:
            with tempfile.TemporaryDirectory() as td:
                case = json.loads(Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8"))
                snapshot, binding = _write_host_checkpoint(case)
                checkpoint = Path(case["fresh_checkpoint_dir"])
                adapter = Path(case["fresh_adapter_path"])
                if mutation == "wrong_case":
                    binding["case_id"] = R2
                elif mutation == "missing_adapter":
                    adapter.unlink()
                elif mutation == "swapped_adapter":
                    _write_reference_adapter(adapter, alpha_bytes=struct.pack("<f", 8.0))
                elif mutation == "wrong_state_dir":
                    case["fresh_checkpoint_dir"] = str(checkpoint / "wrong")
                elif mutation == "altered_optimizer":
                    (checkpoint / "optimizer.bin").write_bytes(b"tampered")
                elif mutation == "progress_mismatch":
                    p = checkpoint / "dts_parameter_policy_progress.json"
                    record = json.loads(p.read_text(encoding="utf-8"))
                    record["checkpoint_id"] = "z" * 32
                    p.write_text(json.dumps(record), encoding="utf-8")
                elif mutation == "wrong_phase_id":
                    snapshot["phase_id"] = "wrong-phase"
                elif mutation == "wrong_scheduler":
                    (checkpoint / "scheduler.bin").unlink()
                elif mutation == "wrong_parent":
                    p = checkpoint / "dts_parameter_policy_progress.json"
                    record = json.loads(p.read_text(encoding="utf-8"))
                    record["resume_source_checkpoint_id"] = "not-a-real-parent"
                    p.write_text(json.dumps(record), encoding="utf-8")
                elif mutation == "wrong_child_optimizer_type":
                    p = checkpoint / "dts_parameter_policy_manifest.json"
                    record = json.loads(p.read_text(encoding="utf-8"))
                    record["optimizers"][0]["optimizer_type"] = "SGD"
                    p.write_text(json.dumps(record), encoding="utf-8")
                    # A matching forged Worker file digest must not bypass
                    # semantic optimizer-child type validation.
                    snapshot["state_file_sha256"][p.name] = _digest(p.read_bytes())
                with self.subTest(mutation=mutation), self.assertRaises(api.FullBF16EvidenceError):
                    api.verify_adapter_checkpoint_binding(
                        case_manifest=case, phase="fresh",
                        worker_snapshot=snapshot, binding=binding,
                        expected_phase_id="host-phase-1",
                    )

    def test_submodule_inventory_records_uninitialized_gitlinks_without_qualification(self):
        gen = _generator()
        from types import SimpleNamespace
        from unittest.mock import patch

        def git_status(command, **kwargs):
            self.assertEqual(command[:3], ["git", "submodule", "status"])
            return SimpleNamespace(stdout=(
                f" {SHA} frontend (heads/main)\n"
                f"-{SHA} mikazuki/dataset-tag-editor\n"
                f" {SHA} sd-scripts (heads/main)\n"
            ))
        with patch.object(gen.subprocess, "run", side_effect=git_status):
            inventory = gen._source_submodule_inventory()
        self.assertEqual(inventory["frontend"]["checkout"], "initialized")
        self.assertEqual(inventory["mikazuki/dataset-tag-editor"]["checkout"], "uninitialized")
        self.assertEqual(inventory["sd-scripts"]["gitlink_sha"], SHA)

    def test_source_manifest_rejects_unknown_or_forged_gitlinks(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            manifest = json.loads(Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8"))
            self.assertIn("sd-scripts", manifest["source_submodules"])
            for kind in ("unknown", "nonhex", "missing", "false_type"):
                with self.subTest(kind=kind):
                    altered = copy.deepcopy(manifest)
                    if kind == "unknown":
                        altered["source_submodules"]["outside"] = {
                            "gitlink_sha": SHA, "checkout": "initialized",
                        }
                    elif kind == "nonhex":
                        altered["source_submodules"]["sd-scripts"]["gitlink_sha"] = "wrong"
                    elif kind == "missing":
                        altered["source_submodules"].pop("frontend")
                    else:
                        altered["source_submodules"]["sd-scripts"]["checkout"] = True
                    with self.assertRaises(api.FullBF16EvidenceError):
                        api.validate_case_manifest(altered)

    def test_git_source_identity_accepts_clean_and_rejects_gitlink_drift(self):
        gen = _generator()
        from unittest.mock import patch
        from types import SimpleNamespace

        def run_git(command, *, gitlink_drift=False, **kwargs):
            if command[1] == "rev-parse":
                return SimpleNamespace(stdout=SHA + "\n")
            if command[1] == "status":
                return SimpleNamespace(stdout="")
            if command[1] == "submodule":
                return SimpleNamespace(
                    stdout=("+1234567 deviating-submodule\n") if gitlink_drift else ""
                )
            raise AssertionError(f"Unexpected Git query: {command}")

        with patch.object(gen.subprocess, "run", side_effect=run_git):
            self.assertEqual(gen._source_commit(), SHA)
        with patch.object(
            gen.subprocess, "run",
            side_effect=lambda command, **kwargs: run_git(
                command, gitlink_drift=True, **kwargs
            ),
        ):
            with self.assertRaises(_support().FullBF16EvidenceError):
                gen._source_commit()

    def test_asset_preflight_integrates_medium_shape_audit_but_rejects_other_variants(self):
        api = _support()
        from unittest.mock import patch
        from tests.test_sd3_full_bf16_scope import shape_fixture
        from mikazuki import sd3_full_bf16_scope as scope
        with tempfile.TemporaryDirectory() as td:
            payload = json.loads(Path(_generate(Path(td), R1)["manifest"]).read_text(encoding="utf-8"))
            with patch.object(scope, "_read_safetensors_shape_index", return_value=shape_fixture()):
                # This intentionally stubs the real model header. Its result
                # is only a Host integration fixture, never physical asset
                # preflight evidence or CUDA qualification.
                integrated = api.validate_case_assets(payload)
                host_fixture = {
                    "status": "host_fixture",
                    "tested_validator_branch": integrated["status"],
                }
                self.assertEqual(host_fixture["status"], "host_fixture")
                self.assertEqual(host_fixture["tested_validator_branch"], "preflight_pass")
            for variant in ("sd3-medium", "sd35-large-topology"):
                with self.subTest(variant=variant):
                    with patch.object(
                        scope, "_read_safetensors_shape_index",
                        return_value=shape_fixture(variant=variant),
                    ):
                        with self.assertRaises(api.FullBF16EvidenceError):
                            api.validate_case_assets(payload)


class B0FinalStrictReviewR2RED(unittest.TestCase):
    """Source-derived negative tests: failures must be from Host Binder itself."""

    def _case(self, root: Path, case_id: str = R1) -> dict:
        return json.loads(Path(_generate(root, case_id)["manifest"]).read_text(encoding="utf-8"))

    def _bind(self, api, case, snapshot, binding, phase="fresh", phase_id="host-phase-1"):
        return api.verify_adapter_checkpoint_binding(
            case_manifest=case, phase=phase, worker_snapshot=snapshot,
            binding=binding, expected_phase_id=phase_id,
        )

    def test_red_checkpoint_missing_accelerate_network_and_rng_cannot_bind(self):
        api = _support()
        for filename in ("model.safetensors", "random_states_0.pkl",
                         "optimizer.bin", "scheduler.bin"):
            with tempfile.TemporaryDirectory() as td:
                case = self._case(Path(td))
                snapshot, binding = _write_host_checkpoint(case)
                (Path(case["fresh_checkpoint_dir"]) / filename).unlink()
                with self.subTest(missing=filename), self.assertRaises(api.FullBF16EvidenceError):
                    self._bind(api, case, snapshot, binding)

    def test_state_rejects_an_extra_accelerate_checkpoint_file(self):
        api = _support()
        for filename in ("optimizer_1.bin", "custom_checkpoint_0.pkl", "unknown.pt"):
            with tempfile.TemporaryDirectory() as td:
                case = self._case(Path(td))
                snapshot, binding = _write_host_checkpoint(case)
                path = Path(case["fresh_checkpoint_dir"]) / filename
                path.write_bytes(b"forged-extra-accelerate-state")
                snapshot["state_file_sha256"][filename] = _digest(path.read_bytes())
                with self.subTest(extra=filename), self.assertRaises(api.FullBF16EvidenceError):
                    self._bind(api, case, snapshot, binding)

    def test_network_state_tensor_swapped_even_with_updated_file_digest(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td))
            snapshot, binding = _write_host_checkpoint(case)
            path = Path(case["fresh_checkpoint_dir"]) / "model.safetensors"
            _write_reference_adapter(path, alpha_bytes=struct.pack("<f", 9.0))
            snapshot["state_file_sha256"][path.name] = _digest(path.read_bytes())
            with self.assertRaises(api.FullBF16EvidenceError):
                self._bind(api, case, snapshot, binding)

    def test_red_resume_coupled_false_parent_is_rejected_against_actual_fresh(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td), R2)
            fresh_snapshot, fresh_binding = _write_host_checkpoint(
                case, phase="fresh", phase_id="fresh-coupled",
            )
            accepted_fresh = self._bind(
                api, case, fresh_snapshot, fresh_binding,
                phase="fresh", phase_id="fresh-coupled",
            )
            forged = "c" * 32
            snapshot, binding = _write_host_checkpoint(
                case, phase="resume", phase_id="resume-coupled",
                parent_id=forged,
            )
            # Two self-consistent false claims cannot establish real lineage.
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_adapter_checkpoint_binding(
                    case_manifest=case, phase="resume", worker_snapshot=snapshot,
                    binding=binding, expected_phase_id="resume-coupled",
                    accepted_fresh_evidence=accepted_fresh,
                )

    def test_red_all_frozen_roles_cannot_evict_required_bf16_components(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td))
            snapshot, binding = _write_host_checkpoint(case)
            adapter = Path(case["fresh_adapter_path"])
            file_sha, inventory = _write_reference_adapter(adapter, weight_dtype="F32")
            for key, entry in inventory.items():
                if key.endswith(".weight"):
                    entry["role"] = "frozen_weight"
            snapshot["adapter_tensors"] = inventory
            binding["adapter_file_sha256"] = file_sha
            binding["adapter_tensor_inventory_sha256"] = api._json_sha256(inventory)
            with self.assertRaises(api.FullBF16EvidenceError):
                self._bind(api, case, snapshot, binding)

    def test_red_forward_semantics_wrong_schema_hint_or_version_not_accepted(self):
        api = _support()
        for field, bad in (("schema", "not-the-forward-schema"),
                           ("version", 2), ("model_type_hint", "wrong-model")):
            with tempfile.TemporaryDirectory() as td:
                case = self._case(Path(td))
                snapshot, binding = _write_host_checkpoint(case)
                path = Path(case["fresh_checkpoint_dir"]) / "dts_sd3_forward_semantics.json"
                sem = json.loads(path.read_text(encoding="utf-8"))
                sem[field] = bad
                path.write_text(json.dumps(sem), encoding="utf-8")
                snapshot["state_file_sha256"][path.name] = _digest(path.read_bytes())
                with self.subTest(field=field), self.assertRaises(api.FullBF16EvidenceError):
                    self._bind(api, case, snapshot, binding)

    def test_red_scheduler_self_signed_wrong_reference_config_rejected(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td))
            snapshot, binding = _write_host_checkpoint(case)
            path = Path(case["fresh_checkpoint_dir"]) / "dts_parameter_policy_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["scheduler_identity"]["lr_scheduler"] = "cosine"
            manifest["scheduler_signature"] = api._json_sha256(manifest["scheduler_identity"])
            path.write_text(json.dumps(manifest), encoding="utf-8")
            snapshot["state_file_sha256"][path.name] = _digest(path.read_bytes())
            with self.assertRaises(api.FullBF16EvidenceError):
                self._bind(api, case, snapshot, binding)

    def test_parameter_ownership_rejects_wrong_optimizer_child_and_frozen_trick(self):
        api = _support()
        mutations = ("all_frozen", "empty_component", "duplicate_child",
                     "wrong_profile", "split_qkv", "wrong_saved_name")
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as td:
                case = self._case(Path(td), R2)
                snap, bind = _write_host_checkpoint(case)
                owner = snap["parameter_ownership"]
                rows = owner["parameter_rows"]
                if mutation == "all_frozen":
                    for key, row in rows.items():
                        row["requires_grad"] = False
                        row["route_kind"] = "frozen"
                        row["optimizer_profile"] = None
                        snap["adapter_tensors"][key]["role"] = "frozen_weight"
                    bind["adapter_tensor_inventory_sha256"] = api._json_sha256(snap["adapter_tensors"])
                elif mutation == "empty_component":
                    for row in rows.values():
                        if row["component_id"] == "mmdit.mlp.adapter":
                            row["component_id"] = "mmdit.attention.adapter"
                elif mutation == "duplicate_child":
                    k = next(iter(owner["optimizer_child_parameter_names"]))
                    names = owner["optimizer_child_parameter_names"][k]
                    names.append(names[0])
                elif mutation == "wrong_profile":
                    first = next(iter(rows.values()))
                    first["optimizer_profile"] = "forged_optimizer"
                elif mutation == "split_qkv":
                    owner["network_split_qkv"] = True
                elif mutation == "wrong_saved_name":
                    first = next(iter(rows.values()))
                    first["canonical_name"] = "network.another_module.weight"
                with self.subTest(mutation=mutation), self.assertRaises(api.FullBF16EvidenceError):
                    self._bind(api, case, snap, bind)

    def test_resume_checks_fresh_digest_parent_identity_and_same_id(self):
        api = _support()
        mutations = ("fresh_file_changed", "coupled_wrong_parent", "same_checkpoint_id")
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as td:
                case = self._case(Path(td), R2)
                fresh_snap, fresh_bind = _write_host_checkpoint(
                    case, phase="fresh", phase_id="host-fresh-r2",
                )
                accepted = api.verify_adapter_checkpoint_binding(
                    case_manifest=case, phase="fresh", worker_snapshot=fresh_snap,
                    binding=fresh_bind, expected_phase_id="host-fresh-r2",
                )
                resume_snap, resume_bind = _write_host_checkpoint(
                    case, phase="resume", phase_id="host-resume-r2",
                    parent_id=accepted["checkpoint_id"],
                )
                if mutation == "fresh_file_changed":
                    path = Path(case["fresh_checkpoint_dir"]) / "random_states_0.pkl"
                    path.write_bytes(b"changed-after-fresh-accepted")
                elif mutation == "coupled_wrong_parent":
                    bogus = "c" * 32
                    path = Path(case["resume_checkpoint_dir"]) / "dts_parameter_policy_progress.json"
                    progress = json.loads(path.read_text(encoding="utf-8"))
                    progress["resume_source_checkpoint_id"] = bogus
                    path.write_text(json.dumps(progress), encoding="utf-8")
                    resume_snap["state_file_sha256"][path.name] = _digest(path.read_bytes())
                    resume_snap["expected_parent_checkpoint_id"] = bogus
                elif mutation == "same_checkpoint_id":
                    path = Path(case["resume_checkpoint_dir"]) / "dts_parameter_policy_progress.json"
                    progress = json.loads(path.read_text(encoding="utf-8"))
                    progress["checkpoint_id"] = accepted["checkpoint_id"]
                    path.write_text(json.dumps(progress), encoding="utf-8")
                    resume_snap["state_file_sha256"][path.name] = _digest(path.read_bytes())
                    resume_snap["checkpoint_id"] = accepted["checkpoint_id"]
                    resume_bind["checkpoint_id"] = accepted["checkpoint_id"]
                with self.subTest(mutation=mutation), self.assertRaises(api.FullBF16EvidenceError):
                    api.verify_adapter_checkpoint_binding(
                        case_manifest=case, phase="resume", worker_snapshot=resume_snap,
                        binding=resume_bind, expected_phase_id="host-resume-r2",
                        accepted_fresh_evidence=accepted,
                    )

    def test_forward_and_scheduler_self_consistent_forgery_still_rejected(self):
        api = _support()
        for scenario in ("forged_forward", "forged_scheduler", "wrong_child_mode"):
            with tempfile.TemporaryDirectory() as td:
                case = self._case(Path(td))
                snap, bind = _write_host_checkpoint(case)
                checkpoint = Path(case["fresh_checkpoint_dir"])
                if scenario == "forged_forward":
                    path = checkpoint / "dts_sd3_forward_semantics.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["model_type_hint"] = "incorrect-but-internally-agreed"
                    snap["forward_semantics"] = payload
                else:
                    path = checkpoint / "dts_parameter_policy_manifest.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if scenario == "forged_scheduler":
                        payload["scheduler_identity"]["lr_scheduler"] = "cosine"
                        payload["scheduler_signature"] = api._json_sha256(payload["scheduler_identity"])
                        snap["scheduler_identity"] = payload["scheduler_identity"]
                        snap["scheduler_signature"] = payload["scheduler_signature"]
                    else:
                        payload["schedulers"][0]["mode"] = "optimizer_managed"
                path.write_text(json.dumps(payload), encoding="utf-8")
                snap["state_file_sha256"][path.name] = _digest(path.read_bytes())
                with self.subTest(scenario=scenario), self.assertRaises(api.FullBF16EvidenceError):
                    self._bind(api, case, snap, bind)

    def test_red_step_zero_progress_cannot_prove_completed_step(self):
        api = _support()
        with tempfile.TemporaryDirectory() as td:
            case = self._case(Path(td))
            snapshot, binding = _write_host_checkpoint(case)
            path = Path(case["fresh_checkpoint_dir"]) / "dts_parameter_policy_progress.json"
            progress = json.loads(path.read_text(encoding="utf-8"))
            for child in progress["optimizer_profiles"].values():
                child["step_values"] = [0]
            progress["scheduler"]["step_count"] = 0
            path.write_text(json.dumps(progress), encoding="utf-8")
            snapshot["state_file_sha256"][path.name] = _digest(path.read_bytes())
            with self.assertRaises(api.FullBF16EvidenceError):
                self._bind(api, case, snapshot, binding)


if __name__ == "__main__":
    unittest.main()
