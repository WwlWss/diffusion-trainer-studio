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
            self.assertEqual(Path(result["manifest"]), work / "manifest.json")
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
            sha, inventory = _write_tiny_bf16_safetensors(
                path, payload=b"\x00\x3f\x00\x40\x80\x40\x00\x41"
            )
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
            _write_tiny_bf16_safetensors(
                path, payload=b"\x00\x3f\x00\x40\x80\x40\x00\x42"
            )
            with self.assertRaises(api.FullBF16EvidenceError):
                api.verify_adapter_file(
                    path, expected_file_sha256=sha, expected_tensors=inventory
                )

    def test_checkpoint_binding_rejects_wrong_phase_step_or_checkpoint_id(self):
        api = _support()
        binding = {
            "case_id": R1, "phase_id": "fresh-unique", "logical_step": 1,
            "checkpoint_id": "c" * 32,
            "adapter_path": "/outside/dts-step00000001.safetensors",
            "adapter_file_sha256": "a" * 64,
            "adapter_tensor_inventory_sha256": "b" * 64,
        }
        progress = {"schema": "dts.parameter-policy.checkpoint-progress",
                    "version": 1, "checkpoint_id": "c" * 32}
        self.assertIsNotNone(api.validate_adapter_checkpoint_binding(
            binding, progress, expected_phase_id="fresh-unique", expected_step=1
        ))
        for field, value in (("logical_step", 2),
                             ("phase_id", "resume-unique"),
                             ("checkpoint_id", "d" * 32)):
            with self.subTest(field=field):
                wrong = dict(binding)
                wrong[field] = value
                with self.assertRaises(api.FullBF16EvidenceError):
                    api.validate_adapter_checkpoint_binding(
                        wrong, progress, expected_phase_id="fresh-unique",
                        expected_step=1,
                    )


if __name__ == "__main__":
    unittest.main()
