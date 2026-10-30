"""Fix C C1 RED: restricted Worker/Host timeline contract.

These are intentionally failing tests until Fix C GREEN. No CUDA, Torch
or model checkpoint is needed; do not mask missing modules as SKIP.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


R1 = "r1-adamw-accum1"
R2 = "r2-muon-adamw-fallback-accum2"
SHA = "a" * 40


def worker():
    return importlib.import_module("tools.sd3_full_bf16_evidence_worker")


def support():
    return importlib.import_module("tools.sd3_full_bf16_evidence_support")


def ticket(root: Path, *, case_id=R1, phase="fresh"):
    return {
        "schema": "dts.sd3c2b.phase-ticket",
        "version": 1,
        "run_id": "run-red-0001",
        "phase_id": "fresh-red-0001" if phase == "fresh" else "resume-red-0001",
        "phase": phase,
        "case_id": case_id,
        "source_commit": SHA,
        "manifest_sha256": "b" * 64,
        "input_identity_sha256": "c" * 64,
        "effective_argv_sha256": "d" * 64,
        "expected_resume_checkpoint_id": None if phase == "fresh" else "e" * 32,
        "worker_entrypoint": "tools/sd3_full_bf16_evidence_worker.py",
        "output_dir": str(root / phase),
        "nonce": "e" * 64,
    }


def event(kind, seq, *, phase="fresh", case=R1, step=1, batch=None, sample=None, sync=None, profile=None):
    e = {
        "run_id": "run-red-0001",
        "phase_id": "fresh-red-0001" if phase == "fresh" else "resume-red-0001",
        "case_id": case,
        "source_commit": SHA,
        "event_seq": seq,
        "event_type": kind,
        "logical_step": step,
    }
    if batch is not None:
        e["microbatch_index"] = batch
    if sample is not None:
        e["sample_identity"] = sample
    if sync is not None:
        e["sync_gradients"] = sync
    if profile is not None:
        e["optimizer_profile"] = profile
    if kind == "checkpoint_saved":
        e["checkpoint_id"] = "f" * 32
    return e


class WorkerEntryContractRED(unittest.TestCase):
    def test_entrypoint_has_no_torch_import_at_module_import(self):
        import sys
        baseline = {k for k in sys.modules if k == "torch" or k.startswith("torch.")}
        api = worker()
        self.assertTrue(callable(api.parse_worker_cli))
        after = {k for k in sys.modules if k == "torch" or k.startswith("torch.")}
        self.assertEqual(after, baseline, "Host Worker import must not load Torch")

    def test_worker_cli_accepts_one_absolute_ticket_only(self):
        api = worker()
        path = Path(tempfile.gettempdir()).resolve() / "one.ticket.json"
        self.assertEqual(api.parse_worker_cli(["--ticket", str(path)]), path)
        for argv in (
            [], ["--ticket", "relative.json"], ["--ticket", str(path), "--full_bf16"],
            ["--ticket", str(path), "--ticket", str(path)],
            ["--config_file", "anything"], [str(path)],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises((ValueError, SystemExit)):
                    api.parse_worker_cli(argv)

    def test_ticket_manifest_recomputed_digests_and_output(self):
        api = worker()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = ticket(root)
            manifest = {
                "case_id": R1,
                "source_commit": SHA,
                "work_root": str(root),
                "fresh_command": ["scripts/dev/sd3_train_network.py", "--output_dir", str(root / "fresh")],
                "resume_command": ["scripts/dev/sd3_train_network.py", "--output_dir", str(root / "resume")],
                "input_identity": {"dataset_signature": "x"},
            }
            digest = hashlib.sha256(b"manifest bytes").hexdigest()
            payload["manifest_sha256"] = digest
            payload["input_identity_sha256"] = support()._json_sha256(manifest["input_identity"])
            payload["effective_argv_sha256"] = support()._json_sha256(manifest["fresh_command"])
            valid = api.validate_ticket_manifest_binding(payload, manifest, manifest_file_sha256=digest)
            self.assertEqual(valid["case_id"], R1)
            for key, invalid in (
                ("manifest_sha256", "0" * 64),
                ("input_identity_sha256", "0" * 64),
                ("effective_argv_sha256", "0" * 64),
                ("output_dir", str(root / "outside")),
                ("source_commit", "b" * 40),
                ("case_id", R2),
            ):
                bad = dict(payload)
                bad[key] = invalid
                with self.subTest(field=key):
                    with self.assertRaises(ValueError):
                        api.validate_ticket_manifest_binding(bad, manifest, manifest_file_sha256=digest)


class TimelineContractRED(unittest.TestCase):
    def test_r1_minimal_complete_step_accepted_as_metadata_only(self):
        api = support()
        stream = [
            event("batch_started", 1, batch=1, sample="dts_c2b_sample_0001", sync=True),
            event("backward_completed", 2, batch=1, sample="dts_c2b_sample_0001", sync=True),
            event("physical_child_stepped", 3, profile="adamw_main"),
            event("physical_scheduler_stepped", 4),
            event("checkpoint_saved", 5),
            event("train_completed", 6),
        ]
        result = api.validate_worker_event_stream(case_id=R1, phase="fresh", events=stream)
        self.assertEqual(result["status"], "timeline_metadata_valid")
        self.assertNotIn("qualified", result)
        self.assertNotIn("cuda_pass", result)

    def test_duplicate_sequence_and_step_before_backward_rejected(self):
        api = support()
        good = [
            event("batch_started", 1, batch=1, sample="dts_c2b_sample_0001", sync=True),
            event("backward_completed", 2, batch=1, sample="dts_c2b_sample_0001", sync=True),
            event("physical_child_stepped", 3, profile="adamw_main"),
            event("physical_scheduler_stepped", 4),
            event("checkpoint_saved", 5),
            event("train_completed", 6),
        ]
        duplicate = [dict(item) for item in good]
        duplicate[1]["event_seq"] = duplicate[0]["event_seq"]
        step_first = [dict(item) for item in good]
        step_first = [step_first[2], step_first[0], step_first[1], *step_first[3:]]
        wrong_order = [dict(item) for item in good]
        wrong_order = [wrong_order[0], wrong_order[2], wrong_order[1], *wrong_order[3:]]
        for label, broken in (
            ("duplicate_seq", duplicate),
            ("step_before_batch", step_first),
            ("step_before_backward", wrong_order),
        ):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    api.validate_worker_event_stream(case_id=R1, phase="fresh", events=broken)

    def test_r2_first_microbatch_must_not_have_physical_step(self):
        api = support()
        events = [
            event("batch_started", 1, case=R2, batch=1, sample="dts_c2b_sample_0001", sync=False),
            event("backward_completed", 2, case=R2, batch=1, sample="dts_c2b_sample_0001", sync=False),
            event("physical_child_stepped", 3, case=R2, profile="muon_primary"),
            event("batch_started", 4, case=R2, batch=2, sample="dts_c2b_sample_0002", sync=True),
            event("backward_completed", 5, case=R2, batch=2, sample="dts_c2b_sample_0002", sync=True),
            event("checkpoint_saved", 6, case=R2),
            event("train_completed", 7, case=R2),
        ]
        with self.assertRaises(ValueError):
            api.validate_worker_event_stream(case_id=R2, phase="fresh", events=events)

    def test_r2_reused_caption_identity_rejected(self):
        api = support()
        events = [
            event("batch_started", 1, case=R2, batch=1, sample="dts_c2b_sample_0001", sync=False),
            event("backward_completed", 2, case=R2, batch=1, sample="dts_c2b_sample_0001", sync=False),
            event("batch_started", 3, case=R2, batch=2, sample="dts_c2b_sample_0001", sync=True),
            event("backward_completed", 4, case=R2, batch=2, sample="dts_c2b_sample_0001", sync=True),
            event("physical_child_stepped", 5, case=R2, profile="muon_primary"),
            event("physical_child_stepped", 6, case=R2, profile="adamw_fallback"),
            event("physical_scheduler_stepped", 7, case=R2),
            event("checkpoint_saved", 8, case=R2),
            event("train_completed", 9, case=R2),
        ]
        with self.assertRaises(ValueError):
            api.validate_worker_event_stream(case_id=R2, phase="fresh", events=events)


if __name__ == "__main__":
    unittest.main()
