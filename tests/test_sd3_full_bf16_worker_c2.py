"""Fix C2 RED-first Host tests for the real restricted Worker admission and lease.

All fixtures are synthetic/Host and cannot qualify Full BF16 or CUDA.  Missing
Worker functions are intended RED until the corresponding C2 implementation.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_sd3_full_bf16_gpu_evidence import _generate, R1, R2, SHA
from tools import sd3_full_bf16_evidence_support as host


def api():
    from tools import sd3_full_bf16_evidence_worker
    return sd3_full_bf16_evidence_worker


def fixture(root: Path, case_id: str = R1):
    generated = _generate(root, case_id)
    manifest_file = Path(generated["manifest"])
    manifest = host._read_json(manifest_file)
    work_root = Path(manifest["work_root"])
    events = work_root / "events"
    events.mkdir()
    ticket = {
        "schema": host.TICKET_SCHEMA,
        "version": host.TICKET_VERSION,
        "run_id": "run-c2red-0001",
        "phase_id": "fresh-c2red-0001",
        "phase": "fresh",
        "case_id": case_id,
        "source_commit": SHA,
        "manifest_sha256": host._sha256_file(manifest_file),
        "input_identity_sha256": host._json_sha256(manifest["input_identity"]),
        "effective_argv_sha256": host._json_sha256(manifest["fresh_command"]),
        "expected_resume_checkpoint_id": None,
        "worker_entrypoint": host.WORKER_ENTRYPOINT,
        "output_dir": str(work_root / "fresh"),
        "nonce": "a" * 64,
    }
    ticket_path = events / "fresh-fresh-c2red-0001.ticket.json"
    host.issue_phase_ticket(ticket_path, ticket)
    return manifest, ticket, manifest_file, ticket_path


class C2TicketAndAssetGateRED(unittest.TestCase):
    def test_resolve_rejects_invalid_model_before_ticket_claim(self):
        """Generator accepts synthetic Host model; Worker MUST reject it."""
        with tempfile.TemporaryDirectory() as td:
            manifest, _ticket, _mf, path = fixture(Path(td))
            with self.assertRaises((ValueError, RuntimeError)):
                api().resolve_phase_invocation(path)
            self.assertFalse(path.with_name(path.name + ".consumed").exists())
            self.assertFalse((Path(manifest["work_root"]) / "fresh").exists())

    def test_resolve_valid_host_preflight_is_read_only_and_bound(self):
        """Stub only costly physical asset/source gates; never the ticket logic."""
        with tempfile.TemporaryDirectory() as td:
            manifest, ticket, mf, path = fixture(Path(td))
            with (
                patch.object(host, "validate_case_assets",
                             return_value={"status": "preflight_pass"}),
                patch.object(api(), "_verify_source_identity",
                             return_value={"source_commit": SHA}),
            ):
                invocation = api().resolve_phase_invocation(path)
            self.assertEqual(invocation.case_id, R1)
            self.assertEqual(invocation.source_commit, SHA)
            self.assertEqual(invocation.phase_id, ticket["phase_id"])
            self.assertEqual(invocation.manifest_path, mf)
            self.assertEqual(invocation.output_dir, Path(manifest["work_root"]) / "fresh")
            self.assertEqual(invocation.trainer_argv, manifest["fresh_command"])
            self.assertFalse(path.with_name(path.name + ".consumed").exists())
            claim = host.claim_phase_ticket(path, expected=ticket)
            self.assertEqual(claim["ticket_sha256"], invocation.ticket_sha256)
            with self.assertRaises(host.FullBF16EvidenceError):
                host.claim_phase_ticket(path, expected=ticket)

    def test_source_drift_fails_before_claim(self):
        with tempfile.TemporaryDirectory() as td:
            _manifest, ticket, _mf, path = fixture(Path(td))
            with (
                patch.object(host, "validate_case_assets",
                             return_value={"status": "preflight_pass"}),
                patch.object(api(), "_verify_source_identity",
                             side_effect=host.FullBF16EvidenceError("dirty source")),
            ):
                with self.assertRaisesRegex(ValueError, "dirty source"):
                    api().resolve_phase_invocation(path)
            self.assertFalse(path.with_name(path.name + ".consumed").exists())

    def test_malformed_ticket_name_and_root_layout_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            _manifest, ticket, _mf, path = fixture(Path(td))
            alias = path.parent / "incorrect.ticket.json"
            alias.write_bytes(path.read_bytes())
            with self.assertRaises(ValueError):
                api().resolve_phase_invocation(alias)
            self.assertFalse(alias.with_name(alias.name + ".consumed").exists())


class C2LeaseRED(unittest.TestCase):
    def setUp(self):
        from mikazuki import parameter_policy_execution as execution
        self.execution = execution
        self.before_backend = dict(execution.FULL_BF16_BACKEND_QUALIFICATIONS)
        self.before_optimizer = dict(execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS)

    def tearDown(self):
        self.assertEqual(dict(self.execution.FULL_BF16_BACKEND_QUALIFICATIONS),
                         self.before_backend, "C2 test leaked a backend qualification")
        self.assertEqual(dict(self.execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS),
                         self.before_optimizer, "C2 test leaked an optimizer qualification")

    def test_r1_backend_only_lease_releases_on_success(self):
        def inspect(lease):
            self.assertEqual(
                self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"].status,
                "qualified",
            )
            self.assertTrue(lease["backend_lease_only"])
            self.assertEqual(
                self.execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS,
                self.before_optimizer,
            )
            return "observed-only"
        result = api().run_with_temporary_qualification(R1, inspect)
        self.assertEqual(result["value"], "observed-only")
        self.assertTrue(result["lease"]["backend_lease_only"])

    def test_r2_both_children_already_qualified(self):
        def inspect(_):
            return dict(self.execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS)
        result = api().run_with_temporary_qualification(R2, inspect)
        self.assertEqual(result["value"], self.before_optimizer)

    def test_missing_d0_optimizer_qualification_fails_before_callback(self):
        orig = self.execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"]
        seen = []
        self.execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"] = replace(
            orig, status="pending", evidence_case_id=None,
        )
        try:
            with self.assertRaises(ValueError):
                api().run_with_temporary_qualification(R2, lambda _: seen.append(1))
            self.assertEqual(seen, [])
            self.assertEqual(self.execution.FULL_BF16_BACKEND_QUALIFICATIONS,
                             self.before_backend)
        finally:
            self.execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS["Muon"] = orig

    def test_backend_already_qualified_cannot_run(self):
        orig = self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"]
        self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"] = replace(
            orig, status="qualified", evidence_case_id="not-release-evidence",
        )
        try:
            with self.assertRaises(ValueError):
                api().run_with_temporary_qualification(R1, lambda _: self.fail("callback ran"))
        finally:
            self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"] = orig

    def test_exception_preserved_and_all_lease_state_restored(self):
        class TrainingFailed(RuntimeError):
            pass
        def crash(_):
            self.assertEqual(
                self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sd3-lora"].status,
                "qualified",
            )
            raise TrainingFailed("intentional observer failure")
        with self.assertRaisesRegex(TrainingFailed, "intentional observer failure"):
            api().run_with_temporary_qualification(R1, crash)

    def test_out_of_scope_qualification_mutation_rejected(self):
        original = self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sdxl-lora"]
        def mutate(_):
            self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sdxl-lora"] = replace(
                original, status="pending", evidence_case_id=None,
            )
        try:
            with self.assertRaises(ValueError):
                api().run_with_temporary_qualification(R1, mutate)
        finally:
            self.execution.FULL_BF16_BACKEND_QUALIFICATIONS["sdxl-lora"] = original


class C2FailurePublicationRED(unittest.TestCase):
    def test_failure_payload_never_claims_qualification(self):
        with tempfile.TemporaryDirectory() as td:
            result = Path(td) / "result.json"
            info = {
                "run_id": "run-c2red-0001", "phase_id": "fresh-c2red-0001",
                "case_id": R1, "phase": "fresh", "source_commit": SHA,
            }
            payload = api().publish_failure_result(
                result, info, RuntimeError("trainer exception"),
            )
            self.assertEqual(payload["status"], "fail")
            self.assertFalse(payload["production_qualification_mutated"])
            self.assertFalse(payload["backend_qualification_eligible"])
            self.assertEqual(json.loads(result.read_text()), payload)
            with self.assertRaises(ValueError):
                api().publish_failure_result(
                    result, info, RuntimeError("duplicate publication"),
                )


class C2ClaimAndFailureRED(unittest.TestCase):
    def test_post_claim_runtime_failure_is_recorded_and_ticket_cannot_replay(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, ticket, _mf, path = fixture(Path(td))
            with (
                patch.object(host, "validate_case_assets",
                             return_value={"status": "preflight_pass"}),
                patch.object(api(), "_verify_source_identity",
                             return_value={"source_commit": SHA}),
                patch.object(api(), "validate_runtime_environment",
                             side_effect=RuntimeError("CUDA required (injected C2 Host test)")),
            ):
                self.assertEqual(api().main(["--ticket", str(path)]), 1)
            claim_path = path.with_name(path.name + ".consumed")
            result_path = path.parent / "fresh-fresh-c2red-0001.result.json"
            self.assertTrue(claim_path.is_file())
            evidence = host._read_json(result_path)
            self.assertEqual(evidence["status"], "fail")
            self.assertEqual(evidence["error"]["type"], "RuntimeError")
            self.assertIn("CUDA required", evidence["error"]["message"])
            self.assertFalse(evidence["backend_qualification_eligible"])
            self.assertFalse((Path(manifest["work_root"]) / "fresh").exists())
            with self.assertRaises(ValueError):
                api().resolve_phase_invocation(path)
            self.assertEqual(host._read_json(result_path), evidence)

    def test_source_submodule_checkout_drift_and_missing_runtime_gitlink(self):
        from tools import prepare_sd3_full_bf16_cases as generator
        submodule_names = ("frontend", "mikazuki/dataset-tag-editor", "sd-scripts")
        good = {
            name: {"gitlink_sha": SHA, "checkout": "initialized"}
            for name in submodule_names
        }
        manifest = {"source_commit": SHA, "source_submodules": good}
        with (
            patch.object(generator, "_source_commit", return_value=SHA),
            patch.object(generator, "_source_submodule_inventory",
                         return_value=copy.deepcopy(good)),
        ):
            self.assertTrue(api()._verify_source_identity(manifest)["source_clean"])
        bad = copy.deepcopy(good)
        bad["sd-scripts"]["checkout"] = "uninitialized"
        with (
            patch.object(generator, "_source_commit", return_value=SHA),
            patch.object(generator, "_source_submodule_inventory",
                         return_value=bad),
        ):
            with self.assertRaises(ValueError):
                api()._verify_source_identity(manifest)

    def test_source_invalid_commit_rejected(self):
        from tools import prepare_sd3_full_bf16_cases as generator
        with patch.object(generator, "_source_commit", return_value="b" * 40):
            with self.assertRaises(ValueError):
                api()._verify_source_identity({"source_commit": SHA})


if __name__ == "__main__":
    unittest.main()
