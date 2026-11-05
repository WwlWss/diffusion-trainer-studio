"""C4-C2-P1-B1: standalone Runner authority and durable replay Tests-only RED.

Pure Host mechanisms only; no CUDA, no original Load, no physical parent.
Ephemeral Ed25519 test keys never enter a Worker or production provisioning.
Missing *source contracts* are targeted FAIL, never import ERROR or SKIP.
"""
from __future__ import annotations

from dataclasses import fields
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.test_sd3_full_bf16_worker_c4c_p1_host import (
    NOW, _context, _envelope, _keys, _payload,
)
from tools.sd3_full_bf16_evidence_worker import WorkerInvocation
from tools.sd3_full_bf16_runner_trust import verify_provisional_step1_signature


_MISSING = "C4-C2-P1-B HOST CONTRACT ABSENT"
_RUNNER = "tools.sd3_full_bf16_independent_runner"
_LEDGER = "tools.sd3_full_bf16_runner_replay"
_ISSUER = "runner-test-issuer-01"
_KWARGS = {
    "issuer_id": _ISSUER,
    "audience": "dts.sd3c2b.c4c2.resume-load",
    "ticket_sha256": "a" * 64,
    "one_use_nonce": "b" * 64,
    "token_id": "c" * 64,
    "issued_at_unix": NOW - 5,
    "expires_at_unix": NOW + 55,
    "now_unix": NOW,
}


def _module(test, name):
    test.assertIsNotNone(
        importlib.util.find_spec(name),
        f"{_MISSING}: reviewed module {name} missing",
    )
    return importlib.import_module(name)


def _api(test, name, attribute):
    module = _module(test, name)
    target = getattr(module, attribute, None)
    test.assertTrue(
        callable(target),
        f"{_MISSING}: reviewed {name}.{attribute} missing",
    )
    return target


def _ledger_type(test):
    return _api(test, _LEDGER, "RunnerNonceLedger")


def _run(args, *, env=None, timeout=15):
    return subprocess.run(
        [sys.executable, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def _consume(ledger, **changes):
    args = dict(_KWARGS)
    args.update(changes)
    with ledger:
        return ledger.consume_once(**args)


class C4C2P1BRunnerAuthorityHostRED(unittest.TestCase):
    def test_b101_existing_ed25519_test_signature_is_unqualified(self):
        private, public = _keys()
        result = verify_provisional_step1_signature(
            envelope=_envelope(private),
            pinned_public_key=public,
            expected_context=_context(),
            now_unix=NOW,
            seen_nonces=set(),
        )
        self.assertEqual(result, {
            "status": "signature_valid_unqualified",
            "physical_cuda_qualified": False,
        })

    def test_b102_existing_worker_has_no_runner_secret_fields(self):
        names = {f.name for f in fields(WorkerInvocation)}
        for forbidden in (
            "runner_private_key", "signing_seed", "issuer_secret",
            "runner_signing_key", "trusted_root_path",
        ):
            self.assertNotIn(forbidden, names)

    def test_b103_test_signature_has_no_physical_cuda_entitlement(self):
        private, public = _keys()
        p = _payload()
        self.assertEqual(p["phase"], "fresh")
        self.assertEqual(p["logical_step"], 1)
        result = verify_provisional_step1_signature(
            envelope=_envelope(private, p),
            pinned_public_key=public,
            expected_context=_context(),
            now_unix=NOW,
            seen_nonces=set(),
        )
        self.assertFalse(result["physical_cuda_qualified"])
        self.assertNotIn("checkpoint_loaded", result)
        self.assertNotIn("worker_complete", result)

    def test_b104_standalone_unprovisioned_runner_process_cannot_issue(self):
        _api(self, _RUNNER, "issue_provisional_fresh_step1_attestation")
        done = _run(["-m", _RUNNER, "--probe-unprovisioned"])
        self.assertEqual(done.returncode, 0, done.stderr)
        out = json.loads(done.stdout)
        self.assertEqual(out, {
            "schema": "dts.sd3c2b.c4c2p1.runner-probe-v1",
            "status": "unprovisioned",
            "may_issue": False,
            "physical_cuda_qualified": False,
        })

    def test_b105_external_pinned_root_rejects_worker_selected_configuration(self):
        get_root = _api(self, _RUNNER, "get_pinned_runner_trust_root")
        with patch.dict(os.environ, {
            "DTS_RUNNER_PUBLIC_KEY": "f" * 64,
            "DTS_RUNNER_PUBLIC_KEY_PATH": "/tmp/worker-public-key",
            "DTS_RUNNER_TRUST_ROOT": "worker-selected",
            "DTS_RUNNER_PRIVATE_KEY": "worker-selected",
        }):
            with self.assertRaises(ValueError):
                get_root()
        self.assertNotIn("runner_public_key", vars(WorkerInvocation))

    def test_b106_untrusted_worker_dict_cannot_mint_runner_attestation(self):
        issue = _api(self, _RUNNER, "issue_provisional_fresh_step1_attestation")
        with self.assertRaises(ValueError):
            issue({"status": "physical_binding_valid", "payload": _payload(),
                   "claimed_issuer": "independent-runner"})
        with self.assertRaises(ValueError):
            issue(_payload())

    def test_b107_durable_ledger_accepts_once_only_as_unqualified_mechanism(self):
        ledger_type = _ledger_type(self)
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "runner-nonces.sqlite3"
            with ledger_type(ledger_path) as ledger:
                first = ledger.consume_once(**_KWARGS)
                self.assertEqual(first, {
                    "status": "nonce_consumed_unqualified",
                    "physical_cuda_qualified": False,
                })
                with self.assertRaises(ValueError):
                    ledger.consume_once(**_KWARGS)

    def test_b108_replay_rejected_in_new_python_process_after_restart(self):
        ledger_type = _ledger_type(self)
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "nonce.sqlite3"
            with ledger_type(db) as ledger:
                first = ledger.consume_once(**_KWARGS)
                self.assertEqual(first["status"], "nonce_consumed_unqualified")
            script = (
                "import json,sys\n"
                "from pathlib import Path\n"
                f"from {_LEDGER} import RunnerNonceLedger\n"
                "args=json.loads(sys.argv[2])\n"
                "try:\n"
                " with RunnerNonceLedger(Path(sys.argv[1])) as l:\n"
                "  l.consume_once(**args)\n"
                "except ValueError:\n"
                " print('REPLAY_REJECTED')\n"
                "else:\n"
                " raise SystemExit('REPLAY_ACCEPTED')\n"
            )
            finished = _run(["-c", script, str(db), json.dumps(_KWARGS)])
            self.assertEqual(finished.returncode, 0, finished.stderr)
            self.assertEqual(finished.stdout.strip(), "REPLAY_REJECTED")

    def test_b109_concurrent_distinct_process_claims_are_atomic(self):
        _ledger_type(self)
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "nonce.sqlite3"
            gate = Path(td) / "go"
            script = (
                "import json,sys,time\n"
                "from pathlib import Path\n"
                f"from {_LEDGER} import RunnerNonceLedger\n"
                "gate=Path(sys.argv[2]); deadline=time.monotonic()+10\n"
                "while not gate.exists() and time.monotonic()<deadline:\n"
                " time.sleep(0.01)\n"
                "if not gate.exists(): raise SystemExit('GATE_TIMEOUT')\n"
                "try:\n"
                " with RunnerNonceLedger(Path(sys.argv[1])) as ledger:\n"
                "  ledger.consume_once(**json.loads(sys.argv[3]))\n"
                "except ValueError:\n"
                " print('REJECT')\n"
                "else:\n"
                " print('ACCEPT')\n"
            )
            jobs = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(db), str(gate),
                     json.dumps(_KWARGS)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                for _ in range(4)
            ]
            try:
                gate.touch()
                outcomes = [job.communicate(timeout=20) for job in jobs]
                self.assertEqual(
                    [job.returncode for job in jobs], [0] * 4,
                    repr(outcomes),
                )
                results = [out.strip() for out, _ in outcomes]
                self.assertEqual(sorted(results), ["ACCEPT", "REJECT",
                                                   "REJECT", "REJECT"])
            finally:
                for job in jobs:
                    if job.poll() is None:
                        job.kill()
                        job.communicate(timeout=5)

    def test_b110_ticket_audience_token_changes_do_not_reset_nonce(self):
        ledger_type = _ledger_type(self)
        with tempfile.TemporaryDirectory() as td:
            with ledger_type(Path(td) / "nonce.sqlite3") as ledger:
                ledger.consume_once(**_KWARGS)
                for change in (
                    {"ticket_sha256": "d" * 64},
                    {"audience": "dts.other.service"},
                    {"token_id": "e" * 64},
                    {"issuer_id": "foreign-issuer"},
                ):
                    with self.subTest(change=change):
                        with self.assertRaises(ValueError):
                            ledger.consume_once(**{**_KWARGS, **change})

    def test_b111_corrupted_existing_ledger_fails_closed(self):
        ledger_type = _ledger_type(self)
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "nonce.sqlite3"
            db.write_bytes(b"not-a-sqlite-ledger")
            with self.assertRaises(ValueError):
                with ledger_type(db) as ledger:
                    ledger.consume_once(**_KWARGS)
            self.assertEqual(db.read_bytes(), b"not-a-sqlite-ledger")

    def test_b112_inherited_fake_signer_environment_never_activates_runner(self):
        _api(self, _RUNNER, "get_pinned_runner_trust_root")
        marker = "P1B-FAKE-SECRET-NEVER-LOG-" + "z" * 32
        env = dict(os.environ)
        env.update({
            "DTS_RUNNER_PRIVATE_KEY": marker,
            "DTS_RUNNER_SIGNING_SEED": marker,
            "DTS_RUNNER_PUBLIC_KEY_PATH": marker,
            "DTS_RUNNER_TRUST_ROOT": marker,
        })
        done = _run(["-m", _RUNNER, "--probe-unprovisioned"], env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn(marker, done.stdout + done.stderr)
        report = json.loads(done.stdout)
        self.assertEqual(report["status"], "unprovisioned")
        self.assertIs(report["may_issue"], False)
        self.assertIs(report["physical_cuda_qualified"], False)


if __name__ == "__main__":
    unittest.main()
