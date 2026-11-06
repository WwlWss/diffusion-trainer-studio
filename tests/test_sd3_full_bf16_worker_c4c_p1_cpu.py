"""C4-C2-P1 independent Runner source boundary, pinned real CPU diagnostics.

No GPU is forged. Worker has no issuer key or ability to create provisional
Fresh evidence. Original Accelerate.load_state still refuses untrusted parent
prior to prehooks; the genuine CPU four-file new-process Load is a baseline.
"""
from __future__ import annotations

from dataclasses import fields
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from tests.test_sd3_full_bf16_worker_c4a_cpu import _r1_real_step
from tests.test_sd3_full_bf16_worker_c4c_cpu import (
    _new_process_restore, _resume_adapter,
)
from tools.sd3_full_bf16_evidence_worker import WorkerInvocation


def _runner_api(test, name):
    module_name = "tools.sd3_full_bf16_independent_runner"
    spec = importlib.util.find_spec(module_name)
    test.assertIsNotNone(
        spec,
        "C4-C2-P1 CPU CONTRACT ABSENT: no independent Runner "
        "executable / protected Fresh Step-1 source",
    )
    module = importlib.import_module(module_name)
    method = getattr(module, name, None)
    test.assertTrue(
        callable(method),
        "C4-C2-P1 CPU CONTRACT ABSENT: missing independent Runner " + name,
    )
    return method


class C4C2P1RunnerOriginCpuRED(unittest.TestCase):
    def test_r101_independent_runner_requires_real_issuer_entrypoint(self):
        _runner_api(self, "issue_provisional_fresh_step1_attestation")

    def test_r102_provisional_source_requires_real_physical_snapshot_check(self):
        _runner_api(self, "verify_original_provisional_step1_source")

    def test_r103_runner_verifier_trust_root_must_be_outside_worker(self):
        _runner_api(self, "get_pinned_runner_trust_root")

    def test_r104_worker_invocation_has_no_private_signing_key_fields(self):
        names = {f.name for f in fields(WorkerInvocation)}
        for forbidden in (
            "runner_private_key", "signing_seed", "issuer_secret",
            "runner_signing_key", "trusted_root_path",
        ):
            self.assertNotIn(forbidden, names)

    def test_r105_current_adapter_load_remains_closed_without_runner_parent(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            acc = bundle[1]
            seen = []
            hook = acc.register_load_state_pre_hook(
                lambda models, source: seen.append(str(source))
            )
            self.addCleanup(hook.remove)
            with self.assertRaises(ValueError):
                acc.load_state(str(Path(td) / "self-declared-state"))
            self.assertEqual(seen, [])
            self.assertEqual(adapter._load_original_calls, 0)
            self.assertFalse(bundle[5].events)

    def test_r106_genuine_cpu_four_file_new_process_restore_is_diagnostic(self):
        fresh = _r1_real_step(self)
        with tempfile.TemporaryDirectory() as td:
            result = _new_process_restore(self, fresh, "r1", Path(td))
            self.assertEqual(set(result["restored"]), {
                "model", "optimizer", "scheduler",
            })


if __name__ == "__main__":
    unittest.main()
