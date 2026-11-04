"""C4-B-FSR: isolated source-authority Save negative controls (pinned CPU).

Tests-only RED. Existing C4-A, C4-B and G1/G2 suites stay frozen.
Real CPU BF16 physical Optimizer Steps, original Accelerator.save_state and
production LoRANetwork.save_weights are used where relevant. These are NOT
full SD3.5 Medium CUDA admissions or completed Checkpoint receipts.
No import/fixture ERROR may masquerade as a source-contract RED.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from tests.test_sd3_full_bf16_worker_c4a_cpu import _r1_real_step
from tests.test_sd3_full_bf16_worker_c4b_cpu import (
    _bound_adapter, _production_one_lora, _real_cpu_state_save,
)
from tools.sd3_full_bf16_worker_step_adapter import C4BVerifiedStepSave
from tools import sd3_full_bf16_evidence_support as host


FILES = (
    "model.safetensors", "optimizer.bin", "scheduler.bin",
    "random_states_0.pkl", "train_state.json",
    "dts_parameter_policy_manifest.json",
    "dts_parameter_policy_progress.json",
    "dts_sd3_resume_progress.json",
    "dts_sd3_forward_semantics.json",
)


def _contract_reject(test, callback, *, tag, pattern=None):
    try:
        callback()
    except ValueError as exc:
        if pattern is not None:
            test.assertRegex(str(exc), pattern)
        return
    test.fail("C4-B-FSR CONTRACT ABSENT: " + tag)


def _require_interface(test, owner, name):
    function = getattr(owner, name, None)
    test.assertTrue(
        callable(function),
        "C4-B-FSR CONTRACT ABSENT: reviewed " + name + " interface is missing",
    )
    return function


def _forged_receipt(adapter, probe, root, *, wrong_identity=False):
    state = root / "fresh" / "dts_c4a_cpu-step00000001-state"
    state.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        (state / filename).write_bytes(b"synthetic-untrusted-" + filename.encode())
    adapter_file = root / "fresh" / "dts_c4a_cpu-step00000001.safetensors"
    adapter_file.write_bytes(b"synthetic-pretend-LoRA")
    receipt = C4BVerifiedStepSave(
        case_id=probe.case_id,
        phase="resume" if wrong_identity else probe.phase,
        run_id=probe.run_id,
        phase_id=probe.phase_id,
        source_commit="f" * 40 if wrong_identity else probe.source_commit,
        logical_step=1,
        anchor_sha256="a" * 64,
        checkpoint_id="b" * 32,
        adapter_path=str(adapter_file),
        adapter_file_sha256=host._sha256_file(adapter_file),
        tensor_inventory_sha256="c" * 64,
        state_dir=str(state),
        state_file_sha256=tuple(sorted(
            (name, host._sha256_file(state / name)) for name in FILES
        )),
    )
    adapter._verified_step_records[1] = receipt
    return receipt


def _rng_payload(acc, *, step_delta=0, changed_python=False):
    python_state = random.getstate()
    if changed_python:
        modified = list(python_state)
        modified[-1] = (modified[-1] or 0) + 1
        python_state = tuple(modified)
    result = {
        "step": acc.step + step_delta,
        "random_state": python_state,
        "numpy_random_seed": np.random.get_state(),
        "torch_manual_seed": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    return result


class C4BFSRSourceAuthorityCpuRED(unittest.TestCase):
    def _verifier(self):
        return importlib.import_module(
            "tools.sd3_full_bf16_worker_save_verifier"
        )

    def test_fsr01_unsaved_forged_step_receipt_does_not_unlock_parent(self):
        bundle = _r1_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            _forged_receipt(adapter, probe, root)
            self.assertEqual(adapter._save_call_count, 0)
            _contract_reject(
                self, lambda: adapter.verified_step_save(logical_step=1),
                tag="synthetic nine-file hashes accepted as original Save",
                pattern=r"C4-B|C4-A|committed|provenance|original",
            )

    def test_fsr02_forged_foreign_phase_and_source_receipt_rejected(self):
        bundle = _r1_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            forged = _forged_receipt(
                adapter, probe, root, wrong_identity=True,
            )
            self.assertNotEqual(forged.source_commit, probe.source_commit)
            _contract_reject(
                self, lambda: adapter.verified_step_save(logical_step=1),
                tag="foreign-phase source-mismatched receipt trusted",
            )

    def test_fsr03_event_commit_requires_real_event_append(self):
        bundle = _r1_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            adapter = _bound_adapter(bundle, Path(dirname), self)
            inspect = _require_interface(
                self, adapter, "_assert_checkpoint_saved_event_appended"
            )
            before = len(probe.events)
            # No genuine checkpoint_saved append has happened.
            _contract_reject(
                self,
                lambda: inspect(
                    before_count=before, logical_step=1,
                    checkpoint_id="b" * 32, anchor_sha256="a" * 64,
                ),
                tag="callback return accepted without append",
            )

    def test_fsr04_original_rng_file_wrong_accelerator_step_rejected(self):
        bundle = _r1_real_step(self)
        acc = bundle[1]
        with tempfile.TemporaryDirectory() as dirname:
            state = Path(dirname) / "rng-save"
            _real_cpu_state_save(acc, state)
            verifier = self._verifier()
            snapshot = _require_interface(
                self, verifier, "snapshot_accelerate_rng_expectation"
            )
            validate = _require_interface(
                self, verifier, "verify_saved_rng_payload"
            )
            expected = snapshot(acc)
            torch.save(
                _rng_payload(acc, step_delta=1),
                state / "random_states_0.pkl",
            )
            _contract_reject(
                self, lambda: validate(
                    state / "random_states_0.pkl", expected=expected,
                ),
                tag="actual serialized RNG step differs from Accelerator.step",
            )

    def test_fsr05_actual_rng_file_changed_python_state_rejected(self):
        bundle = _r1_real_step(self)
        acc = bundle[1]
        with tempfile.TemporaryDirectory() as dirname:
            state = Path(dirname) / "rng-save"
            _real_cpu_state_save(acc, state)
            verifier = self._verifier()
            snapshot = _require_interface(
                self, verifier, "snapshot_accelerate_rng_expectation"
            )
            validate = _require_interface(
                self, verifier, "verify_saved_rng_payload"
            )
            expected = snapshot(acc)
            torch.save(
                _rng_payload(acc, changed_python=True),
                state / "random_states_0.pkl",
            )
            _contract_reject(
                self, lambda: validate(
                    state / "random_states_0.pkl", expected=expected,
                ),
                tag="changed Python RNG accepted as original-state payload",
            )

    def test_fsr06_real_rng_file_matches_exact_original_snapshot(self):
        bundle = _r1_real_step(self)
        acc = bundle[1]
        with tempfile.TemporaryDirectory() as dirname:
            verifier = self._verifier()
            snapshot = _require_interface(
                self, verifier, "snapshot_accelerate_rng_expectation"
            )
            validate = _require_interface(
                self, verifier, "verify_saved_rng_payload"
            )
            expected = snapshot(acc)
            state = Path(dirname) / "rng-save"
            _real_cpu_state_save(acc, state)
            payload = validate(
                state / "random_states_0.pkl", expected=expected,
            )
            self.assertEqual(payload["step"], acc.step)
            self.assertEqual(
                payload["file_sha256"],
                host._sha256_file(state / "random_states_0.pkl"),
            )

    def test_fsr07_preexisting_lora_and_shadowed_original_writer_rejected(self):
        network = _production_one_lora()
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            dest = root / "verified-step00000001.safetensors"
            network.save_weights(
                str(dest), None, {"ss_steps": "1", "ss_epoch": "0"}
            )
            network.save_weights = lambda *args, **kwargs: None
            verifier = self._verifier()
            observer_type = _require_interface(
                self, verifier, "C4BLoRAWriteObserver"
            )
            _contract_reject(
                self, lambda: observer_type(
                    network=network, output_dir=root,
                    output_name="verified", step_provider=lambda: 1,
                ),
                tag="pre-existing LoRA accepted with shadowed original writer",
            )

    def test_fsr08_wrong_step_lora_metadata_rejected_despite_equal_tensors(self):
        network = _production_one_lora()
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "verified-step00000001.safetensors"
            network.save_weights(
                str(path), None, {"ss_steps": "2", "ss_epoch": "0"}
            )
            verifier = self._verifier()
            check = _require_interface(
                self, verifier, "verify_lora_step_metadata"
            )
            _contract_reject(
                self, lambda: check(path, expected_step=1),
                tag="correct Tensor bytes but wrong ss_steps metadata",
            )

    def test_fsr09_real_original_lora_writer_observed_and_restored(self):
        network = _production_one_lora()
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            verifier = self._verifier()
            observer_type = _require_interface(
                self, verifier, "C4BLoRAWriteObserver"
            )
            before_slot = vars(network).get("save_weights")
            had_slot = "save_weights" in vars(network)
            observer = observer_type(
                network=network, output_dir=root,
                output_name="verified", step_provider=lambda: 1,
            )
            observer.install()
            try:
                file = root / "verified-step00000001.safetensors"
                network.save_weights(
                    str(file), None, {"ss_steps": "1", "ss_epoch": "0"}
                )
                observation = observer.completed_step_write(logical_step=1)
                self.assertTrue(observation.original_returned)
                self.assertEqual(observation.file_sha256, host._sha256_file(file))
            finally:
                observer.close()
            self.assertEqual("save_weights" in vars(network), had_slot)
            self.assertIs(vars(network).get("save_weights"), before_slot)

    def test_fsr10_automatic_checkpoint_naming_rejected_before_original(self):
        bundle = _r1_real_step(self)
        acc = bundle[1]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            # Set a valid automatic naming destination; old C4-B does not
            # reject this configuration before original save_state.
            conf = acc.project_configuration
            old_auto, old_project = conf.automatic_checkpoint_naming, conf.project_dir
            conf.automatic_checkpoint_naming = True
            conf.project_dir = str(root / "auto")
            self.addCleanup(
                lambda: (
                    setattr(conf, "automatic_checkpoint_naming", old_auto),
                    setattr(conf, "project_dir", old_project),
                )
            )
            path = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            try:
                acc.save_state(str(path), safe_serialization=True)
            except ValueError as exc:
                self.assertRegex(
                    str(exc), r"automatic|naming|save_location|destination",
                )
                return
            except Exception as exc:
                self.fail(
                    "C4-B-FSR CONTRACT ABSENT: original Save reached "
                    "incompatible automatic naming configuration: "
                    + type(exc).__name__,
                )
            self.fail(
                "C4-B-FSR CONTRACT ABSENT: automatic Save destination "
                "deviated from approved Step directory",
            )

    def test_fsr11_shadowed_probe_recorder_must_fail_at_save_boundary(self):
        bundle = _r1_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            adapter = _bound_adapter(bundle, Path(dirname), self)
            probe._record = lambda *args, **kwargs: None
            _contract_reject(
                self,
                lambda: adapter.classify_state_save(
                    state_dir=Path(dirname) / "fresh" / "dts_c4a_cpu-state",
                ),
                tag="unreviewed Probe event recorder accepted",
                pattern=r"recorder|event|Probe|unreviewed",
            )

    def test_fsr12_last_is_ancillary_and_cpu_receipt_always_unqualified(self):
        bundle = _r1_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            self.assertEqual(adapter.classify_state_save(
                state_dir=root / "fresh" / "dts_c4a_cpu-state"
            ), "ancillary_last")
            with self.assertRaises(ValueError):
                adapter.verified_step_save(logical_step=1)
            self.assertEqual(
                sum(row["event_type"] == "checkpoint_saved"
                    for row in probe.events),
                0,
            )


if __name__ == "__main__":
    unittest.main()
