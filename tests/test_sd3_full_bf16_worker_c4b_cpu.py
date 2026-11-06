"""C4-B Tests-only RED: real pinned CPU save_state and LoRA bytes.

Real Torch 2.7.0, Accelerate 1.6.0, pytorch-optimizer 3.10.0.
This suite executes real BF16 physical C3 R1/R2 steps and original
Accelerator.save_state; it never substitutes a fake save for positive I/O.
A tiny CPU 4-file Accelerate directory, even with actual BF16 LoRA
triplets, is ALWAYS unqualified for the fixed 789-tensor/9-file CUDA case.

The C4-B tests intentionally fail with explicit targeted assertions where
post-return observation, original-method binding or file attestation is
not yet implemented. No ImportError/AttributeError as purported RED.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from tools import sd3_full_bf16_evidence_support as host
from tests.test_sd3_full_bf16_worker_c4a_cpu import (
    _r1_real_step, _r2_real_step, _invocation, _manifest,
)
from tools.sd3_full_bf16_worker_step_adapter import C4StepAdapter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from networks.lora_sd3 import LoRANetwork  # noqa: E402

ACCELERATE_CPU_FILES = {
    "model.safetensors", "optimizer.bin", "scheduler.bin",
    "random_states_0.pkl",
}
PRODUCTION_STATE_FILES = {
    *ACCELERATE_CPU_FILES, "train_state.json",
    "dts_parameter_policy_manifest.json",
    "dts_parameter_policy_progress.json",
    "dts_sd3_resume_progress.json",
    "dts_sd3_forward_semantics.json",
}


def _files(directory):
    return {f.name for f in directory.iterdir() if f.is_file()}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_adapter(bundle, root, test):
    model, accelerator, _, _, _, probe = bundle
    adapter = C4StepAdapter(
        probe=probe, accelerator=accelerator, network=model,
        scheduler=probe.scheduler, invocation=_invocation(probe),
        manifest_payload=_manifest(probe, root),
    )
    test.addCleanup(adapter.close)
    adapter.install()
    return adapter


def _real_cpu_state_save(accelerator, directory):
    accelerator.save_state(str(directory), safe_serialization=True)
    return _files(directory)


def _production_one_lora():
    class SingleDiTBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 16)

    class TinyMMDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.joint_blocks = torch.nn.ModuleList([SingleDiTBlock()])

    mmdit = TinyMMDiT()
    network = LoRANetwork(
        [], mmdit, lora_dim=4, alpha=4.0,
        split_qkv=False, train_t5xxl=False,
    )
    network.apply_to([], mmdit, apply_text_encoder=False, apply_unet=True)
    network.to(dtype=torch.bfloat16)
    return network


def _reference_tensors(state):
    descriptors = {}
    for key, value in state.items():
        raw = value.detach().to("cpu").contiguous().reshape(-1).view(
            torch.uint8
        ).numpy().tobytes()
        descriptors[key] = {
            "dtype": "BF16",
            "shape": list(value.shape),
            "role": ("alpha_buffer" if key.endswith(".alpha")
                     else "trainable_weight"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return descriptors


def _real_diagnostic_lora_save(model, path, step):
    """Use production LoRANetwork.save_weights with genuine C3 LoRAModule
    parameters and alpha; the test-only facade is NOT a real LoRANetwork nor
    a production 263-target eligibility claim. Production instance behavior
    is independently covered in B03.
    """
    from types import SimpleNamespace
    tensors = {}
    for lora in model.unet_loras:
        for part, value in lora.state_dict().items():
            tensors[f"{lora.lora_name}.{part}"] = (
                value.detach().cpu().contiguous()
            )
    if len(tensors) != 3 * len(model.unet_loras):
        raise AssertionError("Incomplete physical LoRAModule tensor triplets")
    path.parent.mkdir(parents=True, exist_ok=True)
    provider = SimpleNamespace(state_dict=lambda: tensors)
    LoRANetwork.save_weights(
        provider, str(path), None,
        {"ss_steps": str(step), "dts_cpu_diagnostic_only": "true"},
    )
    return tensors


def _diagnostic_pre_save_hook(accelerator, *, reject=False):
    counter = []
    def hook(models, weights, output_dir):
        del models, weights
        counter.append(Path(output_dir))
        if reject:
            raise RuntimeError("C4-B injected pre-save failure")
    handle = accelerator.register_save_state_pre_hook(hook)
    return counter, handle


def _require_c4b_method(test, adapter, method):
    member = getattr(adapter, method, None)
    test.assertTrue(
        callable(member),
        "C4-B CONTRACT ABSENT: " + method
        + " needs genuine same-instance post-return Save observation",
    )
    return member


class C4BRealSaveCpuRED(unittest.TestCase):
    def test_b01_true_r1_adamw_step_one_real_accelerate_save(self):
        bundle = _r1_real_step(self)
        model, acc, _, _, facade, probe = bundle
        self.assertEqual(probe._last_completed_step, 1)
        with tempfile.TemporaryDirectory() as dirname:
            out = Path(dirname) / "r1-actual"
            self.assertEqual(_real_cpu_state_save(acc, out), ACCELERATE_CPU_FILES)
            self.assertEqual(len(facade.entries), 1)
            with safe_open(str(out / "model.safetensors"),
                           framework="pt", device="cpu") as sf:
                self.assertEqual(set(sf.keys()), set(model.state_dict()))
                for key, current in model.state_dict().items():
                    self.assertTrue(torch.equal(sf.get_tensor(key), current.cpu()))
            self.assertTrue((out / "optimizer.bin").stat().st_size > 0)
            self.assertTrue((out / "scheduler.bin").stat().st_size > 0)
            self.assertTrue((out / "random_states_0.pkl").stat().st_size > 0)

    def test_b02_true_r2_muon_adamw_accum2_one_state_file_per_facade(self):
        bundle = _r2_real_step(self)
        model, acc, _, _, facade, probe = bundle
        self.assertEqual(probe.sync_sequence, (False, True))
        self.assertEqual(tuple(e.profile_name for e in facade.entries),
                         ("muon_primary", "adamw_fallback"))
        self.assertEqual(set(facade.state_dict()["children"]),
                         {"muon_primary", "adamw_fallback"})
        with tempfile.TemporaryDirectory() as dirname:
            out = Path(dirname) / "r2-actual"
            self.assertEqual(_real_cpu_state_save(acc, out), ACCELERATE_CPU_FILES)
            self.assertEqual(probe._last_completed_step, 1)
            self.assertEqual(len(model.state_dict()), 18)
            self.assertEqual(set(_files(out)).intersection(
                {"optimizer_1.bin", "scheduler_1.bin"}
            ), set())
            with safe_open(str(out / "model.safetensors"),
                           framework="pt", device="cpu") as sf:
                self.assertEqual(set(sf.keys()), set(model.state_dict()))

    def test_b03_actual_production_lora_save_weights_bf16_alpha_triplet(self):
        network = _production_one_lora()
        state = network.state_dict()
        self.assertEqual(len(state), 3)
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in state.values()))
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "genuine-provider.safetensors"
            network.save_weights(
                str(path), None, {"ss_steps": "1", "ss_epoch": "0"}
            )
            expected = _reference_tensors(state)
            accepted = host.verify_adapter_file(
                path, expected_file_sha256=_sha256(path),
                expected_tensors=expected,
            )
            self.assertEqual(accepted["file_sha256"], _sha256(path))
            with safe_open(str(path), framework="pt", device="cpu") as sf:
                self.assertEqual(sf.metadata().get("ss_steps"), "1")
                self.assertEqual(sum(k.endswith(".alpha") for k in sf.keys()), 1)

    def test_b04_host_rejects_actual_safetensors_byte_tamper(self):
        network = _production_one_lora()
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "genuine-provider.safetensors"
            network.save_weights(str(path), None, {"ss_steps": "1"})
            baseline = _sha256(path)
            refs = _reference_tensors(network.state_dict())
            host.verify_adapter_file(
                path, expected_file_sha256=baseline, expected_tensors=refs
            )
            with open(path, "r+b") as handle:
                handle.seek(-1, 2)
                last = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([last[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                host.verify_adapter_file(
                    path, expected_file_sha256=baseline,
                    expected_tensors=refs,
                )

    def test_b05_same_instance_save_state_wrapper_installed_and_removed(self):
        bundle = _r1_real_step(self)
        acc = bundle[1]
        original_slot = vars(acc).get("save_state")
        original_had_slot = "save_state" in vars(acc)
        with tempfile.TemporaryDirectory() as dirname:
            adapter = _bound_adapter(bundle, Path(dirname), self)
            wrapped = vars(acc).get("save_state")
            self.assertTrue(
                callable(wrapped),
                "C4-B CONTRACT ABSENT: same Accelerator instance needs"
                " a managed save_state post-return wrapper",
            )
            self.assertIsNot(wrapped, original_slot)
            adapter.close()
        self.assertEqual("save_state" in vars(acc), original_had_slot)
        self.assertIs(vars(acc).get("save_state"), original_slot)

    def test_b06_real_step_save_must_report_unqualified_cpu_after_return(self):
        bundle = _r2_real_step(self)
        model, acc, _, _, _, probe = bundle
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            adapter_path = root / "fresh" / "dts_c4a_cpu-step00000001.safetensors"
            state_path = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            actual_tensors = _real_diagnostic_lora_save(model, adapter_path, 1)
            self.assertEqual(len(actual_tensors), 18)
            error = None
            try:
                acc.save_state(str(state_path), safe_serialization=True)
            except ValueError as exc:
                error = exc
            self.assertEqual(
                _files(state_path), ACCELERATE_CPU_FILES,
                "Original real Accelerator.save_state did not write all four"
                " CPU state files before optional post-return rejection",
            )
            observer = _require_c4b_method(self, adapter, "save_attempt_observation")
            result = observer()
            self.assertTrue(result.original_returned)
            self.assertFalse(result.qualified)
            if error is not None:
                self.assertRegex(str(error), r"C4-B|unqualified|incomplete")
            self.assertFalse(any(
                row["event_type"] == "checkpoint_saved"
                for row in probe.events
            ))

    def test_b07_real_pre_save_hook_failure_cannot_become_saved(self):
        bundle = _r1_real_step(self)
        acc, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            counter, handle = _diagnostic_pre_save_hook(acc, reject=True)
            self.addCleanup(handle.remove)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            state = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            with self.assertRaisesRegex(RuntimeError, "injected pre-save failure"):
                acc.save_state(str(state), safe_serialization=True)
            self.assertEqual(len(counter), 1)
            observer = _require_c4b_method(self, adapter, "save_attempt_observation")
            attempt = observer()
            self.assertFalse(attempt.original_returned)
            self.assertFalse(attempt.qualified)
            self.assertFalse(any(
                row["event_type"] == "checkpoint_saved" for row in probe.events
            ))

    def test_b08_real_r2_full_cpu_lora_tensor_inventory_requires_alpha(self):
        bundle = _r2_real_step(self)
        model = bundle[0]
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "real-r2-diagnostic.safetensors"
            actual = _real_diagnostic_lora_save(model, path, 1)
            self.assertEqual(len(actual), 18)
            self.assertEqual(sum(k.endswith(".alpha") for k in actual), 6)
            self.assertTrue(all(t.dtype == torch.bfloat16 for t in actual.values()))
            module = "tools.sd3_full_bf16_worker_save_verifier"
            self.assertIsNotNone(
                importlib.util.find_spec(module),
                "C4-B CONTRACT ABSENT: trusted serialized 18-tensor CPU"
                " diagnostic inventory and 789-tensor production inventory",
            )
            verifier = importlib.import_module(module)
            snapshot = getattr(verifier, "snapshot_serialized_tensor_inventory", None)
            self.assertTrue(callable(snapshot),
                            "C4-B CONTRACT ABSENT: actual full LoRA Alpha+Weight snapshot")
            info = snapshot(model, diagnostic_cpu=True)
            self.assertEqual(set(info), set(actual))
            self.assertEqual(
                set(info), set(_reference_tensors(actual)),
                "Diagnostic live tensor inventory lost the genuine Alpha buffer",
            )

    def test_b09_unreviewed_state_dir_rejected_before_original_save(self):
        bundle = _r1_real_step(self)
        acc, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            unauthorized = root / "fresh" / "unknown-step00000001-state"
            counter, handle = _diagnostic_pre_save_hook(acc)
            self.addCleanup(handle.remove)
            try:
                acc.save_state(str(unauthorized), safe_serialization=True)
            except ValueError as exc:
                self.assertRegex(str(exc), "C4|unreviewed|Save")
                self.assertFalse(unauthorized.exists())
                self.assertEqual(counter, [])
            else:
                self.fail(
                    "C4-B CONTRACT ABSENT: real Accelerator.save_state wrote"
                    " an unreviewed Step path before authorization",
                )
            self.assertFalse(any(
                row["event_type"] == "checkpoint_saved" for row in probe.events
            ))

    def test_b10_missing_real_rng_file_cannot_create_verified_receipt(self):
        bundle = _r1_real_step(self)
        acc, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            directory = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            try:
                acc.save_state(str(directory), safe_serialization=True)
            except ValueError as exc:
                self.assertRegex(str(exc), "C4-B|unqualified|incomplete")
            self.assertEqual(_files(directory), ACCELERATE_CPU_FILES)
            (directory / "random_states_0.pkl").unlink()
            verifier = _require_c4b_method(self, adapter, "verified_step_save")
            with self.assertRaises(ValueError):
                verifier(logical_step=1)
            self.assertFalse(any(
                row["event_type"] == "checkpoint_saved" for row in probe.events
            ))

    def test_b11_true_accelerator_save_failure_does_not_publish_success(self):
        bundle = _r1_real_step(self)
        acc, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            conflict = root / "fresh" / "dts_c4a_cpu-step00000001-state"
            conflict.parent.mkdir(parents=True, exist_ok=True)
            conflict.write_text("not a directory", encoding="utf-8")
            with self.assertRaises((OSError, ValueError)):
                acc.save_state(str(conflict), safe_serialization=True)
            self.assertTrue(conflict.is_file())
            self.assertFalse(any(row["event_type"] == "checkpoint_saved"
                                 for row in probe.events))

    def test_b12_real_last_save_is_ancillary_not_extra_step(self):
        bundle = _r1_real_step(self)
        acc, probe = bundle[1], bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            last = root / "fresh" / "dts_c4a_cpu-state"
            self.assertEqual(
                adapter.classify_state_save(state_dir=last), "ancillary_last"
            )
            self.assertEqual(_real_cpu_state_save(acc, last), ACCELERATE_CPU_FILES)
            self.assertEqual(probe._last_completed_step, 1)
            self.assertFalse(any(
                row["event_type"] in {"checkpoint_saved", "checkpoint_loaded",
                                      "train_completed"}
                for row in probe.events
            ))


if __name__ == "__main__":
    unittest.main()
