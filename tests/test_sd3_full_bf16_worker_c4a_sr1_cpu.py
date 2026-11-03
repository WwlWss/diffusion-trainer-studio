"""C4-A SR1 independent strict-review regressions, tests-only RED first.

Eight pinned REAL CPU tests (Torch BF16, actual Accelerator backward,
Muon + AdamW Children, CompositeOptimizer, CompositeLRScheduler).
Six negative controls require new fail-closed behavior; two are pre-existing
positive/negative controls. No mocked optimizer/scheduler Steps, CUDA, I/O
checkpoint authentication or worker_complete.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_sd3_full_bf16_worker import R1
from tests.test_sd3_full_bf16_worker_c3_f8_cpu import _make_r1_cpu_probe
from tests.test_sd3_full_bf16_worker_c3_f9_cpu import _lora_bundle
from tests.test_sd3_full_bf16_worker_c4a_cpu import (
    _r1_real_step, _r2_real_step, _invocation, _manifest,
)
from tools.sd3_full_bf16_worker_step_adapter import (
    C4PhysicalStepAnchor, C4StepAdapter, canonical_c4_anchor_sha256,
)


def _must_reject(test, operation, *, case, expected):
    try:
        operation()
    except ValueError as error:
        test.assertRegex(str(error), expected)
        return
    test.fail("C4-A-SR1 " + case + ": unaudited state was incorrectly accepted")


def _r1_batch(bundle, number):
    model, accelerator, prepared_opt, prepared_sched, _, probe = bundle
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    with accelerator.accumulate(model):
        probe.begin_batch(f"dts_c2b_sample_{number:04d}")
        accelerator.backward(model(x).square().float().sum())
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        prepared_opt.step()
        prepared_sched.step()
        prepared_opt.zero_grad(set_to_none=True)
    probe.assert_step_complete()


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


class C4ASR1RealCpuReview(unittest.TestCase):
    def test_sr1_a01_same_value_parameter_map_rebind_must_reject(self):
        probe = _r2_real_step(self)[5]
        name, original = next(iter(probe.params.items()))
        probe.params[name] = torch.nn.Parameter(
            original.detach().clone(), requires_grad=original.requires_grad,
        )
        _must_reject(
            self, lambda: probe.export_completed_step_anchor(1),
            case="P1-01 same-value mapping rebind",
            expected=r"Prepared Parameter identity drift|canonical Parameter|parameter mapping",
        )

    def test_sr1_a02_stale_parameter_map_masks_real_bf16_mutation(self):
        bundle = _r2_real_step(self)
        probe = bundle[5]
        name, original = next(iter(probe.params.items()))
        probe.params[name] = torch.nn.Parameter(
            original.detach().clone(), requires_grad=original.requires_grad,
        )
        with torch.no_grad():
            original.add_(torch.ones_like(original))
        _must_reject(
            self, lambda: probe.export_completed_step_anchor(1),
            case="P1-01 original BF16 weights changed but clone looks unchanged",
            expected=r"Prepared Parameter identity drift|canonical Parameter|parameter mapping|Adapter bytes drift",
        )

    def test_sr1_b01_injected_probe_exporter_cannot_mint_physical_step(self):
        bundle = _make_r1_cpu_probe()
        model, accelerator, _, _, _, probe = bundle
        probe.install()
        self.addCleanup(probe.close)
        with tempfile.TemporaryDirectory() as dirname:
            adapter = _bound_adapter(bundle, Path(dirname), self)
            payload = dict(
                case_id=R1, phase="fresh", run_id=probe.run_id,
                phase_id=probe.phase_id, source_commit=probe.source_commit,
                logical_step=1, accumulation=1,
                samples=("dts_c2b_sample_0001",),
                child_profiles=("baseline",),
                physical_scheduler_event_seq=7,
                trained_adapter_parameter_sha256="a" * 64,
                child_optimizer_state_sha256=(("baseline", "a" * 64),),
                scheduler_state_sha256="a" * 64,
            )
            fake = C4PhysicalStepAnchor(
                **payload, anchor_sha256=canonical_c4_anchor_sha256(payload)
            )
            probe.export_completed_step_anchor = lambda step: fake
            _must_reject(
                self, lambda: adapter.reserve_step_save(logical_step=1),
                case="P1-02 an instance-exporter override forged an unstepped Anchor",
                expected=r"Probe exporter|override|unreviewed|physical Step",
            )
            self.assertFalse(any(row.get("event_type") == "checkpoint_saved"
                                 for row in probe.events))

    def test_sr1_b02_anchor_digest_and_requested_step_checked_separately(self):
        bundle = _r1_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            adapter = _bound_adapter(bundle, Path(dirname), self)
            validator = getattr(adapter, "_validate_anchor", None)
            if not callable(validator):
                self.fail("C4-A-SR1 P1-02: missing independent Anchor validator")
            genuine = probe.export_completed_step_anchor(1)
            bad_digest = replace(genuine, anchor_sha256="b" * 64)
            _must_reject(
                self, lambda: validator(bad_digest, logical_step=1),
                case="P1-02 arbitrary Anchor checksum",
                expected=r"Anchor|digest|sha256|hash",
            )
            changed = asdict(genuine)
            changed["logical_step"] = 2
            changed.pop("anchor_sha256")
            changed["samples"] = tuple(changed["samples"])
            changed["child_profiles"] = tuple(changed["child_profiles"])
            changed["child_optimizer_state_sha256"] = tuple(
                tuple(pair) for pair in changed["child_optimizer_state_sha256"]
            )
            tampered = C4PhysicalStepAnchor(
                **changed, anchor_sha256=canonical_c4_anchor_sha256(changed)
            )
            _must_reject(
                self, lambda: validator(tampered, logical_step=1),
                case="P1-02 internally consistent but wrong logical Step",
                expected=r"logical Step|Anchor|Step mismatch",
            )

    def test_sr1_c01_old_step_reservation_stale_after_real_step_two(self):
        bundle = _make_r1_cpu_probe()
        probe = bundle[5]
        probe.install()
        self.addCleanup(probe.close)
        _r1_batch(bundle, 1)
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            first = adapter.reserve_step_save(logical_step=1)
            self.assertEqual(first.logical_step, 1)
            self.assertEqual(adapter.classify_state_save(
                state_dir=root / "fresh" / "dts_c4a_cpu-step00000001-state"
            ), "authoritative_step_1")
            _r1_batch(bundle, 2)
            self.assertEqual(probe._last_completed_step, 2)
            _must_reject(
                self, lambda: adapter.classify_state_save(
                    state_dir=root / "fresh" / "dts_c4a_cpu-step00000001-state"
                ),
                case="P2-01 Step-1 pending reservation is stale after real Step-2",
                expected=r"Stale|expired|physical Step|reservation",
            )

    def test_sr1_c02_failed_classification_does_not_publish_success(self):
        bundle = _r2_real_step(self)
        probe = bundle[5]
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            adapter = _bound_adapter(bundle, root, self)
            adapter.reserve_step_save(logical_step=1)
            with self.assertRaises(ValueError):
                adapter.classify_state_save(state_dir=root / "fresh" / "unknown")
            self.assertEqual(adapter.classify_state_save(
                state_dir=root / "fresh" / "dts_c4a_cpu-step00000001-state"
            ), "authoritative_step_1")
            self.assertFalse(any(row["event_type"] in (
                "checkpoint_saved", "checkpoint_loaded", "train_completed"
            ) for row in probe.events))

    def test_sr1_d01_real_child_order_reversal_before_step_rejected(self):
        bundle = _lora_bundle()
        model, accelerator, opt, sched, facade, probe = bundle
        probe.install()
        self.addCleanup(probe.close)
        x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
        for number in (1, 2):
            with accelerator.accumulate(model):
                probe.begin_batch(f"dts_c2b_sample_{number:04d}")
                accelerator.backward(model(x))
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    original_entries = facade.entries
                    facade.entries = tuple(reversed(original_entries))
                    try:
                        _must_reject(
                            self, opt.step,
                            case="P2-02 live physical Child order swapped",
                            expected=r"Child entry order drift|Child execution order drift|Prepared.*[Oo]rder",
                        )
                    finally:
                        facade.entries = original_entries
                else:
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

    def test_sr1_d02_real_r2_canonical_child_order_exports_anchor(self):
        bundle = _r2_real_step(self)
        probe = bundle[5]
        anchor = probe.export_completed_step_anchor(1)
        self.assertEqual(anchor.child_profiles,
                         ("muon_primary", "adamw_fallback"))
        observed = tuple(
            e["optimizer_profile"] for e in probe.events
            if e["event_type"] == "physical_child_stepped"
        )
        self.assertEqual(observed, anchor.child_profiles)
        with tempfile.TemporaryDirectory() as dirname:
            adapter = _bound_adapter(bundle, Path(dirname), self)
            self.assertEqual(
                adapter.reserve_step_save(logical_step=1).anchor_sha256,
                anchor.anchor_sha256,
            )


if __name__ == "__main__":
    unittest.main()
