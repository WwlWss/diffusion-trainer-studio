"""C4-A Tests-only RED: pinned real-CPU Step Adapter contract.

Target: 12 independently discovered tests; zero ERROR/SKIP/xfail.
This suite runs *actual* BF16 LoRAModule, Accelerator.backward, Muon,
AdamW, CompositeOptimizer and CompositeLRScheduler from the existing C3
fixtures. Missing C4-A APIs must produce targeted assertion FAIL, never
accidental ImportError/AttributeError. Tiny CPU fixtures do NOT qualify CUDA.

The production Save/Load byte/lineage verification belongs to C4-B/C;
this file neither fabricates checkpoints nor permits worker_complete.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from tests.test_sd3_full_bf16_worker import R1, R2
from tests.test_sd3_full_bf16_worker_c3_f8_cpu import _make_r1_cpu_probe
from tests.test_sd3_full_bf16_worker_c3_f9_cpu import _lora_bundle
from tools import sd3_full_bf16_worker_observer as observer


API = "tools.sd3_full_bf16_worker_step_adapter"
REQUIRED_ANCHOR_FIELDS = (
    "case_id", "phase", "run_id", "phase_id", "source_commit",
    "logical_step", "accumulation", "samples", "child_profiles",
    "physical_scheduler_event_seq", "trained_adapter_parameter_sha256",
    "child_optimizer_state_sha256", "scheduler_state_sha256",
    "anchor_sha256",
)


def _r1_real_step(test, *, count=1):
    """Keep C3 installed through Snapshot; the original F8 helper closes it."""
    bundle = _make_r1_cpu_probe()
    model, accelerator, prepared_opt, prepared_sched, _, probe = bundle
    probe.install()
    test.addCleanup(probe.close)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    for number in range(1, count + 1):
        with accelerator.accumulate(model):
            probe.begin_batch(f"dts_c2b_sample_{number:04d}")
            accelerator.backward(model(x).square().float().sum())
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            prepared_opt.step()
            prepared_sched.step()
            prepared_opt.zero_grad(set_to_none=True)
        probe.assert_step_complete()
    test.assertEqual(probe._last_completed_step, count)
    test.assertTrue(any(e["event_type"] == "physical_scheduler_stepped"
                        and e["logical_step"] == count for e in probe.events))
    return bundle


def _r2_real_step(test, *, complete=True):
    """The real F9 LoRAModule+Muon+AdamW fixture, without closing C3."""
    bundle = _lora_bundle()
    model, accelerator, prepared_opt, prepared_sched, _, probe = bundle
    probe.install()
    test.addCleanup(probe.close)
    x = torch.tensor([[0.25, -0.5, 1.0, -1.5]], dtype=torch.bfloat16)
    for number in ((1, 2) if complete else (1,)):
        with accelerator.accumulate(model):
            probe.begin_batch(f"dts_c2b_sample_{number:04d}")
            accelerator.backward(model(x))
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            prepared_opt.step()
            prepared_sched.step()
            prepared_opt.zero_grad(set_to_none=True)
    if complete:
        probe.assert_step_complete()
        test.assertEqual(probe._step_scheduler[1], 1)
        test.assertEqual(probe._step_children[1],
                         {"muon_primary": 1, "adamw_fallback": 1})
        test.assertEqual(probe._last_completed_step, 1)
    else:
        test.assertFalse(any(e["event_type"] == "physical_scheduler_stepped"
                             for e in probe.events))
        test.assertIsNone(probe._last_completed_step)
    return bundle


def _invocation(probe):
    return SimpleNamespace(
        case_id=probe.case_id, phase=probe.phase,
        run_id=probe.run_id, phase_id=probe.phase_id,
        source_commit=probe.source_commit,
    )


def _manifest(probe, root):
    # Minimal *CPU-only* identity fixture; never a CUDA-ready CaseManifest.
    return {
        "case_id": probe.case_id, "phase": probe.phase,
        "run_id": probe.run_id, "phase_id": probe.phase_id,
        "source_commit": probe.source_commit,
        "work_root": str(root), "output_name": "dts_c4a_cpu",
        "output_dir": str(root / probe.phase),
        "reference": {"max_train_steps": 2},
    }


class C4ACpuStepAdapterRED(unittest.TestCase):
    def _api(self):
        self.assertIsNotNone(
            importlib.util.find_spec(API),
            "C4-A CONTRACT ABSENT: independent Step Adapter module is missing",
        )
        api = importlib.import_module(API)
        self.assertTrue(hasattr(api, "C4PhysicalStepAnchor"),
                        "C4-A CONTRACT ABSENT: immutable Step Anchor type")
        self.assertTrue(hasattr(api, "C4StepAdapter"),
                        "C4-A CONTRACT ABSENT: C4StepAdapter")
        return api

    def _export(self, probe, step=1):
        exporter = getattr(probe, "export_completed_step_anchor", None)
        self.assertTrue(
            callable(exporter),
            "C4-A CONTRACT ABSENT: genuine C3 Probe must export a read-only "
            "completed physical Step; fabricated event JSON is not an Anchor",
        )
        return exporter(step)

    def _adapter(self, bundle, root):
        api = self._api()
        model, accelerator, _, _, _, probe = bundle
        adapter = api.C4StepAdapter(
            probe=probe, accelerator=accelerator,
            network=model, scheduler=probe.scheduler,
            invocation=_invocation(probe),
            manifest_payload=_manifest(probe, root),
        )
        self.addCleanup(adapter.close)
        return adapter

    def test_a01_r1_real_adamw_step_exports_frozen_source_bound_anchor(self):
        probe = _r1_real_step(self)[5]
        anchor = self._export(probe)
        api = self._api()
        self.assertIsInstance(anchor, api.C4PhysicalStepAnchor)
        self.assertTrue(is_dataclass(anchor))
        self.assertTrue(anchor.__dataclass_params__.frozen)
        self.assertEqual(tuple(f.name for f in fields(anchor)), REQUIRED_ANCHOR_FIELDS)
        self.assertEqual((anchor.case_id, anchor.phase, anchor.logical_step),
                         (R1, "fresh", 1))
        self.assertEqual(anchor.accumulation, 1)
        self.assertEqual(anchor.child_profiles, ("baseline",))
        self.assertEqual(anchor.samples, ("dts_c2b_sample_0001",))
        self.assertEqual(len(anchor.anchor_sha256), 64)
        with self.assertRaises(FrozenInstanceError):
            anchor.logical_step = 2

    def test_a02_r2_first_unsynced_microbatch_cannot_export_step(self):
        probe = _r2_real_step(self, complete=False)[5]
        self.assertEqual(probe.sync_sequence, (False,))
        with self.assertRaises(ValueError):
            self._export(probe)

    def test_a03_r2_two_real_batches_two_children_one_scheduler_anchor(self):
        bundle = _r2_real_step(self)
        probe = bundle[5]
        anchor = self._export(probe)
        self.assertEqual((anchor.case_id, anchor.logical_step), (R2, 1))
        self.assertEqual(anchor.accumulation, 2)
        self.assertEqual(anchor.samples,
                         ("dts_c2b_sample_0001", "dts_c2b_sample_0002"))
        self.assertEqual(set(anchor.child_profiles),
                         {"muon_primary", "adamw_fallback"})
        self.assertEqual(len(anchor.child_optimizer_state_sha256), 2)
        self.assertEqual(len(anchor.scheduler_state_sha256), 64)

    def test_a04_forged_event_without_real_child_step_cannot_export(self):
        bundle = _make_r1_cpu_probe()
        probe = bundle[5]
        probe.install()
        self.addCleanup(probe.close)
        # A fake JSON row cannot substitute for real CompositeOptimizer.step.
        probe.events.append({
            "event_type": "physical_scheduler_stepped",
            "logical_step": 1, "event_seq": 1,
            "case_id": R1, "phase_id": probe.phase_id,
            "run_id": probe.run_id, "source_commit": probe.source_commit,
        })
        self.assertEqual(probe._step_children, {})
        with self.assertRaises(ValueError):
            self._export(probe)

    def test_a05_missing_or_duplicate_real_child_rejected(self):
        probe = _r2_real_step(self)[5]
        accepted = dict(probe._step_children[1])
        for count in (0, 2):
            probe._step_children[1]["muon_primary"] = count
            with self.subTest(muon_count=count), self.assertRaises(ValueError):
                self._export(probe)
        probe._step_children[1].update(accepted)

    def test_a06_missing_or_duplicate_real_scheduler_rejected(self):
        probe = _r1_real_step(self)[5]
        for count in (0, 2):
            probe._step_scheduler[1] = count
            with self.subTest(scheduler_count=count), self.assertRaises(ValueError):
                self._export(probe)

    def test_a07_wrong_logical_step_and_phase_identity_not_rebased(self):
        probe = _r1_real_step(self)[5]
        with self.assertRaises(ValueError):
            self._export(probe, step=2)
        anchor = self._export(probe, step=1)
        self.assertEqual((anchor.run_id, anchor.phase_id, anchor.source_commit),
                         (probe.run_id, probe.phase_id, probe.source_commit))
        self.assertEqual(anchor.phase, "fresh")
        self.assertNotEqual(anchor.phase_id, "resume-" + probe.phase_id)

    def test_a08_one_use_step_save_reservation_rejects_duplicate(self):
        bundle = _r2_real_step(self)
        with tempfile.TemporaryDirectory() as td:
            adapter = self._adapter(bundle, Path(td))
            adapter.install()
            anchor = adapter.completed_step_anchor(logical_step=1)
            self.assertEqual(anchor.logical_step, 1)
            reserve = getattr(adapter, "reserve_step_save", None)
            self.assertTrue(callable(reserve),
                            "C4-A CONTRACT ABSENT: one-use pending Save reservation")
            first = reserve(logical_step=1)
            self.assertEqual(first.anchor_sha256, anchor.anchor_sha256)
            with self.assertRaises(ValueError):
                reserve(logical_step=1)
            self.assertFalse(any(
                e["event_type"] == "checkpoint_saved"
                for e in bundle[5].events
            ), "A pending reservation falsely published a saved checkpoint")

    def test_a09_post_step_adapter_or_optimizer_drift_cannot_be_accepted(self):
        # Actual BF16 Parameter mutation after completed Scheduler boundary.
        bundle = _r2_real_step(self)
        probe = bundle[5]
        with torch.no_grad():
            first = next(iter(probe.params.values()))
            first.add_(torch.ones_like(first))
        with self.assertRaises(ValueError):
            self._export(probe)
        # Recheck independently with a real optimizer state drift.
        other = _r1_real_step(self)
        other_probe = other[5]
        adamw = other[4].entries[0].optimizer
        weight = other[0].weight
        self.assertIn("exp_avg", adamw.state[weight])
        with torch.no_grad():
            adamw.state[weight]["exp_avg"].add_(1)
        with self.assertRaises(ValueError):
            self._export(other_probe)

    def test_a10_last_tail_classifies_ancillary_never_step_saved(self):
        bundle = _r2_real_step(self)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            adapter = self._adapter(bundle, root)
            adapter.install()
            classifier = getattr(adapter, "classify_state_save", None)
            self.assertTrue(callable(classifier),
                            "C4-A CONTRACT ABSENT: exact last/Step Save classification")
            classification = classifier(
                state_dir=root / "fresh" / "dts_c4a_cpu-state"
            )
            self.assertEqual(classification, "ancillary_last")
            self.assertFalse(any(
                e["event_type"] == "checkpoint_saved" for e in bundle[5].events
            ), "last Save was misrepresented as an authoritative Step")

    def test_a11_resume_prebatch_load_path_must_not_fake_begin_batch(self):
        bundle = _lora_bundle()
        model, accelerator, _, _, facade, old = bundle
        probe = observer.PhysicalStepProbe(
            accelerator=accelerator, composite_optimizer=facade,
            composite_scheduler=old.scheduler,
            parameter_names=old.params,
            component_by_parameter=old.component_by_parameter,
            case_id=R2, run_id="run-c4a-resume-0001",
            phase_id="resume-c4a-resume-0001",
            source_commit="a" * 40, phase="resume", start_step=1,
            lora_network=model,
        )
        probe.install()
        self.addCleanup(probe.close)
        self.assertIsNone(probe._current_step)
        with self.assertRaises(ValueError):
            probe._record("checkpoint_loaded", details={"checkpoint_id": "a" * 32})
        self.assertEqual(probe.events, [])
        with tempfile.TemporaryDirectory() as td:
            adapter = self._adapter(
                (model, accelerator, None, None, facade, probe), Path(td)
            )
            adapter.install()
            self.assertIsNone(probe._current_step)
            self.assertEqual(probe.events, [],
                             "Installing C4-A must never invent a Resume load")
            self.assertFalse(hasattr(adapter, "emit"),
                             "Arbitrary public event injection is forbidden")

    def test_a12_aborted_adapter_restores_instance_methods_and_stays_closed(self):
        bundle = _r2_real_step(self)
        model, accelerator, _, _, _, probe = bundle
        before_save = vars(accelerator).get("save_state")
        before_load = vars(accelerator).get("load_state")
        before_events = len(probe.events)
        with tempfile.TemporaryDirectory() as td:
            adapter = self._adapter(bundle, Path(td))
            adapter.install()
            adapter.close()
        self.assertIs(vars(accelerator).get("save_state"), before_save)
        self.assertIs(vars(accelerator).get("load_state"), before_load)
        self.assertEqual(len(probe.events), before_events)
        self.assertFalse(any(
            e["event_type"] in {"checkpoint_saved", "train_completed"}
            for e in probe.events
        ), "C4-A close cannot certify Save or worker_complete")


if __name__ == "__main__":
    unittest.main()
