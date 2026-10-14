"""SD3-only fail-closed resume: logical steps, epoch cursor and saved geometry."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "dev" / "library" / "sd3_resume_cursor.py"


def _load():
    import sys
    spec = importlib.util.spec_from_file_location("sd3_resume_cursor", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Accelerator:
    is_main_process = True
    num_processes = 1

    def __init__(self):
        self.save_hooks, self.load_hooks = [], []

    def register_save_state_pre_hook(self, hook):
        self.save_hooks.append(hook)

    def register_load_state_pre_hook(self, hook):
        self.load_hooks.append(hook)


class SD3ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def plan(self, **overrides):
        kw = dict(
            saved_step=1, saved_epoch=1, max_train_steps=2,
            dataloader_batches=1, gradient_accumulation_steps=1,
        )
        kw.update(overrides)
        return self.m.plan_sd3_lora_resume_cursor(**kw)

    def test_one_to_two_epoch_boundary(self):
        cursor = self.plan()
        self.assertEqual((cursor.completed_steps, cursor.epoch_to_start, cursor.batches_to_skip, cursor.remaining_steps), (1, 1, 0, 1))

    def test_accumulation_two_two_batches_is_one_optimizer_step(self):
        cursor = self.plan(dataloader_batches=2, gradient_accumulation_steps=2)
        self.assertEqual((cursor.completed_steps, cursor.epoch_to_start, cursor.batches_to_skip, cursor.remaining_steps), (1, 1, 0, 1))

    def test_partial_final_accumulation_group_at_epoch_boundary(self):
        cursor = self.plan(dataloader_batches=3, gradient_accumulation_steps=4)
        self.assertEqual(cursor.epoch_to_start, 1)
        self.assertEqual(cursor.batches_to_skip, 0)

    def test_middle_of_epoch_explicitly_rejected(self):
        with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "mid-epoch"):
            self.plan(dataloader_batches=4, gradient_accumulation_steps=1)

    def test_epoch_step_mismatch(self):
        with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "epoch/step mismatch"):
            self.plan(saved_epoch=2)

    def test_no_remaining_steps(self):
        with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "no remaining"):
            self.plan(saved_step=2)

    def test_competing_lifecycle_overrides_and_multiprocess(self):
        for x in ("initial_step", "initial_epoch", "max_train_epochs"):
            with self.subTest(field=x), self.assertRaisesRegex(self.m.SD3ResumeCursorError, "overrides"):
                self.plan(**{x: 1})
        with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "overrides"):
            self.plan(skip_until_initial_step=True)
        with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "one process"):
            self.plan(num_processes=2)

    def test_bool_and_missing_checkpoint_progress_rejected(self):
        for value in (None, False, 0, -1, "1"):
            with self.subTest(value=value), self.assertRaises(self.m.SD3ResumeCursorError):
                self.plan(saved_step=value)

    def test_save_load_sidecar_matches_train_state(self):
        accelerator = Accelerator()
        args = SimpleNamespace(gradient_accumulation_steps=2)
        current_epoch, current_step = SimpleNamespace(value=1), SimpleNamespace(value=0)
        self.m.register_sd3_resume_progress_hooks(
            accelerator, args=args, train_dataloader=[1, 2],
            current_epoch=current_epoch, current_step=current_step,
        )
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "train_state.json").write_text(
                json.dumps({"current_epoch": 1, "current_step": 1}), encoding="utf-8",
            )
            accelerator.save_hooks[0]([], [], td)
            progress = self.m.validate_sd3_resume_progress(
                td, dataloader_batches=2, gradient_accumulation_steps=2, num_processes=1,
            )
            self.assertEqual(progress["completed_optimizer_steps"], 1)
            accelerator.load_hooks[0]([], td)

    def test_missing_modified_accumulation_or_train_state_rejected(self):
        accelerator = Accelerator()
        args = SimpleNamespace(gradient_accumulation_steps=1)
        self.m.register_sd3_resume_progress_hooks(
            accelerator, args=args, train_dataloader=[0],
            current_epoch=SimpleNamespace(value=1), current_step=SimpleNamespace(value=0),
        )
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "train_state.json").write_text(
                json.dumps({"current_step": 1, "current_epoch": 1}), encoding="utf-8",
            )
            with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "sidecar"):
                accelerator.load_hooks[0]([], td)
            accelerator.save_hooks[0]([], [], td)
            with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "conflicts"):
                self.m.validate_sd3_resume_progress(
                    td, dataloader_batches=1, gradient_accumulation_steps=2,
                    num_processes=1,
                )
            (folder / "train_state.json").write_text(
                json.dumps({"current_step": 2, "current_epoch": 1}), encoding="utf-8",
            )
            with self.assertRaisesRegex(self.m.SD3ResumeCursorError, "conflicts"):
                accelerator.load_hooks[0]([], td)

    def test_scoped_production_wiring_preserves_other_dev_trainers(self):
        base = (ROOT / "scripts" / "dev" / "train_network.py").read_text(encoding="utf-8")
        sd3 = (ROOT / "scripts" / "dev" / "sd3_train_network.py").read_text(encoding="utf-8")
        flux = (ROOT / "scripts" / "dev" / "flux_train_network.py").read_text(encoding="utf-8")
        self.assertIn("backend_resume_cursor = self.plan_backend_resume_cursor(", base)
        self.assertIn("global_step = backend_resume_cursor.completed_steps", base)
        self.assertIn("backend_resume_cursor.remaining_steps", base)
        self.assertIn("register_sd3_resume_progress_hooks(", sd3)
        self.assertIn("plan_sd3_lora_resume_cursor(", sd3)
        self.assertNotIn("plan_sd3_lora_resume_cursor", flux)


if __name__ == "__main__":
    unittest.main()
