"""SD3 WebUI duration and resumed-checkpoint compatibility regressions.

Use real DTS config compilation, but small temporary PR #60 sidecars instead
of loading a multi-gigabyte MMDiT on CI.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mikazuki.sd3_resume_ui import (
    inspect_sd3_resume_checkpoint,
    normalize_sd3_training_duration,
)
from mikazuki.training_config import prepare_training_config
from mikazuki.training_rehydrate import rehydrate_trainer_config
from mikazuki.training_validation import validate_prepared_config


def _resolver(config, requested):
    return requested, f"./{requested}.py"


class Sd3UiResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "step-one-state"

    def checkpoint(self, *, step=3, epoch=1, batches=5, accumulation=2, processes=1,
                   version=1, corrupt=False):
        self.folder.mkdir(exist_ok=True)
        progress = {
            "schema": "dts.sd3.resume-progress",
            "version": version,
            "completed_optimizer_steps": step,
            "current_epoch": epoch,
            "dataloader_batches": batches,
            "gradient_accumulation_steps": accumulation,
            "num_processes": processes,
        }
        (self.folder / "dts_sd3_resume_progress.json").write_text(
            "{bad" if corrupt else json.dumps(progress), encoding="utf-8"
        )
        (self.folder / "train_state.json").write_text(
            json.dumps({"current_epoch": epoch, "current_step": step}),
            encoding="utf-8",
        )
        return str(self.folder)

    def prepare(self, overrides=None, *, launch=False):
        raw = {"optimizer_type": "AdamW"}
        raw.update(overrides or {})
        return prepare_training_config(
            raw, page_train_type="sd3-lora", resolve_backend=_resolver, launch=launch
        )

    def test_fresh_gui_epoch_default_semantics(self):
        prepared = self.prepare({
            "sd3_training_target_mode": "epochs",
            "sd3_training_target_epochs": 20,
            "sd3_training_target_steps": 500,  # stale inactive branch
        })
        self.assertEqual(prepared.config["max_train_epochs"], 20)
        self.assertNotIn("max_train_steps", prepared.config)
        self.assertFalse(any(key.startswith("sd3_training_") for key in prepared.config))

    def test_fresh_gui_steps_exclude_epoch_override(self):
        prepared = self.prepare({
            "sd3_training_target_mode": "steps",
            "sd3_training_target_steps": 1200,
            "sd3_training_target_epochs": 20,
        })
        self.assertEqual(prepared.config["max_train_steps"], 1200)
        self.assertNotIn("max_train_epochs", prepared.config)

    def test_epoch_resume_converts_to_absolute_steps_using_saved_geometry(self):
        resume = self.checkpoint()
        raw = {
            "resume": resume,
            "sd3_training_target_mode": "epochs",
            "sd3_training_target_epochs": 3,
            "gradient_accumulation_steps": 2,
        }
        prepared = self.prepare(raw)
        self.assertEqual(prepared.config["max_train_steps"], 9)
        self.assertNotIn("max_train_epochs", prepared.config)
        self.assertTrue(any("3 Epoch" in item and "= 9" in item for item in prepared.warnings))
        self.assertTrue(any("Checkpoint 未记录原训练" in item for item in prepared.warnings))
        self.assertEqual(
            prepared.config,
            self.prepare(raw, launch=True).config,
            "Preview/Start effective config must not diverge",
        )
        validate_prepared_config(prepared, False)  # no model/dataset needed for Preview

    def test_steps_resume_preserves_explicit_total(self):
        resume = self.checkpoint()
        prepared = self.prepare({
            "resume": resume,
            "sd3_training_target_mode": "steps",
            "sd3_training_target_steps": 8,
            "gradient_accumulation_steps": 2,
        })
        self.assertEqual(prepared.config["max_train_steps"], 8)
        self.assertNotIn("max_train_epochs", prepared.config)
        self.assertTrue(any("原训练的目标总步数" in w for w in prepared.warnings))

    def test_legacy_duration_without_ui_mode_uses_original_epoch_precedence(self):
        prepared = self.prepare({"max_train_epochs": 20, "max_train_steps": 100})
        self.assertEqual(prepared.config["max_train_epochs"], 20)
        self.assertEqual(prepared.config["max_train_steps"], 100)
        self.assertTrue(any("Epoch 优先" in w for w in prepared.warnings))
        raw_headless = self.prepare({})
        self.assertNotIn("max_train_epochs", raw_headless.config)
        self.assertNotIn("max_train_steps", raw_headless.config)

    def test_legacy_epoch_resume_compiles_to_absolute_steps(self):
        resume = self.checkpoint()
        prepared = self.prepare({
            "resume": resume,
            "max_train_epochs": 5,
            "gradient_accumulation_steps": 2,
        })
        self.assertEqual(prepared.config["max_train_steps"], 15)
        self.assertNotIn("max_train_epochs", prepared.config)

    def test_resume_requires_explicit_target_and_future_steps(self):
        resume = self.checkpoint()
        with self.assertRaisesRegex(ValueError, "必须明确指定"):
            self.prepare({"resume": resume, "gradient_accumulation_steps": 2})
        with self.assertRaisesRegex(ValueError, "必须大于"):
            self.prepare({
                "resume": resume,
                "sd3_training_target_mode": "steps",
                "sd3_training_target_steps": 3,
                "gradient_accumulation_steps": 2,
            })

    def test_corrupt_missing_and_outdated_resume_sidecars_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "目录不存在"):
            inspect_sd3_resume_checkpoint(str(self.folder))
        resume = self.checkpoint(corrupt=True)
        with self.assertRaisesRegex(ValueError, "有效"):
            inspect_sd3_resume_checkpoint(resume)
        resume = self.checkpoint(version=99)
        with self.assertRaisesRegex(ValueError, "Schema/Version"):
            inspect_sd3_resume_checkpoint(resume)
        progress_path = self.folder / "dts_sd3_resume_progress.json"
        progress_path.unlink()
        with self.assertRaisesRegex(ValueError, "dts_sd3_resume_progress"):
            inspect_sd3_resume_checkpoint(resume)

    def test_mismatched_step_or_mid_epoch_or_multiprocess_fail(self):
        resume = self.checkpoint()
        path = self.folder / "train_state.json"
        path.write_text(json.dumps({"current_epoch": 1, "current_step": 6}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不匹配"):
            inspect_sd3_resume_checkpoint(resume)
        resume = self.checkpoint(step=2)
        with self.assertRaisesRegex(ValueError, "Epoch 中途"):
            inspect_sd3_resume_checkpoint(resume)
        resume = self.checkpoint(processes=2)
        with self.assertRaisesRegex(ValueError, "单进程"):
            inspect_sd3_resume_checkpoint(resume)

    def test_accumulation_and_explicit_multigpu_reject_before_launch(self):
        resume = self.checkpoint()
        base = {
            "resume": resume,
            "sd3_training_target_mode": "steps",
            "sd3_training_target_steps": 10,
        }
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            self.prepare({**base, "gradient_accumulation_steps": 1})
        with self.assertRaisesRegex(ValueError, "单 GPU"):
            self.prepare({**base, "gradient_accumulation_steps": 2, "gpu_ids": ["0", "1"]})

    def test_ui_custom_params_cannot_reintroduce_legacy_duration(self):
        with self.assertRaisesRegex(ValueError, "不能再通过旧字段"):
            self.prepare({
                "sd3_training_target_mode": "steps",
                "sd3_training_target_steps": 20,
                "ui_custom_params": "max_train_epochs = 3",
            })
        with self.assertRaisesRegex(ValueError, "不能再通过旧字段"):
            self.prepare({
                "sd3_training_target_mode": "epochs",
                "sd3_training_target_epochs": 3,
                "ui_custom_params": "max_train_steps = 20",
            })

    def test_read_only_preview_does_not_touch_checkpoint_files(self):
        resume = self.checkpoint()
        before = {file.name: file.read_bytes() for file in self.folder.iterdir()}
        self.prepare({
            "resume": resume,
            "sd3_training_target_mode": "steps",
            "sd3_training_target_steps": 10,
            "gradient_accumulation_steps": 2,
        })
        after = {file.name: file.read_bytes() for file in self.folder.iterdir()}
        self.assertEqual(before, after)

    def test_rehydrate_legacy_and_gui_effective_config_roundtrip(self):
        for raw, expected_mode, expected_total_key, expected_total in (
            ({"max_train_epochs": 20, "max_train_steps": 300}, "epochs", "max_train_epochs", 20),
            ({"max_train_steps": 800}, "steps", "max_train_steps", 800),
        ):
            with self.subTest(raw=raw):
                gui = rehydrate_trainer_config({
                    **raw, "optimizer_type": "AdamW", "network_train_unet_only": True,
                }, "sd3-lora")
                self.assertEqual(gui["sd3_training_target_mode"], expected_mode)
                self.assertNotIn("max_train_epochs", gui)
                self.assertNotIn("max_train_steps", gui)
                prepared = self.prepare(gui)
                self.assertEqual(prepared.config[expected_total_key], expected_total)
                other = "max_train_steps" if expected_total_key == "max_train_epochs" else "max_train_epochs"
                self.assertNotIn(other, prepared.config)

    def test_resume_rehydrate_preserves_absolute_steps(self):
        resume = self.checkpoint()
        raw = {
            "resume": resume, "optimizer_type": "AdamW",
            "network_train_unet_only": True, "max_train_steps": 8,
            "gradient_accumulation_steps": 2,
        }
        gui = rehydrate_trainer_config(raw, "sd3-lora")
        self.assertEqual(gui["sd3_training_target_mode"], "steps")
        self.assertEqual(gui["sd3_training_target_steps"], 8)
        self.assertEqual(self.prepare(gui).config["max_train_steps"], 8)

    def test_other_backends_do_not_enter_sd3_duration_preflight(self):
        other = prepare_training_config(
            {"max_train_epochs": 20, "optimizer_type": "AdamW"},
            page_train_type="flux-lora", resolve_backend=_resolver
        )
        self.assertEqual(other.config["max_train_epochs"], 20)
        self.assertFalse(any("SD3 Resume" in w for w in other.warnings))


if __name__ == "__main__":
    unittest.main()
