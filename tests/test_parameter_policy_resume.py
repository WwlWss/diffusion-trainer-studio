"""D1 SD1 stock-LoRA full-BF16 resume cursor and trainer wiring regression."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from mikazuki.parameter_policy_resume import (
    SdLoraResumeCursorError,
    plan_sd_lora_full_bf16_resume,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "stable" / "train_network.py"
BRIDGE = ROOT / "scripts" / "stable" / "library" / "dts_parameter_policy_bridge.py"


def _cursor(**overrides):
    options = {
        "saved_step": 1,
        "saved_epoch": 1,
        "max_train_steps": 2,
        "dataloader_batches": 8,
        "gradient_accumulation_steps": 1,
    }
    options.update(overrides)
    return plan_sd_lora_full_bf16_resume(**options)


class SdLoraFullBf16ResumeCursorTests(unittest.TestCase):
    def test_adamw_one_step_to_two_skips_one_batch(self):
        cursor = _cursor()
        self.assertEqual(cursor.completed_steps, 1)
        self.assertEqual(cursor.epoch_to_start, 0)
        self.assertEqual(cursor.batches_to_skip, 1)
        self.assertEqual(cursor.remaining_steps, 1)
        self.assertEqual(cursor.skipped_batches_for_epoch(0), 1)
        self.assertEqual(cursor.skipped_batches_for_epoch(1), 0)

    def test_muon_adamw_accumulation_two_skips_two_batches(self):
        cursor = _cursor(gradient_accumulation_steps=2)
        self.assertEqual(cursor.completed_steps, 1)
        self.assertEqual(cursor.epoch_to_start, 0)
        self.assertEqual(cursor.batches_to_skip, 2)
        self.assertEqual(cursor.remaining_steps, 1)

    def test_epoch_boundary_starts_next_epoch_with_no_skip(self):
        for accumulation in (1, 2):
            with self.subTest(accumulation=accumulation):
                cursor = _cursor(
                    saved_step=2 if accumulation == 1 else 1,
                    max_train_steps=3 if accumulation == 1 else 2,
                    dataloader_batches=2,
                    gradient_accumulation_steps=accumulation,
                )
                self.assertEqual(cursor.epoch_to_start, 1)
                self.assertEqual(cursor.batches_to_skip, 0)
                self.assertEqual(cursor.skipped_batches_for_epoch(0), 0)
                self.assertEqual(cursor.skipped_batches_for_epoch(1), 0)

    def test_partial_last_accumulation_group_never_skips_past_epoch(self):
        # Five microbatches and accumulation 2 => steps at batch 2, 4, 5.
        cursor = _cursor(
            saved_step=2,
            saved_epoch=1,
            max_train_steps=3,
            dataloader_batches=5,
            gradient_accumulation_steps=2,
        )
        self.assertEqual(cursor.epoch_to_start, 0)
        self.assertEqual(cursor.batches_to_skip, 4)
        self.assertEqual(cursor.remaining_steps, 1)

        end_of_epoch = _cursor(
            saved_step=3,
            saved_epoch=1,
            max_train_steps=4,
            dataloader_batches=5,
            gradient_accumulation_steps=2,
        )
        self.assertEqual(end_of_epoch.epoch_to_start, 1)
        self.assertEqual(end_of_epoch.batches_to_skip, 0)
        self.assertEqual(end_of_epoch.remaining_steps, 1)

    def test_multi_epoch_interior_resume(self):
        cursor = _cursor(
            saved_step=5,
            saved_epoch=2,
            max_train_steps=7,
            dataloader_batches=4,
            gradient_accumulation_steps=1,
        )
        self.assertEqual(cursor.epoch_to_start, 1)
        self.assertEqual(cursor.batches_to_skip, 1)
        self.assertEqual(cursor.skipped_batches_for_epoch(1), 1)
        self.assertEqual(cursor.skipped_batches_for_epoch(2), 0)
        self.assertEqual(cursor.remaining_steps, 2)

    def test_one_remaining_logical_step_for_both_d1_cases(self):
        # Mimic the production loop's use of skipped_batches_for_epoch:
        # data-loader position advances by one/two microbatches, whereas
        # global_step remains the *completed* optimizer step after load_state.
        for accumulation in (1, 2):
            with self.subTest(accumulation=accumulation):
                cursor = _cursor(gradient_accumulation_steps=accumulation)
                epoch_batches = list(range(8))
                start = cursor.skipped_batches_for_epoch(cursor.epoch_to_start)
                resumed_batches = epoch_batches[start:]
                global_step = cursor.completed_steps
                consumed = []
                for batch in resumed_batches:
                    consumed.append(batch)
                    if len(consumed) % accumulation == 0:
                        global_step += 1
                        if global_step >= 2:
                            break
                self.assertEqual(global_step, 2)
                self.assertEqual(
                    consumed,
                    list(range(accumulation, accumulation * 2)),
                )

    def test_requires_positive_integer_checkpoint_step_and_epoch(self):
        for field in ("saved_step", "saved_epoch"):
            for invalid in (None, -1, 0, False, True, 1.5, "1"):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(
                        SdLoraResumeCursorError, "positive integer"
                    ):
                        _cursor(**{field: invalid})

    def test_rejects_invalid_training_lengths(self):
        for field in (
            "max_train_steps",
            "dataloader_batches",
            "gradient_accumulation_steps",
        ):
            for invalid in (None, 0, -1, True, 1.5, "2"):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(
                        SdLoraResumeCursorError, "positive integer"
                    ):
                        _cursor(**{field: invalid})

    def test_rejects_no_remaining_steps(self):
        for saved_step in (2, 3):
            with self.subTest(saved_step=saved_step):
                with self.assertRaisesRegex(
                    SdLoraResumeCursorError, "no remaining optimizer steps"
                ):
                    _cursor(saved_step=saved_step)

    def test_rejects_epoch_step_inconsistency(self):
        with self.assertRaisesRegex(
            SdLoraResumeCursorError, "epoch/optimizer-step mismatch"
        ):
            _cursor(saved_step=1, saved_epoch=2)

    def test_rejects_competing_lifecycle_authorities(self):
        overrides = (
            {"initial_step": 1},
            {"initial_epoch": 1},
            {"max_train_epochs": 1},
            {"skip_until_initial_step": True},
        )
        for extra in overrides:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(
                    SdLoraResumeCursorError, "explicit lifecycle override"
                ):
                    _cursor(**extra)


class SdLoraFullBf16ProductionWiringTests(unittest.TestCase):
    def test_production_trainer_is_syntactically_valid_and_uses_cursor(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NetworkTrainer"
        )
        train_fn = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        calls = {
            node.func.attr
            for node in ast.walk(train_fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("plan_sd_lora_full_bf16_resume", calls)
        self.assertIn("skipped_batches_for_epoch", calls)

        self.assertIn('parameter_policy_train_type == "sd-lora"', source)
        self.assertIn('args.network_module == "networks.lora"', source)
        self.assertIn("and args.full_bf16", source)
        self.assertIn("and not args.v2", source)
        self.assertIn("and bool(args.resume)", source)
        self.assertIn(
            "global_step = d1_resume_cursor.completed_steps", source
        )
        self.assertIn("epoch_to_start = d1_resume_cursor.epoch_to_start", source)
        self.assertIn(
            "accelerator.skip_first_batches(\n"
            "                    train_dataloader, d1_resume_skip",
            source,
        )
        # No changes to legacy skip branch or ordinary fresh training path.
        self.assertIn("initial_step = 0  # do not skip", source)
        self.assertIn("elif initial_step > 0:", source)

    def test_bridge_is_lazily_resolved(self):
        source = BRIDGE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("plan_sd_lora_full_bf16_resume", function_names)
        self.assertNotIn(
            "from mikazuki.parameter_policy_resume import",
            source,
        )


if __name__ == "__main__":
    unittest.main()
