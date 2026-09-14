import subprocess
import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_qwen_config import trainer_supports_qwen_training
from mikazuki.anima_qwen_runtime import materialize_qwen_joint_trainer


class AnimaQwenRuntimeTests(unittest.TestCase):
    def test_materializes_isolated_joint_trainer_without_mutating_submodule(self):
        source = Path("sd-scripts").resolve()
        before = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(before.strip(), "")

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Path(materialize_qwen_joint_trainer(source, temp_dir))
            self.assertTrue(trainer.is_file())
            self.assertNotEqual(trainer.parent.resolve(), source)
            self.assertTrue(trainer_supports_qwen_training(str(trainer)))

            text = trainer.read_text(encoding="utf-8")
            self.assertIn("Qwen3 joint finetuning with blocks_to_swap requires AdaFactor + fused_backward_pass", text)
            self.assertIn('"adafactor"', text)
            self.assertIn("optimizer.step_param", text)

            # Content-addressed cache should be reused rather than patched again.
            second = Path(materialize_qwen_joint_trainer(source, temp_dir))
            self.assertEqual(trainer, second)

        after = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(after.strip(), "")


if __name__ == "__main__":
    unittest.main()
