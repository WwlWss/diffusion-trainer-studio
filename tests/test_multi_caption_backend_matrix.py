import ast
from pathlib import Path
import unittest

from mikazuki.multi_caption_config import build_multi_caption_sidecar
from mikazuki.training_config import PAGE_BACKEND_MAP


ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class MultiCaptionBackendMatrixTests(unittest.TestCase):
    def test_all_current_gui_backends_are_covered(self):
        expected = {
            "sd-lora": "./scripts/stable/train_network.py",
            "sdxl-lora": "./scripts/stable/sdxl_train_network.py",
            "sd-dreambooth": "./scripts/stable/train_db.py",
            "sdxl-finetune": "./scripts/stable/sdxl_train.py",
            "sd3-lora": "./scripts/dev/sd3_train_network.py",
            "flux-lora": "./scripts/dev/flux_train_network.py",
            "chroma-lora": "./scripts/dev/flux_train_network.py",
            "flux-finetune": "./scripts/dev/flux_train.py",
            "anima-lora": "./sd-scripts/anima_train_network.py",
            "anima-finetune": "./sd-scripts/anima_train.py",
        }
        mapping = literal_assignment(ROOT / "mikazuki" / "app" / "api.py", "trainer_mapping")
        self.assertEqual(mapping, expected)

        # There is intentionally no Chroma full trainer/page in the repository.
        self.assertNotIn("chroma-finetune", mapping)

    def test_page_backend_map_reaches_every_current_backend(self):
        reached = set(PAGE_BACKEND_MAP.values())
        expected = {
            "sd-lora", "sdxl-lora", "sd-dreambooth", "sdxl-finetune",
            "sd3-lora", "flux-lora", "chroma-lora", "flux-finetune",
            "anima-lora", "anima-finetune",
        }
        self.assertTrue(expected.issubset(reached))

    def test_stable_lora_and_full_entrypoints_have_optional_runtime_hook(self):
        for rel in (
            "scripts/stable/train_network.py",
            "scripts/stable/train_db.py",
            "scripts/stable/sdxl_train.py",
        ):
            with self.subTest(path=rel):
                text = (ROOT / rel).read_text(encoding="utf-8")
                self.assertIn("if args.multi_caption_config:", text)
                self.assertIn("configure_multi_caption_dataset_groups", text)

    def test_dev_lora_and_full_entrypoints_have_optional_runtime_hook(self):
        for rel in (
            "scripts/dev/train_network.py",
            "scripts/dev/flux_train.py",
            "scripts/dev/sd3_train.py",
        ):
            with self.subTest(path=rel):
                text = (ROOT / rel).read_text(encoding="utf-8")
                self.assertIn("if args.multi_caption_config:", text)
                self.assertIn("configure_multi_caption_dataset_groups", text)

    def test_anima_is_runtime_staged_instead_of_changing_submodule_pointer(self):
        runtime = (ROOT / "mikazuki" / "anima_runtime.py").read_text(encoding="utf-8")
        self.assertIn('"anima-lora", "anima-finetune"', runtime)
        self.assertIn("_FEATURE_MULTI", runtime)
        self.assertIn("materialize_anima_runtime_tree", runtime)

    def test_standard_sidecar_compiler_is_noop_for_every_page_type(self):
        pages = (
            "lora-basic", "lora-master", "sdxl-lora", "dreambooth",
            "sdxl-full", "sd3-lora", "flux-lora", "chroma-lora",
            "flux-finetune", "anima-lora", "anima-finetune",
        )
        for page in pages:
            with self.subTest(page=page):
                config = {
                    "caption_mode": "standard",
                    "caption_extension": ".txt",
                    "shuffle_caption": False,
                }
                path, sidecars, policy = build_multi_caption_sidecar(config, page)
                self.assertIsNone(path)
                self.assertEqual(sidecars, {})
                self.assertIsNone(policy)
                self.assertEqual(
                    config,
                    {"caption_extension": ".txt", "shuffle_caption": False},
                )

    def test_dataset_standard_branch_stays_direct(self):
        for rel in (
            "scripts/stable/library/train_util.py",
            "scripts/dev/library/train_util.py",
        ):
            with self.subTest(path=rel):
                text = (ROOT / rel).read_text(encoding="utf-8")
                self.assertIn("if self.multi_caption_resolver is None or image_info.is_reg:", text)
                self.assertIn("caption = self.process_caption(subset, image_info.caption)", text)


if __name__ == "__main__":
    unittest.main()
