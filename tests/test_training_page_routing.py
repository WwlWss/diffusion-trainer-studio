import ast
import json
from pathlib import Path
import re
import unittest

from mikazuki.app.training_pages import (
    VIRTUAL_TRAINING_PAGES,
    page_data_js,
    patch_frontend_app_js,
    virtual_asset,
)


ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = (ROOT / "mikazuki/app/api.py").read_text(encoding="utf-8")
APP_BUNDLE = (ROOT / "frontend/dist/assets/app.547295de.js").read_text(encoding="utf-8")
FLUX_FULL_SCHEMA = (ROOT / "mikazuki/schema/flux-finetune.ts").read_text(encoding="utf-8")
ANIMA_TEMPLATE = (ROOT / "mikazuki/schema/flux-lora.ts").read_text(encoding="utf-8")
DREAMBOOTH_TEMPLATE = (ROOT / "mikazuki/schema/dreambooth.ts").read_text(encoding="utf-8")
LORA_TEMPLATE = (ROOT / "mikazuki/schema/lora-master.ts").read_text(encoding="utf-8")


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name} not found")


class TrainingPageRoutingTests(unittest.TestCase):
    def test_one_train_type_maps_to_one_concrete_trainer(self):
        mapping = _literal_assignment(API_SOURCE, "trainer_mapping")
        expected = {
            "sd-lora": "./scripts/stable/train_network.py",
            "sdxl-lora": "./scripts/stable/sdxl_train_network.py",
            "sd-dreambooth": "./scripts/stable/train_db.py",
            "sdxl-finetune": "./scripts/stable/sdxl_train.py",
            "flux-lora": "./scripts/dev/flux_train_network.py",
            "chroma-lora": "./scripts/dev/flux_train_network.py",
            "flux-finetune": "./scripts/dev/flux_train.py",
            "anima-lora": "./sd-scripts/anima_train_network.py",
            "anima-finetune": "./sd-scripts/anima_train.py",
        }
        for train_type, script in expected.items():
            with self.subTest(train_type=train_type):
                self.assertEqual(mapping[train_type], script)

        # There is no Chroma full trainer in this repository. Do not expose a
        # fictitious page merely because Chroma LoRA shares Flux network code.
        self.assertNotIn("chroma-finetune", mapping)

    def test_virtual_page_train_types_match_backend_schema_names(self):
        expected = {
            "/lora/chroma.html": "chroma-lora",
            "/lora/anima.html": "anima-lora",
            "/finetune/sdxl.html": "sdxl-finetune",
            "/finetune/flux.html": "flux-finetune",
            "/finetune/anima.html": "anima-finetune",
        }
        actual = {page.path: page.train_type for page in VIRTUAL_TRAINING_PAGES}
        self.assertEqual(actual, expected)
        self.assertNotIn("chroma-finetune", actual.values())

        for page in VIRTUAL_TRAINING_PAGES:
            generated = page_data_js(page)
            payload_string = re.search(r"JSON\.parse\((.+)\);export", generated).group(1)
            payload = json.loads(json.loads(payload_string))
            self.assertEqual(payload["frontmatter"]["trainType"], page.train_type)
            self.assertEqual(payload["path"], page.path)

    def test_frontend_bundle_patch_creates_separate_sidebar_groups_and_routes(self):
        patched = patch_frontend_app_js(APP_BUNDLE)

        for fragment in (
            "SD1.5 / SD2 LoRA",
            "SDXL LoRA",
            "Flux LoRA",
            "Chroma LoRA",
            "Anima LoRA",
            "\\u5168\\u53C2\\u5FAE\\u8C03",
            "/finetune/sdxl.md",
            "/finetune/flux.md",
            "/finetune/anima.md",
            "/lora/chroma.md",
            "/lora/anima.md",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, patched)

        self.assertNotIn("Anima / Flux / Chroma", patched)
        self.assertNotIn("chroma-finetune", patched)

        for page in VIRTUAL_TRAINING_PAGES:
            self.assertIn(page.key, patched)
            self.assertIn(page.path, patched)
            self.assertIn(page.data_asset, patched)
            self.assertIn(page.content_asset, patched)
            self.assertIsNotNone(virtual_asset(page.data_asset))
            self.assertIsNotNone(virtual_asset(page.content_asset))

    def test_full_finetune_schema_has_no_lora_network_hyperparameters(self):
        for field in (
            "network_module",
            "network_weights",
            "network_dim",
            "network_alpha",
            "network_dropout",
            "network_args",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, FLUX_FULL_SCHEMA)

        self.assertIn('model_train_type: Schema.string().default("flux-finetune")', FLUX_FULL_SCHEMA)
        self.assertIn('model_type: Schema.string().default("flux")', FLUX_FULL_SCHEMA)

    def test_source_templates_still_have_real_conditional_backend_branches(self):
        # The runtime schema builder freezes these selectors; the source remains
        # one maintainable template with explicit conditional branches.
        self.assertIn('Schema.union(["sd-lora", "sdxl-lora"])', LORA_TEMPLATE)
        self.assertIn('Schema.union(["sd-dreambooth", "sdxl-finetune"])', DREAMBOOTH_TEMPLATE)
        self.assertIn('model_type: Schema.union(["flux", "chroma", "anima"])', ANIMA_TEMPLATE)
        self.assertIn('anima_training_mode: Schema.union(["lora", "finetune"])', ANIMA_TEMPLATE)

    def test_migrated_presets_target_dedicated_pages(self):
        expected = {
            "anima.toml": "anima-lora",
            "anima-2.9b.toml": "anima-lora",
            "anima-finetune.toml": "anima-finetune",
            "anima-2.9b-finetune.toml": "anima-finetune",
            "chroma.toml": "chroma-lora",
        }
        for filename, train_type in expected.items():
            text = (ROOT / "config/presets" / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn(f'train_type = "{train_type}"', text)


if __name__ == "__main__":
    unittest.main()
