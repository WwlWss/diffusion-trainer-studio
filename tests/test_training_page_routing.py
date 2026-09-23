import ast
import json
from pathlib import Path
import re
import unittest

from mikazuki.training_pages import (
    VIRTUAL_TRAINING_PAGES,
    page_data_js,
    patch_frontend_app_js,
    virtual_asset,
)


ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = (ROOT / "mikazuki/app/api.py").read_text(encoding="utf-8")
APP_BUNDLE = (ROOT / "frontend/dist/assets/app.547295de.js").read_text(encoding="utf-8")
FLUX_FULL_SCHEMA = (ROOT / "mikazuki/schema/flux-finetune.ts").read_text(encoding="utf-8")
SDXL_FULL_SCHEMA = (ROOT / "mikazuki/schema/sdxl-full.ts").read_text(encoding="utf-8")
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


def _schema_field_names(source: str) -> set[str]:
    """Return actual object keys from the schema source, not words in descriptions/comments."""
    return set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*Schema\.", source, re.MULTILINE))


class TrainingPageRoutingTests(unittest.TestCase):
    def test_one_backend_key_maps_to_one_concrete_trainer(self):
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
        self.assertNotIn("chroma-finetune", mapping)

    def test_virtual_pages_use_dedicated_schema_names(self):
        expected = {
            "/lora/chroma.html": "chroma-lora",
            "/lora/anima.html": "anima-lora",
            "/finetune/sdxl.html": "sdxl-full",
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

        self.assertIn(
            '{"text":"LoRA\\u8BAD\\u7EC3","collapsible":true,"children":[',
            patched,
        )
        self.assertIn(
            '{"text":"\\u5168\\u53C2\\u5FAE\\u8C03","collapsible":true,"children":[',
            patched,
        )
        self.assertIn('{"text":"LoRA \\u6982\\u89C8","link":"/lora/index.md"}', patched)
        self.assertNotIn(
            '{"text":"LoRA\\u8BAD\\u7EC3","link":"/lora/index.md","collapsible":false',
            patched,
        )

        self.assertNotIn("Anima / Flux / Chroma", patched)
        self.assertNotIn("chroma-finetune", patched)
        for page in VIRTUAL_TRAINING_PAGES:
            self.assertIn(page.key, patched)
            self.assertIn(page.path, patched)
            self.assertIn(page.data_asset, patched)
            self.assertIn(page.content_asset, patched)
            self.assertIsNotNone(virtual_asset(page.data_asset))
            self.assertIsNotNone(virtual_asset(page.content_asset))


    def test_frontend_patch_pins_explicit_union_discriminators_before_full_validation(self):
        patched = patch_frontend_app_js(APP_BUNDLE)
        self.assertIn("function __dtsUnionDiscriminator(e,t)", patched)
        self.assertIn(
            "const __dtsPinned=t.schema.list.find(v=>!v.meta.hidden&&__dtsUnionDiscriminator(v,d));",
            patched,
        )
        self.assertIn("if(__dtsPinned)a.value=__dtsPinned;", patched)
        self.assertIn("let f=!a.value,h=0;", patched)
        self.assertNotIn(
            "const l=oo({input(d){a.value=null;let f=!0,h=0;",
            patched,
        )

    def test_union_discriminator_patch_only_pins_explicit_top_level_consts(self):
        patched = patch_frontend_app_js(APP_BUNDLE)
        helper_start = patched.index("function __dtsUnionDiscriminator")
        helper_end = patched.index("function md(e,t){", helper_start)
        helper = patched[helper_start:helper_end]
        self.assertIn('o.type==="intersect"', helper)
        self.assertIn('o.type!=="object"', helper)
        self.assertIn('i&&i.type==="const"', helper)
        self.assertIn("n.length>0", helper)
        self.assertIn("Object.prototype.hasOwnProperty.call(t,o)&&t[o]===a", helper)

    def test_frontend_patch_forwards_union_entry_menu(self):
        patched = patch_frontend_app_js(APP_BUNDLE)
        self.assertIn('menu:G(()=>[pe(d.$slots,"menu")])', patched)

    def test_frontend_patch_separates_profile_name_and_optimizer_type(self):
        patched = patch_frontend_app_js(APP_BUNDLE)
        self.assertIn('"dts-profile-entry"', patched)
        self.assertIn('"Profile \\u540D\\u79F0"', patched)
        self.assertIn('"例如 muon / fallback"', patched)
        self.assertIn("function __dtsOptimizerProfileUnion(e,t)", patched)
        self.assertIn('__dtsOptimizerProfileUnion(e.schema,e.prefix)?', patched)
        self.assertNotIn(
            'String(e.prefix||"").startsWith("parameter_policy_profiles.")?',
            patched,
        )

    def test_frontend_patch_keeps_collapse_control_visible_after_expand(self):
        patched = patch_frontend_app_js(APP_BUNDLE)
        self.assertIn('Je(Ee(c(o)("collapse")),1)', patched)
        self.assertIn('onClick:h=>n.value=!0', patched)
        self.assertNotIn('collapse:"\\u6298\\u53E0\\u5B50\\u9879"', patched)
        self.assertIn('collapse:"\\u6536\\u8D77"', patched)

    def test_frontend_patch_restores_explicit_schema_collapse(self):
        # The pinned Schemastery renderer forces Schema.intersect children to
        # extra.foldable=false, which suppresses child meta.collapse entirely.
        # DTS only lets children that explicitly carry meta.collapse inherit
        # their foldability. Conditional Schema.union branches stay non-foldable
        # so union.vue cannot render empty selector/collapse rows.
        self.assertIn('extra:{foldable:!1}', APP_BUNDLE)
        patched = patch_frontend_app_js(APP_BUNDLE)
        self.assertNotIn('extra:{foldable:!1}', patched)
        self.assertIn('extra:{foldable:h.meta.collapse?void 0:!1}', patched)

    def test_flux_full_schema_has_no_lora_network_hyperparameters(self):
        fields = _schema_field_names(FLUX_FULL_SCHEMA)
        forbidden = {
            "network_module",
            "network_weights",
            "network_dim",
            "network_alpha",
            "network_dropout",
            "network_args",
        }
        self.assertFalse(fields & forbidden, f"Flux full contains LoRA-only fields: {sorted(fields & forbidden)}")
        self.assertIn("model_train_type", fields)
        self.assertIn("model_type", fields)
        self.assertIn('model_train_type: Schema.string().default("flux-finetune")', FLUX_FULL_SCHEMA)
        self.assertIn('model_type: Schema.string().default("flux")', FLUX_FULL_SCHEMA)

    def test_sdxl_full_schema_is_not_dreambooth_or_lora_ui(self):
        fields = _schema_field_names(SDXL_FULL_SCHEMA)
        forbidden = {
            "network_module",
            "network_weights",
            "network_dim",
            "network_alpha",
            "network_dropout",
            "stop_text_encoder_training",
            "clip_skip",
            "weighted_captions",
        }
        self.assertFalse(fields & forbidden, f"SDXL full contains foreign fields: {sorted(fields & forbidden)}")

        self.assertIn('model_train_type: Schema.string().default("sdxl-finetune")', SDXL_FULL_SCHEMA)
        required = {
            "train_text_encoder",
            "learning_rate_te1",
            "learning_rate_te2",
            "block_lr",
            "fused_optimizer_groups",
            "no_half_vae",
            "cache_text_encoder_outputs",
            "dataset_config",
            "in_json",
            "deepspeed",
        }
        missing = required - fields
        self.assertFalse(missing, f"SDXL full is missing trainer-backed fields: {sorted(missing)}")

    def test_source_templates_remain_for_legacy_compatibility(self):
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
