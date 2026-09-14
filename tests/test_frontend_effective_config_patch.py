from pathlib import Path
import unittest

from mikazuki.frontend_training_patch import patch_training_layout_js

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "frontend" / "dist" / "assets" / "layout.96d49288.js"


@unittest.skipUnless(LAYOUT.is_file(), "frontend submodule is not initialized")
class FrontendEffectiveConfigPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = LAYOUT.read_text(encoding="utf-8")
        cls.patched = patch_training_layout_js(cls.original)

    def test_preview_export_start_all_send_raw_gui_state(self):
        self.assertIn('/api/training/preview', self.patched)
        self.assertIn('/api/training/export', self.patched)
        self.assertIn('__trainingRequest("/api/run",T())', self.patched)
        self.assertIn('__requestEffective(T(),"/api/training/export")', self.patched)
        self.assertNotIn('O=async()=>{const _=parseParams(', self.patched)
        self.assertNotIn('stringify(parseParams(n.value(clone(m.value)),t))', self.patched)

    def test_import_effective_toml_uses_backend_rehydrate(self):
        self.assertIn('/api/training/rehydrate', self.patched)
        self.assertIn('U.data&&U.data.gui_state', self.patched)
        self.assertNotIn('let k=TomlParse(V),U=findChangedDataBySchema(k,n.value)', self.patched)

    def test_preset_and_history_preview_use_effective_backend(self):
        self.assertIn('q=async _=>', self.patched)
        self.assertIn('Z=async(_,m)=>', self.patched)
        self.assertIn('await __requestEffective(n.value(clone(m.value)))', self.patched)

    def test_schema_hot_reload_resets_shared_and_schema_object_caches(self):
        self.assertIn('delete s.schemaObject', self.patched)
        self.assertIn('SHARED_SCHEMAS=void 0,this.loadSharedSchema()', self.patched)

    def test_start_pending_and_stop_backend_task_association(self):
        self.assertIn('__startPending=ref(!1)', self.patched)
        self.assertIn('disabled:__startPending.value', self.patched)
        self.assertIn('k.status=="CREATED"||k.status=="RUNNING"', self.patched)
        self.assertIn('k.page_train_type===t', self.patched)

    def test_patch_fails_closed_when_pinned_bundle_changes(self):
        with self.assertRaises(RuntimeError):
            patch_training_layout_js("not the pinned training layout")


if __name__ == "__main__":
    unittest.main()
