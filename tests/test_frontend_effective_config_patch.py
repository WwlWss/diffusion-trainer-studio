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

    def test_right_panel_and_download_use_backend_effective_preview(self):
        self.assertIn('/api/training/preview', self.patched)
        self.assertIn('__effectiveToml', self.patched)
        self.assertIn('E=async()=>', self.patched)
        self.assertIn('await __requestEffective()', self.patched)

    def test_start_is_gated_by_backend_preview_and_carries_outer_train_type(self):
        self.assertIn('__payload={train_type:t,config:_}', self.patched)
        self.assertIn('post("/api/training/preview",JSON.stringify(__payload)', self.patched)
        self.assertIn('post("/api/run",JSON.stringify(__payload)', self.patched)

    def test_stop_uses_page_scoped_task_id_not_first_running_task(self):
        self.assertIn('sessionStorage.setItem(`current-task:${t}`', self.patched)
        self.assertIn('sessionStorage.getItem(`current-task:${t}`)', self.patched)
        self.assertIn('String(k.id)===String(__taskId)', self.patched)
        self.assertNotIn('.filter(k=>k.status=="RUNNING");if(V.length==0)', self.patched)

    def test_preview_failures_become_visible_errors(self):
        self.assertIn('配置解析失败', self.patched)
        self.assertIn('d.value=[_.message||String(_)]', self.patched)

    def test_patch_fails_closed_when_pinned_bundle_changes(self):
        with self.assertRaises(RuntimeError):
            patch_training_layout_js("not the pinned training layout")


if __name__ == "__main__":
    unittest.main()
