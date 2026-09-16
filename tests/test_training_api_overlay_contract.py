from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APPLICATION = (ROOT / "mikazuki/app/application.py").read_text(encoding="utf-8")
OVERLAY = (ROOT / "mikazuki/app/api_overlay.py").read_text(encoding="utf-8")
TRAINING_API = (ROOT / "mikazuki/app/training_api.py").read_text(encoding="utf-8")
LAUNCHER = (ROOT / "mikazuki/training_launcher.py").read_text(encoding="utf-8")


class TrainingApiOverlayContractTests(unittest.TestCase):
    def test_application_keeps_compatibility_overlay_import(self):
        self.assertIn("from mikazuki.app.api_overlay import router as api_router", APPLICATION)
        self.assertNotIn("from mikazuki.app.api import router as api_router", APPLICATION)
        self.assertIn("from mikazuki.app.training_api import router", OVERLAY)

    def test_branding_overlay_is_installed_after_training_api_import(self):
        training_import = OVERLAY.index("from mikazuki.app.training_api import router")
        branding_import = OVERLAY.index("from mikazuki.frontend_branding import install_frontend_branding_patch")
        branding_install = OVERLAY.index("install_frontend_branding_patch()")
        self.assertLess(training_import, branding_import)
        self.assertLess(branding_import, branding_install)
        self.assertIn('@app.get("/branding/logo.webp"', APPLICATION)

    def test_application_serves_branded_shell_for_document_routes(self):
        self.assertIn("from mikazuki.frontend_branding import patch_branding_index_html", APPLICATION)
        self.assertIn("content = patch_branding_index_html(", APPLICATION)
        self.assertIn("return _frontend_shell_response()", APPLICATION)
        self.assertIn('if path.endswith(".html") or (leaf and "." not in leaf):', APPLICATION)
        self.assertNotIn('return FileResponse(FRONTEND_DIST_DIR / "index.html")', APPLICATION)

    def test_preview_export_rehydrate_and_run_live_in_one_api_module(self):
        for route in (
            '@router.post("/training/preview")',
            '@router.post("/training/export")',
            '@router.post("/training/rehydrate")',
            '@router.post("/run")',
        ):
            with self.subTest(route=route):
                self.assertIn(route, TRAINING_API)
        self.assertGreaterEqual(TRAINING_API.count("prepare_request_config("), 3)
        self.assertIn("rehydrate_trainer_config", TRAINING_API)
        self.assertIn("validate_prepared_config", TRAINING_API)
        self.assertIn("materialize_sidecars", TRAINING_API)

    def test_launch_does_not_reenter_process_normalization(self):
        self.assertIn("run_prepared_train", TRAINING_API)
        self.assertNotIn("process.run_train", TRAINING_API)
        self.assertIn('"task_id": task.task_id', LAUNCHER)
        self.assertIn('"page_train_type": page_train_type', LAUNCHER)
        self.assertIn('"run_id": run_id', LAUNCHER)
        self.assertIn('"toml_path": toml_path', LAUNCHER)

    def test_runtime_schema_and_frontend_patches_are_wired(self):
        self.assertIn("legacy_api._fixed_sd_schema = fixed_sd_schema", TRAINING_API)
        self.assertIn("legacy_api._fixed_flux_family_schema = fixed_flux_family_schema", TRAINING_API)
        self.assertIn("legacy_api._append_schema = _append_overridden_schema", TRAINING_API)
        self.assertIn("install_frontend_training_patch()", TRAINING_API)


if __name__ == "__main__":
    unittest.main()
