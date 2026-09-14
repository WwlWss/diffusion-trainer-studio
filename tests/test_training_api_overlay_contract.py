from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APPLICATION = (ROOT / "mikazuki/app/application.py").read_text(encoding="utf-8")
OVERLAY = (ROOT / "mikazuki/app/api_overlay.py").read_text(encoding="utf-8")


class TrainingApiOverlayContractTests(unittest.TestCase):
    def test_application_uses_overlay_router(self):
        self.assertIn("from mikazuki.app.api_overlay import router as api_router", APPLICATION)
        self.assertNotIn("from mikazuki.app.api import router as api_router", APPLICATION)

    def test_overlay_replaces_only_training_submission(self):
        self.assertIn('getattr(route, "path", None) == "/run"', OVERLAY)
        self.assertIn('@router.post("/run")', OVERLAY)
        self.assertIn("validate_dataset_source(", OVERLAY)

    def test_runtime_schema_factory_is_wired(self):
        self.assertIn("legacy_api._fixed_sd_schema = fixed_sd_schema", OVERLAY)
        self.assertIn("legacy_api._fixed_flux_family_schema = fixed_flux_family_schema", OVERLAY)


if __name__ == "__main__":
    unittest.main()
