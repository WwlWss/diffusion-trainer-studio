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

    def test_generated_frontend_assets_use_media_type_helper(self):
        branding_import = next(
            line
            for line in APPLICATION.splitlines()
            if line.startswith("from mikazuki.frontend_branding import ")
        )
        self.assertIn("frontend_asset_media_type", branding_import)

        frontend_asset_start = APPLICATION.index("async def frontend_asset(asset_name: str):")
        static_asset_start = APPLICATION.index(
            "    asset_path = _safe_frontend_path",
            frontend_asset_start,
        )
        generated_block = APPLICATION[frontend_asset_start:static_asset_start]
        self.assertIn("media_type = frontend_asset_media_type(asset_name)", generated_block)
        self.assertNotIn('media_type="application/javascript"', generated_block)

    def test_application_serves_branded_shell_for_document_routes(self):
        branding_import = next(
            line
            for line in APPLICATION.splitlines()
            if line.startswith("from mikazuki.frontend_branding import ")
        )
        self.assertIn("patch_branding_index_html", branding_import)
        self.assertIn("content = patch_branding_index_html(", APPLICATION)
        self.assertIn("return _frontend_shell_response()", APPLICATION)
        self.assertIn('if path.endswith(".html") or (leaf and "." not in leaf):', APPLICATION)
        self.assertNotIn('return FileResponse(FRONTEND_DIST_DIR / "index.html")', APPLICATION)

    def test_preview_export_rehydrate_and_run_live_in_one_api_module(self):
        for route in (
            '@router.get("/training/parameter-policy/metadata")',
            '@router.post("/training/parameter-policy/bootstrap")',
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

    def test_parameter_policy_editor_api_is_host_only(self):
        self.assertIn("parameter_policy_editor_metadata", TRAINING_API)
        self.assertIn("bootstrap_parameter_policy_editor", TRAINING_API)
        self.assertIn("parameter_policy_editor_preview", TRAINING_API)
        self.assertIn('data={"stage": "parameter-policy-metadata"}', TRAINING_API)
        self.assertIn('data={"stage": "parameter-policy-bootstrap"}', TRAINING_API)
        metadata = TRAINING_API.index('@router.get("/training/parameter-policy/metadata")')
        preview = TRAINING_API.index('@router.post("/training/preview")')
        self.assertLess(metadata, preview)

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


    def test_export_payload_contains_portable_bundle(self):
        self.assertIn('"format": "dts-training-bundle-v1"', TRAINING_API)
        self.assertIn('"bundle": json.dumps(bundle', TRAINING_API)
        self.assertIn("_validated_bundle_sidecars", TRAINING_API)
        self.assertIn("hashlib.sha256(raw_content.encode", TRAINING_API)
        self.assertIn("materialize_sidecars(sidecars)", TRAINING_API)

    def test_preview_and_export_build_payload_inside_error_boundary(self):
        preview = TRAINING_API.index('@router.post("/training/preview")')
        export = TRAINING_API.index('@router.post("/training/export")')
        rehydrate = TRAINING_API.index('@router.post("/training/rehydrate")')

        preview_block = TRAINING_API[preview:export]
        export_block = TRAINING_API[export:rehydrate]

        self.assertIn("payload = _prepared_payload(prepared)", preview_block)
        self.assertLess(
            preview_block.index("payload = _prepared_payload(prepared)"),
            preview_block.index("except (KeyError, TypeError, ValueError, RuntimeError) as exc:"),
        )
        self.assertIn("payload = _prepared_payload(prepared)", export_block)
        self.assertIn("materialize_sidecars(prepared.sidecars)", export_block)
        self.assertLess(
            export_block.index("payload = _prepared_payload(prepared)"),
            export_block.index("materialize_sidecars(prepared.sidecars)"),
        )
        self.assertLess(
            export_block.index("materialize_sidecars(prepared.sidecars)"),
            export_block.index("except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:"),
        )

    def test_parameter_policy_bundle_and_runtime_readiness_are_exposed(self):
        self.assertIn('autosave" / "parameter-policy"', TRAINING_API)
        self.assertIn("def _prepared_parameter_policy_preview(prepared)", TRAINING_API)
        self.assertIn('content = prepared.sidecars.get(str(path))', TRAINING_API)
        self.assertNotIn("Path(str(path)).read_text", TRAINING_API)
        self.assertIn('payload["runtime_ready"] = policy_preview["runtime_ready"]', TRAINING_API)
        self.assertIn('payload["runtime_blockers"] = list(policy_preview["runtime_blockers"])', TRAINING_API)
        self.assertIn('payload["parameter_policy_preview"] = {', TRAINING_API)
        self.assertIn('"profiles": policy_preview["profiles"]', TRAINING_API)
        self.assertIn('"components": policy_preview["components"]', TRAINING_API)
        self.assertIn("RuntimeError, OSError", TRAINING_API)


if __name__ == "__main__":
    unittest.main()
