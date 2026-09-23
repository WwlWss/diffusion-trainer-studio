from pathlib import Path
import unittest

from mikazuki import frontend_branding, frontend_training_patch, training_pages


ASSETS = Path("frontend/dist/assets")
INDEX = Path("frontend/dist/index.html")


class FrontendBrandingTests(unittest.TestCase):
    def test_app_branding_composes_after_training_page_patch(self):
        source = (ASSETS / frontend_branding.APP_ASSET).read_text(encoding="utf-8")
        training_patched = training_pages.patch_frontend_app_js(source)
        branded = frontend_branding.patch_branding_app_js(training_patched)

        self.assertIn('{"text":"Diffusion Trainer Studio","link":"/"}', branded)
        self.assertIn('["v-8daa1a0e","/",{title:"Diffusion Trainer Studio"},', branded)
        self.assertNotIn('{"text":"SD-Trainer","link":"/"}', branded)
        self.assertIn('"text":"Anima LoRA"', branded)
        self.assertIn('"text":"Anima Finetune"', branded)

    def test_layout_branding_composes_after_effective_config_patch(self):
        source = (ASSETS / frontend_branding.LAYOUT_ASSET).read_text(encoding="utf-8")
        training_patched = frontend_training_patch.patch_training_layout_js(source)
        branded = frontend_branding.patch_branding_layout_js(training_patched)

        self.assertIn(
            'href:"https://github.com/WwlWss/diffusion-trainer-studio",target:"_blank","aria-label":"GitHub"',
            branded,
        )
        self.assertNotIn(
            'href:"https://github.com/Akegarasu/lora-scripts",target:"_blank","aria-label":"GitHub"',
            branded,
        )
        self.assertIn('/api/training/preview', branded)
        self.assertIn('/api/training/export', branded)
        self.assertIn('/api/training/rehydrate', branded)

    def test_profile_editor_css_uses_two_distinct_rows(self):
        source = (ASSETS / frontend_branding.STYLE_ASSET).read_text(encoding="utf-8")
        patched = frontend_branding.patch_profile_editor_css(source)
        self.assertIn("/* DTS profile editor two-row layout */", patched)
        self.assertIn("grid-template-rows:auto auto", patched)
        self.assertIn("grid-row:1", patched)
        self.assertIn("grid-row:2", patched)
        self.assertIn(".dts-profile-entry .dts-profile-type-label", patched)

    def test_pre_rendered_shell_is_branded_before_hydration(self):
        source = INDEX.read_text(encoding="utf-8")
        branded = frontend_branding.patch_branding_index_html(source)

        self.assertIn(
            "<title>Diffusion Trainer Studio | 多架构 Diffusion 模型训练工作台</title>",
            branded,
        )
        self.assertIn('href="/branding/logo.webp"', branded)
        self.assertIn('aria-label="Diffusion Trainer Studio"', branded)
        self.assertIn('href="https://github.com/WwlWss/diffusion-trainer-studio"', branded)
        self.assertIn(frontend_branding.home_html(), branded)
        self.assertNotIn("<title>SD-Trainer | SD 训练 UI</title>", branded)
        self.assertNotIn('aria-label="SD-Trainer"', branded)
        self.assertNotIn('<h1 id="sd-trainer"', branded)
        self.assertNotIn("Stable Diffusion 训练 UI v1.13.0", branded)

    def test_shell_patch_fails_closed_when_pinned_html_changes(self):
        with self.assertRaises(RuntimeError):
            frontend_branding.patch_branding_index_html("not the pinned VuePress shell")

    def test_runtime_wrapper_chain_is_ordered_idempotent_and_preserves_virtual_pages(self):
        original = training_pages.virtual_asset
        try:
            while hasattr(training_pages.virtual_asset, "__wrapped__"):
                training_pages.virtual_asset = training_pages.virtual_asset.__wrapped__
            base_wrapper = training_pages.virtual_asset

            frontend_training_patch.install_frontend_training_patch()
            effective_wrapper = training_pages.virtual_asset
            self.assertTrue(getattr(effective_wrapper, "_mikazuki_effective_config_patch", False))
            self.assertIs(getattr(effective_wrapper, "__wrapped__", None), base_wrapper)

            frontend_branding.install_frontend_branding_patch()
            branded_wrapper = training_pages.virtual_asset
            self.assertTrue(getattr(branded_wrapper, "_mikazuki_effective_config_patch", False))
            self.assertTrue(getattr(branded_wrapper, "_mikazuki_branding_patch", False))
            self.assertIs(getattr(branded_wrapper, "__wrapped__", None), effective_wrapper)

            # Installing either layer again must not stack another wrapper.
            frontend_training_patch.install_frontend_training_patch()
            frontend_branding.install_frontend_branding_patch()
            self.assertIs(training_pages.virtual_asset, branded_wrapper)

            app = training_pages.virtual_asset(frontend_branding.APP_ASSET)
            self.assertIn('{"text":"Diffusion Trainer Studio","link":"/"}', app)
            self.assertIn('"text":"Anima LoRA"', app)
            self.assertIn('"text":"Anima Finetune"', app)

            style = training_pages.virtual_asset(frontend_branding.STYLE_ASSET)
            self.assertIn("DTS profile editor two-row layout", style)
            self.assertIn("grid-template-rows:auto auto", style)

            layout = training_pages.virtual_asset(frontend_branding.LAYOUT_ASSET)
            self.assertIn('/api/training/preview', layout)
            self.assertIn('/api/training/export', layout)
            self.assertIn('/api/training/rehydrate', layout)
            self.assertIn(
                'href:"https://github.com/WwlWss/diffusion-trainer-studio",target:"_blank","aria-label":"GitHub"',
                layout,
            )

            anima_page = next(page for page in training_pages.VIRTUAL_TRAINING_PAGES if page.train_type == "anima-lora")
            self.assertEqual(
                training_pages.virtual_asset(anima_page.content_asset),
                training_pages.page_content_js(anima_page),
            )
            self.assertEqual(
                training_pages.virtual_asset(anima_page.data_asset),
                training_pages.page_data_js(anima_page),
            )
        finally:
            training_pages.virtual_asset = original

    def test_branding_refuses_wrong_installation_order(self):
        original = training_pages.virtual_asset
        try:
            while hasattr(training_pages.virtual_asset, "__wrapped__"):
                training_pages.virtual_asset = training_pages.virtual_asset.__wrapped__
            self.assertFalse(
                getattr(training_pages.virtual_asset, "_mikazuki_effective_config_patch", False)
            )
            with self.assertRaisesRegex(RuntimeError, "after the effective-config frontend patch"):
                frontend_branding.install_frontend_branding_patch()
        finally:
            training_pages.virtual_asset = original

    def test_home_content_uses_new_project_identity_and_upstream_credits(self):
        content = frontend_branding.home_content_js()
        self.assertIn("Diffusion Trainer Studio", content)
        self.assertIn("v2.0.0", content)
        self.assertIn("/branding/logo.webp", content)
        self.assertIn("https://github.com/WwlWss/diffusion-trainer-studio", content)
        self.assertIn("https://github.com/Akegarasu/lora-scripts", content)
        self.assertIn("https://github.com/kohya-ss/sd-scripts", content)

    def test_about_replaces_upstream_contact_details_with_project_links(self):
        content = frontend_branding.about_content_js()
        self.assertIn("https://github.com/WwlWss/diffusion-trainer-studio/issues", content)
        self.assertIn("https://github.com/hanamizuki-ai/lora-gui-dist", content)
        self.assertIn("https://github.com/shigma/schemastery", content)
        self.assertNotIn("work@anzu.link", content)
        self.assertNotIn("discord.gg/Uu3syD9PnR", content)

    def test_page_data_uses_new_titles(self):
        self.assertIn('Diffusion Trainer Studio', frontend_branding.home_data_js())
        self.assertIn('Diffusion Trainer Studio', frontend_branding.about_data_js())

    def test_logo_asset_is_present(self):
        self.assertTrue(Path("assets/dts-logo.webp").is_file())


if __name__ == "__main__":
    unittest.main()
