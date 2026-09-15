from pathlib import Path
import unittest

from mikazuki import frontend_branding, frontend_training_patch, training_pages


ASSETS = Path("frontend/dist/assets")


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
            'href:"https://github.com/WwlWss/lora-scripts",target:"_blank","aria-label":"GitHub"',
            branded,
        )
        self.assertNotIn(
            'href:"https://github.com/Akegarasu/lora-scripts",target:"_blank","aria-label":"GitHub"',
            branded,
        )

    def test_home_content_uses_new_project_identity_and_upstream_credits(self):
        content = frontend_branding.home_content_js()
        self.assertIn("Diffusion Trainer Studio", content)
        self.assertIn("v2.0.0", content)
        self.assertIn("/branding/logo.webp", content)
        self.assertIn("https://github.com/WwlWss/lora-scripts", content)
        self.assertIn("https://github.com/Akegarasu/lora-scripts", content)
        self.assertIn("https://github.com/kohya-ss/sd-scripts", content)

    def test_about_replaces_upstream_contact_details_with_project_links(self):
        content = frontend_branding.about_content_js()
        self.assertIn("https://github.com/WwlWss/lora-scripts/issues", content)
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
