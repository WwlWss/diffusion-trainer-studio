from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from mikazuki.frontend_training_patch import patch_training_layout_js


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "frontend" / "dist" / "assets" / "layout.96d49288.js"


@unittest.skipUnless(LAYOUT.is_file(), "frontend submodule is not initialized")
@unittest.skipUnless(shutil.which("node"), "node is required for frontend behavior smoke")
class ParameterPolicyFrontendBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = LAYOUT.read_text(encoding="utf-8")
        cls.patched = patch_training_layout_js(original)
        start = cls.patched.index("__profileNames=")
        end = cls.patched.index(",__presetPolicyMode=", start)
        cls.profile_helpers = cls.patched[start:end]

    def _run_node(self, body: str) -> None:
        script = f"""
const warnings = [];
const ElMessage = {{ warning: (message) => warnings.push(String(message)) }};
const window = globalThis;
const a = {{ value: {{
  parameter_policy_profiles: {{
    muon: {{type: "Muon", args: {{}}}},
    adamw: {{type: "AdamW", args: {{}}}},
  }},
  parameter_policy_components: {{
    self: {{
      train: true,
      optimizer_profile: "muon",
      fallback_optimizer_profile: "adamw",
      fallback_learning_rate: "2.5e-5",
    }},
    base: {{
      train: true,
      optimizer_profile: "adamw",
    }},
  }},
}} }};
const {self.profile_helpers};
{body}
"""
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "node behavior smoke failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    def test_profile_rename_delete_and_empty_name_behavior(self):
        self._run_node(
            """
if (__renamePolicyProfile("muon", "") !== false) throw new Error("empty rename accepted");
if (a.value.parameter_policy_components.self.optimizer_profile !== "muon") {
  throw new Error("empty rename mutated primary reference");
}
if (__renamePolicyProfile("muon", "ADAMW") !== false) throw new Error("duplicate rename accepted");
if (__renamePolicyProfile("muon", "muon_main") !== true) throw new Error("valid rename rejected");
if (a.value.parameter_policy_components.self.optimizer_profile !== "muon_main") {
  throw new Error("primary reference was not retargeted");
}
if (__canDeletePolicyProfile("adamw") !== false) throw new Error("referenced delete allowed");
delete a.value.parameter_policy_components.self.fallback_optimizer_profile;
delete a.value.parameter_policy_components.self.fallback_learning_rate;
a.value.parameter_policy_components.base.optimizer_profile = "muon_main";
if (__canDeletePolicyProfile("adamw") !== true) throw new Error("unreferenced delete blocked");
if (!warnings.some((value) => value.includes("不能为空"))) throw new Error("missing empty-name warning");
"""
        )

    def test_optimizer_type_switch_reconciles_fallback_semantics(self):
        self._run_node(
            """
__onPolicyOptimizerTypeChanged(
  "parameter_policy_profiles.muon.",
  "Muon",
  "AdamW"
);
if ("fallback_optimizer_profile" in a.value.parameter_policy_components.self) {
  throw new Error("Muon -> AdamW kept invalid fallback profile");
}
if ("fallback_learning_rate" in a.value.parameter_policy_components.self) {
  throw new Error("Muon -> AdamW kept invalid fallback LR");
}

a.value.parameter_policy_components.base.fallback_optimizer_profile = "adamw";
a.value.parameter_policy_components.base.fallback_learning_rate = "1e-5";
__onPolicyOptimizerTypeChanged(
  "parameter_policy_profiles.adamw.",
  "AdamW",
  "Muon"
);
if ("fallback_optimizer_profile" in a.value.parameter_policy_components.base) {
  throw new Error("new eligibility-constrained fallback reference survived");
}
"""
        )

    def test_profile_names_hook_exposes_live_keys(self):
        self._run_node(
            """
const before = window.__dtsPolicyProfileHooks.profileNames().sort();
if (JSON.stringify(before) !== JSON.stringify(["adamw", "muon"])) {
  throw new Error("initial profile names mismatch: " + JSON.stringify(before));
}
a.value.parameter_policy_profiles.custom = {type: "AdamW", args: {}};
const after = window.__dtsPolicyProfileHooks.profileNames().sort();
if (JSON.stringify(after) !== JSON.stringify(["adamw", "custom", "muon"])) {
  throw new Error("live profile names did not update: " + JSON.stringify(after));
}
"""
        )


if __name__ == "__main__":
    unittest.main()
