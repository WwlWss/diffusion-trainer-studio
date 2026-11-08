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
        request_start = cls.patched.index("const __trainingRequest=")
        request_end = cls.patched.index(",__requestEffective=", request_start)
        cls.training_request_helper = cls.patched[request_start:request_end]
        start_handler = cls.patched.index("O=async()=>{")
        stop_handler = cls.patched.index(",K=async()=>", start_handler)
        cls.start_handler = cls.patched[start_handler:stop_handler]

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

    def test_patched_layout_is_valid_javascript(self):
        completed = subprocess.run(
            ["node", "--check", "--input-type=module"],
            cwd=ROOT,
            input=self.patched,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"patched frontend syntax failed: {completed.stderr}",
        )

    def test_request_envelope_carries_ack_only_for_explicit_start(self):
        self._run_node(
            f"""
const sent = [];
const post = async (url, body) => {{
  sent.push({{url, body: JSON.parse(body)}});
  return {{ok: true, json: async () => ({{status: "success", data: {{}}}})}};
}};
const t = "sd-lora";
{self.training_request_helper};
const raw = {{full_bf16: true}};
await __trainingRequest("/api/training/preview", raw);
await __trainingRequest("/api/training/export", raw, true);
await __trainingRequest("/api/run", raw, true);
await __trainingRequest("/api/run", raw, false);
if ("experimental_full_bf16_ack" in sent[0].body) throw new Error("Preview leaked ack");
if ("experimental_full_bf16_ack" in sent[1].body) throw new Error("Export leaked ack");
if (sent[2].body.experimental_full_bf16_ack !== true) throw new Error("Start lost explicit ack");
if ("experimental_full_bf16_ack" in sent[3].body) throw new Error("false ack should remain absent");
if ("experimental_full_bf16_ack" in raw) throw new Error("request helper mutated Trainer config");
"""
        )

    def test_ack_checkbox_is_visible_transient_and_start_consumes_once(self):
        self.assertIn("我确认本次 Start 使用实验性 Full-BF16 路径", self.patched)
        self.assertIn("__experimentalFullBf16Ack=ref(!1)", self.patched)
        self.assertIn("__experimentalFullBf16Ack.value=!1", self.patched)
        self.assertIn("__admission.value.requires_experimental_ack===!0", self.patched)
        self._run_node(
            f"""
const sent = [];
const post = async (url, body) => {{
  sent.push({{url, body: JSON.parse(body)}});
  return {{ok: true, json: async () => ({{status: "success", data: {{task_id: "one"}, message: "held"}})}};
}};
const t = "sd-lora";
const __trainingRequest = async (endpoint, raw, experimentalAck) => {{
  let body = {{train_type: t, config: raw}};
  endpoint === "/api/run" && experimentalAck === true &&
    (body.experimental_full_bf16_ack = true);
  const response = await post(endpoint, JSON.stringify(body));
  const result = await response.json();
  if (!response.ok || result.status !== "success") throw new Error(result.message);
  return result;
}};
let __startPending = {{value: false}};
let __policyBootstrapPending = {{value: false}};
let __runtimeReady = {{value: false}};
let __experimentalFullBf16Ack = {{value: true}};
let __admission = {{value: {{requires_experimental_ack: true}}}};
let __isComponentMode = () => true;
let T = () => ({{full_bf16: true}});
let sessionStorage = {{setItem: () => {{}}}};
let ElMessage = {{success: () => {{}}}};
let v = () => "networkError";
let O;
{self.start_handler};
await O();
if (sent.length !== 1 || sent[0].body.experimental_full_bf16_ack !== true) {{
  throw new Error("single Start did not carry the explicit acknowledgement");
}}
if (__experimentalFullBf16Ack.value !== false) throw new Error("ack was not consumed/reset");
await O();
if (sent.length !== 1) throw new Error("ack replayed into a later Start");
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
