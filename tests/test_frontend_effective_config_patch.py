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

    def test_raw_gui_normalization_does_not_mutate_watched_form_state(self):
        self.assertIn('T=()=>{let _=clone(a.value);', self.patched)
        self.assertNotIn('T=()=>{let _=a.value;', self.patched)

    def test_preview_mutable_state_uses_refs_not_const_reassignment(self):
        self.assertIn('__previewTimer=ref(null)', self.patched)
        self.assertIn('__previewGeneration=ref(0)', self.patched)
        self.assertIn('clearTimeout(__previewTimer.value)', self.patched)
        self.assertIn('++__previewGeneration.value', self.patched)
        self.assertIn('__previewTimer.value=setTimeout', self.patched)
        self.assertNotIn('__previewTimer=null', self.patched)
        self.assertNotIn('__previewGeneration=0', self.patched)

    def test_import_effective_toml_uses_backend_rehydrate_and_replaces_state(self):
        self.assertIn('/api/training/rehydrate', self.patched)
        self.assertIn('U.data&&U.data.gui_state', self.patched)
        self.assertIn('a.value=clone(B)', self.patched)
        self.assertNotIn('a.value=Object.assign({},n.value(),B)', self.patched)
        self.assertNotIn('let k=TomlParse(V),U=findChangedDataBySchema(k,n.value)', self.patched)

    def test_preview_ignores_stale_async_responses(self):
        self.assertIn('__previewGeneration=ref(0)', self.patched)
        self.assertIn('const __generation=++__previewGeneration.value', self.patched)
        self.assertGreaterEqual(self.patched.count('if(__generation!==__previewGeneration.value)return'), 2)
        self.assertIn('return D.data||{}', self.patched)

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

    def test_parameter_policy_state_machine_refs_exist(self):
        for anchor in (
            "__runtimeReady=ref(!0)",
            "__runtimeBlockers=ref([])",
            "__policyBootstrapPending=ref(!1)",
            "__policyModeGuard=ref(!1)",
            "__policyBootstrapGeneration=ref(0)",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, self.patched)

    def test_parameter_policy_bootstrap_uses_raw_standard_snapshot(self):
        self.assertIn(
            '__policyBootstrapSnapshot=()=>{let R=clone(a.value||{});delete R.parameter_policy_profiles,delete R.parameter_policy_components,R.optimization_mode="standard";return R}',
            self.patched,
        )
        self.assertIn(
            '__trainingRequest("/api/training/parameter-policy/bootstrap",__policyBootstrapSnapshot())',
            self.patched,
        )
        self.assertNotIn(
            '__trainingRequest("/api/training/parameter-policy/bootstrap",T())',
            self.patched,
        )

    def test_parameter_policy_bootstrap_preserves_current_form_and_existing_policy(self):
        self.assertIn(
            '__hasPolicyState=()=>{let P=a.value&&a.value.parameter_policy_profiles,Cc=a.value&&a.value.parameter_policy_components;',
            self.patched,
        )
        self.assertIn(
            'Object.assign(a.value,{parameter_policy_profiles:clone(B.parameter_policy_profiles),parameter_policy_components:clone(B.parameter_policy_components)})',
            self.patched,
        )
        self.assertNotIn('a.value=clone(B.parameter_policy_profiles)', self.patched)
        self.assertIn(
            '__hasPolicyState()?__refreshPreview():await __bootstrapPolicy()',
            self.patched,
        )

    def test_parameter_policy_bootstrap_cancels_preexisting_preview(self):
        start = self.patched.index('__bootstrapPolicy=async()=>')
        end = self.patched.index(',__syncPolicyMode=async()=>', start)
        block = self.patched[start:end]
        self.assertIn('clearTimeout(__previewTimer.value)', block)
        self.assertIn('++__previewGeneration.value', block)
        pending = block.index('__policyBootstrapPending.value=!0')
        cancel = block.index('clearTimeout(__previewTimer.value)')
        invalidate = block.index('++__previewGeneration.value')
        self.assertLess(cancel, pending)
        self.assertLess(invalidate, pending)

    def test_parameter_policy_bootstrap_rejects_stale_mode_transitions(self):
        self.assertIn('const G=++__policyBootstrapGeneration.value', self.patched)
        self.assertGreaterEqual(
            self.patched.count('G!==__policyBootstrapGeneration.value||!__isComponentMode()'),
            2,
        )
        self.assertIn('++__policyBootstrapGeneration.value,__policyBootstrapPending.value=!1', self.patched)
        self.assertIn(
            'watch(()=>a.value&&a.value.optimization_mode,__syncPolicyMode)',
            self.patched,
        )

    def test_parameter_policy_initial_mount_and_import_use_mode_sync(self):
        self.assertIn(
            'onMounted(async()=>{I(),y(),await nextTick(),await __syncPolicyMode()})',
            self.patched,
        )
        self.assertIn(
            '++__policyBootstrapGeneration.value,a.value=clone(B)',
            self.patched,
        )
        self.assertIn('await nextTick(),await __syncPolicyMode()', self.patched)

    def test_component_preview_invalidates_stale_readiness_before_debounce(self):
        refresh = self.patched.index('__refreshPreview=()=>')
        timer = self.patched.index('__previewTimer.value=setTimeout', refresh)
        invalidate = self.patched.index('if(__isComponentMode())__runtimeReady.value=!1', refresh)
        self.assertLess(invalidate, timer)
        self.assertIn(
            'if(__policyBootstrapPending.value)return',
            self.patched,
        )
        self.assertIn(
            '__runtimeReady.value=R.runtime_ready===!0',
            self.patched,
        )
        self.assertIn(
            '__runtimeBlockers.value=Array.isArray(R.runtime_blockers)?R.runtime_blockers:[]',
            self.patched,
        )
        self.assertIn(
            'd.value=__runtimeReady.value?[]:(__runtimeBlockers.value.length?__runtimeBlockers.value:["Component runtime is not ready."])',
            self.patched,
        )

    def test_component_start_is_guarded_by_bootstrap_and_runtime_readiness(self):
        guard = '__startPending.value||__policyBootstrapPending.value||__isComponentMode()&&!__runtimeReady.value'
        self.assertGreaterEqual(self.patched.count(guard), 2)
        self.assertIn(
            'disabled:__startPending.value||__policyBootstrapPending.value||__isComponentMode()&&!__runtimeReady.value',
            self.patched,
        )

    def test_switching_standard_does_not_delete_component_policy_state(self):
        sync_start = self.patched.index('__syncPolicyMode=async()=>')
        sync_end = self.patched.index(',__refreshPreview=()=>', sync_start)
        sync = self.patched[sync_start:sync_end]
        self.assertNotIn('delete a.value.parameter_policy_profiles', sync)
        self.assertNotIn('delete a.value.parameter_policy_components', sync)
        self.assertIn('__runtimeReady.value=!0,__runtimeBlockers.value=[]', sync)

    def test_patch_fails_closed_when_pinned_bundle_changes(self):
        with self.assertRaises(RuntimeError):
            patch_training_layout_js("not the pinned training layout")


    def test_export_preserves_plain_toml_and_bundles_sidecars(self):
        self.assertIn('.dts.json', self.patched)
        self.assertIn('R.sidecars&&R.sidecars.length>0', self.patched)
        self.assertIn('H?".dts.json":".toml"', self.patched)

    def test_import_accepts_toml_and_portable_bundle(self):
        self.assertIn('_.accept=".toml,.json"', self.patched)
        self.assertIn('JSON.parse(V):TomlParse(V)', self.patched)


if __name__ == "__main__":
    unittest.main()
