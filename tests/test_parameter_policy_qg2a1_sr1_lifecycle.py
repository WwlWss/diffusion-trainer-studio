"""QG-2-A1-SR1 Fix C: request isolation and combined Host/GPU diagnostics.

These are *additional* real regression cases, not manufactured RED cases.
They intentionally do not authorize GUI/Direct Trainer or perform CUDA work.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mikazuki.parameter_policy_admission import (
    assess_component_training_admission,
)
from mikazuki.parameter_policy_editor import parameter_policy_editor_preview
from mikazuki.parameter_policy_execution import FULL_BF16_BACKEND_QUALIFICATIONS
from mikazuki.training_config import RequestAdmissionIntent
from mikazuki.training_rehydrate import rehydrate_trainer_config
from tests.test_parameter_policy_admission_red import _pending
from tests.test_parameter_policy_qg2a1_request_red import _make_preparation
from tests.test_parameter_policy_request_contract import (
    _component_gui_config, _real_prepare_request,
)


class QG2A1SR1LifecycleAndGPU(unittest.TestCase):
    def test_sr201_explicit_failed_start_followed_by_unacknowledged_start(self):
        reports, writes = [], []

        def inspect(policy, **kwargs):
            result = assess_component_training_admission(policy, **kwargs)
            reports.append((
                kwargs["experimental_ack"], result.status, result.purpose,
            ))
            return result

        prepare = _make_preparation(
            source_admission=inspect,
            materializations=writes,
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {
            "sd-lora": _pending()
        }):
            with self.assertRaisesRegex(ValueError, "QG-2-A2"):
                prepare(
                    {}, "sd-lora", launch=True, materialize=True,
                    admission_intent=RequestAdmissionIntent(True, "webui_start"),
                )
            with self.assertRaisesRegex(ValueError, "experimental_full_bf16_ack"):
                prepare({}, "sd-lora", launch=True, materialize=True)
        self.assertEqual(
            reports,
            [(True, "experimental_host_candidate", "start"),
             (False, "ack_required", "start")],
        )
        self.assertEqual(writes, [])

    def test_sr202_export_and_rehydrate_cannot_preserve_positive_intent(self):
        prepare = _real_prepare_request()
        raw = _component_gui_config(
            "sd-lora", ["unet.attention.adapter"],
            full_bf16=True, mixed_precision="bf16",
        )
        with patch.dict(FULL_BF16_BACKEND_QUALIFICATIONS, {
            "sd-lora": _pending()
        }):
            exported = prepare(
                raw, "sd-lora", launch=False, admission_purpose="export",
                admission_intent=RequestAdmissionIntent(True, "webui_start"),
            )
            self.assertEqual(exported.admission.status, "ack_required")
            self.assertFalse(
                exported.admission_intent.experimental_full_bf16_ack
            )

            # Freeze the control-plane exclusion via a real portable JSON
            # copy of Trainer fields + host-owned sidecars. This minimal
            # Host CI intentionally does not install the optional 'toml'
            # package used by the FastAPI Export payload renderer.
            portable_text = json.dumps({
                "train_type": exported.train_type,
                "effective_config": exported.config,
                "sidecars": exported.sidecars,
            }, ensure_ascii=False, sort_keys=True)
            self.assertNotIn("experimental_full_bf16_ack", portable_text)
            self.assertNotIn("admission_intent", portable_text)
            portable = json.loads(portable_text)

            rehydrated = rehydrate_trainer_config(
                portable["effective_config"], "sd-lora",
                sidecars=portable["sidecars"],
            )
            self.assertNotIn("experimental_full_bf16_ack", rehydrated)
            second = prepare(
                rehydrated, "sd-lora", launch=False,
                admission_purpose="preview",
            )
            self.assertEqual(second.admission.status, "ack_required")
            with self.assertRaisesRegex(ValueError, "experimental_full_bf16_ack"):
                prepare(rehydrated, "sd-lora", launch=True)
        self.assertNotIn("experimental_full_bf16_ack", exported.config)

    def test_sr203_host_admissible_does_not_cancel_multigpu_block(self):
        prepare = _real_prepare_request()
        raw = _component_gui_config(
            "sd-lora", ["unet.attention.adapter"],
            full_bf16=True, mixed_precision="bf16",
            gpu_ids=["0", "1"],
        )
        state = prepare(raw, "sd-lora", launch=False)
        self.assertEqual(state.admission.status, "legacy_approved")
        self.assertTrue(state.admission.host_admissible)
        self.assertTrue(
            any("GPU" in x for x in state.runtime_blockers),
            state.runtime_blockers,
        )
        with self.assertRaisesRegex(ValueError, "GPU"):
            prepare(raw, "sd-lora", launch=True, materialize=True)

    def test_sr204_public_runtime_ready_uses_all_blockers_not_host_only(self):
        prepare = _real_prepare_request()
        state = prepare(
            _component_gui_config(
                "sd-lora", ["unet.attention.adapter"],
                full_bf16=True, mixed_precision="bf16",
                gpu_ids=["0", "1"],
            ),
            "sd-lora", launch=False,
        )
        path = str(state.config["parameter_policy_config"])
        policy = json.loads(state.sidecars[path])
        preview = parameter_policy_editor_preview(
            policy, state.train_type, state.runtime_blockers
        )
        self.assertTrue(state.admission.host_admissible)
        self.assertFalse(preview["runtime_ready"])
        self.assertEqual(preview["runtime_blockers"], state.runtime_blockers)


if __name__ == "__main__":
    unittest.main()
