"""B3-C-SR1: real imported Late Session fail-closed CPU Torch regression.

This suite exercises the actual production entrypoint and actual CPU model
root (not a source-extracted prefix), while patching scan/optimizer solely
as sentinels to prove identity rejection happens before parameter mutation.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_admission import prepare_direct_trainer_admission
from tests.test_parameter_policy_qg2b3a_snapshot_red import _config, _policy, _write_policy

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch
    from mikazuki import parameter_policy_trainer as trainer


@unittest.skipUnless(_HAS_TORCH, "CPU Torch runtime is mandatory in SR1 CI")
class QG2B3CSR1RealLateCPUTests(unittest.TestCase):
    def _case(self, *, backend="sd-lora", full_bf16=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        policy_path = Path(temp.name) / "policy.json"
        _write_policy(policy_path, _policy(backend))
        values = _config(full_bf16=full_bf16)
        values["parameter_policy_config"] = str(policy_path)
        args = SimpleNamespace(**values)
        intent = DirectTrainerAdmissionIntent(False)
        snapshot = prepare_direct_trainer_admission(
            policy_path,
            train_type=backend,
            effective_config=dict(vars(args)),
            admission_intent=intent,
        )
        root = torch.nn.Linear(4, 4)
        return policy_path, args, intent, snapshot, root

    def _assert_before_scan(self, args, snapshot, intent, backend, root, error):
        with (
            mock.patch.object(trainer, "scan_parameter_roots") as scan,
            mock.patch.object(trainer, "build_parameter_policy_optimizer") as optimizer,
        ):
            with self.assertRaisesRegex(trainer.ParameterPolicyTrainerRuntimeError, error):
                trainer.create_parameter_policy_session(
                    args=args,
                    train_type=backend,
                    roots={"network": root},
                    admission_snapshot=snapshot,
                    admission_intent=intent,
                )
            scan.assert_not_called()
            optimizer.assert_not_called()

    def test_cpu01_real_late_rejects_policy_content_replacement_before_scan(self):
        path, args, intent, snapshot, model = self._case()
        changed = _policy()
        changed["components"]["unet.attention.adapter"]["learning_rate"] = 2e-4
        _write_policy(path, changed)
        self._assert_before_scan(
            args, snapshot, intent, "sd-lora", model, "snapshot identity mismatch",
        )

    def test_cpu02_real_late_rejects_live_effective_config_drift_before_scan(self):
        _path, args, intent, snapshot, model = self._case()
        args.max_train_steps = 1024
        self._assert_before_scan(
            args, snapshot, intent, "sd-lora", model, "snapshot identity mismatch",
        )

    def test_cpu03_real_late_rejects_backend_and_ack_drift_before_scan(self):
        _path, args, intent, snapshot, model = self._case()
        self._assert_before_scan(
            args, snapshot, intent, "sdxl-lora", model, "snapshot identity mismatch",
        )
        self._assert_before_scan(
            args, snapshot, DirectTrainerAdmissionIntent(True), "sd-lora", model,
            "snapshot identity mismatch",
        )

    def test_cpu04_real_late_shared_loader_rejects_ambiguous_json_before_scan(self):
        path, args, intent, snapshot, model = self._case()
        path.write_text(
            '{"version":1,"version":1,"optimizer_profiles":{},"components":{}}',
            encoding="utf-8",
        )
        self._assert_before_scan(
            args, snapshot, intent, "sd-lora", model,
            "Duplicate Parameter Policy JSON key",
        )

    def test_cpu05_real_late_held_full_bf16_rejects_before_scan(self):
        _path, args, intent, snapshot, model = self._case(
            backend="flux-finetune", full_bf16=True,
        )
        self.assertEqual(snapshot.admission.status, "blocked")
        self._assert_before_scan(
            args, snapshot, intent, "flux-finetune", model,
            r"capability\.backend\.audit_required",
        )

    def test_cpu06_real_late_reclassification_drift_rejects_before_scan(self):
        _path, args, intent, snapshot, model = self._case()
        from mikazuki.parameter_policy_admission import assess_component_training_admission
        changed = assess_component_training_admission(
            _policy("flux-finetune"),
            train_type="flux-finetune",
            effective_config=_config(full_bf16=True),
            purpose="direct_trainer",
            experimental_ack=False,
        )
        self.assertEqual(changed.status, "blocked")
        with mock.patch.object(
            trainer,
            "assess_component_training_admission",
            return_value=changed,
        ):
            self._assert_before_scan(
                args, snapshot, intent, "sd-lora", model,
                "result drifted from the frozen snapshot",
            )


if __name__ == "__main__":
    unittest.main()
