"""QG-2-B3-C dynamic torch-free checks of actual Late gate source.

Execute the exact checked-in late entry prefix, not a reimplementation;
do not import torch-heavy parameter_policy_trainer or scan any real roots.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy import load_parameter_policy_sidecar
from mikazuki.parameter_policy_readiness import evaluate_execution_readiness
from mikazuki.parameter_policy_admission import (
    ComponentAdmission,
    DirectTrainerAdmissionSnapshot,
    assess_component_training_admission,
    fingerprint_effective_training_config,
    prepare_direct_trainer_admission,
)
from tests.test_parameter_policy_qg2b3a_snapshot_red import _config, _policy, _write_policy

ROOT = Path(__file__).resolve().parents[1]
LATE = ROOT / "mikazuki" / "parameter_policy_trainer.py"


class TrainerRuntimeError(RuntimeError):
    pass


def _late_loader():
    raw = LATE.read_text(encoding="utf-8")
    start = raw.index("def load_parameter_policy_file(")
    end = raw.index("\ndef load_parameter_policy_train_flags(", start)
    snippet = raw[start:end]
    context = dict(
        os=os, Any=Any, load_parameter_policy_sidecar=load_parameter_policy_sidecar,
        ParameterPolicyTrainerRuntimeError=TrainerRuntimeError,
    )
    exec(compile("from __future__ import annotations\n" + snippet, str(LATE), "exec"), context)
    return context["load_parameter_policy_file"]


def _late_prefix(*, args, snapshot, intent, backend, classifier=None):
    raw = LATE.read_text(encoding="utf-8")
    start = raw.index("    # Late gate requires the exact out-of-band invocation identity.", raw.index("def create_parameter_policy_session("))
    stop = raw.index("    semantic_blockers = parameter_policy_v1_semantic_blockers(", start)
    body = textwrap.dedent(raw[start:stop])
    program = "def execute():\n" + textwrap.indent(body, "    ")
    config = dict(
        args=args, admission_snapshot=snapshot, admission_intent=intent,
        train_type=backend,
        _effective_config=lambda a: dict(a),
        _policy_path=lambda a: a["parameter_policy_config"],
        load_parameter_policy_file=_late_loader(),
        ParameterPolicyTrainerRuntimeError=TrainerRuntimeError,
        DirectTrainerAdmissionSnapshot=DirectTrainerAdmissionSnapshot,
        DirectTrainerAdmissionIntent=DirectTrainerAdmissionIntent,
        fingerprint_effective_training_config=fingerprint_effective_training_config,
        assess_component_training_admission=classifier or assess_component_training_admission,
    )
    exec(compile(program, str(LATE), "exec"), config)
    return config["execute"]


def _snapshot(policy_path, args, intent, *, backend="sd-lora"):
    return prepare_direct_trainer_admission(
        policy_path, train_type=backend,
        effective_config=args, admission_intent=intent,
    )


class QG2B3CLateDynamicHost(unittest.TestCase):
    def test_cdyn01_actual_shared_late_loader_rejects_nonstandard_json(self):
        loader = _late_loader()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            early, digest = load_parameter_policy_sidecar(path)
            late, late_digest = loader(path)
            self.assertEqual((late, late_digest), (early, digest))
            for raw in (
                '{"version":1,"version":1,"optimizer_profiles":{},"components":{}}',
                '{"version":1,"optimizer_profiles":{"m":{"type":"AdamW","args":{"x":NaN}}},"components":{}}',
                '{"version":1,"optimizer_profiles":{"m":{"type":"AdamW","args":{"x":Infinity}}},"components":{}}',
            ):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(TrainerRuntimeError):
                    loader(path)

    def test_cdyn02_actual_late_gate_accepts_unchanged_inactive_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = _snapshot(path, args, intent)
            self.assertEqual(snapshot.admission.status, "inactive")
            _late_prefix(args=args, snapshot=snapshot, intent=intent, backend="sd-lora")()

    def test_cdyn03_rejects_sidecar_replacement_before_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = _snapshot(path, args, intent)
            edited = _policy()
            edited["components"]["unet.attention.adapter"]["learning_rate"] = 2e-4
            _write_policy(path, edited)
            with self.assertRaisesRegex(TrainerRuntimeError, "identity mismatch"):
                _late_prefix(args=args, snapshot=snapshot, intent=intent, backend="sd-lora")()

    def test_cdyn04_rejects_live_config_change_before_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = _snapshot(path, args, intent)
            args["scale_weight_norms"] = 0.5
            with self.assertRaisesRegex(TrainerRuntimeError, "identity mismatch"):
                _late_prefix(args=args, snapshot=snapshot, intent=intent, backend="sd-lora")()

    def test_cdyn05_rejects_backend_or_ack_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = _snapshot(path, args, intent)
            with self.assertRaisesRegex(TrainerRuntimeError, "identity mismatch"):
                _late_prefix(args=args, snapshot=snapshot, intent=intent, backend="sdxl-lora")()
            with self.assertRaisesRegex(TrainerRuntimeError, "identity mismatch"):
                _late_prefix(args=args, snapshot=snapshot,
                             intent=DirectTrainerAdmissionIntent(True), backend="sd-lora")()

    def test_cdyn06_rejects_untyped_snapshot_and_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = _snapshot(path, args, intent)
            with self.assertRaisesRegex(TrainerRuntimeError, "first-party frozen snapshot"):
                _late_prefix(args=args, snapshot={"policy_hash": snapshot.policy_hash},
                             intent=intent, backend="sd-lora")()
            with self.assertRaisesRegex(TrainerRuntimeError, "typed one-shot"):
                _late_prefix(args=args, snapshot=snapshot, intent=True, backend="sd-lora")()

    def test_cdyn07_first_party_reclassification_rejects_all_held_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            original = _snapshot(path, args, intent)
            for status in ("blocked", "ack_required", "experimental_host_candidate"):
                admission = ComponentAdmission(
                    status=status, purpose="direct_trainer",
                    readiness=evaluate_execution_readiness(capability="implemented"),
                    runtime_blockers=() if status == "experimental_host_candidate" else ("held",),
                    warnings=("advisory",) if status != "blocked" else (),
                    requires_experimental_ack=status == "ack_required",
                    host_admissible=status == "experimental_host_candidate",
                )
                snapshot = DirectTrainerAdmissionSnapshot(
                    policy_hash=original.policy_hash,
                    effective_config_fingerprint=original.effective_config_fingerprint,
                    train_type="sd-lora", experimental_ack=False,
                    admission=admission,
                )
                def classifier(*a, **kw):
                    self.assertEqual(kw["purpose"], "direct_trainer")
                    self.assertIs(kw["experimental_ack"], False)
                    return admission
                with self.subTest(status=status):
                    with self.assertRaisesRegex(TrainerRuntimeError, "remains held"):
                        _late_prefix(args=args, snapshot=snapshot, intent=intent,
                                     backend="sd-lora", classifier=classifier)()

    def test_cdyn08_reclassification_drift_is_rejected_before_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            args = {**_config(full_bf16=False), "parameter_policy_config": str(path)}
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = _snapshot(path, args, intent)
            different = ComponentAdmission(
                status="blocked", purpose="direct_trainer",
                readiness=evaluate_execution_readiness(capability="implemented"),
                runtime_blockers=("changed readiness",), warnings=(),
                requires_experimental_ack=False, host_admissible=False,
            )
            with self.assertRaisesRegex(TrainerRuntimeError, "result drifted"):
                _late_prefix(args=args, snapshot=snapshot, intent=intent,
                             backend="sd-lora", classifier=lambda *a, **kw: different)()


if __name__ == "__main__":
    unittest.main()
