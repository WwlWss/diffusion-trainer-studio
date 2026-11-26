"""CPU runtime probes for the isolated GPU evidence-only Full-BF16 helper.

These tests do NOT qualify flux-finetune, and do not launch a CUDA job.
They check that the tool-only construction uses actual DTS low-level objects
while the production Direct Trainer admission remains fail-closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_admission import prepare_direct_trainer_admission
from mikazuki.parameter_policy_trainer import (
    ParameterPolicyTrainerRuntimeError,
    create_parameter_policy_session,
)
from tools._parameter_policy_gpu_low_level_fixture import (
    build_gpu_evidence_full_bf16_session,
)


_MUON_ARGS = {
    "momentum": 0.95,
    "weight_decay": 0.0,
    "weight_decouple": True,
    "nesterov": True,
    "ns_steps": 5,
    "ns_coeffs": "original",
    "use_adjusted_lr": False,
}


class _TinyFlux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])


class _TinyMuonFlux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList(
            [
                torch.nn.Linear(8, 12, bias=False),
                torch.nn.Linear(12, 8, bias=False),
            ]
        )


class _TinyMuonAdamWFallbackFlux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList(
            [
                torch.nn.Linear(8, 12, bias=True),
                torch.nn.Linear(12, 8, bias=True),
            ]
        )


class _OtherFlux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])


def _policy_for(model: torch.nn.Module) -> dict:
    if type(model).__name__ == "_TinyFlux":
        profiles = {"main": {"type": "AdamW", "args": {}}}
        component = {
            "train": True,
            "optimizer_profile": "main",
            "learning_rate": 1e-2,
        }
    elif type(model).__name__ == "_TinyMuonFlux":
        profiles = {"main": {"type": "Muon", "args": dict(_MUON_ARGS)}}
        component = {
            "train": True,
            "optimizer_profile": "main",
            "learning_rate": 2e-2,
        }
    else:
        profiles = {
            "muon": {"type": "Muon", "args": dict(_MUON_ARGS)},
            "adamw_fallback": {"type": "AdamW", "args": {}},
        }
        component = {
            "train": True,
            "optimizer_profile": "muon",
            "learning_rate": 2e-2,
            "fallback_optimizer_profile": "adamw_fallback",
            "fallback_learning_rate": 1e-2,
        }
    return {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": {"transformer.double_stream": component},
    }


def _args(policy_path: Path, *, full_bf16: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        parameter_policy_config=str(policy_path),
        full_bf16=full_bf16,
        mixed_precision="bf16",
    )


class IsolatedGpuEvidenceHelperCpuTests(unittest.TestCase):
    def _exercise(self, model_type: type[torch.nn.Module]) -> None:
        model = model_type()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(_policy_for(model)), encoding="utf-8")
            session = build_gpu_evidence_full_bf16_session(
                args=_args(path),
                train_type="flux-finetune",
                roots={"transformer": model},
            )
            self.assertEqual(session.execution_contract.features, ("full_bf16",))
            self.assertEqual(
                {type(parameter).__name__ for parameter in session.trainable_parameters},
                {"Parameter"},
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in session.trainable_parameters),
                sum(parameter.numel() for parameter in model.parameters()),
            )
            self.assertTrue(session.routing_plan.is_valid)

    def test_R2G01_tiny_adamw_low_level_session(self):
        self._exercise(_TinyFlux)

    def test_R2G02_tiny_muon_low_level_session(self):
        self._exercise(_TinyMuonFlux)

    def test_R2G03_tiny_muon_adamw_fallback_low_level_session(self):
        self._exercise(_TinyMuonAdamWFallbackFlux)

    def test_R2G04_wrong_model_class_is_rejected_before_root_scan(self):
        model = _OtherFlux()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(_policy_for(_TinyFlux())), encoding="utf-8")
            with mock.patch("mikazuki.parameter_policy_trainer.scan_parameter_roots") as scan:
                with self.assertRaisesRegex(ValueError, "non-synthetic model classes"):
                    build_gpu_evidence_full_bf16_session(
                        args=_args(path),
                        train_type="flux-finetune",
                        roots={"transformer": model},
                    )
                scan.assert_not_called()

    def test_R2G05_inactive_precision_is_rejected_before_root_scan(self):
        model = _TinyFlux()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(_policy_for(model)), encoding="utf-8")
            with mock.patch("mikazuki.parameter_policy_trainer.scan_parameter_roots") as scan:
                with self.assertRaisesRegex(ValueError, "real full_bf16=True"):
                    build_gpu_evidence_full_bf16_session(
                        args=_args(path, full_bf16=False),
                        train_type="flux-finetune",
                        roots={"transformer": model},
                    )
                scan.assert_not_called()

    def test_R2G06_production_full_bf16_flux_remains_blocked_pre_scan(self):
        model = _TinyFlux()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(_policy_for(model)), encoding="utf-8")
            args = _args(path)
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = prepare_direct_trainer_admission(
                str(path),
                train_type="flux-finetune",
                effective_config=vars(args),
                admission_intent=intent,
            )
            self.assertEqual(snapshot.admission.status, "blocked")
            with mock.patch("mikazuki.parameter_policy_trainer.scan_parameter_roots") as scan:
                with self.assertRaisesRegex(
                    ParameterPolicyTrainerRuntimeError,
                    "capability\\.backend\\.audit_required",
                ):
                    create_parameter_policy_session(
                        args=args,
                        train_type="flux-finetune",
                        roots={"transformer": model},
                        admission_snapshot=snapshot,
                        admission_intent=intent,
                    )
                scan.assert_not_called()

    def test_R2G07_inactive_real_admission_still_constructs_legacy_session(self):
        model = _TinyFlux()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(_policy_for(model)), encoding="utf-8")
            args = _args(path, full_bf16=False)
            intent = DirectTrainerAdmissionIntent(False)
            snapshot = prepare_direct_trainer_admission(
                str(path),
                train_type="flux-finetune",
                effective_config=vars(args),
                admission_intent=intent,
            )
            self.assertEqual(snapshot.admission.status, "inactive")
            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
                admission_snapshot=snapshot,
                admission_intent=intent,
            )
            self.assertIsNone(session.execution_contract)
            self.assertEqual(len(session.trainable_parameters), 2)


if __name__ == "__main__":
    unittest.main()
