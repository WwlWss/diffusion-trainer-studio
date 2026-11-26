"""Real extracted Stable/Dev Pre-Apply identity guard Host regression.

Executes the exact statements preceding network.apply_to in both trainers,
without importing any sd-scripts runtime, Torch, dataset or Accelerator.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import tempfile
import textwrap
import unittest

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_admission import (
    assess_component_training_admission,
    prepare_direct_trainer_admission,
)
from tests.test_parameter_policy_qg2b3a_snapshot_red import _config, _policy, _write_policy

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    "scripts/stable/train_network.py",
    "scripts/dev/train_network.py",
)
BEGIN = "        # B3-C-SR1 PRE-APPLY IDENTITY GUARD START"
END = "        # B3-C-SR1 PRE-APPLY IDENTITY GUARD END"


def _run_guard(path, *, args, frozen, intent, bridge, active=True):
    text = (ROOT / path).read_text(encoding="utf-8")
    start = text.index(BEGIN)
    # Execute through the actual attach call to prove denied identity never
    # enters network.apply_to(), not just that the guard raises somewhere.
    apply_position = text.index("network.apply_to(", text.index(END, start))
    stop = text.index("\n", apply_position)
    statements = textwrap.dedent(text[start:stop])
    script = "def check():\n" + textwrap.indent(statements, "    ")
    applied = []
    local = dict(
        args=args,
        admission_snapshot=frozen,
        admission_intent=intent,
        parameter_policy_requested=active,
        parameter_policy_train_type="sd-lora",
        dts_parameter_policy_bridge=bridge,
        self=SimpleNamespace(is_train_text_encoder=lambda _args: True),
        network=SimpleNamespace(apply_to=lambda *params: applied.append(params)),
        train_unet=True,
        train_text_encoder=True,
        text_encoder=object(),
        unet=object(),
    )
    exec(compile(script, path, "exec"), local)
    local["check"]()
    return applied


class QG2B3CSR1PreApplyHost(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "policy.json"
        _write_policy(self.path)
        self.args = SimpleNamespace(
            **_config(full_bf16=False),
            parameter_policy_config=str(self.path),
        )
        self.intent = DirectTrainerAdmissionIntent(False)
        self.snapshot = self._prepare()

    def _prepare(self):
        return prepare_direct_trainer_admission(
            self.path, train_type="sd-lora",
            effective_config=dict(vars(self.args)),
            admission_intent=self.intent,
        )

    def _bridge(self, calls):
        def check(**kwargs):
            calls.append(dict(kwargs))
            return prepare_direct_trainer_admission(
                kwargs["policy_path"],
                train_type=kwargs["train_type"],
                effective_config=kwargs["effective_config"],
                admission_intent=kwargs["admission_intent"],
            )
        return SimpleNamespace(preflight_direct_trainer_admission=check)

    def test_pa01_real_stable_dev_guard_accepts_unchanged_first_party_snapshot(self):
        for path in PATHS:
            with self.subTest(path=path):
                calls = []
                applied = _run_guard(
                    path, args=self.args, frozen=self.snapshot,
                    intent=self.intent, bridge=self._bridge(calls),
                )
                self.assertEqual(len(applied), 1)
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0]["admission_intent"], self.intent)
                self.assertEqual(calls[0]["effective_config"], vars(self.args))

    def test_pa02_rewritten_sidecar_fails_before_mutating_network(self):
        changed = _policy()
        changed["components"]["unet.attention.adapter"]["learning_rate"] = 2e-4
        _write_policy(self.path, changed)
        for path in PATHS:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "Pre-Apply Admission snapshot identity mismatch"):
                    _run_guard(
                        path, args=self.args, frozen=self.snapshot,
                        intent=self.intent, bridge=self._bridge([]),
                    )

    def test_pa03_post_network_args_or_ack_drift_fails_closed(self):
        self.args.max_train_steps = 1000
        for path in PATHS:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "Pre-Apply Admission snapshot identity mismatch"):
                    _run_guard(
                        path, args=self.args, frozen=self.snapshot,
                        intent=self.intent, bridge=self._bridge([]),
                    )
        del self.args.max_train_steps
        for path in PATHS:
            with self.subTest(path=path, drift="ack"):
                with self.assertRaisesRegex(ValueError, "Pre-Apply Admission snapshot identity mismatch"):
                    _run_guard(
                        path, args=self.args, frozen=self.snapshot,
                        intent=DirectTrainerAdmissionIntent(True), bridge=self._bridge([]),
                    )

    def test_pa04_reclassification_drift_fails_even_if_hashes_match(self):
        held = assess_component_training_admission(
            _policy("flux-finetune"),
            train_type="flux-finetune",
            effective_config=_config(full_bf16=True),
            purpose="direct_trainer",
            experimental_ack=False,
        )
        self.assertEqual(held.status, "blocked")
        forged = replace(self.snapshot, admission=held)
        bridge = SimpleNamespace(
            preflight_direct_trainer_admission=lambda **kwargs: forged,
        )
        for path in PATHS:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "first-party Admission classification drift"):
                    _run_guard(
                        path, args=self.args, frozen=self.snapshot,
                        intent=self.intent, bridge=bridge,
                    )

    def test_pa05_standard_path_never_calls_component_bridge(self):
        bridge = SimpleNamespace(
            preflight_direct_trainer_admission=lambda **kwargs: self.fail(
                "Standard path called Component bridge"
            ),
        )
        for path in PATHS:
            with self.subTest(path=path):
                applied = _run_guard(
                    path, args=self.args, frozen=self.snapshot,
                    intent=None, bridge=bridge, active=False,
                )
                self.assertEqual(len(applied), 1)


if __name__ == "__main__":
    unittest.main()
