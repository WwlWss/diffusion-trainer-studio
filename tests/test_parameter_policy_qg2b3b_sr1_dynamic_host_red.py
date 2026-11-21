"""QG-2-B3-B SR1: dynamically execute extracted real Host gates, plus identity guard RED.

Extraction retains actual NetworkTrainer statements and runs them under a pure
Python Host harness; it never imports sd-scripts, torch, or a real model.
"""
import ast
import sys
import runpy
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mikazuki.direct_trainer_admission_intent import DirectTrainerAdmissionIntent
from mikazuki.parameter_policy_admission import fingerprint_effective_training_config

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = ("scripts/stable/train_network.py", "scripts/dev/train_network.py")
MARKER = "QG-2-B3-B-SR1 POST-GATE CONFIG IDENTITY CONTRACT ABSENT"


def _source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def _as_host_function(path, start, end, context):
    source = _source(path)
    left = source.index(start, source.index("    def train(self, args, admission_intent=None):"))
    right = source.index(end, left)
    statements = textwrap.dedent(source[left:right])
    block = "def execute():\n" + textwrap.indent(statements, "    ")
    compile(block, path, "exec")
    namespace = dict(context)
    exec(block, namespace)
    return namespace["execute"]


class _Trainer:
    def __init__(self, mutation=None):
        self.mutation = mutation

    def get_parameter_policy_train_type(self, args):
        return "sd-lora"

    def assert_extra_args(self, args, *rest):
        if self.mutation is not None:
            self.mutation(args)


class _Network:
    def prepare_network(self, args):
        pass


class QG2B3BSR1DynamicHost(unittest.TestCase):
    def _initial(self, path, *, policy, status, intent=None):
        args = types.SimpleNamespace(
            parameter_policy_config="policy.json" if policy else "",
            mixed_precision="bf16", full_bf16=True,
            scale_weight_norms=False,
        )
        calls = []
        bridge = types.ModuleType("library.dts_parameter_policy_bridge")
        def booleans(**kw):
            calls.append(("booleans", kw["train_type"]))
        def admission(**kw):
            calls.append(("admission", dict(kw["effective_config"])))
            return types.SimpleNamespace(
                admission=types.SimpleNamespace(
                    status=status, runtime_blockers=("hard blocker",)
                )
            )
        bridge.preflight_component_runtime_booleans = booleans
        bridge.preflight_direct_trainer_admission = admission
        library = types.ModuleType("library")
        library.dts_parameter_policy_bridge = bridge
        util = types.SimpleNamespace(verify_training_args=lambda *a, **kw: calls.append(("verify", None)))
        fn = _as_host_function(
            path, "        parameter_policy_requested = bool(",
            "        train_util.prepare_dataset_args(args, True)",
            dict(args=args, self=_Trainer(), admission_intent=intent,
                 train_util=util, DirectTrainerAdmissionIntent=DirectTrainerAdmissionIntent),
        )
        with patch.dict(sys.modules, {"library": library, "library.dts_parameter_policy_bridge": bridge}):
            error = None
            try:
                fn()
            except ValueError as exc:
                error = str(exc)
        return calls, error

    def test_sr102_real_stable_dev_early_gate_rejects_three_denied_statuses(self):
        for path in TRAINERS:
            for status in ("blocked", "ack_required", "experimental_host_candidate"):
                with self.subTest(path=path, status=status):
                    calls, error = self._initial(
                        path, policy=True, status=status,
                        intent=DirectTrainerAdmissionIntent(status == "experimental_host_candidate"),
                    )
                    self.assertIsNotNone(error)
                    self.assertIn(status, error)
                    self.assertEqual([item[0] for item in calls], ["booleans", "verify", "admission"])

    def test_sr103_real_stable_dev_standard_path_never_imports_bridge(self):
        for path in TRAINERS:
            calls, error = self._initial(path, policy=False, status="blocked")
            self.assertIsNone(error)
            self.assertEqual([item[0] for item in calls], ["verify"])

    def test_sr104_real_early_gate_rejects_untyped_intent(self):
        for path in TRAINERS:
            calls, error = self._initial(path, policy=True, status="legacy_approved", intent=None)
            self.assertIn("typed one-shot", error)
            self.assertEqual([item[0] for item in calls], ["booleans", "verify"])

    def test_sr105_final_gate_recomputes_post_normalization_config(self):
        for path in TRAINERS:
            args = types.SimpleNamespace(
                parameter_policy_config="policy.json",
                mixed_precision="bf16", full_bf16=True,
                scale_weight_norms=True,
            )
            actual = []
            bridge = types.SimpleNamespace(
                preflight_direct_trainer_admission=lambda **kw: (
                    actual.append(dict(kw["effective_config"]))
                    or types.SimpleNamespace(
                        admission=types.SimpleNamespace(status="legacy_approved", runtime_blockers=())
                    )
                )
            )
            fn = _as_host_function(
                path, "        self.assert_extra_args(args, train_dataset_group",
                "        # acceleratorを準備する",
                dict(args=args, self=_Trainer(lambda a: setattr(a, "scale_weight_norms", False)),
                     train_dataset_group=object(), val_dataset_group=None,
                     parameter_policy_requested=True,
                     dts_parameter_policy_bridge=bridge, admission_intent=DirectTrainerAdmissionIntent(),
                     parameter_policy_train_type="sd-lora"),
            )
            fn()
            self.assertEqual(len(actual), 1)
            self.assertIs(actual[0]["scale_weight_norms"], False)

    def test_sr101_active_policy_requires_no_post_snapshot_args_mutation(self):
        """True RED: source executes network preparation yet may still alter args."""
        for path in TRAINERS:
            args = types.SimpleNamespace(scale_weight_norms=True, parameter_policy_config="policy.json")
            network = _Network()  # No max-norm support: existing fallback flips args.
            logger = types.SimpleNamespace(warning=lambda *a: None)
            start = '        if hasattr(network, "prepare_network"):'
            end = ("        train_unet = not args.network_train_text_encoder_only"
                   if "/stable/" in path else "        self.post_process_network(")
            fn = _as_host_function(
                path, start, end,
                dict(args=args, network=network, logger=logger, parameter_policy_requested=True,
                     admission_snapshot=types.SimpleNamespace(effective_config_fingerprint=fingerprint_effective_training_config(vars(args)))),
            )
            try:
                fn()
            except ValueError:
                # Rejecting unsupported network semantics is a safe GREEN.
                continue
            self.assertIs(
                args.scale_weight_norms, True,
                f"{MARKER}: {path} mutated an identity-bound arg after final Host snapshot",
            )


    def test_sr106_network_prepare_cannot_mutate_snapshot_identity_silently(self):
        class MutatingNetwork:
            def prepare_network(self, args):
                args.network_mutated_flag = True

        unprotected = []
        for path in TRAINERS:
            args = types.SimpleNamespace(
                scale_weight_norms=False, parameter_policy_config="policy.json",
                network_mutated_flag=False,
            )
            logger = types.SimpleNamespace(warning=lambda *a: None)
            start = '        if hasattr(network, "prepare_network"):'
            end = ("        train_unet = not args.network_train_text_encoder_only"
                   if "/stable/" in path else "        self.post_process_network(")
            fn = _as_host_function(
                path, start, end,
                dict(args=args, network=MutatingNetwork(), logger=logger,
                     parameter_policy_requested=True,
                     admission_snapshot=types.SimpleNamespace(
                         effective_config_fingerprint=fingerprint_effective_training_config(vars(args))
                     )),
            )
            try:
                fn()
            except ValueError as exc:
                if not any(token in str(exc).lower() for token in ("identity", "fingerprint", "mismatch")):
                    unprotected.append(path + ": wrong reason " + str(exc))
            else:
                unprotected.append(path + ": no identity guard")
        if unprotected:
            self.fail(MARKER + ": " + "; ".join(unprotected))


    def test_sr107_real_pure_bridges_reload_changed_sidecar_and_rehash(self):
        from tests.test_parameter_policy_qg2b3a_snapshot_red import _policy, _write_policy, _config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            _write_policy(path)
            for bridge_path in (
                "scripts/stable/library/dts_parameter_policy_bridge.py",
                "scripts/dev/library/dts_parameter_policy_bridge.py",
            ):
                bridge = runpy.run_path(str(ROOT / bridge_path))
                gate = bridge["preflight_direct_trainer_admission"]
                first = gate(
                    policy_path=path, train_type="sd-lora",
                    effective_config=_config(),
                    admission_intent=DirectTrainerAdmissionIntent(False),
                )
                updated = _policy()
                updated["components"]["unet.attention.adapter"]["learning_rate"] = 2e-4
                _write_policy(path, updated)
                changed = gate(
                    policy_path=path, train_type="sd-lora",
                    effective_config=_config(),
                    admission_intent=DirectTrainerAdmissionIntent(False),
                )
                self.assertNotEqual(first.policy_hash, changed.policy_hash)
                self.assertEqual(first.effective_config_fingerprint, changed.effective_config_fingerprint)
                _write_policy(path)

    def test_sr108_supported_network_does_not_trigger_false_identity_error(self):
        class SupportedNetwork:
            def prepare_network(self, args):
                pass
            def apply_max_norm_regularization(self):
                pass

        for path in TRAINERS:
            args = types.SimpleNamespace(
                scale_weight_norms=True, parameter_policy_config="policy.json",
                unchanged=True,
            )
            snapshot = types.SimpleNamespace(
                effective_config_fingerprint=fingerprint_effective_training_config(vars(args))
            )
            start = '        if hasattr(network, "prepare_network"):'
            end = ("        train_unet = not args.network_train_text_encoder_only"
                   if "/stable/" in path else "        self.post_process_network(")
            fn = _as_host_function(
                path, start, end,
                dict(args=args, network=SupportedNetwork(),
                     logger=types.SimpleNamespace(warning=lambda *a: None),
                     parameter_policy_requested=True, admission_snapshot=snapshot),
            )
            fn()
            self.assertTrue(args.scale_weight_norms)


if __name__ == "__main__":
    unittest.main()
