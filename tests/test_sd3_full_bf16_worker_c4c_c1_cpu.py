"""C4-C1 original Accelerator.load_state wrapper: narrow RED-first guards.

Genuine pinned CPU Accelerator + C3 Probe; ONLY the wrapper, prehook
deny-before-load, class/instance identity and restoration are in scope.
A self-declared Fresh parent is NOT Host accepted. No fake CUDA proof,
no forged checkpoint_loaded, no C4-C2 authentication bypass.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_sd3_full_bf16_worker_c3_f8_cpu import _make_r1_cpu_probe
from tests.test_sd3_full_bf16_worker_c4a_cpu import _manifest, _invocation
from tests.test_sd3_full_bf16_worker_c4c_cpu import _resume_adapter
from tools.sd3_full_bf16_worker_step_adapter import C4StepAdapter


def _real_prepared_resume(test, root, *, advertised_parent=None):
    bundle = _make_r1_cpu_probe()
    model, acc, _, _, _, probe = bundle
    probe.phase = "resume"
    probe.phase_id = "resume-c4c-c1-0001"
    probe.start_step = 1
    probe.install()
    test.addCleanup(probe.close)
    manifest = _manifest(probe, root)
    if advertised_parent is not None:
        manifest["fresh_checkpoint_dir"] = str(advertised_parent)
    adapter = C4StepAdapter(
        probe=probe, accelerator=acc, network=model,
        scheduler=probe.scheduler, invocation=_invocation(probe),
        manifest_payload=manifest,
    )
    test.addCleanup(adapter.close)
    adapter.install()
    return bundle, adapter


class C4C1SourceBoundaryCpu(unittest.TestCase):
    def test_c101_exact_original_accelerator_load_slot_restored_on_close(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            acc = bundle[1]
            had, slot = adapter._original_accelerator_slots["load_state"]
            load = vars(acc).get("load_state")
            self.assertTrue(
                callable(load),
                "C4-C1 CONTRACT ABSENT: same-instance managed original "
                "Accelerator.load_state wrapper is not installed",
            )
            self.assertIsNot(load, slot)
            adapter.close()
            self.assertEqual("load_state" in vars(acc), had)
            self.assertIs(vars(acc).get("load_state"), slot)
            self.assertFalse(any(
                row["event_type"] == "checkpoint_loaded"
                for row in bundle[5].events
            ))

    def test_c102_self_declared_parent_cannot_enter_original_load_prehooks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = _make_r1_cpu_probe()
            saved = root / "fresh-self-declared-state"
            bundle[1].save_state(str(saved), safe_serialization=True)
            resume, adapter = _real_prepared_resume(
                self, root, advertised_parent=saved,
            )
            acc = resume[1]
            entered = []
            hook = acc.register_load_state_pre_hook(
                lambda models, input_dir: entered.append(str(input_dir))
            )
            self.addCleanup(hook.remove)
            try:
                acc.load_state(str(saved))
            except ValueError:
                pass
            else:
                self.fail(
                    "C4-C1 CONTRACT ABSENT: self-declared Fresh parent "
                    "triggered original Accelerator Load",
                )
            self.assertEqual(
                entered, [],
                "C4-C1 CONTRACT ABSENT: original prehook invoked before "
                "independent Fresh-parent authentication",
            )
            self.assertFalse(resume[5].events)

    def test_c103_class_level_original_load_method_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            acc = bundle[1]
            original_class_method = type(acc).load_state

            def replaced(self, *args, **kwargs):
                raise AssertionError("untrusted load class replacement")

            with mock.patch.object(type(acc), "load_state", replaced):
                try:
                    adapter._check_bound_identity()
                except ValueError as exc:
                    self.assertRegex(
                        str(exc), "original|Original|drift|load_state|C4-C1",
                    )
                else:
                    self.fail(
                        "C4-C1 CONTRACT ABSENT: original Accelerator.load_state "
                        "class method was replaced after binding",
                    )
            self.assertIs(type(acc).load_state, original_class_method)
            self.assertFalse(bundle[5].events)

    def test_c104_instance_override_cannot_replace_managed_load_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, adapter = _resume_adapter(self, Path(td))
            acc = bundle[1]
            old = vars(acc).get("load_state")
            self.assertTrue(
                callable(old),
                "C4-C1 CONTRACT ABSENT: no original Load wrapper to protect",
            )
            acc.load_state = lambda *_args, **_kwargs: None
            with self.assertRaisesRegex(
                ValueError, "load_state|method drift|Load",
            ):
                adapter._check_bound_identity()
            vars(acc)["load_state"] = old
            self.assertFalse(bundle[5].events)

    def test_c105_duplicate_untrusted_load_attempt_never_enters_original(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = _make_r1_cpu_probe()
            state = root / "self-claimed-parent"
            fresh[1].save_state(str(state), safe_serialization=True)
            bundle, _adapter = _real_prepared_resume(
                self, root, advertised_parent=state,
            )
            acc = bundle[1]
            seen = []
            hook = acc.register_load_state_pre_hook(
                lambda models, input_dir: seen.append(input_dir)
            )
            self.addCleanup(hook.remove)
            for _ in range(2):
                try:
                    acc.load_state(str(state))
                except ValueError:
                    pass
                else:
                    self.fail(
                        "C4-C1 CONTRACT ABSENT: repeated unauthenticated "
                        "Load reached original Accelerator",
                    )
            self.assertEqual(
                seen, [],
                "C4-C1 CONTRACT ABSENT: duplicate Load reached prehooks",
            )
            self.assertFalse(bundle[5].events)

    def test_c106_preinstall_instance_override_rejected_without_orphan_save(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = _make_r1_cpu_probe()
            model, acc, _, _, _, probe = bundle
            probe.phase = "resume"
            probe.phase_id = "resume-c4c-c1-0002"
            probe.start_step = 1
            probe.install()
            self.addCleanup(probe.close)
            adapter = C4StepAdapter(
                probe=probe, accelerator=acc, network=model,
                scheduler=probe.scheduler, invocation=_invocation(probe),
                manifest_payload=_manifest(probe, root),
            )
            self.addCleanup(adapter.close)
            old_save = vars(acc).get("save_state")
            old_load = vars(acc).get("load_state")
            acc.load_state = lambda *_a, **_k: None
            with self.assertRaises(ValueError):
                adapter.install()
            self.assertIs(vars(acc).get("save_state"), old_save)
            self.assertNotIn("save_state", vars(acc))
            self.assertIs(vars(acc).get("load_state"), acc.load_state)
            delattr(acc, "load_state")
            self.assertIs(vars(acc).get("load_state"), old_load)


if __name__ == "__main__":
    unittest.main()
