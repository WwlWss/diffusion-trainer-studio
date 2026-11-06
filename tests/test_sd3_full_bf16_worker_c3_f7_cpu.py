"""C3-F7 RED-first CPU contracts: frozen provider semantics and AdamW continuity.

Real pinned Torch 2.7.0 / pytorch-optimizer 3.10.0 / Accelerate 1.6.0.
Never upgrades CPU evidence to physical CUDA qualification.
"""
from __future__ import annotations

import copy
import unittest

from tests.test_sd3_full_bf16_worker_c3_strict_cpu import (
    R1, R2, three_component_cpu_probe, two_batches,
)
from tests.test_sd3_full_bf16_worker_c3_f6_cpu import (
    fake_r1_reference_session, fake_reference_session,
)
from tools import sd3_full_bf16_worker_observer as observer


def _child(facade, kind):
    matches = [entry.optimizer for entry in facade.entries if entry.optimizer_type == kind]
    if len(matches) != 1:
        raise AssertionError("Expected exactly one real optimizer child: " + kind)
    return matches[0]


class F7AReferenceProviderSemantics(unittest.TestCase):
    def test_r1_real_adamw_default_reference_positive(self):
        session = fake_r1_reference_session()
        result = observer.validate_reference_optimizer_group_semantics(
            case_id=R1, session=session,
        )
        self.assertEqual(set(result), {"baseline"})
        self.assertEqual(result["baseline"]["optimizer_type"], "AdamW")

    def test_r2_real_muon_and_independent_adamw_reference_positive(self):
        session = fake_reference_session()
        result = observer.validate_reference_optimizer_group_semantics(
            case_id=R2, session=session,
        )
        self.assertEqual(set(result), {"muon_primary", "adamw_fallback"})
        self.assertEqual(result["muon_primary"]["optimizer_type"], "Muon")
        self.assertEqual(result["adamw_fallback"]["optimizer_type"], "AdamW")

    def test_self_consistent_lr_cannot_hide_wrong_adamw_betas(self):
        session = fake_r1_reference_session()
        _child(session.optimizer, "AdamW").param_groups[0]["betas"] = (0.5, 0.9)
        # Existing F6-B accepts the correct LR; this is specifically an F7 gap.
        observer.validate_reference_group_learning_rates(case_id=R1, session=session)
        with self.assertRaisesRegex(ValueError, "semantic|reference|AdamW|Group"):
            observer.validate_reference_optimizer_group_semantics(
                case_id=R1, session=session,
            )

    def test_adamw_epsilon_and_weight_decay_mutation_rejected(self):
        for field, value in (("eps", 1e-5), ("weight_decay", 0.25)):
            with self.subTest(field=field):
                session = fake_reference_session()
                _child(session.optimizer, "AdamW").param_groups[0][field] = value
                observer.validate_reference_group_learning_rates(case_id=R2, session=session)
                with self.assertRaisesRegex(ValueError, "semantic|reference|AdamW|Group"):
                    observer.validate_reference_optimizer_group_semantics(
                        case_id=R2, session=session,
                    )

    def test_adamw_execution_mode_mutations_rejected(self):
        for field, value in (
            ("maximize", True),
            ("foreach", True),
            ("capturable", True),
            ("differentiable", True),
            ("fused", True),
            ("decoupled_weight_decay", False),
        ):
            with self.subTest(field=field):
                session = fake_r1_reference_session()
                _child(session.optimizer, "AdamW").param_groups[0][field] = value
                with self.assertRaisesRegex(ValueError, "semantic|reference|AdamW|Group"):
                    observer.validate_reference_optimizer_group_semantics(
                        case_id=R1, session=session,
                    )

    def test_muon_momentum_nesterov_newton_schulz_steps_rejected(self):
        for field, value in (
            ("momentum", 0.8), ("nesterov", False), ("ns_steps", 2),
        ):
            with self.subTest(field=field):
                session = fake_reference_session()
                _child(session.optimizer, "Muon").param_groups[0][field] = value
                with self.assertRaisesRegex(ValueError, "semantic|reference|Muon|Group"):
                    observer.validate_reference_optimizer_group_semantics(
                        case_id=R2, session=session,
                    )

    def test_muon_coefficients_and_routing_flags_rejected(self):
        for field, value in (
            ("ns_coeffs", [(1.0, 2.0, 3.0)]),
            ("use_adjusted_lr", True),
            ("weight_decouple", False),
            ("weight_decay", 0.2),
            ("use_muon", False),
        ):
            with self.subTest(field=field):
                session = fake_reference_session()
                _child(session.optimizer, "Muon").param_groups[0][field] = value
                with self.assertRaisesRegex(ValueError, "semantic|reference|Muon|Group"):
                    observer.validate_reference_optimizer_group_semantics(
                        case_id=R2, session=session,
                    )

    def test_muon_optimizer_instance_maximize_rejected(self):
        session = fake_reference_session()
        _child(session.optimizer, "Muon").maximize = True
        with self.assertRaisesRegex(ValueError, "semantic|reference|Muon|Group"):
            observer.validate_reference_optimizer_group_semantics(
                case_id=R2, session=session,
            )

    def test_muon_unreviewed_cautious_mode_rejected(self):
        session = fake_reference_session()
        _child(session.optimizer, "Muon").param_groups[0]["cautious"] = True
        with self.assertRaisesRegex(ValueError, "semantic|reference|Muon|Group"):
            observer.validate_reference_optimizer_group_semantics(
                case_id=R2, session=session,
            )

    def test_after_probe_init_pre_child_drift_rejected_before_physical_step(self):
        bundle = three_component_cpu_probe()
        adamw = _child(bundle[4], "AdamW")
        original = adamw.step
        called = []

        def counting_step(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        adamw.step = counting_step
        adamw.param_groups[0]["betas"] = (0.5, 0.9)
        try:
            with self.assertRaisesRegex(ValueError, "semantic|reference|drift|Group"):
                two_batches(bundle)
            self.assertEqual(called, [], "Invalid Group must be rejected before original Child.step")
        finally:
            delattr(adamw, "step")

    def test_mutation_inside_real_child_step_cannot_publish_success(self):
        bundle = three_component_cpu_probe()
        adamw = _child(bundle[4], "AdamW")
        original = adamw.step
        called = []

        def corrupt_inside_step(*args, **kwargs):
            called.append(True)
            output = original(*args, **kwargs)
            adamw.param_groups[0]["betas"] = (0.5, 0.9)
            return output

        adamw.step = corrupt_inside_step
        try:
            with self.assertRaisesRegex(ValueError, "semantic|reference|drift|Group"):
                two_batches(bundle)
            self.assertEqual(len(called), 1)
            self.assertFalse(any(
                e["event_type"] == "physical_child_stepped"
                and e.get("optimizer_profile") == "adamw_fallback"
                for e in bundle[5].events
            ))
        finally:
            delattr(adamw, "step")

    def test_constant_scheduler_mutating_non_lr_semantics_is_rejected(self):
        bundle = three_component_cpu_probe()
        scheduler = bundle[5].scheduler
        adamw = _child(bundle[4], "AdamW")
        original = scheduler.step

        def corrupt_after_scheduler(*args, **kwargs):
            result = original(*args, **kwargs)
            adamw.param_groups[0]["weight_decay"] = 0.4
            return result

        scheduler.step = corrupt_after_scheduler
        try:
            with self.assertRaisesRegex(ValueError, "semantic|reference|drift|Group"):
                two_batches(bundle)
            self.assertFalse(any(
                e["event_type"] == "physical_scheduler_stepped"
                for e in bundle[5].events
            ))
        finally:
            delattr(scheduler, "step")

    def test_unmodified_real_cpu_r2_steps_remain_valid(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        events = two_batches(bundle)
        children = [
            e for e in events
            if e["event_type"] == "physical_child_stepped"
            and e["logical_step"] == 2
        ]
        self.assertEqual(len(children), 2)
        self.assertTrue(all("physical_group_semantics_sha256" in e for e in children))


class F7BAdamWInProcessContinuity(unittest.TestCase):
    def test_unmodified_fresh_step1_to_step2_positive(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        probe = bundle[5]
        self.assertIn("adamw_fallback", probe._last_adamw_moment_sha)
        first = probe._last_adamw_moment_sha["adamw_fallback"]
        two_batches(bundle)
        self.assertIn("adamw_fallback", probe._last_adamw_moment_sha)
        self.assertNotEqual(first, probe._last_adamw_moment_sha["adamw_fallback"])

    def test_finite_exp_avg_modified_between_real_steps_rejected(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        adamw = _child(bundle[4], "AdamW")
        p = adamw.param_groups[0]["params"][0]
        adamw.state[p]["exp_avg"].add_(0.5)
        with self.assertRaisesRegex(ValueError, "AdamW|moment|between|continuity"):
            two_batches(bundle)

    def test_finite_exp_avg_sq_modified_between_real_steps_rejected(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        adamw = _child(bundle[4], "AdamW")
        p = adamw.param_groups[0]["params"][0]
        adamw.state[p]["exp_avg_sq"].add_(0.5)
        with self.assertRaisesRegex(ValueError, "AdamW|moment|between|continuity"):
            two_batches(bundle)

    def test_fresh_second_step_missing_previous_digest_rejected(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        bundle[5]._last_adamw_moment_sha.clear()
        with self.assertRaisesRegex(ValueError, "AdamW|moment|baseline|continuity"):
            two_batches(bundle)

    def test_new_resume_probe_does_not_require_in_memory_fresh_digest(self):
        bundle = three_component_cpu_probe()
        two_batches(bundle)
        model, accelerator, prepared_opt, prepared_sched, facade, previous = bundle
        resumed = observer.PhysicalStepProbe(
            accelerator=accelerator,
            composite_optimizer=facade,
            composite_scheduler=previous.scheduler,
            parameter_names=previous.params,
            component_by_parameter=previous.component_by_parameter,
            case_id=R2,
            run_id="run-c3-f7-resume-0001",
            phase_id="resume-c3-f7-0001",
            source_commit="a" * 40,
            phase="resume",
            start_step=1,
        )
        self.assertEqual(resumed._last_adamw_moment_sha, {})
        events = two_batches((
            model, accelerator, prepared_opt, prepared_sched, facade, resumed,
        ))
        self.assertEqual(len([
            e for e in events if e["event_type"] == "physical_child_stepped"
            and e["logical_step"] == 2
        ]), 2)


if __name__ == "__main__":
    unittest.main()
