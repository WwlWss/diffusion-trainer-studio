from __future__ import annotations

import unittest
from unittest.mock import patch

from mikazuki.optimizer_profiles import OptimizerCapability
from mikazuki.parameter_policy_runtime import (
    ParameterPolicyRuntimeSpecError,
    compile_parameter_policy_runtime_spec,
)
from mikazuki.parameter_routing import (
    ParameterStat,
    RoutingAssignment,
    RoutingIssue,
    RoutingPlan,
    RoutingStats,
)


class FakeParameter:
    def __init__(self, shape=(4, 4)):
        self.shape = tuple(shape)


def _stats():
    zero = ParameterStat(0, 0)
    return RoutingStats(
        total=zero,
        assigned=zero,
        unassigned=zero,
        conflicts=zero,
        unroutable=zero,
        by_component={},
        by_route={},
        by_parameter_class={},
    )


def _plan(*assignments, issues=()):
    return RoutingPlan(tuple(assignments), tuple(issues), _stats())


def _assignment(
    parameter,
    *,
    name,
    component,
    profile=None,
    lr=None,
    route="primary",
    parameter_id=None,
):
    return RoutingAssignment(
        parameter=parameter,
        parameter_id=id(parameter) if parameter_id is None else parameter_id,
        canonical_name=name,
        parameter_class="matrix_weight",
        component_id=component,
        route_kind=route,
        optimizer_profile=profile,
        learning_rate=lr,
        reason="test",
    )


def _policy(*, profiles, components):
    return {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": components,
    }


def _train(profile, lr, *, fallback=None, fallback_lr=None):
    result = {
        "train": True,
        "optimizer_profile": profile,
        "learning_rate": lr,
    }
    if fallback is not None:
        result["fallback_optimizer_profile"] = fallback
    if fallback_lr is not None:
        result["fallback_learning_rate"] = fallback_lr
    return result


class ParameterPolicyRuntimeSpecTests(unittest.TestCase):
    def test_same_profile_same_lr_merges_components_into_one_group(self):
        a = FakeParameter()
        b = FakeParameter()
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 1e-4), "b": _train("main", 1e-4)},
        )
        spec = compile_parameter_policy_runtime_spec(
            policy,
            _plan(
                _assignment(a, name="root.z.weight", component="a", profile="main", lr=1e-4),
                _assignment(b, name="root.a.weight", component="b", profile="main", lr=1e-4),
            ),
        )
        self.assertEqual(len(spec.optimizers), 1)
        self.assertEqual(len(spec.optimizers[0].groups), 1)
        self.assertEqual(
            [p.canonical_name for p in spec.optimizers[0].groups[0].parameters],
            ["root.a.weight", "root.z.weight"],
        )

    def test_same_profile_different_lrs_become_groups_not_optimizers(self):
        a = FakeParameter()
        b = FakeParameter()
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 2e-4), "b": _train("main", 1e-4)},
        )
        spec = compile_parameter_policy_runtime_spec(
            policy,
            _plan(
                _assignment(a, name="a.weight", component="a", profile="main", lr=2e-4),
                _assignment(b, name="b.weight", component="b", profile="main", lr=1e-4),
            ),
        )
        self.assertEqual(len(spec.optimizers), 1)
        self.assertEqual(
            [group.learning_rate for group in spec.optimizers[0].groups],
            [1e-4, 2e-4],
        )

    def test_primary_and_fallback_share_final_reusable_profile(self):
        a = FakeParameter()
        b = FakeParameter()
        policy = _policy(
            profiles={
                "muon": {"type": "Muon", "args": {}},
                "shared": {"type": "AdamW", "args": {}},
            },
            components={
                "a": _train("shared", 5e-5),
                "b": _train("muon", 2e-4, fallback="shared", fallback_lr=5e-5),
            },
        )
        spec = compile_parameter_policy_runtime_spec(
            policy,
            _plan(
                _assignment(a, name="a.weight", component="a", profile="shared", lr=5e-5),
                _assignment(
                    b,
                    name="b.weight",
                    component="b",
                    profile="shared",
                    lr=5e-5,
                    route="fallback",
                ),
            ),
        )
        self.assertEqual([item.profile_name for item in spec.optimizers], ["shared"])
        self.assertEqual(spec.optimizers[0].tensor_count, 2)

    def test_order_is_deterministic(self):
        a = FakeParameter()
        b = FakeParameter()
        policy = _policy(
            profiles={
                "Zulu": {"type": "AdamW", "args": {}},
                "alpha": {"type": "Lion", "args": {}},
            },
            components={"a": _train("Zulu", 2e-4), "b": _train("alpha", 1e-4)},
        )
        first = _plan(
            _assignment(a, name="z.weight", component="a", profile="Zulu", lr=2e-4),
            _assignment(b, name="a.weight", component="b", profile="alpha", lr=1e-4),
        )
        spec1 = compile_parameter_policy_runtime_spec(policy, first)
        spec2 = compile_parameter_policy_runtime_spec(policy, _plan(*reversed(first.assignments)))
        self.assertEqual(spec1, spec2)
        self.assertEqual(spec1.topology_fingerprint, spec2.topology_fingerprint)
        self.assertEqual([item.profile_name for item in spec1.optimizers], ["alpha", "Zulu"])

    def test_runtime_object_ids_do_not_enter_fingerprint(self):
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 1e-4)},
        )
        first = FakeParameter((8, 4))
        second = FakeParameter((8, 4))
        spec1 = compile_parameter_policy_runtime_spec(
            policy,
            _plan(_assignment(first, name="a.weight", component="a", profile="main", lr=1e-4)),
        )
        spec2 = compile_parameter_policy_runtime_spec(
            policy,
            _plan(_assignment(second, name="a.weight", component="a", profile="main", lr=1e-4)),
        )
        self.assertEqual(spec1.topology_fingerprint, spec2.topology_fingerprint)

    def test_topology_changes_on_name_shape_lr_type_or_args(self):
        def compile_one(name="a.weight", shape=(4, 4), lr=1e-4, opt_type="AdamW", args=None):
            p = FakeParameter(shape)
            policy = _policy(
                profiles={"main": {"type": opt_type, "args": args or {}}},
                components={"a": _train("main", lr)},
            )
            return compile_parameter_policy_runtime_spec(
                policy,
                _plan(_assignment(p, name=name, component="a", profile="main", lr=lr)),
            ).topology_fingerprint

        base = compile_one()
        for value in (
            compile_one(name="b.weight"),
            compile_one(shape=(8, 4)),
            compile_one(lr=2e-4),
            compile_one(opt_type="Lion"),
            compile_one(args={"weight_decay": 0.1}),
        ):
            self.assertNotEqual(base, value)

    def test_component_and_route_labels_do_not_change_optimizer_topology_hash(self):
        p1 = FakeParameter()
        p2 = FakeParameter()
        primary_policy = _policy(
            profiles={"shared": {"type": "AdamW", "args": {}}},
            components={"old": _train("shared", 5e-5)},
        )
        fallback_policy = _policy(
            profiles={
                "muon": {"type": "Muon", "args": {}},
                "shared": {"type": "AdamW", "args": {}},
            },
            components={"new": _train("muon", 2e-4, fallback="shared", fallback_lr=5e-5)},
        )
        primary = compile_parameter_policy_runtime_spec(
            primary_policy,
            _plan(_assignment(p1, name="same.weight", component="old", profile="shared", lr=5e-5)),
        )
        fallback = compile_parameter_policy_runtime_spec(
            fallback_policy,
            _plan(
                _assignment(
                    p2,
                    name="same.weight",
                    component="new",
                    profile="shared",
                    lr=5e-5,
                    route="fallback",
                )
            ),
        )
        self.assertNotEqual(
            primary.optimizers[0].groups[0].parameters[0].component_id,
            fallback.optimizers[0].groups[0].parameters[0].component_id,
        )
        self.assertNotEqual(
            primary.optimizers[0].groups[0].parameters[0].route_kind,
            fallback.optimizers[0].groups[0].parameters[0].route_kind,
        )
        self.assertEqual(primary.topology_fingerprint, fallback.topology_fingerprint)

    def test_duplicate_physical_parameter_and_duplicate_name_fail_closed(self):
        p = FakeParameter()
        q = FakeParameter()
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 1e-4), "b": _train("main", 1e-4)},
        )
        with self.assertRaises(ParameterPolicyRuntimeSpecError) as physical:
            compile_parameter_policy_runtime_spec(
                policy,
                _plan(
                    _assignment(p, name="a.weight", component="a", profile="main", lr=1e-4),
                    _assignment(p, name="b.weight", component="b", profile="main", lr=1e-4),
                ),
            )
        self.assertIn("duplicate_physical_parameter", {x.code for x in physical.exception.issues})

        with self.assertRaises(ParameterPolicyRuntimeSpecError) as names:
            compile_parameter_policy_runtime_spec(
                policy,
                _plan(
                    _assignment(p, name="same.weight", component="a", profile="main", lr=1e-4),
                    _assignment(q, name="same.weight", component="b", profile="main", lr=1e-4),
                ),
            )
        self.assertIn("duplicate_canonical_parameter_name", {x.code for x in names.exception.issues})

    def test_parameter_id_mismatch_fails_closed(self):
        p = FakeParameter()
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 1e-4)},
        )
        with self.assertRaises(ParameterPolicyRuntimeSpecError) as ctx:
            compile_parameter_policy_runtime_spec(
                policy,
                _plan(
                    _assignment(
                        p,
                        name="a.weight",
                        component="a",
                        profile="main",
                        lr=1e-4,
                        parameter_id=id(p) + 1,
                    )
                ),
            )
        self.assertIn("parameter_identity_mismatch", {x.code for x in ctx.exception.issues})

    def test_policy_plan_coherence_and_fallback_default_lr(self):
        p = FakeParameter()
        policy = _policy(
            profiles={
                "muon": {"type": "Muon", "args": {}},
                "fallback": {"type": "AdamW", "args": {}},
            },
            components={"a": _train("muon", 2e-4, fallback="fallback")},
        )
        spec = compile_parameter_policy_runtime_spec(
            policy,
            _plan(
                _assignment(
                    p,
                    name="a.weight",
                    component="a",
                    profile="fallback",
                    lr=2e-4,
                    route="fallback",
                )
            ),
        )
        self.assertEqual(spec.optimizers[0].groups[0].learning_rate, 2e-4)

        with self.assertRaises(ParameterPolicyRuntimeSpecError) as mismatch:
            compile_parameter_policy_runtime_spec(
                policy,
                _plan(
                    _assignment(
                        p,
                        name="a.weight",
                        component="a",
                        profile="fallback",
                        lr=1e-4,
                        route="fallback",
                    )
                ),
            )
        self.assertIn(
            "assignment_learning_rate_mismatch",
            {x.code for x in mismatch.exception.issues},
        )

    def test_invalid_plan_no_trainable_and_bad_inactive_route_fail_closed(self):
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": {"train": False}},
        )
        issue = RoutingIssue(severity="error", code="missing_fallback", message="test")
        with self.assertRaises(ParameterPolicyRuntimeSpecError) as invalid:
            compile_parameter_policy_runtime_spec(policy, _plan(issues=(issue,)))
        self.assertEqual(invalid.exception.issues[0].code, "invalid_routing_plan")

        p = FakeParameter()
        with self.assertRaises(ParameterPolicyRuntimeSpecError) as empty:
            compile_parameter_policy_runtime_spec(
                policy,
                _plan(_assignment(p, name="a.weight", component="a", route="frozen")),
            )
        self.assertIn(
            "no_trainable_parameter_assignments",
            {x.code for x in empty.exception.issues},
        )

        with self.assertRaises(ParameterPolicyRuntimeSpecError) as bad_frozen:
            compile_parameter_policy_runtime_spec(
                policy,
                _plan(
                    _assignment(
                        p,
                        name="a.weight",
                        component="a",
                        profile="main",
                        lr=1e-4,
                        route="frozen",
                    )
                ),
            )
        self.assertIn(
            "inactive_route_has_optimizer_metadata",
            {x.code for x in bad_frozen.exception.issues},
        )

    def test_used_restricted_is_rejected_unused_planned_is_ignored(self):
        p = FakeParameter()
        policy = _policy(
            profiles={
                "main": {"type": "AdamW", "args": {}},
                "unused": {"type": "pytorch_optimizer.CAME", "args": {}},
            },
            components={"a": _train("main", 1e-4)},
        )
        spec = compile_parameter_policy_runtime_spec(
            policy,
            _plan(_assignment(p, name="a.weight", component="a", profile="main", lr=1e-4)),
        )
        self.assertEqual([x.profile_name for x in spec.optimizers], ["main"])

        restricted = _policy(
            profiles={"restricted": {"type": "AdaFactor", "args": {}}},
            components={"a": _train("restricted", 1e-4)},
        )
        with self.assertRaises(ParameterPolicyRuntimeSpecError) as ctx:
            compile_parameter_policy_runtime_spec(
                restricted,
                _plan(
                    _assignment(
                        p,
                        name="a.weight",
                        component="a",
                        profile="restricted",
                        lr=1e-4,
                    )
                ),
            )
        self.assertIn("optimizer_profile_not_runnable", {x.code for x in ctx.exception.issues})

    def test_group_lr_capability_is_enforced(self):
        a = FakeParameter()
        b = FakeParameter()
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 1e-4), "b": _train("main", 2e-4)},
        )
        fake = OptimizerCapability(
            "AdamW", "supported", False, True, "normal", implementation="torch.optim.AdamW"
        )
        with patch(
            "mikazuki.parameter_policy_runtime.require_unrestricted_component_optimizer",
            return_value=fake,
        ):
            with self.assertRaises(ParameterPolicyRuntimeSpecError) as ctx:
                compile_parameter_policy_runtime_spec(
                    policy,
                    _plan(
                        _assignment(a, name="a.weight", component="a", profile="main", lr=1e-4),
                        _assignment(b, name="b.weight", component="b", profile="main", lr=2e-4),
                    ),
                )
        self.assertIn(
            "optimizer_profile_multiple_lrs_unsupported",
            {x.code for x in ctx.exception.issues},
        )

    def test_shape_and_numel_are_structural_only(self):
        p = FakeParameter((2, 3, 4))
        policy = _policy(
            profiles={"main": {"type": "AdamW", "args": {}}},
            components={"a": _train("main", 1e-4)},
        )
        spec = compile_parameter_policy_runtime_spec(
            policy,
            _plan(_assignment(p, name="a.weight", component="a", profile="main", lr=1e-4)),
        )
        item = spec.optimizers[0].groups[0].parameters[0]
        self.assertEqual(item.shape, (2, 3, 4))
        self.assertEqual(item.numel, 24)
        self.assertEqual(spec.tensor_count, 1)
        self.assertEqual(spec.numel, 24)


if __name__ == "__main__":
    unittest.main()
