from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_method(path: Path, class_name: str, method_name: str, **globals_):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class _CapturedCacheStrategy:
    def __init__(self, *args, is_partial=False, **kwargs):
        self.is_partial = is_partial


def _write_policy(path: Path, components: dict[str, bool]) -> None:
    payload = {
        "version": 1,
        "optimizer_profiles": {
            "main": {"type": "AdamW", "args": {}},
        },
        "components": {
            component_id: (
                {
                    "train": True,
                    "optimizer_profile": "main",
                    "learning_rate": 1e-4,
                }
                if train
                else {"train": False}
            )
            for component_id, train in components.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bridge_module():
    def load_parameter_policy_train_flags(path, component_ids):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        components = payload["components"]
        missing = [item for item in component_ids if item not in components]
        if missing:
            raise ValueError(f"missing components: {missing!r}")
        return {
            item: bool(components[item].get("train"))
            for item in component_ids
        }

    return SimpleNamespace(
        load_parameter_policy_train_flags=load_parameter_policy_train_flags,
    )


_RUNTIME_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "accelerate")
)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator

    from mikazuki.parameter_policy_trainer import (
        ParameterPolicyTrainerRuntimeError,
        create_parameter_policy_session,
        load_parameter_policy_train_flags,
        make_legacy_scheduler_factory,
    )


def _args(policy_path: Path):
    return SimpleNamespace(
        parameter_policy_config=str(policy_path),
        optimizer_type="AdamW",
        lr_scheduler="constant",
        lr_scheduler_type="",
        lr_scheduler_args=None,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        max_train_steps=4,
    )


if _RUNTIME_DEPS_AVAILABLE:
    class TinyFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.single_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
else:
    TinyFlux = object


class ParameterPolicyCacheLifecycleContractTests(unittest.TestCase):
    def test_flux_policy_cache_partial_hint_tracks_sidecar_train_flags(self):
        method = _load_method(
            ROOT / "scripts" / "dev" / "flux_train_network.py",
            "FluxNetworkTrainer",
            "get_text_encoder_outputs_caching_strategy",
            strategy_flux=SimpleNamespace(
                FluxTextEncoderOutputsCachingStrategy=_CapturedCacheStrategy,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            args = SimpleNamespace(
                cache_text_encoder_outputs=True,
                parameter_policy_config=str(policy_path),
                cache_text_encoder_outputs_to_disk=False,
                text_encoder_batch_size=1,
                skip_cache_check=False,
                apply_t5_attn_mask=False,
            )
            library_module = SimpleNamespace(
                dts_parameter_policy_bridge=_bridge_module(),
            )
            with mock.patch.dict(
                "sys.modules",
                {
                    "library": library_module,
                    "library.dts_parameter_policy_bridge": library_module.dts_parameter_policy_bridge,
                },
            ):
                trainer = SimpleNamespace(
                    train_clip_l=False,
                    train_t5xxl=False,
                    use_clip_l=True,
                )

                _write_policy(
                    policy_path,
                    {
                        "clip_l.adapter": False,
                        "t5xxl.adapter": False,
                    },
                )
                self.assertFalse(method(trainer, args).is_partial)

                _write_policy(
                    policy_path,
                    {
                        "clip_l.adapter": True,
                        "t5xxl.adapter": False,
                    },
                )
                self.assertTrue(method(trainer, args).is_partial)

                _write_policy(
                    policy_path,
                    {
                        "clip_l.adapter": False,
                        "t5xxl.adapter": True,
                    },
                )
                with self.assertRaisesRegex(ValueError, "T5XXL"):
                    method(trainer, args)

                trainer.use_clip_l = False
                _write_policy(
                    policy_path,
                    {"t5xxl.adapter": False},
                )
                self.assertFalse(method(trainer, args).is_partial)

    def test_sd3_policy_cache_partial_hint_tracks_sidecar_train_flags(self):
        method = _load_method(
            ROOT / "scripts" / "dev" / "sd3_train_network.py",
            "Sd3NetworkTrainer",
            "get_text_encoder_outputs_caching_strategy",
            strategy_sd3=SimpleNamespace(
                Sd3TextEncoderOutputsCachingStrategy=_CapturedCacheStrategy,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            args = SimpleNamespace(
                cache_text_encoder_outputs=True,
                parameter_policy_config=str(policy_path),
                cache_text_encoder_outputs_to_disk=False,
                text_encoder_batch_size=1,
                skip_cache_check=False,
                apply_lg_attn_mask=False,
                apply_t5_attn_mask=False,
            )
            library_module = SimpleNamespace(
                dts_parameter_policy_bridge=_bridge_module(),
            )
            with mock.patch.dict(
                "sys.modules",
                {
                    "library": library_module,
                    "library.dts_parameter_policy_bridge": library_module.dts_parameter_policy_bridge,
                },
            ):
                trainer = SimpleNamespace(train_clip=False, train_t5xxl=False)

                _write_policy(
                    policy_path,
                    {
                        "clip_l.adapter": False,
                        "clip_g.adapter": False,
                        "t5xxl.adapter": False,
                    },
                )
                self.assertFalse(method(trainer, args).is_partial)

                for trained_component in ("clip_l.adapter", "clip_g.adapter"):
                    _write_policy(
                        policy_path,
                        {
                            "clip_l.adapter": trained_component == "clip_l.adapter",
                            "clip_g.adapter": trained_component == "clip_g.adapter",
                            "t5xxl.adapter": False,
                        },
                    )
                    self.assertTrue(method(trainer, args).is_partial)

                _write_policy(
                    policy_path,
                    {
                        "clip_l.adapter": False,
                        "clip_g.adapter": False,
                        "t5xxl.adapter": True,
                    },
                )
                with self.assertRaisesRegex(ValueError, "T5XXL"):
                    method(trainer, args)


    def test_partial_cache_keeps_dataset_tokenization_available(self):
        source = (
            ROOT / "scripts" / "dev" / "library" / "train_util.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "self.text_encoder_output_caching_strategy is None or "
            "self.text_encoder_output_caching_strategy.is_partial",
            source,
        )


    def test_policy_cache_residency_rules_match_backend_topology(self):
        flux_source = (
            ROOT / "scripts" / "dev" / "flux_train_network.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if policy_cache or not self.is_train_text_encoder(args):",
            flux_source,
        )

        sd3_source = (
            ROOT / "scripts" / "dev" / "sd3_train_network.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if not policy_cache and not self.is_train_text_encoder(args):",
            sd3_source,
        )

    def test_sd3_policy_routing_keeps_clip_pair_co_resident_when_either_trains(self):
        method = _load_method(
            ROOT / "scripts" / "dev" / "sd3_train_network.py",
            "Sd3NetworkTrainer",
            "configure_parameter_policy_training",
        )

        class FakeEncoder:
            def __init__(self, device="cuda"):
                self.device = device

            def to(self, device, *args, **kwargs):
                self.device = str(device)
                return self

        class FakeSession:
            def __init__(self, trained):
                self.trained = set(trained)

            def trains_component(self, component_id):
                return component_id in self.trained

            def trains_prefix(self, prefix):
                return False

        args = SimpleNamespace(cache_text_encoder_outputs=True)
        cases = (
            ({"clip_l.adapter"}, ("cuda", "cuda")),
            ({"clip_g.adapter"}, ("cuda", "cuda")),
            ({"clip_l.adapter", "clip_g.adapter"}, ("cuda", "cuda")),
            (set(), ("cpu", "cpu")),
        )
        for trained, expected_devices in cases:
            with self.subTest(trained=trained):
                trainer = SimpleNamespace()
                encoders = [
                    FakeEncoder("cuda"),
                    FakeEncoder("cuda"),
                    FakeEncoder("cpu"),
                ]
                train_unet, train_text_encoder = method(
                    trainer,
                    args,
                    FakeSession(trained),
                    encoders,
                )
                self.assertFalse(train_unet)
                self.assertEqual(
                    train_text_encoder,
                    bool(trained),
                )
                self.assertEqual(
                    (encoders[0].device, encoders[1].device),
                    expected_devices,
                )
                self.assertEqual(
                    trainer.train_clip,
                    bool(
                        {"clip_l.adapter", "clip_g.adapter"}.intersection(trained)
                    ),
                )

                get_models = _load_method(
                    ROOT / "scripts" / "dev" / "sd3_train_network.py",
                    "Sd3NetworkTrainer",
                    "get_models_for_text_encoding",
                )
                models = get_models(
                    trainer,
                    args,
                    None,
                    encoders,
                )
                if trainer.train_clip:
                    self.assertIsNotNone(models)
                    self.assertIs(models[0], encoders[0])
                    self.assertIs(models[1], encoders[1])
                    self.assertIsNone(models[2])
                    self.assertEqual(encoders[0].device, encoders[1].device)
                else:
                    self.assertIsNone(models)


@unittest.skipUnless(_RUNTIME_DEPS_AVAILABLE, "runtime dependencies are not installed")
class ParameterPolicyNetworkRuntimeSmokeTests(unittest.TestCase):
    def test_pre_routing_train_flags_use_validated_policy_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            _write_policy(
                policy_path,
                {
                    "clip_l.adapter": True,
                    "clip_g.adapter": False,
                    "t5xxl.adapter": False,
                },
            )
            flags = load_parameter_policy_train_flags(
                policy_path,
                ("clip_l.adapter", "clip_g.adapter", "t5xxl.adapter"),
            )
            self.assertEqual(
                flags,
                {
                    "clip_l.adapter": True,
                    "clip_g.adapter": False,
                    "t5xxl.adapter": False,
                },
            )
            with self.assertRaises(ParameterPolicyTrainerRuntimeError):
                load_parameter_policy_train_flags(
                    policy_path,
                    ("missing.adapter",),
                )

    def test_sd3_clip_pair_survives_real_accelerator_prepare(self):
        for train_side in ("clip_l", "clip_g"):
            with self.subTest(train_side=train_side):
                accelerator = Accelerator(cpu=True)
                try:
                    clip_l = torch.nn.Linear(4, 4)
                    clip_g = torch.nn.Linear(4, 4)
                    if train_side == "clip_l":
                        clip_g.requires_grad_(False)
                        clip_l = accelerator.prepare(clip_l)
                    else:
                        clip_l.requires_grad_(False)
                        clip_g = accelerator.prepare(clip_g)

                    l_device = next(clip_l.parameters()).device
                    g_device = next(clip_g.parameters()).device
                    self.assertEqual(l_device, g_device)

                    value = torch.randn(2, 4, device=l_device)
                    l_out = clip_l(value)
                    g_out = clip_g(value.to(g_device))
                    combined = torch.cat((l_out, g_out.to(l_device)), dim=-1)
                    self.assertEqual(tuple(combined.shape), (2, 8))
                finally:
                    accelerator.end_training()

    def test_component_lr_logging_survives_accelerate_scheduler_wrapper(self):
        policy = {
            "version": 1,
            "optimizer_profiles": {
                "main": {"type": "AdamW", "args": {}},
            },
            "components": {
                "transformer.double_stream": {
                    "train": True,
                    "optimizer_profile": "main",
                    "learning_rate": 1e-3,
                },
                "transformer.single_stream": {
                    "train": True,
                    "optimizer_profile": "main",
                    "learning_rate": 5e-4,
                },
                "transformer.modulation_norm_other": {"train": False},
                "transformer.input_conditioning": {"train": False},
                "transformer.final": {"train": False},
                "transformer.other": {"train": False},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            args = _args(policy_path)
            model = TinyFlux()
            session = create_parameter_policy_session(
                args=args,
                train_type="flux-finetune",
                roots={"transformer": model},
            )

            def get_scheduler_fix(child_args, optimizer, _num_processes):
                self.assertEqual(child_args.optimizer_type, "AdamW")
                return torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda _step: 1.0,
                )

            factory = make_legacy_scheduler_factory(
                args=args,
                get_scheduler_fix=get_scheduler_fix,
                num_processes=1,
            )
            scheduler = session.build_scheduler(factory)

            accelerator = Accelerator(cpu=True)
            model, optimizer, scheduler = accelerator.prepare(
                model,
                session.optimizer,
                scheduler,
            )
            session.audit_after_prepare(
                accelerator=accelerator,
                optimizer=optimizer,
            )

            logs = session.component_lr_logs(scheduler)
            self.assertEqual(logs["lr/transformer.double_stream"], 1e-3)
            self.assertEqual(logs["lr/transformer.single_stream"], 5e-4)
            self.assertNotIn("lr/transformer.other", logs)
            accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
