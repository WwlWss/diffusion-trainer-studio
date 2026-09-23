from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
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


_RUNTIME_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "accelerate")
)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator

    from mikazuki.parameter_policy_trainer import (
        create_parameter_policy_session,
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
    def test_flux_policy_cache_forces_partial_strategy_before_session_exists(self):
        method = _load_method(
            ROOT / "scripts" / "dev" / "flux_train_network.py",
            "FluxNetworkTrainer",
            "get_text_encoder_outputs_caching_strategy",
            strategy_flux=SimpleNamespace(
                FluxTextEncoderOutputsCachingStrategy=_CapturedCacheStrategy,
            ),
        )
        trainer = SimpleNamespace(train_clip_l=False, train_t5xxl=False)
        args = SimpleNamespace(
            cache_text_encoder_outputs=True,
            parameter_policy_config="policy.json",
            cache_text_encoder_outputs_to_disk=False,
            text_encoder_batch_size=1,
            skip_cache_check=False,
            apply_t5_attn_mask=False,
        )
        strategy = method(trainer, args)
        self.assertTrue(strategy.is_partial)

        args.parameter_policy_config = ""
        strategy = method(trainer, args)
        self.assertFalse(strategy.is_partial)

        trainer.train_clip_l = True
        strategy = method(trainer, args)
        self.assertTrue(strategy.is_partial)

    def test_sd3_policy_cache_forces_partial_strategy_before_session_exists(self):
        method = _load_method(
            ROOT / "scripts" / "dev" / "sd3_train_network.py",
            "Sd3NetworkTrainer",
            "get_text_encoder_outputs_caching_strategy",
            strategy_sd3=SimpleNamespace(
                Sd3TextEncoderOutputsCachingStrategy=_CapturedCacheStrategy,
            ),
        )
        trainer = SimpleNamespace(
            train_clip=False,
            train_t5xxl=False,
        )
        args = SimpleNamespace(
            cache_text_encoder_outputs=True,
            parameter_policy_config="policy.json",
            cache_text_encoder_outputs_to_disk=False,
            text_encoder_batch_size=1,
            skip_cache_check=False,
            apply_lg_attn_mask=False,
            apply_t5_attn_mask=False,
        )
        strategy = method(trainer, args)
        self.assertTrue(strategy.is_partial)

        args.parameter_policy_config = ""
        strategy = method(trainer, args)
        self.assertFalse(strategy.is_partial)

        trainer.train_clip = True
        strategy = method(trainer, args)
        self.assertTrue(strategy.is_partial)

    def test_partial_cache_keeps_dataset_tokenization_available(self):
        source = (
            ROOT / "scripts" / "dev" / "library" / "train_util.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "self.text_encoder_output_caching_strategy is None or "
            "self.text_encoder_output_caching_strategy.is_partial",
            source,
        )


@unittest.skipUnless(_RUNTIME_DEPS_AVAILABLE, "runtime dependencies are not installed")
class ParameterPolicyNetworkRuntimeSmokeTests(unittest.TestCase):
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
