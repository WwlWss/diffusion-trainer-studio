from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


_RUNTIME_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "accelerate")
)

if _RUNTIME_DEPS_AVAILABLE:
    import torch
    from accelerate import Accelerator

    from mikazuki.parameter_policy_trainer import create_parameter_policy_session


def _args(policy_path: Path, **overrides):
    values = {
        "parameter_policy_config": str(policy_path),
        "stop_text_encoder_training": None,
        "train_text_encoder": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _policy(components):
    return {
        "version": 1,
        "optimizer_profiles": {
            "main": {"type": "AdamW", "args": {}},
        },
        "components": components,
    }


if _RUNTIME_DEPS_AVAILABLE:
    class TinyDreamUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x, cond):
            return self.proj(x + cond)

    class TinyTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x)

    class TinyEncoderContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(4, 4),
                torch.nn.Linear(4, 4),
            ])

    class TinyTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyEncoderContainer()
            self.final_layer_norm = torch.nn.LayerNorm(4)

    class TinySDXLTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_model = TinyTextModel()
            self.extra = torch.nn.Linear(4, 4)

        def forward(self, x):
            x = self.text_model.encoder.layers[0](x)
            return self.extra(x)

    class TinyFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.single_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def forward(self, x):
            return self.single_blocks[0](self.double_blocks[0](x))
else:
    TinyDreamUNet = TinyTextEncoder = TinySDXLTextEncoder = TinyFlux = object


@unittest.skipUnless(_RUNTIME_DEPS_AVAILABLE, "runtime dependencies are not installed")
class ParameterPolicyFullRuntimeSmokeTests(unittest.TestCase):
    def _write_policy(self, directory: str, policy: dict) -> Path:
        path = Path(directory) / "policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path

    def test_dreambooth_text_encoder_only_backpropagates_through_frozen_unet(self):
        components = {
            "unet.transformer": {"train": False},
            "unet.conv_resnet": {"train": False},
            "unet.norm_bias_other": {"train": False},
            "unet.base_other": {"train": False},
            "text_encoder": {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 1e-3,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy(temp_dir, _policy(components))
            unet = TinyDreamUNet()
            text_encoder = TinyTextEncoder()
            session = create_parameter_policy_session(
                args=_args(policy_path),
                train_type="sd-dreambooth",
                roots={"unet": unet, "text_encoder": text_encoder},
            )
            self.assertFalse(session.trains_prefix("unet."))
            self.assertTrue(session.trains_component("text_encoder"))

            accelerator = Accelerator(cpu=True)
            text_encoder, optimizer = accelerator.prepare(text_encoder, session.optimizer)
            session.audit_after_prepare(accelerator=accelerator, optimizer=optimizer)

            x = torch.ones(2, 4)
            cond = text_encoder(x)
            loss = unet(x, cond).sum()
            accelerator.backward(loss)

            self.assertTrue(any(p.grad is not None for p in text_encoder.parameters()))
            self.assertTrue(all(p.grad is None for p in unet.parameters()))
            accelerator.end_training()

    def test_sdxl_structural_freeze_never_enters_optimizer(self):
        components = {
            "unet.transformer": {"train": False},
            "unet.conv_resnet": {"train": False},
            "unet.norm_bias_other": {"train": False},
            "unet.base_other": {"train": False},
            "text_encoder_1": {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 2e-4,
            },
            "text_encoder_2": {"train": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy(temp_dir, _policy(components))
            unet = TinyDreamUNet()
            te1 = TinySDXLTextEncoder()
            te2 = TinySDXLTextEncoder()
            structural = tuple(te1.text_model.encoder.layers[-1].parameters()) + tuple(
                te1.text_model.final_layer_norm.parameters()
            )
            session = create_parameter_policy_session(
                args=_args(policy_path, train_text_encoder=True),
                train_type="sdxl-finetune",
                roots={
                    "unet": unet,
                    "text_encoder_1": te1,
                    "text_encoder_2": te2,
                },
                structural_frozen_parameters=structural,
            )

            owned = session.trainable_parameter_ids
            self.assertTrue(session.trains_component("text_encoder_1"))
            self.assertFalse(session.trains_component("text_encoder_2"))
            for parameter in structural:
                self.assertFalse(parameter.requires_grad)
                self.assertNotIn(id(parameter), owned)

            accelerator = Accelerator(cpu=True)
            te1, optimizer = accelerator.prepare(te1, session.optimizer)
            session.audit_after_prepare(accelerator=accelerator, optimizer=optimizer)
            accelerator.end_training()

    def test_flux_partial_component_owns_only_double_stream(self):
        components = {
            "transformer.double_stream": {
                "train": True,
                "optimizer_profile": "main",
                "learning_rate": 3e-4,
            },
            "transformer.single_stream": {"train": False},
            "transformer.modulation_norm_other": {"train": False},
            "transformer.input_conditioning": {"train": False},
            "transformer.final": {"train": False},
            "transformer.other": {"train": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy(temp_dir, _policy(components))
            flux = TinyFlux()
            session = create_parameter_policy_session(
                args=_args(policy_path),
                train_type="flux-finetune",
                roots={"transformer": flux},
            )

            self.assertTrue(session.trains_component("transformer.double_stream"))
            self.assertFalse(session.trains_component("transformer.single_stream"))
            self.assertTrue(all(p.requires_grad for p in flux.double_blocks.parameters()))
            self.assertTrue(all(not p.requires_grad for p in flux.single_blocks.parameters()))

            accelerator = Accelerator(cpu=True)
            flux, optimizer = accelerator.prepare(flux, session.optimizer)
            session.audit_after_prepare(accelerator=accelerator, optimizer=optimizer)
            self.assertEqual(
                set(session.trainable_parameter_ids),
                {id(p) for p in flux.double_blocks.parameters()},
            )
            accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
