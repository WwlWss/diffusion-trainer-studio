"""Generate new, external SD3.5 Medium Component R0 evidence inputs.

This produces a canonical seven-Component AdamW Policy and a two-task
fresh/resume manifest.  It does not execute training, import Torch or enable
full-BF16 qualification.  Input data must contain one effective batch/epoch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy import serialize_parameter_policy, validate_parameter_policy
from tools.run_sd3_runtime_gpu_evidence import _outside_repo


def make_policy() -> dict:
    trained = {
        "mmdit.attention.adapter",
        "mmdit.mlp.adapter",
        "mmdit.modulation_norm.adapter",
    }
    policy = {
        "version": 1,
        "optimizer_profiles": {
            "baseline": {
                "type": "AdamW",
                "args": {"weight_decay": 0.01},
            },
        },
        "components": {
            name: (
                {
                    "train": True, "optimizer_profile": "baseline",
                    "learning_rate": 0.0001,
                }
                if name in trained else {"train": False}
            )
            for name in get_model_component_profile("sd3-lora").components
        },
    }
    return validate_parameter_policy(policy)


def generate(
    *,
    model: Path,
    dataset: Path,
    work_dir: Path,
    external_models: dict[str, Path],
    cache_latents: bool,
    cache_text_encoder_outputs: bool,
) -> dict:
    model = model.expanduser().resolve(strict=True)
    dataset = dataset.expanduser().resolve(strict=True)
    work_dir = _outside_repo(str(work_dir), name="R0 work_dir")
    if not model.is_file() or not dataset.is_dir():
        raise ValueError("R0 requires a real SD3.5 Medium file and training data directory")
    if work_dir.exists():
        raise ValueError(f"R0 work_dir must not exist (no stale evidence): {work_dir}")
    for key, path in external_models.items():
        if not path.expanduser().resolve(strict=True).is_file():
            raise ValueError(f"{key} must be a real model file")
    work_dir.mkdir(parents=True, exist_ok=False)

    policy = make_policy()
    policy_path = work_dir / "component-policy.json"
    _unused, canonical_text = serialize_parameter_policy(policy)
    policy_path.write_text(canonical_text, encoding="utf-8")

    output_name = "sd3c1-r0"
    fresh_dir, resume_dir = work_dir / "fresh", work_dir / "resume"
    common = [
        "scripts/dev/sd3_train_network.py",
        "--pretrained_model_name_or_path", str(model),
        "--train_data_dir", str(dataset),
        "--parameter_policy_config", str(policy_path),
        "--output_name", output_name,
        "--max_train_steps", "2",
        "--save_every_n_steps", "1",
        "--save_state",
        "--mixed_precision", "bf16",
        "--optimizer_type", "AdamW",
        "--network_module", "networks.lora_sd3",
        "--network_train_unet_only",
        "--network_dim", "4",
        "--network_alpha", "4",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--resolution", "512",
        "--save_model_as", "safetensors",
        "--lr_scheduler", "constant",
        "--seed", "123456",
        "--max_data_loader_n_workers", "0",
    ]
    for flag in ("--clip_l", "--clip_g", "--t5xxl", "--vae"):
        if flag in external_models:
            common.extend((flag, str(external_models[flag].expanduser().resolve(strict=True))))
    if cache_latents:
        common.append("--cache_latents")
    if cache_text_encoder_outputs:
        common.append("--cache_text_encoder_outputs")
    payload = {
        "schema": "dts.sd3-component-baseline-manifest",
        "version": 1,
        "fresh_command": [*common, "--output_dir", str(fresh_dir)],
        "resume_command": [
            *common,
            "--resume", str(fresh_dir / f"{output_name}-step00000001-state"),
            "--output_dir", str(resume_dir),
        ],
        "fresh_checkpoint_dir": str(fresh_dir / f"{output_name}-step00000001-state"),
        "resume_checkpoint_dir": str(resume_dir / f"{output_name}-step00000002-state"),
        "fresh_adapter_path": str(fresh_dir / f"{output_name}-step00000001.safetensors"),
        "resume_adapter_path": str(resume_dir / f"{output_name}-step00000002.safetensors"),
    }
    (work_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "manifest": str(work_dir / "manifest.json"),
        "policy": str(policy_path),
        "evidence": str(work_dir / "result.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    for flag in ("clip-l", "clip-g", "t5xxl", "vae"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--cache-latents", action="store_true")
    parser.add_argument("--cache-text-encoder-outputs", action="store_true")
    args = parser.parse_args(argv)
    external = {}
    for key, option in (("--clip_l", "clip_l"), ("--clip_g", "clip_g"), ("--t5xxl", "t5xxl"), ("--vae", "vae")):
        value = getattr(args, option)
        if value is not None:
            external[key] = value
    try:
        result = generate(
            model=args.model, dataset=args.data_dir, work_dir=args.work_dir,
            external_models=external,
            cache_latents=args.cache_latents,
            cache_text_encoder_outputs=args.cache_text_encoder_outputs,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"SD3C1 input generator failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
