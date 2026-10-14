"""Exact-head, non-qualification SD3.5 Medium LoRA fresh/resume evidence runner.

This tool never edits the production backend-feature qualification table.
Only single-process, mixed BF16, AdamW LoRA reference paths are accepted.
It executes the *real* SD3 network trainer via a new Python subprocess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dts.sd3b-runtime-evidence"
VERSION = 1
TRIGGER = ROOT / "scripts" / "dev" / "sd3_train_network.py"
BLOCKED_OPTIONS = frozenset({
    "--full_bf16", "--full_fp16", "--fp8_base", "--fp8_base_unet",
    "--deepspeed", "--blocks_to_swap", "--cpu_offload_checkpointing",
    "--torch_compile", "--fused_backward_pass", "--blockwise_fused_optimizers",
    "--initial_epoch", "--initial_step", "--max_train_epochs",
    "--skip_until_initial_step", "--resume_from_huggingface",
    "--config_file", "--gradient_checkpointing",  # first reference is isolated
})


class SD3RuntimeEvidenceError(RuntimeError):
    pass


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _check_head(expected: str) -> None:
    if _git("rev-parse", "HEAD") != expected:
        raise SD3RuntimeEvidenceError("SD3B CUDA evidence must run on exact HEAD")
    if _git("status", "--porcelain"):
        raise SD3RuntimeEvidenceError("SD3B CUDA evidence requires a clean working tree")


def _outside_repo(raw: str, *, name: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    if not path.is_absolute() or path == ROOT or ROOT in path.parents:
        raise SD3RuntimeEvidenceError(f"{name} must be absolute and outside the source repository")
    return path


def _read_options(command: list[str]) -> dict[str, str | bool]:
    if not command or not all(isinstance(x, str) and x for x in command):
        raise SD3RuntimeEvidenceError("Trainer command must be a nonempty string array")
    script = (ROOT / command[0]).resolve(strict=False)
    if script != TRIGGER:
        raise SD3RuntimeEvidenceError("Only the actual SD3 NetworkTrainer is an accepted entrypoint")
    result: dict[str, str | bool] = {}
    i = 1
    while i < len(command):
        token = command[i]
        if not token.startswith("--"):
            raise SD3RuntimeEvidenceError(f"Unexpected argument {token!r}")
        key, delimiter, inline = token.partition("=")
        if key in result:
            raise SD3RuntimeEvidenceError(f"Duplicate trainer option {key!r}")
        if delimiter:
            result[key] = inline
        elif i + 1 < len(command) and not command[i + 1].startswith("--"):
            result[key] = command[i + 1]
            i += 1
        else:
            result[key] = True
        i += 1
    return result


def _required(options: dict[str, str | bool], key: str, expected: str) -> None:
    if options.get(key) != expected:
        raise SD3RuntimeEvidenceError(f"SD3B reference requires {key}={expected!r}")


def _strip_runtime_options(options: dict[str, str | bool]) -> dict[str, str | bool]:
    return {k: v for k, v in options.items() if k not in ("--resume", "--output_dir")}


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise SD3RuntimeEvidenceError("Unrecognized SD3B runtime evidence manifest")
    for key in ("fresh_command", "resume_command"):
        if not isinstance(payload.get(key), list):
            raise SD3RuntimeEvidenceError(f"Manifest requires {key}")
    fresh = _read_options(payload["fresh_command"])
    resume = _read_options(payload["resume_command"])
    if _strip_runtime_options(fresh) != _strip_runtime_options(resume):
        raise SD3RuntimeEvidenceError("Fresh/Resume training inputs differ")
    for label, options in (("fresh", fresh), ("resume", resume)):
        bad = sorted(set(options) & BLOCKED_OPTIONS)
        if bad:
            raise SD3RuntimeEvidenceError(f"{label} has unqualified feature(s): {bad!r}")
        _required(options, "--max_train_steps", "2")
        _required(options, "--save_every_n_steps", "1")
        _required(options, "--mixed_precision", "bf16")
        _required(options, "--optimizer_type", "AdamW")
        _required(options, "--network_module", "networks.lora_sd3")
        if options.get("--save_state") is not True:
            raise SD3RuntimeEvidenceError(f"{label} must use --save_state")
        if options.get("--network_train_unet_only") is not True:
            raise SD3RuntimeEvidenceError(f"{label} must train MMDiT LoRA only")
        if options.get("--gradient_accumulation_steps", "1") not in ("1", "2"):
            raise SD3RuntimeEvidenceError("SD3B reference accumulation must be 1 or 2")
    if "--resume" in fresh or not isinstance(resume.get("--resume"), str):
        raise SD3RuntimeEvidenceError("Only the second phase may use --resume")
    if not isinstance(fresh.get("--output_dir"), str) or not isinstance(resume.get("--output_dir"), str):
        raise SD3RuntimeEvidenceError("Both phases require explicit output directories")
    keys = (
        "fresh_checkpoint_dir", "resume_checkpoint_dir",
        "fresh_adapter_path", "resume_adapter_path",
    )
    paths: dict[str, Path] = {}
    for k in keys:
        if not isinstance(payload.get(k), str):
            raise SD3RuntimeEvidenceError(f"Missing evidence output path {k}")
        paths[k] = _outside_repo(payload[k], name=k)
    for label in ("fresh", "resume"):
        out = _outside_repo(str((fresh if label == "fresh" else resume)["--output_dir"]), name=f"{label} output")
        if out.exists():
            raise SD3RuntimeEvidenceError(f"{label} output already exists: {out}")
        if not paths[f"{label}_checkpoint_dir"].is_relative_to(out):
            raise SD3RuntimeEvidenceError(f"{label} checkpoint must be under its output")
        if not paths[f"{label}_adapter_path"].is_relative_to(out):
            raise SD3RuntimeEvidenceError(f"{label} adapter must be under its output")
        paths[label + "_output"] = out
    if paths["fresh_output"] == paths["resume_output"]:
        raise SD3RuntimeEvidenceError("Fresh and Resume output directories must differ")
    if Path(str(resume["--resume"])).expanduser().resolve(strict=False) != paths["fresh_checkpoint_dir"]:
        raise SD3RuntimeEvidenceError("Resume must point to the exact Step-1 fresh state")
    base = fresh.get("--pretrained_model_name_or_path")
    if not isinstance(base, str) or not Path(base).is_file():
        raise SD3RuntimeEvidenceError("The real SD3.5 Medium model must exist locally")
    for key in ("--clip_l", "--clip_g", "--t5xxl", "--vae"):
        if key in fresh and (not isinstance(fresh[key], str) or not Path(fresh[key]).is_file()):
            raise SD3RuntimeEvidenceError(f"Missing external SD3 input {key}")
    return {"paths": paths, "fresh": fresh, "resume": resume, "base_model": Path(base)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SD3RuntimeEvidenceError(f"Missing/corrupt CUDA evidence {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SD3RuntimeEvidenceError(f"Evidence must be an object: {path}")
    return data


def _checkpoint_evidence(path: Path, *, step: int) -> dict[str, Any]:
    progress = _read_json(path / "dts_sd3_resume_progress.json")
    state = _read_json(path / "train_state.json")
    semantics = _read_json(path / "dts_sd3_forward_semantics.json")
    if progress.get("completed_optimizer_steps") != step or state.get("current_step") != step:
        raise SD3RuntimeEvidenceError(f"Expected optimizer step {step} at {path}")
    if progress.get("current_epoch") != state.get("current_epoch"):
        raise SD3RuntimeEvidenceError("Checkpoint progress and train_state disagree")
    if semantics.get("attention_contract") != "extra-attention-head-dim-v1":
        raise SD3RuntimeEvidenceError("Checkpoint does not prove corrected SD3.5 extra-attention")
    if semantics.get("head_dim") != 64 or semantics.get("extra_attention_layers") != list(range(13)):
        raise SD3RuntimeEvidenceError("SD3.5 Medium forward semantics differ from reference")
    optimizer_paths = sorted(path.glob("optimizer*.bin"))
    if not optimizer_paths:
        raise SD3RuntimeEvidenceError(f"Missing Accelerate optimizer state at {path}")
    optimizer_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in optimizer_paths
    }
    policy = path / "dts_parameter_policy_manifest.json"
    policy_manifest = _read_json(policy) if policy.is_file() else None
    if policy_manifest is not None and policy_manifest.get("version") != 2:
        raise SD3RuntimeEvidenceError("Component checkpoint manifest must remain v2")
    return {
        "progress": progress,
        "train_state": state,
        "forward_semantics": semantics,
        "optimizer_state_sha256": optimizer_hashes,
        "parameter_policy_manifest": policy_manifest,
    }


def _compare_adapter_weights(fresh_path: Path, resumed_path: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise SD3RuntimeEvidenceError("Torch and Safetensors are required for adapter evidence") from exc
    if not fresh_path.is_file() or not resumed_path.is_file():
        raise SD3RuntimeEvidenceError("One or both trained LoRA adapter files are missing")
    with safe_open(fresh_path, framework="pt", device="cpu") as base, safe_open(resumed_path, framework="pt", device="cpu") as result:
        names = sorted(base.keys())
        if not names or names != sorted(result.keys()):
            raise SD3RuntimeEvidenceError("Fresh and resumed LoRA target keys differ")
        changed, max_delta = 0, 0.0
        for name in names:
            a, b = base.get_tensor(name), result.get_tensor(name)
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise SD3RuntimeEvidenceError(f"Nonfinite or shape-mismatched adapter tensor: {name}")
            delta = (a.float() - b.float()).abs().max().item()
            if delta > 0:
                changed += 1
                max_delta = max(max_delta, delta)
        if changed == 0:
            raise SD3RuntimeEvidenceError("Resumed optimizer step did not update LoRA weights")
    return {"checked_tensors": len(names), "changed_tensors": changed, "max_abs_delta": max_delta}


def _execute(command: list[str], *, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code != 0:
        raise SD3RuntimeEvidenceError(
            f"Production trainer exited {code}; inspect {log_path}"
        )


def run(payload: dict[str, Any], *, expected_commit: str, output: Path) -> dict[str, Any]:
    validated = validate_manifest(payload)
    _check_head(expected_commit)
    import torch
    if not torch.cuda.is_available():
        raise SD3RuntimeEvidenceError("A real CUDA device is required for SD3B evidence")
    output = _outside_repo(str(output), name="Evidence JSON")
    output.parent.mkdir(parents=True, exist_ok=True)
    proof: dict[str, Any] = {
        "schema": "dts.sd3b-runtime-gpu-result",
        "version": 1,
        "source_commit": expected_commit,
        "cuda_device": torch.cuda.get_device_name(),
        "qualifies_full_bf16": False,
        "production_qualification_mutated": False,
        "base_model_path": str(validated["base_model"]),
        "base_model_bytes": validated["base_model"].stat().st_size,
        "status": "incomplete",
    }
    try:
        _execute(payload["fresh_command"], log_path=output.with_name(output.stem + "-fresh.log"))
        fresh = _checkpoint_evidence(validated["paths"]["fresh_checkpoint_dir"], step=1)
        _check_head(expected_commit)
        _execute(payload["resume_command"], log_path=output.with_name(output.stem + "-resume.log"))
        resumed = _checkpoint_evidence(validated["paths"]["resume_checkpoint_dir"], step=2)
        _check_head(expected_commit)
        if fresh["forward_semantics"] != resumed["forward_semantics"]:
            raise SD3RuntimeEvidenceError("Forward semantics drifted over resume")
        if fresh["progress"]["gradient_accumulation_steps"] != resumed["progress"]["gradient_accumulation_steps"]:
            raise SD3RuntimeEvidenceError("Gradient accumulation changed over resume")
        if fresh["optimizer_state_sha256"].keys() != resumed["optimizer_state_sha256"].keys():
            raise SD3RuntimeEvidenceError("Optimizer topology changed over resume")
        if fresh["optimizer_state_sha256"] == resumed["optimizer_state_sha256"]:
            raise SD3RuntimeEvidenceError("Optimizer states have not advanced after resume")
        if fresh["parameter_policy_manifest"] != resumed["parameter_policy_manifest"]:
            raise SD3RuntimeEvidenceError("Parameter Policy manifest changed over resume")
        proof["adapter_updates"] = _compare_adapter_weights(
            validated["paths"]["fresh_adapter_path"],
            validated["paths"]["resume_adapter_path"],
        )
        proof["fresh"] = fresh
        proof["resumed"] = resumed
        proof["status"] = "pass"
    finally:
        output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate manifest and exact source only; no CUDA run")
    args = parser.parse_args(argv)
    try:
        payload = _read_json(args.manifest)
        if args.preflight:
            summary = validate_manifest(payload)
            _check_head(args.expected_commit)
            print(f"SD3B evidence preflight PASS: {summary['base_model']}")
        else:
            proof = run(payload, expected_commit=args.expected_commit, output=args.output)
            print(f"SD3B evidence: {proof['status']}")
    except (SD3RuntimeEvidenceError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"SD3B evidence FAIL: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
