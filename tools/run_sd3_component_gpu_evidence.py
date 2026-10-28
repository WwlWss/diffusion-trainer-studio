"""D1-SD3C1: exact-head SD3.5 Medium Component mixed-BF16 GPU baseline.

This is a NON-QUALIFICATION R0 runner.  It reuses the production SD3B
fresh/resume evidence lifecycle and checks the actual Parameter Policy v2
checkpoint manifest.  It NEVER enables or qualifies Component full_bf16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mikazuki.model_component_profiles import get_model_component_profile
from mikazuki.parameter_policy import serialize_parameter_policy, validate_parameter_policy
from mikazuki.sd3_architecture_contract import validate_sd3_mmdit_checkpoint
from tools import run_sd3_runtime_gpu_evidence as sd3b
from tools.audit_sd3_architecture import read_safetensors_shapes

SCHEMA = "dts.sd3-component-baseline-manifest"
VERSION = 1
EVIDENCE_SCHEMA = "dts.sd3-component-baseline-evidence"
MMDIT_REQUIRED = frozenset({"mmdit.attention.adapter", "mmdit.mlp.adapter"})
FROZEN_REQUIRED = frozenset({
    "mmdit.other.adapter", "clip_l.adapter", "clip_g.adapter", "t5xxl.adapter",
})


class SD3ComponentBaselineError(RuntimeError):
    pass


def _read_policy(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        canonical = validate_parameter_policy(raw)
        _canonical_path, canonical_text = serialize_parameter_policy(canonical)
    except (OSError, ValueError, TypeError) as exc:
        raise SD3ComponentBaselineError(f"Invalid SD3 Component Policy {path}: {exc}") from exc

    expected_components = set(get_model_component_profile("sd3-lora").components)
    if set(canonical["components"]) != expected_components:
        raise SD3ComponentBaselineError("SD3 Component Policy must include all seven canonical components")
    trained = {
        name for name, route in canonical["components"].items()
        if route["train"]
    }
    if not MMDIT_REQUIRED.issubset(trained):
        raise SD3ComponentBaselineError("R0 must train MMDiT attention and MLP adapters")
    if trained & FROZEN_REQUIRED:
        raise SD3ComponentBaselineError("R0 requires MMDiT Other and all three Text Encoders frozen")

    profiles = canonical["optimizer_profiles"]
    if len(profiles) != 1:
        raise SD3ComponentBaselineError("R0 requires exactly one AdamW Optimizer Profile")
    name, profile = next(iter(profiles.items()))
    if profile["type"] != "AdamW":
        raise SD3ComponentBaselineError("R0 is an AdamW-only Component baseline")
    for component in trained:
        route = canonical["components"][component]
        if route["optimizer_profile"] != name or route.get("fallback_optimizer_profile") is not None:
            raise SD3ComponentBaselineError("R0 requires explicit AdamW primary ownership without fallback")

    digest = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    return canonical, digest


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before loading Torch or executing a production trainer."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise SD3ComponentBaselineError("Unknown D1-SD3C1 baseline manifest")
    if any(key in payload for key in ("qualifies_full_bf16", "backend_qualification_eligible", "qualification_override")):
        raise SD3ComponentBaselineError("Manifest cannot declare or override production qualification")

    # Preserve the previously accepted SD3B lifecycle checks rather than
    # reimplementing a second optimizer-step / resume command parser.
    reference = dict(payload)
    reference["schema"] = sd3b.SCHEMA
    reference["version"] = sd3b.VERSION
    try:
        validated = sd3b.validate_manifest(reference)
    except sd3b.SD3RuntimeEvidenceError as exc:
        raise SD3ComponentBaselineError(str(exc)) from exc

    fresh, resume = validated["fresh"], validated["resume"]
    for label, options in (("fresh", fresh), ("resume", resume)):
        for key, expected in (
            ("--network_dim", "4"),
            ("--network_alpha", "4"),
            ("--train_batch_size", "1"),
            ("--gradient_accumulation_steps", "1"),
            ("--resolution", "512"),
        ):
            if options.get(key) != expected:
                raise SD3ComponentBaselineError(f"{label} R0 requires {key}={expected}")
        for key in (
            "--network_args", "--network_weights", "--base_weights",
            "--optimizer_args", "--network_train_text_encoder_only",
            "--fused_optimizer_groups", "--unsloth_offload_checkpointing",
        ):
            if key in options:
                raise SD3ComponentBaselineError(f"{label} R0 forbids unreviewed option {key}")
        if options.get("--save_model_as", "safetensors") != "safetensors":
            raise SD3ComponentBaselineError("R0 requires actual safetensors LoRA weights")
        if "--parameter_policy_config" not in options or not isinstance(options["--parameter_policy_config"], str):
            raise SD3ComponentBaselineError(f"{label} must explicitly select Component Policy")
        if not isinstance(options.get("--train_data_dir"), str):
            raise SD3ComponentBaselineError(f"{label} must specify a real --train_data_dir")

    policy_path = sd3b._outside_repo(fresh["--parameter_policy_config"], name="Parameter Policy")
    if not policy_path.is_file():
        raise SD3ComponentBaselineError(f"Missing Parameter Policy sidecar: {policy_path}")
    if policy_path != sd3b._outside_repo(resume["--parameter_policy_config"], name="Resume Parameter Policy"):
        raise SD3ComponentBaselineError("Fresh/Resume must use one immutable Policy path")
    dataset = Path(fresh["--train_data_dir"]).expanduser().resolve(strict=False)
    if not dataset.is_dir():
        raise SD3ComponentBaselineError(f"Missing training dataset directory: {dataset}")
    if dataset != Path(resume["--train_data_dir"]).expanduser().resolve(strict=False):
        raise SD3ComponentBaselineError("Fresh/Resume must use the same training dataset")
    canonical, policy_hash = _read_policy(policy_path)
    return {
        **validated,
        "policy": canonical,
        "policy_hash": policy_hash,
        "policy_path": policy_path,
        "policy_bytes_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "dataset": dataset,
    }


def _assert_unchanged_inputs(validated: dict[str, Any]) -> None:
    if hashlib.sha256(validated["policy_path"].read_bytes()).hexdigest() != validated["policy_bytes_sha256"]:
        raise SD3ComponentBaselineError("Parameter Policy changed during the evidence run")


def _check_architecture(model: Path) -> dict[str, Any]:
    shapes = read_safetensors_shapes(model)
    audited = validate_sd3_mmdit_checkpoint(shapes, require_loader_compatible_root=True)
    report = audited.architecture.report()
    if report["variant"] != "sd35-medium" or not audited.loader_prefix_compatible:
        raise SD3ComponentBaselineError("R0 permits only DTS-loader-compatible SD3.5 Medium MMDiT")
    return report


def verify_component_checkpoint(
    evidence: dict[str, Any],
    *,
    policy: dict[str, Any],
    policy_hash: str,
    expected_step: int,
) -> dict[str, Any]:
    """Validate both the SD3 cursor and actual CompositeOptimizer ownership."""
    policy_manifest = evidence.get("parameter_policy_manifest")
    if not isinstance(policy_manifest, dict) or policy_manifest.get("version") != 2:
        raise SD3ComponentBaselineError("R0 requires Parameter Policy checkpoint Manifest v2")
    if policy_manifest.get("train_type") != "sd3-lora" or policy_manifest.get("policy_hash") != policy_hash:
        raise SD3ComponentBaselineError("R0 checkpoint train_type / canonical Policy hash mismatch")
    expected_train = sorted(
        name for name, route in policy["components"].items() if route["train"]
    )
    expected_frozen = sorted(
        name for name, route in policy["components"].items() if not route["train"]
    )
    if policy_manifest.get("trainable_components") != expected_train:
        raise SD3ComponentBaselineError("R0 trainable Component ownership mismatch")
    if policy_manifest.get("frozen_components") != expected_frozen:
        raise SD3ComponentBaselineError("R0 frozen Component ownership mismatch")
    if (
        type(policy_manifest.get("trainable_parameter_tensors")) is not int
        or policy_manifest["trainable_parameter_tensors"] <= 0
        or not isinstance(policy_manifest.get("runtime_topology_fingerprint"), str)
        or not policy_manifest["runtime_topology_fingerprint"]
    ):
        raise SD3ComponentBaselineError("R0 lacks physical parameter / topology evidence")

    name = next(iter(policy["optimizer_profiles"]))
    expected_profiles = {name: "AdamW"}
    if policy_manifest.get("optimizer_profiles") != expected_profiles:
        raise SD3ComponentBaselineError("R0 checkpoint does not own the expected AdamW profile")
    optimizers = policy_manifest.get("optimizers")
    if (
        not isinstance(optimizers, list)
        or len(optimizers) != 1
        or optimizers[0].get("profile_name") != name
        or optimizers[0].get("optimizer_type") != "AdamW"
        or not optimizers[0].get("topology_fingerprint")
    ):
        raise SD3ComponentBaselineError("R0 CompositeOptimizer child topology is invalid")
    if any(k in policy_manifest for k in ("execution_identity", "execution_signature")):
        raise SD3ComponentBaselineError("Mixed-BF16 R0 must not be labeled full-BF16 execution")
    if not isinstance(policy_manifest.get("scheduler_identity"), dict) or not policy_manifest.get("scheduler_signature"):
        raise SD3ComponentBaselineError("R0 checkpoint lacks CompositeLRScheduler identity")

    progress = evidence["progress"]
    if (
        progress.get("completed_optimizer_steps") != expected_step
        or progress.get("gradient_accumulation_steps") != 1
        or progress.get("num_processes") != 1
        or progress.get("dataloader_batches") != 1
    ):
        raise SD3ComponentBaselineError("R0 must save at a single-process Epoch boundary, accum=1")
    return policy_manifest


def compare_component_checkpoints(first: dict[str, Any], second: dict[str, Any]) -> None:
    one, two = first["parameter_policy_manifest"], second["parameter_policy_manifest"]
    if one != two:
        raise SD3ComponentBaselineError("R0 Component/optimizer/scheduler manifest changed across resume")
    if first["forward_semantics"] != second["forward_semantics"]:
        raise SD3ComponentBaselineError("SD3.5 extra-attention semantics changed across resume")
    if first["optimizer_state_sha256"].keys() != second["optimizer_state_sha256"].keys():
        raise SD3ComponentBaselineError("Optimizer checkpoint file topology changed")
    if first["optimizer_state_sha256"] == second["optimizer_state_sha256"]:
        raise SD3ComponentBaselineError("Optimizer state did not advance after Resume")



def summarize_component_weight_updates(
    changed_flags: dict[str, bool], *, policy: dict[str, Any],
) -> dict[str, Any]:
    """Torch-free ownership/update contract, unit-testable on host CI."""
    group_map = {
        "mmdit.attention.adapter": "attention",
        "mmdit.mlp.adapter": "mlp",
        "mmdit.modulation_norm.adapter": "modulation",
    }
    required = {
        group for component, group in group_map.items()
        if policy["components"][component]["train"]
    }
    changed = dict.fromkeys(("attention", "mlp", "modulation", "other"), 0)
    total = dict.fromkeys(changed, 0)
    for name, advanced in changed_flags.items():
        if not name.endswith(".weight"):
            continue
        if type(advanced) is not bool:
            raise SD3ComponentBaselineError(f"R0 has invalid weight advance flag: {name}")
        if not name.startswith("lora_unet_"):
            raise SD3ComponentBaselineError(
                f"R0 adapter includes non-MMDiT trainable weights: {name}"
            )
        lower = name.lower()
        if "_attn_" in lower or "_attn2_" in lower:
            group = "attention"
        elif "_mlp_" in lower:
            group = "mlp"
        elif "_adaln_modulation_" in lower:
            group = "modulation"
        else:
            group = "other"
        total[group] += 1
        if advanced:
            changed[group] += 1
    if any(changed[group] <= 0 for group in required):
        raise SD3ComponentBaselineError(
            f"R0 trained MMDiT groups did not all advance: required={sorted(required)} "
            f"changed={changed}, totals={total}"
        )
    return {"changed_by_group": changed, "total_by_group": total}


def compare_component_adapter_weights(
    fresh_path: Path, resumed_path: Path, *, policy: dict[str, Any],
) -> dict[str, Any]:
    """Require actual MMDiT changes per trained group, not just any adapter."""
    result = sd3b._compare_adapter_weights(fresh_path, resumed_path)
    from safetensors import safe_open
    import torch

    with safe_open(str(fresh_path), framework="pt", device="cpu") as first, (
        safe_open(str(resumed_path), framework="pt", device="cpu")
    ) as second:
        changed_flags = {
            name: not torch.equal(first.get_tensor(name), second.get_tensor(name))
            for name in first.keys()
            if name.endswith(".weight")
        }
    return {
        **result,
        **summarize_component_weight_updates(changed_flags, policy=policy),
    }


def run(payload: dict[str, Any], *, expected_commit: str, output: Path) -> dict[str, Any]:
    validated = validate_manifest(payload)
    sd3b._check_head(expected_commit)
    _assert_unchanged_inputs(validated)
    architecture = _check_architecture(validated["base_model"])
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise SD3ComponentBaselineError("R0 is single-process only")

    import torch
    if not torch.cuda.is_available():
        raise SD3ComponentBaselineError("R0 requires a real CUDA device")
    out = sd3b._outside_repo(str(output), name="R0 evidence JSON")
    if out.exists():
        raise SD3ComponentBaselineError("R0 evidence output must be new; old results cannot be reused")
    out.parent.mkdir(parents=True, exist_ok=True)
    proof: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "version": VERSION,
        "source_commit": expected_commit,
        "status": "incomplete",
        "scope": "sd35-medium-component-mixed-bf16-adamw-r0",
        "architecture": architecture,
        "policy_hash": validated["policy_hash"],
        "policy_bytes_sha256": validated["policy_bytes_sha256"],
        "cuda_device": torch.cuda.get_device_name(),
        "qualifies_full_bf16": False,
        "backend_qualification_eligible": False,
        "production_qualification_mutated": False,
    }
    try:
        _assert_unchanged_inputs(validated)
        sd3b._execute(payload["fresh_command"], log_path=out.with_name(out.stem + "-fresh.log"))
        fresh = sd3b._checkpoint_evidence(validated["paths"]["fresh_checkpoint_dir"], step=1)
        verify_component_checkpoint(
            fresh, policy=validated["policy"], policy_hash=validated["policy_hash"], expected_step=1,
        )
        sd3b._check_head(expected_commit)
        _assert_unchanged_inputs(validated)

        sd3b._execute(payload["resume_command"], log_path=out.with_name(out.stem + "-resume.log"))
        resumed = sd3b._checkpoint_evidence(validated["paths"]["resume_checkpoint_dir"], step=2)
        verify_component_checkpoint(
            resumed, policy=validated["policy"], policy_hash=validated["policy_hash"], expected_step=2,
        )
        compare_component_checkpoints(fresh, resumed)
        proof["adapter_updates"] = compare_component_adapter_weights(
            validated["paths"]["fresh_adapter_path"],
            validated["paths"]["resume_adapter_path"],
            policy=validated["policy"],
        )
        sd3b._check_head(expected_commit)
        _assert_unchanged_inputs(validated)
        proof["fresh"] = fresh
        proof["resumed"] = resumed
        proof["status"] = "pass"
    except Exception as exc:
        proof["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        out.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = sd3b._read_json(args.manifest)
        validated = validate_manifest(payload)
        sd3b._check_head(args.expected_commit)
        _assert_unchanged_inputs(validated)
        architecture = _check_architecture(validated["base_model"])
        if args.preflight:
            print(f"SD3C1 R0 preflight PASS: {architecture['variant']}; Full-BF16 remains pending")
        else:
            proof = run(payload, expected_commit=args.expected_commit, output=args.output)
            print(f"SD3C1 Component baseline: {proof['status']}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"SD3C1 R0 FAIL: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
