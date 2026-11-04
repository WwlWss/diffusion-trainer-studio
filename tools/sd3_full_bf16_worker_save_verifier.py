"""C4-B real same-Step Save verifier, with a hard CPU diagnostic boundary.

The module imports Torch lazily, and never changes production train, LoRA
serialization, Optimizer, Scheduler, or original Accelerator.save_state.
A Host MUST NOT unpickle untrusted checkpoint bytes; the only weights_only
torch.load here is an in-process check of the just-produced trusted files.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Any, Mapping

from tools import sd3_full_bf16_evidence_support as host


def _fail(message: str) -> None:
    raise host.FullBF16EvidenceError("C4-B saved-state verification: " + message)


def _raw_tensor_sha256(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
        _fail("Serialized LoRA source is not an actual materialized Tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.bfloat16:
        _fail("Serialized LoRA Tensor must be a real strided BF16 tensor")
    # Process exactly one Tensor at a time. Never persist full GPU model clones.
    cpu = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def snapshot_serialized_tensor_inventory(
    network: Any, *, diagnostic_cpu: bool = False,
) -> dict[str, dict[str, Any]]:
    """Full actual LoRAModule down/up/alpha inventory (not just parameters).

    In production require EXACT actual LoRANetwork class, canonical 263 target
    modules, their shapes and BF16 alpha=4. In tiny pinned CPU diagnostics
    return the actual six-LoRAModule 18-Tensor inventory with no authority.
    """
    import torch

    modules = getattr(network, "unet_loras", None)
    if not isinstance(modules, (list, tuple)) or not modules:
        _fail("Prepared LoRA Network has no actual LoRAModule targets")
    te = getattr(network, "text_encoder_loras", ())
    if te:
        _fail("Unexpected text encoder LoRA in Unet-only reference")
    if getattr(network, "split_qkv", None) is not False:
        _fail("Reference LoRA Network must have split_qkv=False")
    if not diagnostic_cpu:
        from networks.lora_sd3 import LoRANetwork

        if type(network) is not LoRANetwork:
            _fail("Unreviewed production LoRA Network class")
        expected_targets = host.build_sd35_medium_lora_target_spec()
        if len(modules) != 263 or len(expected_targets) != 263:
            _fail("Full reference has no exact 263 target LoRAModules")
    else:
        expected_targets = None

    result: dict[str, dict[str, Any]] = {}
    actual_names: set[str] = set()
    for module in modules:
        name = getattr(module, "lora_name", None)
        if (not isinstance(name, str)
                or re.fullmatch(r"lora_unet_[A-Za-z0-9_]+", name) is None
                or name in actual_names):
            _fail("Duplicate or unrecognized real LoRA target name")
        actual_names.add(name)
        state = module.state_dict()
        if set(state) != {"lora_down.weight", "lora_up.weight", "alpha"}:
            _fail("Unreviewed LoRAModule keys; missing BF16 Alpha/weight")
        if expected_targets is not None and name not in expected_targets:
            _fail("Production LoRA target outside independently frozen MMDiT spec")
        for part in ("lora_down.weight", "lora_up.weight", "alpha"):
            tensor = state[part]
            digest = _raw_tensor_sha256(tensor)
            key = name + "." + part
            result[key] = {
                "dtype": "BF16",
                "shape": list(tensor.shape),
                "role": "alpha_buffer" if part == "alpha" else "trainable_weight",
                "sha256": digest,
            }
            if part == "alpha":
                if tensor.ndim != 0 or float(tensor.item()) != 4.0:
                    _fail("Unreviewed Alpha scalar or value")
            elif tensor.ndim not in (2, 4):
                _fail("Unreviewed LoRA projection weight geometry")
            if expected_targets is not None:
                target = expected_targets[name]
                if part == "alpha":
                    shape = []
                elif part == "lora_down.weight":
                    shape = [target["rank"], target["input_features"]]
                else:
                    shape = [target["output_features"], target["rank"]]
                if result[key]["shape"] != shape:
                    _fail("Production LoRA Tensor differs from canonical target shape")
    if expected_targets is not None:
        if actual_names != set(expected_targets) or len(result) != 789:
            _fail("Missing/extra complete production 789-Tensor inventory")
        state_keys = set(network.state_dict())
        if state_keys != set(result):
            _fail("Production Network state_dict differs from actual LoRAModule inventory")
    elif len(result) != 3 * len(modules):
        _fail("CPU diagnostic lost a LoRA Tensor or Alpha Buffer")
    return dict(sorted(result.items()))


def _json_file(path: Path) -> dict[str, Any]:
    payload = host._read_json(path)
    if not isinstance(payload, dict):
        _fail("Actual checkpoint JSON sidecar must be a mapping")
    return payload


def verify_original_post_save(
    *, network: Any, probe: Any, anchor: Any, state_dir: Path,
    adapter_path: Path, before_tensors: Mapping[str, Any],
    before_optimizer_sha: str, before_scheduler_sha: str,
    before_rng: Mapping[str, str],
) -> dict[str, Any]:
    """Validate exact trusted production post-save files and same-Step bytes.

    No Caller/Worker result can substitute original save_state's successful
    return. This function intentionally never grants tiny CPU authority.
    """
    import torch
    from tools.sd3_full_bf16_worker_observer import (
        fingerprint_state_tree, snapshot_rng_fingerprint,
    )
    from tools.parameter_policy_backend_feature_gpu_support import (
        validate_full_bf16_checkpoint_manifest, validate_checkpoint_progress,
    )

    if (probe.accelerator.device.type != "cuda"
            or len(probe.params) != 526
            or not bool(probe._installed)):
        _fail("CPU diagnostic or incomplete prepared reference is unqualified")
    live = snapshot_serialized_tensor_inventory(network, diagnostic_cpu=False)
    if live != before_tensors:
        _fail("LoRA weights/Alpha drifted during original Save")
    if probe._adapter_parameter_sha256() != anchor.trained_adapter_parameter_sha256:
        _fail("Real physical BF16 Adapter drift since Step Anchor")
    if fingerprint_state_tree(probe.optimizer.state_dict()) != before_optimizer_sha:
        _fail("Optimizer Child state drifted during original Save")
    if fingerprint_state_tree(probe.scheduler.state_dict()) != before_scheduler_sha:
        _fail("Scheduler state drifted during original Save")
    if snapshot_rng_fingerprint() != dict(before_rng):
        _fail("RNG state drifted during Save; cannot attest saved RNG")

    if not state_dir.is_dir() or state_dir.is_symlink():
        _fail("No ordinary actual saved Checkpoint directory")
    adapter_path = host._real_path(
        str(adapter_path), name="C4-B genuine Step Adapter", must_exist=True,
    )
    host._assert_ordinary_dataset_entry(adapter_path, is_directory=False)
    files = host._checkpoint_file_inventory(state_dir)
    if len(files) != 9:
        _fail("Incomplete pinned Accelerate/SD3 nine-file state inventory")
    for key in files:
        if (state_dir / key).stat().st_size <= 0:
            _fail("Empty actual checkpoint state file: " + key)

    adapter_sha = host._sha256_file(adapter_path)
    actual_adapter = host.verify_b0_reference_adapter(
        adapter_path, expected_file_sha256=adapter_sha,
        expected_tensors=before_tensors,
        expected_targets=host.build_sd35_medium_lora_target_spec(),
    )
    actual_model = host.verify_b0_reference_adapter(
        state_dir / "model.safetensors",
        expected_file_sha256=files["model.safetensors"],
        expected_tensors=before_tensors,
        expected_targets=host.build_sd35_medium_lora_target_spec(),
    )
    if actual_adapter["tensor_inventory_sha256"] != actual_model["tensor_inventory_sha256"]:
        _fail("External Step LoRA and actual Accelerate model Tensor bytes differ")

    # weights_only=True prevents arbitrary code execution via checkpoint pickle;
    # this is still a trusted in-process *original Save output*, never a Host
    # loading of an externally supplied optimizer.bin.
    try:
        optimizer_saved = torch.load(
            state_dir / "optimizer.bin", map_location="cpu", weights_only=True,
        )
        scheduler_saved = torch.load(
            state_dir / "scheduler.bin", map_location="cpu", weights_only=True,
        )
    except Exception as exc:
        _fail("Cannot read original trusted optimizer/scheduler with weights_only=True: "
              + type(exc).__name__)
    if fingerprint_state_tree(optimizer_saved) != before_optimizer_sha:
        _fail("Saved CompositeOptimizer state mismatches same-Step physical state")
    if fingerprint_state_tree(scheduler_saved) != before_scheduler_sha:
        _fail("Saved CompositeScheduler state mismatches same-Step physical state")

    pmanifest = _json_file(state_dir / "dts_parameter_policy_manifest.json")
    progress = _json_file(state_dir / "dts_parameter_policy_progress.json")
    sd3 = _json_file(state_dir / "dts_sd3_resume_progress.json")
    train = _json_file(state_dir / "train_state.json")
    semantics = _json_file(state_dir / "dts_sd3_forward_semantics.json")
    validate_full_bf16_checkpoint_manifest(
        pmanifest, train_type="sd3-lora", manifest_version=2,
    )
    validate_checkpoint_progress(progress, checkpoint_manifest=pmanifest)
    host._verify_checkpoint_progress_consistency(
        progress, expected_step=anchor.logical_step,
    )
    scheduler_ref = host._expected_reference_scheduler_identity()
    host._verify_reference_scheduler_identity(
        pmanifest, {
            "scheduler_identity": scheduler_ref,
            "scheduler_signature": host._json_sha256(scheduler_ref),
        },
    )
    host._verify_reference_forward_identity(
        semantics, {"forward_semantics": semantics},
    )
    spec = host.get_case_spec(anchor.case_id)
    expected_cursor = {
        "schema": "dts.sd3.resume-progress", "version": 1,
        "completed_optimizer_steps": anchor.logical_step,
        "current_epoch": anchor.logical_step,
        "dataloader_batches": spec["effective_batches_per_epoch"],
        "gradient_accumulation_steps": spec["gradient_accumulation_steps"],
        "num_processes": 1,
    }
    if (sd3 != expected_cursor
            or type(train.get("current_epoch")) is not int
            or type(train.get("current_step")) is not int
            or train["current_step"] != anchor.logical_step
            or train["current_epoch"] != anchor.logical_step):
        _fail("Actual SD3 progress/train_state differs from physical Step")
    checkpoint_id = progress.get("checkpoint_id")
    if (type(checkpoint_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", checkpoint_id) is None):
        _fail("No original on-disk generated checkpoint_id")
    if (progress.get("resume_source_checkpoint_id") is not None
            and anchor.phase == "fresh"):
        _fail("Fresh Step Checkpoint falsely declares Resume ancestry")
    from mikazuki.parameter_policy import serialize_parameter_policy

    _policy_name, policy_json = serialize_parameter_policy(
        host.build_reference_policy(anchor.case_id)
    )
    expected_policy_sha = hashlib.sha256(
        policy_json.encode("utf-8")
    ).hexdigest()
    if pmanifest.get("policy_hash") != expected_policy_sha:
        _fail("Checkpoint Policy hash differs from independently fixed R1/R2 policy")
    # Source/Phase Ticket/CaseManifest authentication stays with external Fix D:
    # do not infer it from this same-process Save evidence.
    return {
        "status": "original_save_post_return_files_verified",
        "checkpoint_id": checkpoint_id,
        "adapter_file_sha256": actual_adapter["file_sha256"],
        "tensor_inventory_sha256": actual_adapter["tensor_inventory_sha256"],
        "state_file_sha256": files,
    }


__all__ = ["snapshot_serialized_tensor_inventory", "verify_original_post_save"]
