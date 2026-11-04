"""C4-B real same-Step Save verifier, with a hard CPU diagnostic boundary.

The module imports Torch lazily, and never changes production train, LoRA
serialization, Optimizer, Scheduler, or original Accelerator.save_state.
A Host MUST NOT unpickle untrusted checkpoint bytes; the only weights_only
torch.load here is an in-process check of the just-produced trusted files.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class C4BObservedLoRAWrite:
    logical_step: int
    output_path: str
    file_sha256: str
    ss_steps: int
    ss_epoch: int
    original_returned: bool


def verify_lora_step_metadata(
    path: Path | str, *, expected_step: int, expected_epoch: int | None = None,
) -> dict[str, Any]:
    """Read the *actual* safetensors metadata; Tensor bytes are checked later.

    Original train_network.save_model stamps ss_steps and ss_epoch immediately
    before calling LoRANetwork.save_weights. Equal Tensor bytes alone do not
    prove a correct checkpoint Step.
    """
    actual = Path(path)
    if (not actual.is_file() or actual.is_symlink()
            or type(expected_step) is not int or expected_step not in (1, 2)):
        _fail("C4-B invalid original LoRA Step file or physical Step")
    header, _data_start = host._safetensors_header(actual)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        _fail("Original LoRA Step metadata is missing")
    step_s, epoch_s = metadata.get("ss_steps"), metadata.get("ss_epoch")
    if (type(step_s) is not str or type(epoch_s) is not str
            or re.fullmatch(r"(0|[1-9][0-9]*)", step_s) is None
            or re.fullmatch(r"(0|[1-9][0-9]*)", epoch_s) is None
            or int(step_s) != expected_step):
        _fail("Original LoRA ss_steps/ss_epoch metadata mismatch")
    if expected_epoch is not None and (
        type(expected_epoch) is not int or int(epoch_s) != expected_epoch
    ):
        _fail("Original LoRA ss_epoch differs from actual physical progress")
    return {
        "ss_steps": int(step_s),
        "ss_epoch": int(epoch_s),
        "file_sha256": host._sha256_file(actual),
    }



def verify_lora_writer_progress(
    observation: C4BObservedLoRAWrite, *, expected_epoch: int,
) -> dict[str, Any]:
    """Bind a real original LoRA Writer return to actual saved train_state Epoch.

    This is byte/metadata consistency only; no CPU diagnostic may become
    a full-reference physical CUDA checkpoint or cross-process admission.
    """
    if (type(observation) is not C4BObservedLoRAWrite
            or not observation.original_returned
            or type(observation.logical_step) is not int
            or type(expected_epoch) is not int
            or expected_epoch < 0):
        _fail("C4-B untrusted original LoRA Writer/Progress Epoch source")
    details = verify_lora_step_metadata(
        observation.output_path,
        expected_step=observation.logical_step,
        expected_epoch=expected_epoch,
    )
    if (details["file_sha256"] != observation.file_sha256
            or details["ss_steps"] != observation.ss_steps
            or details["ss_epoch"] != observation.ss_epoch):
        _fail("C4-B original LoRA Writer metadata/bytes differ from Progress")
    return details



class C4BLoRAWriteObserver:
    """Observe the original per-instance LoRANetwork.save_weights once.

    This class is only a *writer observation*; even a genuine one-module CPU
    result cannot mint a C4-B production Saved Checkpoint receipt.
    """

    def __init__(
        self, *, network: Any, output_dir: Path | str,
        output_name: str, step_provider: Any,
    ) -> None:
        from networks.lora_sd3 import LoRANetwork

        if type(network) is not LoRANetwork:
            _fail("Original LoRA writer requires exact production LoRANetwork")
        if ("save_weights" in vars(network)
                or type(network).save_weights is not LoRANetwork.save_weights):
            _fail("Unreviewed or shadowed original LoRA save_weights method")
        if (type(output_name) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", output_name) is None
                or not callable(step_provider)):
            _fail("Unreviewed original LoRA writer output / step source")
        self.network = network
        self.output_dir = host._real_path(
            str(output_dir), name="C4-B LoRA writer directory",
        )
        self.output_name = output_name
        self._step_provider = step_provider
        self._original = network.save_weights
        self._reviewed_method = LoRANetwork.save_weights
        self._original_slot = ("save_weights" in vars(network),
                               vars(network).get("save_weights"))
        self._installed = False
        self._closed = False
        self._wrapper = None
        self._attempted: set[int] = set()
        self._completed: dict[int, C4BObservedLoRAWrite] = {}

    def _check_method(self):
        if (type(self.network).save_weights is not self._reviewed_method
                or ("save_weights" in vars(self.network)) != self._installed):
            _fail("Original LoRA writer was replaced or detached")
        if self._installed and vars(self.network)["save_weights"] is not self._wrapper:
            _fail("Unreviewed LoRA instance writer override")

    def install(self) -> None:
        if self._closed or self._installed:
            _fail("LoRA writer observer cannot be installed twice")
        self._check_method()

        def wrapped(*args, **kwargs):
            return self._observe(*args, **kwargs)

        self._wrapper = wrapped
        try:
            vars(self.network)["save_weights"] = wrapped
            self._installed = True
            self._check_method()
        except BaseException:
            self.close()
            raise

    def _observe(self, *args, **kwargs):
        if not self._installed or self._closed:
            _fail("LoRA writer observer is not installed")
        self._check_method()
        if not args and "file" not in kwargs:
            _fail("LoRA writer called without output path")
        if args and "file" in kwargs:
            _fail("LoRA writer has duplicate file argument")
        if len(args) > 3:
            _fail("Unreviewed LoRA writer argument count")
        file = args[0] if args else kwargs["file"]
        if not isinstance(file, (str, Path)):
            _fail("Unreviewed original LoRA output path type")
        literal = Path(file).expanduser().absolute()
        if literal != literal.resolve(strict=False):
            _fail("Unreviewed LoRA writer linked or noncanonical output")
        if literal.parent != self.output_dir:
            _fail("LoRA output outside approved Phase directory")
        if literal.suffix != ".safetensors":
            _fail("Authoritative LoRA must use original safetensors writer")
        if literal.name in (self.output_name + ".safetensors",
                            self.output_name + "-last.safetensors"):
            # Genuine ancillary last still delegates original once, but
            # cannot create an additional Step-N observation.
            return self._original(*args, **kwargs)
        step = self._step_provider()
        if (type(step) is not int or step not in (1, 2)
                or literal != self.output_dir /
                   f"{self.output_name}-step{step:08d}.safetensors"):
            _fail("Original LoRA output is not the authorized physical Step")
        if step in self._attempted or literal.exists():
            _fail("Original LoRA Step file already exists or was attempted")
        metadata = args[2] if len(args) >= 3 else kwargs.get("metadata")
        if (not isinstance(metadata, dict)
                or str(metadata.get("ss_steps")) != str(step)
                or type(metadata.get("ss_epoch")) is not str
                or re.fullmatch(r"[0-9]+", metadata["ss_epoch"]) is None):
            _fail("Original LoRA ss_steps/ss_epoch metadata is unbound")
        self._attempted.add(step)
        result = self._original(*args, **kwargs)
        if not literal.is_file() or literal.is_symlink():
            _fail("Original LoRA writer returned without a regular Step file")
        meta = verify_lora_step_metadata(literal, expected_step=step)
        observation = C4BObservedLoRAWrite(
            logical_step=step, output_path=str(literal),
            file_sha256=meta["file_sha256"],
            ss_steps=meta["ss_steps"], ss_epoch=meta["ss_epoch"],
            original_returned=True,
        )
        self._completed[step] = observation
        return result

    def completed_step_write(self, *, logical_step: int) -> C4BObservedLoRAWrite:
        if not self._installed or self._closed:
            _fail("LoRA writer observer is not active")
        self._check_method()
        observation = self._completed.get(logical_step)
        if (observation is None or not observation.original_returned
                or observation.logical_step != logical_step
                or host._sha256_file(Path(observation.output_path))
                   != observation.file_sha256):
            _fail("No authentic original Step LoRA writer completion")
        verify_lora_step_metadata(
            observation.output_path, expected_step=logical_step,
            expected_epoch=observation.ss_epoch,
        )
        return observation

    def close(self) -> None:
        had, original = self._original_slot
        if had:
            vars(self.network)["save_weights"] = original
        else:
            vars(self.network).pop("save_weights", None)
        self._installed = False
        self._closed = True
        self._wrapper = None


def snapshot_accelerate_rng_expectation(accelerator: Any) -> dict[str, Any]:
    """Capture the actual pinned Accelerate 1.6.0 Save RNG source before I/O.

    Accelerator.step is its INTERNAL counter, not necessarily C3 logical_step.
    Only typed digests are returned; there are no long-lived GPU RNG clones.
    """
    import random
    import numpy as np
    import torch
    from tools.sd3_full_bf16_worker_observer import fingerprint_state_tree

    step = accelerator.step
    if type(step) is not int or step < 0:
        _fail("Accelerate internal RNG Step is not an integer")
    states = {
        "random_state": random.getstate(),
        "numpy_random_seed": np.random.get_state(),
        "torch_manual_seed": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    return {
        "step": step,
        "field_sha256": {
            key: fingerprint_state_tree(value)
            for key, value in states.items()
        },
    }


def verify_saved_rng_payload(
    path: Path | str, *, expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate actual original Accelerate RNG file with restricted loading.

    Never call this on untrusted Host inputs. The caller must have observed
    the original Save of this exact path in its own active process. NumPy's
    small pinned 1.x/2.x array pickle helpers are allowlisted in a scoped
    weights-only loader, never as global unsafe pickle permissions.
    """
    import numpy as np
    import torch
    from tools.sd3_full_bf16_worker_observer import fingerprint_state_tree

    file = Path(path)
    if (not file.is_file() or file.is_symlink()
            or file.stat().st_size <= 0 or file.stat().st_size > 16 * 1024 * 1024):
        _fail("Missing/linked/oversized original RNG state file")
    if (not isinstance(expected, Mapping)
            or type(expected.get("step")) is not int
            or not isinstance(expected.get("field_sha256"), Mapping)):
        _fail("Original Accelerate RNG expectation is not typed")
    try:
        # NumPy's RNG state is a tuple including a uint32 ndarray. A plain
        # torch.load(weights_only=True) rejects its safe NumPy reconstruction
        # objects, so allow exactly those types for this trusted in-process
        # original Save output, NEVER a general Python class loader.
        multiarray = getattr(getattr(np, "_core", None), "multiarray", None)
        if multiarray is None:
            multiarray = np.core.multiarray
        allowlist = [
            multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype("uint32")),
        ]
        with torch.serialization.safe_globals(allowlist):
            payload = torch.load(
                str(file), map_location="cpu", weights_only=True,
            )
    except Exception as exc:
        _fail("Cannot safely decode original RNG payload (weights_only): "
              + type(exc).__name__)
    required_fields = {"step", *expected["field_sha256"]}
    if not isinstance(payload, dict) or set(payload) != required_fields:
        _fail("Actual RNG payload missing/extra Accelerate state fields")
    if type(payload["step"]) is not int or payload["step"] != expected["step"]:
        _fail("Actual saved Accelerate internal RNG Step mismatch")
    for key in sorted(required_fields - {"step"}):
        digest = expected["field_sha256"][key]
        host._require_sha(digest, name="expected original RNG " + key)
        if fingerprint_state_tree(payload[key]) != digest:
            _fail("Actual original RNG payload state drift: " + key)
    return {"step": payload["step"], "file_sha256": host._sha256_file(file)}


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
    before_rng_expectation: Mapping[str, Any],
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
    verified_rng = verify_saved_rng_payload(
        state_dir / "random_states_0.pkl", expected=before_rng_expectation,
    )
    if verified_rng["file_sha256"] != files["random_states_0.pkl"]:
        _fail("Original RNG file drifted during the same Save verification")
    if snapshot_accelerate_rng_expectation(probe.accelerator) != dict(before_rng_expectation):
        _fail("Original Accelerate Step/RNG expectation drift after Save")

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
        "current_epoch": train["current_epoch"],
        "adapter_file_sha256": actual_adapter["file_sha256"],
        "tensor_inventory_sha256": actual_adapter["tensor_inventory_sha256"],
        "state_file_sha256": files,
    }


__all__ = ["snapshot_serialized_tensor_inventory", "verify_original_post_save",
           "verify_lora_step_metadata", "C4BObservedLoRAWrite", "C4BLoRAWriteObserver",
           "snapshot_accelerate_rng_expectation", "verify_saved_rng_payload",
           "verify_lora_writer_progress"]
