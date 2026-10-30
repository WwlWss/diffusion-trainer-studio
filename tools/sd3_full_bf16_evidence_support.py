"""Host-only SD3C2-B candidate evidence contracts (no CUDA or Torch import).

This module validates *reference evidence*, not production Full-BF16
qualification. The restricted Worker and production-trainer observations are
separate B0 stages. No function here can mark a backend qualified.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import stat
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dts.sd3c2b.full-bf16-case-manifest"
VERSION = 1
PHASE_SCHEMA = "dts.sd3c2b.full-bf16-worker-phase"
TICKET_SCHEMA = "dts.sd3c2b.phase-ticket"
TICKET_VERSION = 1
PROGRESS_SCHEMA = "dts.parameter-policy.checkpoint-progress"
REFERENCE_SEED = 123456
TRAINER_ENTRYPOINT = "scripts/dev/sd3_train_network.py"
WORKER_ENTRYPOINT = "tools/sd3_full_bf16_evidence_worker.py"

_CASES = {
    "r1-adamw-accum1": {
        "gradient_accumulation_steps": 1,
        "effective_batches_per_epoch": 1,
        "optimizer_types": ["AdamW"],
        "output_name": "sd3c2b-r1",
    },
    "r2-muon-adamw-fallback-accum2": {
        "gradient_accumulation_steps": 2,
        "effective_batches_per_epoch": 2,
        "optimizer_types": ["Muon", "AdamW"],
        "output_name": "sd3c2b-r2",
    },
}
_INPUT_FILES = ("--clip_l", "--clip_g", "--t5xxl", "--vae")
_BOOLEAN_FLAGS = frozenset({
    "--save_state", "--network_train_unet_only", "--full_bf16",
    "--cache_latents", "--cache_text_encoder_outputs",
})
_REQUIRED_VALUES = {
    "--max_train_steps": "2",
    "--save_every_n_steps": "1",
    "--mixed_precision": "bf16",
    "--optimizer_type": "AdamW",
    "--network_module": "networks.lora_sd3",
    "--network_dim": "4",
    "--network_alpha": "4",
    "--train_batch_size": "1",
    "--resolution": "512",
    "--save_model_as": "safetensors",
    "--lr_scheduler": "constant",
    "--seed": str(REFERENCE_SEED),
    "--max_data_loader_n_workers": "0",
    "--max_grad_norm": "1.0",
}
_PATH_OPTIONS = frozenset({
    "--pretrained_model_name_or_path", "--train_data_dir",
    "--parameter_policy_config", "--output_dir", "--resume", *_INPUT_FILES,
})
_COMMON_VALUES = frozenset({
    *_REQUIRED_VALUES, "--gradient_accumulation_steps", "--output_name",
    "--pretrained_model_name_or_path", "--train_data_dir",
    "--parameter_policy_config", *_INPUT_FILES,
})
_CASE_FLAGS = _BOOLEAN_FLAGS | _COMMON_VALUES | {"--output_dir", "--resume"}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_PHASE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}\Z")
_CHECKPOINT_IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")


class FullBF16EvidenceError(ValueError):
    """The source or the evidence violates the candidate-only B0 contract."""


def _fail(message: str) -> None:
    raise FullBF16EvidenceError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FullBF16EvidenceError(f"Unserializable evidence JSON: {exc}") from exc


def _json_sha256(payload: Any) -> str:
    return _sha256(_canonical_json_bytes(payload))


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda value: _fail(f"Non-finite JSON value: {value}"),
        )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise FullBF16EvidenceError(f"Unreadable JSON input {path}: {exc}") from exc
    if not isinstance(result, dict):
        _fail(f"JSON root must be an object: {path}")
    return result


def _require_sha(value: object, *, name: str, commit: bool = False) -> str:
    pattern = _HEX40 if commit else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase {40 if commit else 64}-hex digest")
    return value


def _real_path(value: object, *, name: str, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a nonempty absolute path")
    p = Path(value).expanduser()
    if not p.is_absolute():
        _fail(f"{name} must be absolute: {value!r}")
    # Existing symlinks, including intermediate directories, change authority.
    for ancestor in (p, *p.parents):
        if ancestor.is_symlink():
            _fail(f"{name} includes an unreviewed symlink: {ancestor}")
    try:
        actual = p.resolve(strict=must_exist)
    except OSError as exc:
        raise FullBF16EvidenceError(f"Cannot resolve {name}: {exc}") from exc
    if must_exist and not actual.exists():
        _fail(f"{name} does not exist: {actual}")
    return actual


def _external_work_path(value: object, *, name: str) -> Path:
    path = _real_path(value, name=name)
    if path == ROOT or ROOT in path.parents:
        _fail(f"{name} must be outside Git checkout: {path}")
    return path


def get_case_spec(case_id: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(_CASES[case_id])
    except (KeyError, TypeError) as exc:
        raise FullBF16EvidenceError(f"Unrecognized B0 candidate case: {case_id!r}") from exc


def build_reference_policy(case_id: str) -> dict[str, Any]:
    """Reuse reviewed R0 canonical R1; independently build the C2-A R2 shape."""
    get_case_spec(case_id)
    from mikazuki.parameter_policy import validate_parameter_policy
    from tools.prepare_sd3_component_baseline import make_policy
    from mikazuki.sd3_full_bf16_scope import SD3_FULL_BF16_TRAINED_COMPONENTS

    policy = make_policy()
    if case_id == "r2-muon-adamw-fallback-accum2":
        policy["optimizer_profiles"] = {
            "muon_primary": {"type": "Muon", "args": {}},
            "adamw_fallback": {"type": "AdamW", "args": {}},
        }
        for component in SD3_FULL_BF16_TRAINED_COMPONENTS:
            policy["components"][component] = {
                "train": True,
                "optimizer_profile": "muon_primary",
                "learning_rate": 0.0001,
                "fallback_optimizer_profile": "adamw_fallback",
                "fallback_learning_rate": 0.0001,
            }
    return validate_parameter_policy(policy)


def decoded_rgb_pixel_sha256(path: Path) -> str:
    """Versioned hash of *decoded* 512x512 RGB pixels, not PNG metadata."""
    try:
        from PIL import Image

        p = Path(path)
        with Image.open(p) as image:
            if image.format != "PNG" or image.size != (512, 512):
                _fail(f"Reference image must be 512x512 PNG: {p}")
            if image.mode != "RGB":
                _fail(f"Reference image must already be RGB: {p}")
            if image.getexif().get(274, 1) != 1:
                _fail(f"EXIF orientation transforms are not reference inputs: {p}")
            pixels = image.tobytes()
        if len(pixels) != 512 * 512 * 3:
            _fail(f"Unexpected decoded RGB byte length: {p}")
        prefix = b"dts.sd3c2b.decoded-rgb.v1\x00" + struct.pack("<II", 512, 512)
        return _sha256(prefix + pixels)
    except FullBF16EvidenceError:
        raise
    except (OSError, ValueError, ImportError) as exc:
        raise FullBF16EvidenceError(f"Cannot decode RGB reference image {path}: {exc}") from exc


def _parse_command(command: object) -> dict[str, str | bool]:
    if (not isinstance(command, list) or not command
            or any(not isinstance(x, str) or not x for x in command)):
        _fail("Trainer command must be a nonempty array of nonempty strings")
    if command[0] != TRAINER_ENTRYPOINT:
        _fail("Only the fixed production SD3 trainer can be specified")
    options: dict[str, str | bool] = {}
    index = 1
    while index < len(command):
        key = command[index]
        if not key.startswith("--") or "=" in key or key not in _CASE_FLAGS:
            _fail(f"Unknown, inline or positional trainer option: {key!r}")
        if key in options:
            _fail(f"Duplicate trainer option: {key!r}")
        if key in _BOOLEAN_FLAGS:
            options[key] = True
            index += 1
            continue
        if index + 1 >= len(command):
            _fail(f"Missing value for {key}")
        value = command[index + 1]
        if value.startswith("--"):
            _fail(f"Missing/non-string value for {key}")
        if key in _PATH_OPTIONS:
            _real_path(value, name=key)
        options[key] = value
        index += 2
    return options


def _assert_digest_mapping(value: Any, *, name: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{name} must be a JSON object")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            _fail(f"Invalid {name} entry: {key!r}")
        _require_sha(item.get("sha256"), name=f"{name}.{key}.sha256")


def _assert_reference_flags(options: dict[str, str | bool], case_id: str) -> None:
    spec = get_case_spec(case_id)
    if set(options) != _BOOLEAN_FLAGS | _COMMON_VALUES | {"--output_dir"} | (
        {"--resume"} if "--resume" in options else set()
    ):
        missing = (_BOOLEAN_FLAGS | _COMMON_VALUES | {"--output_dir"}) - set(options)
        extra = set(options) - (_CASE_FLAGS)
        _fail(f"Incorrect reference trainer flags, missing={sorted(missing)}, extra={sorted(extra)}")
    for key in _BOOLEAN_FLAGS:
        if options.get(key) is not True:
            _fail(f"{key} must be an actual bare boolean flag")
    for key, value in _REQUIRED_VALUES.items():
        if options.get(key) != value:
            _fail(f"Reference requires {key}={value}")
    for key, expected in (
        ("--gradient_accumulation_steps", str(spec["gradient_accumulation_steps"])),
        ("--output_name", spec["output_name"]),
    ):
        if options.get(key) != expected:
            _fail(f"Wrong {key} in {case_id}")
    # C2-A is a separate, production-owned semantic gate; no fake asset header
    # or physical model load is used in the manifest-only phase.
    from mikazuki.sd3_full_bf16_scope import sd3_full_bf16_config_blockers
    config = {key[2:]: value for key, value in options.items()}
    problems = sd3_full_bf16_config_blockers(build_reference_policy(case_id), config)
    if problems:
        _fail("C2-A reference scope mismatch: " + "; ".join(problems))


def validate_case_manifest(payload: object, *, phase: str = "fresh") -> dict[str, Any]:
    """Pure Host phase-aware policy/options gate; never loads model tensors.

    fresh: both trainer output directories must be absent;
    resume: verified Step-1 Fresh artifacts must exist, Resume output absent;
    complete: both declared Step-1/Step-2 outputs must exist.
    """
    if phase not in ("fresh", "resume", "complete"):
        _fail(f"Unknown evidence validation phase: {phase!r}")
    if not isinstance(payload, dict):
        _fail("Case manifest must be a JSON object")
    if payload.get("schema") != SCHEMA or type(payload.get("version")) is not int or payload["version"] != VERSION:
        _fail("Unknown SD3C2-B case manifest schema/version")
    if any(k in payload for k in (
        "qualifies_full_bf16", "qualification_override",
        "backend_qualification_eligible", "production_qualification_mutated",
    )):
        _fail("Case manifest cannot declare production backend qualification")
    case_id = payload.get("case_id")
    spec = get_case_spec(case_id)
    _require_sha(payload.get("source_commit"), name="source_commit", commit=True)
    submodules = payload.get("source_submodules")
    required_submodules = {"frontend", "mikazuki/dataset-tag-editor", "sd-scripts"}
    if (not isinstance(submodules, dict)
            or not required_submodules.issubset(submodules)
            or any(not isinstance(name, str)
                   or not (name in required_submodules
                           or any(name.startswith(parent + "/") for parent in required_submodules))
                   or not isinstance(info, dict)
                   or set(info) != {"gitlink_sha", "checkout"}
                   or info["checkout"] not in ("initialized", "uninitialized")
                   for name, info in submodules.items())):
        _fail("Incomplete or unreviewed B0 source submodule inventory")
    for name, info in submodules.items():
        _require_sha(info["gitlink_sha"], name=f"{name} Gitlink", commit=True)
    ref = payload.get("reference")
    if (not isinstance(ref, dict)
            or any(type(ref.get(k)) is not int for k in (
                "train_batch_size", "effective_batches_per_epoch",
                "gradient_accumulation_steps", "max_train_steps",
            ))
            or ref != {
        "train_type": "sd3-lora", "model_variant": "sd35-medium",
        "resolution": [512, 512], "train_batch_size": 1,
        "effective_batches_per_epoch": spec["effective_batches_per_epoch"],
        "gradient_accumulation_steps": spec["gradient_accumulation_steps"],
        "max_train_steps": 2, "optimizer_profiles": spec["optimizer_types"],
    }):
        _fail("Reference must match exactly the reviewed R1/R2 case geometry")
    identity = payload.get("input_identity")
    if not isinstance(identity, dict):
        _fail("Missing input_identity")
    for key in ("model_sha256", "policy_bytes_sha256", "canonical_policy_hash", "dataset_signature"):
        _require_sha(identity.get(key), name=key)
    _assert_digest_mapping(identity.get("assets"), name="assets")
    if set(identity["assets"]) != {key[2:] for key in _INPUT_FILES}:
        _fail("Reference requires exactly the three independent encoders and VAE")
    _assert_digest_mapping(identity.get("dataset_files"), name="dataset_files")
    inventory = identity["dataset_files"]
    if _json_sha256(inventory) != identity["dataset_signature"]:
        _fail("Dataset inventory signature mismatch")
    count = spec["effective_batches_per_epoch"]
    expected_dataset_names = {
        f"data/1_dts_sd3c2b/sample_{index:04d}.{suffix}"
        for index in range(1, count + 1) for suffix in ("png", "caption")
    }
    if set(inventory) != expected_dataset_names:
        _fail("Dataset inventory differs from exactly one R1/two R2 paired samples")
    for relative, item in inventory.items():
        if relative.endswith(".png"):
            _require_sha(item.get("rgb_pixel_sha256"), name="RGB pixel digest")
            if item.get("kind") != "image" or item.get("dimensions") != [512, 512]:
                _fail("Incorrect image inventory")
        else:
            if item.get("kind") != "caption" or not isinstance(item.get("identity"), str):
                _fail("Incorrect caption inventory")
    if count == 2:
        pngs = [inventory[n] for n in sorted(inventory) if n.endswith(".png")]
        caps = [inventory[n] for n in sorted(inventory) if n.endswith(".caption")]
        if len({v["sha256"] for v in pngs}) != 2 or len({v["rgb_pixel_sha256"] for v in pngs}) != 2:
            _fail("R2 requires different image files AND different decoded RGB pixels")
        if len({v["identity"] for v in caps}) != 2:
            _fail("R2 requires distinct caption markers")
    fresh = _parse_command(payload.get("fresh_command"))
    resume = _parse_command(payload.get("resume_command"))
    if "--resume" in fresh or "--resume" not in resume:
        _fail("Only Resume phase may select a checkpoint")
    _assert_reference_flags(fresh, case_id)
    _assert_reference_flags(resume, case_id)
    strip = lambda d: {k: v for k, v in d.items() if k not in {"--output_dir", "--resume"}}
    if strip(fresh) != strip(resume):
        _fail("Fresh/Resume reference options differ")
    root = _external_work_path(payload.get("work_root"), name="work_root")
    dataset_root = _real_path(fresh["--train_data_dir"], name="dataset_root")
    if dataset_root != root / "data":
        _fail("Trainer dataset path must be the isolated generated data parent")
    policy_path = _real_path(fresh["--parameter_policy_config"], name="policy")
    if policy_path != root / "policy.json" or policy_path != _real_path(resume["--parameter_policy_config"], name="resume policy"):
        _fail("Both phases must share one canonical isolated policy")
    if not policy_path.is_file():
        _fail("Missing generated canonical policy")
    from mikazuki.parameter_policy import serialize_parameter_policy, validate_parameter_policy
    policy = validate_parameter_policy(_read_json(policy_path))
    if policy != build_reference_policy(case_id):
        _fail("Reference Policy differs from the C2-A R1/R2 contract")
    _name, canonical = serialize_parameter_policy(policy)
    if _sha256(canonical.encode("utf-8")) != identity["canonical_policy_hash"]:
        _fail("Canonical Policy digest mismatch")
    if _sha256_file(policy_path) != identity["policy_bytes_sha256"]:
        _fail("Policy file digest mismatch")
    model = _real_path(fresh["--pretrained_model_name_or_path"], name="model")
    if model != _real_path(resume["--pretrained_model_name_or_path"], name="resume model"):
        _fail("Fresh/Resume model mismatch")
    for key in _INPUT_FILES:
        name = key[2:]
        path = _real_path(fresh[key], name=name)
        if path != _real_path(resume[key], name="resume "+name):
            _fail(f"Fresh/Resume {name} input mismatch")
        if identity["assets"][name].get("path") != str(path):
            _fail(f"Manifest {name} identity path mismatch")
    for output_phase, opts in (("fresh", fresh), ("resume", resume)):
        output = _external_work_path(opts["--output_dir"], name=f"{output_phase} output")
        if output != root / output_phase:
            _fail(f"{output_phase} output must be under the isolated work root")
        expected_checkpoint = output / f"{spec['output_name']}-step{1 if output_phase=='fresh' else 2:08d}-state"
        expected_adapter = output / f"{spec['output_name']}-step{1 if output_phase=='fresh' else 2:08d}.safetensors"
        for suffix, expected_path in (("checkpoint_dir", expected_checkpoint), ("adapter_path", expected_adapter)):
            key = f"{output_phase}_{suffix}"
            if _real_path(payload.get(key), name=key) != expected_path:
                _fail(f"Wrong {key}: expected exact physical Step path")
        should_exist = (phase == "complete" or (phase == "resume" and output_phase == "fresh"))
        if should_exist:
            if not output.is_dir() or not expected_checkpoint.is_dir() or not expected_adapter.is_file():
                _fail(f"{output_phase} real Step output/adapter is missing for {phase} validation")
        elif output.exists():
            _fail(f"Stale {output_phase} output directory already exists before {phase} phase")
    if _real_path(resume["--resume"], name="resume source") != _real_path(payload["fresh_checkpoint_dir"], name="checkpoint source"):
        _fail("Resume must read exact Fresh Step-1 checkpoint")
    return {
        "case_spec": spec, "case_id": case_id, "phase": phase,
        "fresh": fresh, "resume": resume, "policy": policy,
        "work_root": root, "input_identity": identity,
        "policy_hash": identity["canonical_policy_hash"],
    }


def _assert_ordinary_dataset_entry(path: Path, *, is_directory: bool) -> None:
    """Reject linked, reparse, and nonstandard Dataset entries."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise FullBF16EvidenceError(f"Unstatable Dataset path {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or (
        os.name == "nt" and getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        _fail(f"Unreviewed linked/reparse Dataset path: {path}")
    if is_directory and not stat.S_ISDIR(info.st_mode):
        _fail(f"Dataset subset must be an ordinary directory: {path}")
    if not is_directory and not stat.S_ISREG(info.st_mode):
        _fail(f"Dataset image/caption must be an ordinary file: {path}")


def _verify_reference_dataset_tree(
    work: Path, expected_inventory: Mapping[str, Any],
) -> None:
    """Audit the full train_data_dir as parsed by production DreamBooth."""
    data_root = work / "data"
    _assert_ordinary_dataset_entry(data_root, is_directory=True)
    entries = list(data_root.iterdir())
    if len(entries) != 1 or entries[0].name != "1_dts_sd3c2b":
        _fail("DreamBooth data parent must contain exactly one approved subset")
    subset = entries[0]
    _assert_ordinary_dataset_entry(subset, is_directory=True)
    observed: set[str] = set()
    for item in subset.iterdir():
        _assert_ordinary_dataset_entry(item, is_directory=False)
        observed.add(item.relative_to(work).as_posix())
    if observed != set(expected_inventory):
        _fail("DreamBooth directory has an unrecorded, missing or altered member")


def verify_case_inputs(payload: dict[str, Any], *, phase: str = "fresh") -> dict[str, Any]:
    """Rehash immutable inputs for every phase without requiring model header.

    This is intentionally not a GPU PASS or even a full model-asset preflight.
    It supports before-Fresh, before-Resume and after-Resume integrity checks.
    """
    verified = validate_case_manifest(payload, phase=phase)
    identity = verified["input_identity"]
    options = verified["fresh"]
    model = _real_path(options["--pretrained_model_name_or_path"], name="model", must_exist=True)
    if not model.is_file() or _sha256_file(model) != identity["model_sha256"]:
        _fail("Full-file model SHA256 mismatch")
    for key in _INPUT_FILES:
        path = _real_path(options[key], name=key, must_exist=True)
        if not path.is_file() or _sha256_file(path) != identity["assets"][key[2:]]["sha256"]:
            _fail(f"Full-file {key} SHA256 mismatch")
    work = verified["work_root"]
    _verify_reference_dataset_tree(work, identity["dataset_files"])
    for relative, recorded in identity["dataset_files"].items():
        path = _real_path(str(work / relative), name=relative, must_exist=True)
        if not path.is_file() or _sha256_file(path) != recorded["sha256"]:
            _fail(f"Dataset changed after generation: {relative}")
        if recorded["kind"] == "image" and decoded_rgb_pixel_sha256(path) != recorded["rgb_pixel_sha256"]:
            _fail(f"Image pixels changed after generation: {relative}")
        if recorded["kind"] == "caption":
            try:
                caption = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise FullBF16EvidenceError(f"Cannot read caption: {exc}") from exc
            if not caption.startswith(recorded["identity"] + ", "):
                _fail(f"Caption identity marker disappeared: {relative}")
    return {"status": "identity_pass", "case_id": verified["case_id"],
            "source_commit": payload["source_commit"], "phase": phase,
            "input_identity_sha256": _json_sha256(identity)}


def validate_case_assets(payload: dict[str, Any], *, phase: str = "fresh") -> dict[str, Any]:
    """Real full-file identity and SD3 safetensors-header preflight (no CUDA)."""
    verified = verify_case_inputs(payload, phase=phase)
    fresh = validate_case_manifest(payload, phase=phase)["fresh"]
    from mikazuki.sd3_full_bf16_scope import sd3_full_bf16_asset_blockers
    config = {key[2:]: value for key, value in fresh.items()}
    errors = sd3_full_bf16_asset_blockers(config)
    if errors:
        _fail("SD3.5 Medium architecture/asset validation failed: " + "; ".join(errors))
    return {**verified, "status": "preflight_pass"}


def _record_value(record: object, key: str) -> Any:
    return record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)


def validate_qualification_preconditions(
    case_id: str, backend: object, optimizers: Mapping[str, object],
) -> dict[str, Any]:
    """Only sd3-lora backend may be pending before its *process-local* lease."""
    spec = get_case_spec(case_id)
    from mikazuki.parameter_policy_execution import FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID

    if _record_value(backend, "status") != "pending" or _record_value(backend, "evidence_case_id") is not None:
        _fail("SD3 backend must be pending without release evidence")
    if not isinstance(optimizers, Mapping) or set(optimizers) != set(spec["optimizer_types"]):
        _fail("Unexpected Full-BF16 optimizer qualification set")
    for name in spec["optimizer_types"]:
        record = optimizers[name]
        if (_record_value(record, "status") != "qualified"
                or _record_value(record, "evidence_case_id") != FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID):
            _fail(f"{name} requires preexisting qualified D0 evidence (no optimizer lease)")
    return {"status": "pass", "case_id": case_id, "backend_lease_only": True}


def validate_qualification_lease_records(
    case_id: str, records: object,
    optimizers_before: Mapping[str, object], optimizers_after: Mapping[str, object],
) -> dict[str, Any]:
    get_case_spec(case_id)
    if not isinstance(records, list):
        _fail("Temporary qualification lease must report records")
    before = {k: (
        _record_value(v, "status"), _record_value(v, "reason"),
        _record_value(v, "evidence_case_id")
    ) for k, v in optimizers_before.items()}
    after = {k: (
        _record_value(v, "status"), _record_value(v, "reason"),
        _record_value(v, "evidence_case_id")
    ) for k, v in optimizers_after.items()}
    if before != after:
        _fail("Optimizer qualification records changed across Worker lease")
    from mikazuki.parameter_policy_execution import FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID
    required = {( "backend", "sd3-lora")}
    required.update({("optimizer", name) for name in get_case_spec(case_id)["optimizer_types"]})
    seen: set[tuple[str, str]] = set()
    backend_leases = 0
    for record in records:
        if not isinstance(record, dict):
            _fail("Invalid lease record")
        identity = (record.get("kind"), record.get("name"))
        if identity in seen or identity not in required:
            _fail("Duplicate or unrecognized qualification lease record")
        seen.add(identity)
        if identity[0] == "backend":
            if (record.get("before_status") != "pending"
                    or record.get("before_evidence_case_id") is not None
                    or record.get("lease_applied") is not True):
                _fail("Backend-only pending lease violated")
            backend_leases += 1
        elif (record.get("before_status") != "qualified"
              or record.get("before_evidence_case_id") != FULL_BF16_SHARED_OPTIMIZER_EVIDENCE_ID
              or record.get("lease_applied") is not False):
            _fail("Temporary lease attempted to qualify an optimizer")
    if seen != required or backend_leases != 1:
        _fail("One backend lease and exact unchanged optimizer records are required")
    return {"status": "pass", "backend_lease_only": True}


def _validate_ticket(ticket: object) -> dict[str, Any]:
    if not isinstance(ticket, dict):
        _fail("Phase ticket must be an object")
    if ticket.get("schema") != TICKET_SCHEMA or type(ticket.get("version")) is not int or ticket["version"] != TICKET_VERSION:
        _fail("Unknown Worker phase ticket schema")
    if ticket.get("case_id") not in _CASES or ticket.get("phase") not in ("fresh", "resume"):
        _fail("Invalid Worker case/phase")
    for k in ("run_id", "phase_id"):
        if (not isinstance(ticket.get(k), str)
                or not _PHASE_IDENTIFIER.fullmatch(ticket[k])):
            _fail(f"Ticket {k} must be a safe bounded identifier")
    _require_sha(ticket.get("nonce"), name="one-shot ticket nonce")
    for k in ("manifest_sha256", "input_identity_sha256", "effective_argv_sha256"):
        _require_sha(ticket.get(k), name=k)
    _require_sha(ticket.get("source_commit"), name="source_commit", commit=True)
    if ticket.get("worker_entrypoint") != WORKER_ENTRYPOINT:
        _fail("Ticket cannot choose arbitrary Worker entrypoint")
    if ticket["phase"] == "fresh" and ticket.get("expected_resume_checkpoint_id") is not None:
        _fail("Fresh ticket cannot inherit a Checkpoint")
    if ticket["phase"] == "resume" and (
        not isinstance(ticket.get("expected_resume_checkpoint_id"), str)
        or not _CHECKPOINT_IDENTIFIER.fullmatch(ticket["expected_resume_checkpoint_id"])
    ):
        _fail("Resume ticket must bind the exact Fresh 32-hex checkpoint ID")
    _external_work_path(ticket.get("output_dir"), name="ticket output")
    _canonical_json_bytes(ticket)
    return ticket


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    raw = _canonical_json_bytes(payload) + b"\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        raise FullBF16EvidenceError(f"Exclusive ticket/claim creation failed: {path}: {exc}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def issue_phase_ticket(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    ticket = _validate_ticket(payload)
    path = _external_work_path(str(Path(path).absolute()), name="ticket")
    if not path.parent.is_dir():
        _fail("Ticket directory must exist before issuance")
    if path.with_name(path.name + ".consumed").exists():
        _fail("Existing Worker ticket claim prevents issuance")
    _exclusive_json(path, ticket)
    return {"path": str(path), "sha256": _sha256_file(path)}


def claim_phase_ticket(path: Path, *, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate before *atomic* claim; no replay even after Worker crash."""
    expected = _validate_ticket(expected)
    path = _external_work_path(str(Path(path).absolute()), name="ticket")
    actual = _read_json(path)
    _validate_ticket(actual)
    if actual != expected:
        _fail("Worker ticket does not match expected source/manifest/phase identity")
    digest = _sha256_file(path)
    claim = path.with_name(path.name + ".consumed")
    _exclusive_json(claim, {"ticket_sha256": digest,
                            "case_id": actual["case_id"],
                            "phase_id": actual["phase_id"],
                            "run_id": actual["run_id"],
                            "pid": os.getpid()})
    return {"claim_path": str(claim), "ticket_sha256": digest}


def publish_phase_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Flush then atomically publish *without replacing* a final result."""
    path = _external_work_path(str(Path(path).absolute()), name="phase result")
    if not isinstance(payload, dict) or payload.get("schema") != PHASE_SCHEMA or type(payload.get("version")) is not int or payload["version"] != 1:
        _fail("Invalid phase-result schema")
    if payload.get("status") not in ("incomplete", "fail", "worker_complete"):
        _fail("Publishing a Worker result cannot declare final CUDA PASS")
    if path.exists():
        _fail("Refusing to overwrite existing phase result")
    raw = _canonical_json_bytes(payload) + b"\n"
    fd, staging = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic hardlink creation fails with EEXIST; os.replace/rename would
        # overwrite an existing result and is intentionally forbidden.
        os.link(staging, path)
        return {"path": str(path), "sha256": _sha256_file(path)}
    except OSError as exc:
        raise FullBF16EvidenceError(f"Atomic no-overwrite publication failed: {exc}") from exc
    finally:
        try:
            os.unlink(staging)
        except OSError:
            pass


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as stream:
            header_size_bytes = stream.read(8)
            if len(header_size_bytes) != 8:
                _fail("Safetensors is missing its 8-byte header length")
            length = struct.unpack("<Q", header_size_bytes)[0]
            if length == 0 or length > 64 * 1024 * 1024:
                _fail("Unreasonable Safetensors header length")
            raw = stream.read(length)
            if len(raw) != length:
                _fail("Safetensors header truncated")
        header = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(header, dict):
            _fail("Safetensors header must be a mapping")
        return header, 8 + length
    except FullBF16EvidenceError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise FullBF16EvidenceError(f"Invalid Safetensors header: {exc}") from exc


_ADAPTER_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I32": 4, "I64": 8,
}


def _adapter_key_role(key: str) -> tuple[str, str]:
    for suffix, kind in (
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".alpha", "alpha"),
    ):
        if key.endswith(suffix):
            name = key[:-len(suffix)]
            if name.startswith("lora_") and "." not in name:
                return name, kind
    _fail(f"Unrecognized production SD3 LoRA state_dict key: {key}")


def verify_adapter_file(
    path: Path, *, expected_file_sha256: str,
    expected_tensors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify real SD3 LoRA down/up/alpha triplets, with BF16 trainable weights.

    Alpha is a 0-dimensional nontrainable registered buffer whose *actual*
    dtype/bytes must agree with the Worker snapshot. Never cast on Host.
    """
    path = _real_path(str(path), name="adapter", must_exist=True)
    _require_sha(expected_file_sha256, name="adapter file sha256")
    if not path.is_file() or _sha256_file(path) != expected_file_sha256:
        _fail("Adapter file SHA256 does not match saved phase identity")
    if not isinstance(expected_tensors, Mapping) or not expected_tensors:
        _fail("Expected live adapter inventory must be nonempty")
    try:
        from safetensors import safe_open

        with safe_open(path, framework="np", device="cpu") as opened:
            keys = set(opened.keys())
            if keys != set(expected_tensors):
                _fail("Full saved adapter tensor key set differs from live state")
            modules: dict[str, set[str]] = {}
            for key in sorted(keys):
                ref = expected_tensors[key]
                if not isinstance(ref, Mapping):
                    _fail(f"Missing reference tensor descriptor for {key}")
                module, kind = _adapter_key_role(key)
                modules.setdefault(module, set()).add(kind)
                slice_ = opened.get_slice(key)
                shape, dtype = slice_.get_shape(), slice_.get_dtype()
                if shape != ref.get("shape") or dtype != ref.get("dtype"):
                    _fail(f"Saved tensor shape/dtype differ from live snapshot: {key}")
                role = ref.get("role")
                if kind == "alpha":
                    if (role != "alpha_buffer" or shape != []
                            or dtype not in _ADAPTER_DTYPE_BYTES):
                        _fail(f"Unsupported SD3 scalar alpha buffer: {key}")
                else:
                    if (role not in ("trainable_weight", "frozen_weight")
                            or len(shape) not in (2, 4)
                            or any(type(x) is not int or x <= 0 for x in shape)):
                        _fail(f"Unreviewed SD3 LoRA weight role/dimensions: {key}")
                    if role == "trainable_weight" and dtype != "BF16":
                        _fail(f"Trainable LoRA weight is not BF16: {key}")
                    if dtype not in ("BF16", "F32"):
                        _fail(f"Unreviewed LoRA weight dtype: {key}")
            if not modules or any(v != {"down", "up", "alpha"} for v in modules.values()):
                _fail("Incomplete saved SD3 LoRA down/up/alpha module triplet")

        header, data_start = _safetensors_header(path)
        if set(header) - {"__metadata__"} != keys:
            _fail("Raw Safetensors tensor keys disagree with checked keys")
        byte_ranges: list[tuple[int, int]] = []
        with path.open("rb") as stream:
            for key in sorted(keys):
                ref, entry = expected_tensors[key], header[key]
                offsets, shape, dtype = entry.get("data_offsets"), entry.get("shape"), entry.get("dtype")
                if (not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(x) is not int for x in offsets)
                        or offsets[0] < 0 or offsets[1] <= offsets[0]
                        or not isinstance(shape, list)
                        or any(type(x) is not int or x <= 0 for x in shape)
                        or dtype != ref.get("dtype") or shape != ref.get("shape")
                        or dtype not in _ADAPTER_DTYPE_BYTES):
                    _fail(f"Malformed or mismatched Safetensors header: {key}")
                _module, kind = _adapter_key_role(key)
                if kind == "alpha" and shape != []:
                    _fail(f"Alpha must be scalar: {key}")
                if kind != "alpha" and len(shape) not in (2, 4):
                    _fail(f"Unsupported weight rank: {key}")
                elements = 1
                for axis in shape:
                    elements *= axis
                length = elements * _ADAPTER_DTYPE_BYTES[dtype]
                if offsets[1] - offsets[0] != length:
                    _fail(f"Invalid dtype-aware tensor byte count: {key}")
                byte_ranges.append((offsets[0], offsets[1]))
                stream.seek(data_start + offsets[0])
                digest = hashlib.sha256()
                left = length
                while left:
                    piece = stream.read(min(left, 1024 * 1024))
                    if not piece:
                        _fail(f"Truncated tensor data: {key}")
                    digest.update(piece)
                    left -= len(piece)
                _require_sha(ref.get("sha256"), name=f"{key} live tensor hash")
                if digest.hexdigest() != ref["sha256"]:
                    _fail(f"Tensor data does not equal live snapshot: {key}")
        end = 0
        for start, stop in sorted(byte_ranges):
            if start != end:
                _fail("Safetensors contains unaccounted or overlapping data")
            end = stop
        if data_start + end != path.stat().st_size:
            _fail("Safetensors trailing data is not in the live inventory")
    except FullBF16EvidenceError:
        raise
    except (OSError, RuntimeError, ImportError, ValueError, KeyError, TypeError) as exc:
        raise FullBF16EvidenceError(f"Cannot verify SD3 adapter file: {exc}") from exc
    return {
        "status": "file_verified",
        "file_sha256": expected_file_sha256,
        "tensor_inventory_sha256": _json_sha256(expected_tensors),
        "tensor_count": len(expected_tensors),
    }



def build_sd35_medium_lora_target_spec() -> dict[str, dict[str, Any]]:
    """Pure Host, fixed SD3.5 Medium reference target geometry.

    24 joint blocks, pre_only context at block 23, x.attn2 in 0..12,
    hidden 1536, MLP expansion 4, rank 4. Independently check every
    target name/shape against the *real* LoRANetwork in the CPU test.
    """
    hidden, mlp, rank = 1536, 6144, 4
    targets: dict[str, dict[str, Any]] = {}

    def put(block: int, side: str, subpath: str, component: str,
            inputs: int, outputs: int) -> None:
        name = f"lora_unet_joint_blocks_{block}_{side}_{subpath}"
        if name in targets:
            _fail(f"Duplicate LoRA reference module: {name}")
        targets[name] = {
            "component_id": component, "input_features": inputs,
            "output_features": outputs, "rank": rank,
        }

    for block in range(24):
        for side in ("context_block", "x_block"):
            context_last = side == "context_block" and block == 23
            extra = side == "x_block" and block <= 12
            put(block, side, "attn_qkv", "mmdit.attention.adapter", hidden, 3 * hidden)
            if not context_last:
                put(block, side, "attn_proj", "mmdit.attention.adapter", hidden, hidden)
            if extra:
                put(block, side, "attn2_qkv", "mmdit.attention.adapter", hidden, 3 * hidden)
                put(block, side, "attn2_proj", "mmdit.attention.adapter", hidden, hidden)
            if not context_last:
                put(block, side, "mlp_fc1", "mmdit.mlp.adapter", hidden, mlp)
                put(block, side, "mlp_fc2", "mmdit.mlp.adapter", mlp, hidden)
            n_mods = 2 if context_last else 9 if extra else 6
            put(block, side, "adaLN_modulation_1", "mmdit.modulation_norm.adapter",
                hidden, n_mods * hidden)
    counts: dict[str, int] = {}
    for entry in targets.values():
        key = entry["component_id"]
        counts[key] = counts.get(key, 0) + 1
    if (len(targets) != 263 or counts != {
        "mmdit.attention.adapter": 121,
        "mmdit.mlp.adapter": 94,
        "mmdit.modulation_norm.adapter": 48,
    }):
        _fail("SD3.5 Medium production target enumeration has drifted")
    return dict(sorted(targets.items()))


def verify_b0_reference_adapter(
    path: Path, *, expected_file_sha256: str,
    expected_tensors: Mapping[str, Mapping[str, Any]],
    expected_targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Strict B0-only rank 4 / alpha 4 / full BF16 / Unet-only file gate."""
    file_result = verify_adapter_file(
        path, expected_file_sha256=expected_file_sha256,
        expected_tensors=expected_tensors,
    )
    canonical = build_sd35_medium_lora_target_spec()
    if expected_targets != canonical:
        _fail("Reference target spec differs from frozen audited SD3.5 Medium")
    expected_keys = {
        name + suffix for name in canonical
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")
    }
    if set(expected_tensors) != expected_keys:
        _fail(f"B0 complete target tensor set differs: "
              f"missing={len(expected_keys-set(expected_tensors))}; "
              f"extra={len(set(expected_tensors)-expected_keys)}")
    bf16_four_digest = _sha256(b"\x80\x40")
    for module, target in canonical.items():
        if not module.startswith("lora_unet_"):
            _fail(f"Non-Unet LoRA target is not allowed: {module}")
        rank = target["rank"]
        for suffix, shape, role in (
            (".lora_down.weight", [rank, target["input_features"]], "trainable_weight"),
            (".lora_up.weight", [target["output_features"], rank], "trainable_weight"),
            (".alpha", [], "alpha_buffer"),
        ):
            key = module + suffix
            ref = expected_tensors[key]
            if (not isinstance(ref, Mapping)
                    or ref.get("dtype") != "BF16"
                    or ref.get("shape") != shape
                    or ref.get("role") != role):
                _fail(f"Full BF16/R4/reference geometry mismatch: {key}")
            if suffix == ".alpha" and ref.get("sha256") != bf16_four_digest:
                _fail(f"Noncanonical BF16 alpha=4 scalar: {key}")
    # verify_adapter_file has already hashed the actual raw saved Tensor
    # bytes for every key. Matching BF16(4) sha means actual scalar=4.
    return {
        **file_result,
        "status": "b0_reference_file_verified",
        "reference_target_spec_sha256": _json_sha256(canonical),
        "module_count": len(canonical),
        "target_tensor_count": len(expected_keys),
    }

def validate_adapter_checkpoint_metadata(
    binding: object, progress: object, *,
    expected_case_id: str, expected_phase_id: str,
    expected_step: int, expected_adapter_path: str | Path,
) -> dict[str, Any]:
    """Metadata-only comparison: NEVER returns physical evidence PASS."""
    if not isinstance(binding, dict) or not isinstance(progress, dict):
        _fail("Adapter metadata and checkpoint progress must be objects")
    get_case_spec(expected_case_id)
    if type(expected_step) is not int or expected_step not in (1, 2):
        _fail("Invalid reference optimizer step")
    if (progress.get("schema") != PROGRESS_SCHEMA
            or type(progress.get("version")) is not int or progress["version"] != 1):
        _fail("Wrong checkpoint progress schema/version")
    if (binding.get("case_id") != expected_case_id
            or binding.get("phase_id") != expected_phase_id
            or type(binding.get("logical_step")) is not int
            or binding["logical_step"] != expected_step):
        _fail("Adapter metadata case/phase/step differs from trusted expectation")
    checkpoint_id = progress.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id or binding.get("checkpoint_id") != checkpoint_id:
        _fail("Adapter checkpoint ID differs from progress record")
    adapter = _real_path(binding.get("adapter_path"), name="adapter metadata")
    expected = _real_path(str(expected_adapter_path), name="expected adapter")
    if adapter != expected or not adapter.name.endswith(f"-step{expected_step:08d}.safetensors"):
        _fail("Adapter path does not match exact Manifest step artifact")
    for key in ("adapter_file_sha256", "adapter_tensor_inventory_sha256"):
        _require_sha(binding.get(key), name=key)
    return {"status": "metadata_valid", "case_id": expected_case_id,
            "checkpoint_id": checkpoint_id, "logical_step": expected_step}


_SD3_STATE_FILES = frozenset({
    # Single-process Accelerate 1.6.0: one prepared Network and one
    # CompositeOptimizer/Scheduler, NOT one state file per optimizer Child.
    "model.safetensors", "optimizer.bin", "scheduler.bin",
    "random_states_0.pkl", "train_state.json",
    "dts_parameter_policy_manifest.json",
    "dts_parameter_policy_progress.json",
    "dts_sd3_resume_progress.json",
    "dts_sd3_forward_semantics.json",
})
_SD3_TRAINED_COMPONENTS = frozenset({
    "mmdit.attention.adapter", "mmdit.mlp.adapter",
    "mmdit.modulation_norm.adapter",
})


def _checkpoint_file_inventory(directory: Path) -> dict[str, str]:
    """Exact pinned Accelerate+SD3 Hook layout. Never unpickle on Host.

    Missing RNG does not always raise on Accelerate load_state, but is
    forbidden here because deterministic restored RNG is an evidence goal.
    """
    result: dict[str, str] = {}
    for item in directory.iterdir():
        _assert_ordinary_dataset_entry(item, is_directory=False)
        if item.name not in _SD3_STATE_FILES:
            _fail(f"Unreviewed Accelerate/SD3 state file: {item.name}")
        result[item.name] = _sha256_file(item)
    if set(result) != _SD3_STATE_FILES:
        _fail(f"Incomplete single-process SD3 state: missing={sorted(_SD3_STATE_FILES - set(result))}")
    return dict(sorted(result.items()))


def _expected_reference_scheduler_identity() -> dict[str, Any]:
    """Frozen R1/R2 fixed CLI + production argparse defaults and identity v1."""
    return {
        "schema": "dts.parameter-policy.scheduler-identity",
        "version": 1, "provider": "sd-scripts.get_scheduler_fix",
        "lr_scheduler": "constant", "lr_scheduler_type": "",
        "lr_scheduler_args": {}, "max_train_steps": 2,
        "num_processes": 1, "num_training_steps": 2,
        "num_warmup_steps": 0, "num_decay_steps": 0,
        "num_stable_steps": 2, "num_cycles": 1,
        "power": 1.0, "timescale": None, "min_lr_ratio": None,
    }


def _verify_reference_scheduler_identity(
    manifest: Mapping[str, Any], snapshot: Mapping[str, Any],
) -> None:
    expected = _expected_reference_scheduler_identity()
    if (manifest.get("scheduler_identity") != expected
            or manifest.get("scheduler_signature") != _json_sha256(expected)
            or snapshot.get("scheduler_identity") != expected
            or snapshot.get("scheduler_signature") != _json_sha256(expected)):
        _fail("Self-signed Scheduler differs from the reference Constant configuration")
    children = manifest.get("schedulers")
    if not isinstance(children, list):
        _fail("Missing physical Scheduler children")
    for child in children:
        if (not isinstance(child, Mapping) or child.get("mode") != "external"
                or child.get("scheduler_type") != "torch.optim.lr_scheduler.LambdaLR"):
            _fail("Every R1/R2 child must use the pinned external Constant LambdaLR")


def _verify_reference_forward_identity(
    saved: Mapping[str, Any], snapshot: Mapping[str, Any],
) -> None:
    # sd3_utils.analyze_state_dict sd35-medium => model_type "5-medium".
    # Full forward_semantics_for_mmdit() is checked, not just selected fields.
    expected = {
        "schema": "dts.sd3.forward-semantics", "version": 1,
        "attention_contract": "extra-attention-head-dim-v1",
        "model_type_hint": "5-medium",
        "depth": 24, "hidden_size": 1536,
        "num_heads": 24, "head_dim": 64,
        "extra_attention_layers": list(range(13)),
    }
    if saved != expected or snapshot.get("forward_semantics") != expected:
        _fail("Saved/observed full SD3.5 Medium Forward Semantics mismatch")


def verify_reference_parameter_ownership(
    *, case_manifest: Mapping[str, Any], worker_snapshot: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any], adapter_tensors: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure Host coherence for prepared parameter/Child ownership metadata.

    Fix C MUST source the input from real Session.routing_plan, RuntimeSpec
    and physical AcceleratedOptimizer child parameter IDs. Host evidence
    shape validation alone does not prove genuine physical ownership.
    """
    policy = build_reference_policy(case_manifest.get("case_id"))
    profiles = {k: v["type"] for k, v in policy["optimizer_profiles"].items()}
    ownership = worker_snapshot.get("parameter_ownership")
    if (not isinstance(ownership, Mapping)
            or ownership.get("source") != "prepared-runtime-ownership-v1"
            or ownership.get("network_split_qkv") is not False
            or ownership.get("runtime_topology_fingerprint")
            != checkpoint_manifest.get("runtime_topology_fingerprint")):
        _fail("Unverified prepared, non-Split-QKV parameter ownership")
    rows = ownership.get("parameter_rows")
    children = ownership.get("optimizer_child_parameter_names")
    if not isinstance(rows, Mapping) or not isinstance(children, Mapping):
        _fail("Missing prepared parameter rows/physical child groups")
    saved_parameters = {k for k in adapter_tensors if k.endswith(".weight")}
    if set(rows) != saved_parameters or set(children) != set(profiles):
        _fail("Saved LoRA weight keys or physical Optimizer Profiles differ")
    seen: set[str] = set()
    by_component: dict[str, int] = {}
    expected_children: dict[str, set[str]] = {name: set() for name in profiles}
    trainable_count = trainable_elements = 0
    for key in sorted(saved_parameters):
        row, tensor = rows[key], adapter_tensors[key]
        if not isinstance(row, Mapping) or not isinstance(tensor, Mapping):
            _fail("Missing ownership row or tensor descriptor")
        canonical = row.get("canonical_name")
        component = row.get("component_id")
        route_kind = row.get("route_kind")
        child = row.get("optimizer_profile")
        if (canonical != "network." + key or canonical in seen
                or row.get("shape") != tensor.get("shape")
                or row.get("dtype") != tensor.get("dtype")
                or row.get("parameter_class") != "matrix_weight"
                or component not in policy["components"]):
            _fail(f"Prepared ownership differs from canonical saved parameter: {key}")
        seen.add(canonical)
        route = policy["components"][component]
        if route["train"] is True:
            expected_child = (route["optimizer_profile"] if route_kind == "primary"
                              else route.get("fallback_optimizer_profile") if route_kind == "fallback"
                              else None)
            if (row.get("requires_grad") is not True
                    or tensor.get("role") != "trainable_weight"
                    or tensor.get("dtype") != "BF16"
                    or child != expected_child or child not in profiles):
                _fail(f"Trainable BF16 weight/Optimizer ownership was falsely frozen: {key}")
            expected_children[child].add(canonical)
            by_component[component] = by_component.get(component, 0) + 1
            trainable_count += 1
            elements = 1
            for dim in tensor["shape"]:
                if type(dim) is not int or dim <= 0:
                    _fail(f"Invalid trainable Parameter geometry: {key}")
                elements *= dim
            trainable_elements += elements
        elif (row.get("requires_grad") is not False
              or tensor.get("role") != "frozen_weight"
              or child is not None
              or route_kind not in ("frozen", "unavailable")):
            _fail(f"Frozen Parameter leaked into physical Optimizer: {key}")
    if set(by_component) != _SD3_TRAINED_COMPONENTS:
        _fail("At least one mandatory trained MMDiT Component has no BF16 parameters")
    for name, expected in expected_children.items():
        actual = children.get(name)
        if (not isinstance(actual, list)
                or any(not isinstance(t, str) for t in actual)
                or len(actual) != len(set(actual))
                or set(actual) != expected or not actual):
            _fail(f"Duplicate, missing or wrong actual Optimizer Child parameters: {name}")
    if (checkpoint_manifest.get("trainable_parameter_tensors") != trainable_count
            or checkpoint_manifest.get("trainable_parameter_elements") != trainable_elements):
        _fail("Checkpoint trainable tensor counts disagree with ownership")
    return {
        "status": "ownership_metadata_valid",
        "trainable_parameter_tensors": trainable_count,
        "trainable_parameter_elements": trainable_elements,
        "trained_components": dict(sorted(by_component.items())),
        "optimizer_children": {k: sorted(v) for k, v in sorted(expected_children.items())},
    }


def _verify_checkpoint_progress_consistency(
    progress: Mapping[str, Any], *, expected_step: int,
) -> None:
    scheduler = progress.get("scheduler")
    if (not isinstance(scheduler, Mapping)
            or type(scheduler.get("step_count")) is not int
            or type(scheduler.get("last_epoch")) is not int
            or scheduler["step_count"] != expected_step + 1
            or scheduler["last_epoch"] != expected_step):
        _fail("Constant LambdaLR Scheduler progress differs from initialization offset + real steps")
    profiles = progress.get("optimizer_profiles")
    if not isinstance(profiles, Mapping):
        _fail("No actual Optimizer Profile progress")
    for name, child in profiles.items():
        if not isinstance(child, Mapping):
            _fail("Invalid child optimizer progress")
        values = child.get("step_values")
        if (not isinstance(values, list)
                or any(type(t) is not int or t < 1 or t > expected_step for t in values)):
            _fail(f"Invalid optimizer state Step value for {name}")
        # AdamW state.step is reliable; Muon 3.10.0 may not retain it.
        if child.get("optimizer_type") == "AdamW" and (
            not values or set(values) != {expected_step}
        ):
            _fail(f"AdamW child progress does not equal actual completed steps: {name}")


def _verify_resume_fresh_parent(
    *, case_manifest: Mapping[str, Any],
    accepted_fresh_evidence: Mapping[str, Any] | None,
    resume_progress: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> str:
    if (not isinstance(accepted_fresh_evidence, Mapping)
            or accepted_fresh_evidence.get("status") != "physical_binding_valid"
            or accepted_fresh_evidence.get("case_id") != case_manifest.get("case_id")
            or accepted_fresh_evidence.get("logical_step") != 1):
        _fail("Resume requires independently accepted Fresh Step-1 evidence")
    root = _real_path(case_manifest.get("fresh_checkpoint_dir"), name="Fresh Step-1", must_exist=True)
    if not root.is_dir():
        _fail("Fresh Step-1 checkpoint disappeared")
    files = _checkpoint_file_inventory(root)
    if files != accepted_fresh_evidence.get("checkpoint_file_sha256"):
        _fail("Fresh Step-1 files changed since Host acceptance")
    progress = _read_json(root / "dts_parameter_policy_progress.json")
    manifest = _read_json(root / "dts_parameter_policy_manifest.json")
    try:
        from tools.parameter_policy_backend_feature_gpu_support import (
            validate_full_bf16_checkpoint_manifest, validate_checkpoint_progress,
        )
        validate_full_bf16_checkpoint_manifest(manifest, train_type="sd3-lora", manifest_version=2)
        validate_checkpoint_progress(progress, checkpoint_manifest=manifest)
    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
        raise FullBF16EvidenceError(f"Actual Fresh parent sidecars invalid: {exc}") from exc
    parent_id = progress.get("checkpoint_id")
    if (not isinstance(parent_id, str) or not parent_id
            or parent_id != accepted_fresh_evidence.get("checkpoint_id")
            or progress.get("resume_source_checkpoint_id") is not None):
        _fail("Fresh parent ID/lineage is not the accepted Step-1 state")
    if (manifest.get("policy_hash") != case_manifest["input_identity"]["canonical_policy_hash"]
            or manifest.get("execution_signature") != accepted_fresh_evidence.get("execution_signature")
            or manifest.get("runtime_topology_fingerprint")
            != accepted_fresh_evidence.get("runtime_topology_fingerprint")):
        _fail("Fresh Parent Policy/Execution/Topology differs from accepted evidence")
    if (resume_progress.get("resume_source_checkpoint_id") != parent_id
            or snapshot.get("expected_parent_checkpoint_id") != parent_id
            or resume_progress.get("checkpoint_id") == parent_id):
        _fail("Resumed checkpoint does not descend from actual Fresh Step-1")
    return parent_id


def verify_adapter_checkpoint_binding(
    *, case_manifest: dict[str, Any], phase: str,
    worker_snapshot: Mapping[str, Any], binding: Mapping[str, Any],
    expected_phase_id: str | None = None,
    accepted_fresh_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Physical Host file/lineage binding, NOT a CUDA qualification decision.

    The future Runner must independently supply expected_phase_id from its
    already-claimed phase ticket and verify the Worker event stream/exit.
    """
    if phase not in ("fresh", "resume") or not expected_phase_id:
        _fail("Physical binding requires a trusted phase and phase-ticket ID")
    if not isinstance(worker_snapshot, Mapping) or not isinstance(binding, Mapping):
        _fail("Missing actual Worker phase snapshot/binding")
    # Fresh Step-1 completion is Resume preflight; resumed Step-2 is complete.
    boundary = "resume" if phase == "fresh" else "complete"
    inputs = verify_case_inputs(case_manifest, phase=boundary)
    case_id = inputs["case_id"]
    expected_step = 1 if phase == "fresh" else 2
    manifest_view = validate_case_manifest(case_manifest, phase=boundary)
    checkpoint = _real_path(
        case_manifest["fresh_checkpoint_dir" if phase == "fresh" else "resume_checkpoint_dir"],
        name="expected state directory", must_exist=True,
    )
    expected_adapter_path = _real_path(
        case_manifest["fresh_adapter_path" if phase == "fresh" else "resume_adapter_path"],
        name="expected adapter", must_exist=True,
    )
    if (not checkpoint.is_dir() or not expected_adapter_path.is_file()
            or worker_snapshot.get("case_id") != case_id
            or worker_snapshot.get("phase_id") != expected_phase_id
            or worker_snapshot.get("logical_step") != expected_step
            or worker_snapshot.get("source_commit") != case_manifest["source_commit"]
            or worker_snapshot.get("input_identity_sha256") != inputs["input_identity_sha256"]):
        _fail("Physical Worker snapshot is not bound to the expected source/case/phase")
    pmanifest = _read_json(checkpoint / "dts_parameter_policy_manifest.json")
    progress = _read_json(checkpoint / "dts_parameter_policy_progress.json")
    sd3_progress = _read_json(checkpoint / "dts_sd3_resume_progress.json")
    train_state = _read_json(checkpoint / "train_state.json")
    semantics = _read_json(checkpoint / "dts_sd3_forward_semantics.json")
    validate_adapter_checkpoint_metadata(
        dict(binding), progress, expected_case_id=case_id,
        expected_phase_id=expected_phase_id, expected_step=expected_step,
        expected_adapter_path=expected_adapter_path,
    )
    try:
        from tools.parameter_policy_backend_feature_gpu_support import (
            validate_full_bf16_checkpoint_manifest,
            validate_checkpoint_progress,
        )
        validate_full_bf16_checkpoint_manifest(
            pmanifest, train_type="sd3-lora", manifest_version=2,
        )
        validate_checkpoint_progress(progress, checkpoint_manifest=pmanifest)
    except FullBF16EvidenceError:
        raise
    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
        raise FullBF16EvidenceError(f"Full-BF16 Manifest/Progress contract failed: {exc}") from exc

    canonical_policy = manifest_view["policy"]
    expected_train = sorted(
        name for name, entry in canonical_policy["components"].items() if entry["train"]
    )
    expected_frozen = sorted(
        name for name, entry in canonical_policy["components"].items() if not entry["train"]
    )
    expected_profiles = {
        name: record["type"] for name, record in canonical_policy["optimizer_profiles"].items()
    }
    if (pmanifest.get("policy_hash") != manifest_view["policy_hash"]
            or pmanifest.get("trainable_components") != expected_train
            or pmanifest.get("frozen_components") != expected_frozen
            or pmanifest.get("optimizer_profiles") != expected_profiles
            or pmanifest.get("runtime_topology_fingerprint") != worker_snapshot.get("runtime_topology_fingerprint")):
        _fail("Physical checkpoint ownership/Policy/runtime topology does not match reference")
    actual_optimizers = pmanifest.get("optimizers")
    actual_schedulers = pmanifest.get("schedulers")
    if (not isinstance(actual_optimizers, list) or not isinstance(actual_schedulers, list)
            or {o.get("profile_name") for o in actual_optimizers if isinstance(o, dict)} != set(expected_profiles)
            or {o.get("profile_name") for o in actual_schedulers if isinstance(o, dict)} != set(expected_profiles)
            or len(actual_optimizers) != len(expected_profiles)
            or len(actual_schedulers) != len(expected_profiles)):
        _fail("Physical checkpoint Optimizer/Scheduler child Profile topology mismatch")
    for child in actual_optimizers:
        if (not isinstance(child, dict)
                or child.get("optimizer_type") != expected_profiles.get(child.get("profile_name"))
                or not isinstance(child.get("topology_fingerprint"), str)
                or not child["topology_fingerprint"]):
            _fail("Physical Optimizer child type/topology differs from canonical Policy")
    for child in actual_schedulers:
        if (not isinstance(child, dict)
                or not isinstance(child.get("scheduler_type"), str)
                or not child["scheduler_type"]
                or not isinstance(child.get("mode"), str)
                or not child["mode"]
                or not isinstance(child.get("topology_fingerprint"), str)
                or not child["topology_fingerprint"]):
            _fail("Physical Scheduler child identity is missing required topology evidence")

    expected_parent = worker_snapshot.get("expected_parent_checkpoint_id")
    if phase == "fresh":
        if expected_parent is not None or progress.get("resume_source_checkpoint_id") is not None:
            _fail("Fresh checkpoint must not claim a resume parent")
    else:
        _verify_resume_fresh_parent(
            case_manifest=case_manifest,
            accepted_fresh_evidence=accepted_fresh_evidence,
            resume_progress=progress,
            snapshot=worker_snapshot,
        )
    if progress.get("checkpoint_id") != worker_snapshot.get("checkpoint_id"):
        _fail("Worker checkpoint ID differs from actual on-disk Progress sidecar")
    spec = manifest_view["case_spec"]
    expected_sd3 = {
        "schema": "dts.sd3.resume-progress",
        "version": 1, "completed_optimizer_steps": expected_step,
        "current_epoch": expected_step,
        "dataloader_batches": spec["effective_batches_per_epoch"],
        "gradient_accumulation_steps": spec["gradient_accumulation_steps"],
        "num_processes": 1,
    }
    if (sd3_progress != expected_sd3
            or type(train_state.get("current_step")) is not int
            or type(train_state.get("current_epoch")) is not int
            or train_state["current_step"] != expected_step
            or train_state["current_epoch"] != expected_step):
        _fail("Actual SD3 Resume cursor does not match physical reference geometry")
    _verify_reference_forward_identity(semantics, worker_snapshot)
    _verify_reference_scheduler_identity(pmanifest, worker_snapshot)
    _verify_checkpoint_progress_consistency(progress, expected_step=expected_step)

    all_files = _checkpoint_file_inventory(checkpoint)
    if (not isinstance(worker_snapshot.get("state_file_sha256"), dict)
            or all_files != worker_snapshot["state_file_sha256"]):
        _fail("Real checkpoint state-file digest inventory differs from Worker snapshot")
    expected_tensors = worker_snapshot.get("adapter_tensors")
    if not isinstance(expected_tensors, Mapping) or not expected_tensors:
        _fail("Missing real prepared-network adapter Tensor snapshot")
    ownership_check = verify_reference_parameter_ownership(
        case_manifest=case_manifest, worker_snapshot=worker_snapshot,
        checkpoint_manifest=pmanifest, adapter_tensors=expected_tensors,
    )
    if _json_sha256(expected_tensors) != binding.get("adapter_tensor_inventory_sha256"):
        _fail("Adapter Tensor inventory hash differs from real Worker snapshot")
    if binding.get("adapter_file_sha256") != _sha256_file(expected_adapter_path):
        _fail("Actual Step adapter file bytes differ from bound evidence")
    adapter_check = verify_adapter_file(
        expected_adapter_path, expected_file_sha256=binding["adapter_file_sha256"],
        expected_tensors=expected_tensors,
    )
    # Accelerate's prepared Network state must match the same-step LoRA
    # snapshot, without assuming file-level metadata is byte-identical.
    network_check = verify_adapter_file(
        checkpoint / "model.safetensors",
        expected_file_sha256=all_files["model.safetensors"],
        expected_tensors=expected_tensors,
    )
    return {
        "status": "physical_binding_valid",
        "case_id": case_id, "phase_id": expected_phase_id,
        "logical_step": expected_step, "checkpoint_id": progress["checkpoint_id"],
        "adapter_file_sha256": adapter_check["file_sha256"],
        "adapter_tensor_inventory_sha256": adapter_check["tensor_inventory_sha256"],
        "checkpoint_file_sha256": all_files,
        "network_state_sha256": network_check["file_sha256"],
        "execution_signature": pmanifest["execution_signature"],
        "runtime_topology_fingerprint": pmanifest["runtime_topology_fingerprint"],
        "scheduler_signature": pmanifest["scheduler_signature"],
        "parameter_ownership_summary": ownership_check,
    }


__all__ = [
    "FullBF16EvidenceError", "get_case_spec", "build_reference_policy",
    "validate_case_manifest", "validate_case_assets", "verify_case_inputs",
    "decoded_rgb_pixel_sha256", "validate_qualification_preconditions",
    "validate_qualification_lease_records", "issue_phase_ticket",
    "claim_phase_ticket", "publish_phase_result", "verify_adapter_file",
    "validate_adapter_checkpoint_metadata",
    "verify_reference_parameter_ownership", "verify_adapter_checkpoint_binding",
]
