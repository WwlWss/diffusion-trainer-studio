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
import secrets
import struct
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
    actual_dataset = work / "data" / "1_dts_sd3c2b"
    actual = {p.relative_to(work).as_posix() for p in actual_dataset.iterdir()}
    if actual != set(identity["dataset_files"]):
        _fail("Unrecorded, removed or extra DreamBooth dataset file")
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
    for k in ("run_id", "phase_id", "nonce"):
        if not isinstance(ticket.get(k), str) or not ticket[k]:
            _fail(f"Missing ticket {k}")
    for k in ("manifest_sha256", "input_identity_sha256", "effective_argv_sha256"):
        _require_sha(ticket.get(k), name=k)
    _require_sha(ticket.get("source_commit"), name="source_commit", commit=True)
    if ticket.get("worker_entrypoint") != WORKER_ENTRYPOINT:
        _fail("Ticket cannot choose arbitrary Worker entrypoint")
    if ticket["phase"] == "fresh" and ticket.get("expected_resume_checkpoint_id") is not None:
        _fail("Fresh ticket cannot inherit a Checkpoint")
    if ticket["phase"] == "resume" and not isinstance(ticket.get("expected_resume_checkpoint_id"), str):
        _fail("Resume ticket must bind a real Fresh checkpoint ID")
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


def verify_adapter_file(
    path: Path, *, expected_file_sha256: str,
    expected_tensors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check *all* raw BF16 tensor bytes without importing Torch on Host."""
    path = _real_path(str(path), name="adapter", must_exist=True)
    _require_sha(expected_file_sha256, name="adapter file sha256")
    if not path.is_file() or _sha256_file(path) != expected_file_sha256:
        _fail("Adapter file bytes differ from the same-step snapshot")
    if not isinstance(expected_tensors, Mapping) or not expected_tensors:
        _fail("Adapter reference tensor inventory must be nonempty")
    try:
        from safetensors import safe_open

        with safe_open(path, framework="np", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != set(expected_tensors):
                _fail("Saved adapter full tensor-key set differs from live state")
            for key in keys:
                part = handle.get_slice(key)
                if part.get_dtype() != "BF16":
                    _fail(f"Adapter parameter must be BF16: {key}")
                ref = expected_tensors[key]
                if (ref.get("dtype") != "BF16" or list(ref.get("shape", ())) != part.get_shape()):
                    _fail(f"Adapter dtype/shape differs from live state: {key}")
        header, data_start = _safetensors_header(path)
        if set(header) - {"__metadata__"} != keys:
            _fail("Safetensors header keys differ from payload keys")
        with path.open("rb") as file:
            for key in sorted(keys):
                entry = header[key]
                offsets = entry.get("data_offsets")
                shape = entry.get("shape")
                if (entry.get("dtype") != "BF16" or not isinstance(offsets, list)
                        or len(offsets) != 2 or any(type(o) is not int for o in offsets)
                        or offsets[0] < 0 or offsets[1] <= offsets[0]
                        or type(shape) is not list or any(type(o) is not int or o < 1 for o in shape)):
                    _fail(f"Malformed BF16 tensor header: {key}")
                if len(shape) != 2 and len(shape) != 1 and len(shape) != 3 and len(shape) != 4:
                    _fail(f"Unreviewed Adapter tensor rank: {key}")
                n = 1
                for axis in shape:
                    n *= axis
                if offsets[1] - offsets[0] != n * 2:
                    _fail(f"BF16 tensor byte size mismatch: {key}")
                file.seek(data_start + offsets[0])
                left = offsets[1] - offsets[0]
                h = hashlib.sha256()
                while left:
                    part = file.read(min(1024 * 1024, left))
                    if not part:
                        _fail(f"Truncated Adapter tensor data: {key}")
                    h.update(part)
                    left -= len(part)
                _require_sha(expected_tensors[key].get("sha256"), name=f"live {key} fingerprint")
                if h.hexdigest() != expected_tensors[key]["sha256"]:
                    _fail(f"Adapter saved tensor differs from live BF16 state: {key}")
    except FullBF16EvidenceError:
        raise
    except (OSError, RuntimeError, ImportError, ValueError, KeyError, TypeError) as exc:
        raise FullBF16EvidenceError(f"Cannot verify real BF16 Adapter file: {exc}") from exc
    return {"status": "pass", "file_sha256": expected_file_sha256,
            "tensor_inventory_sha256": _json_sha256(expected_tensors),
            "tensor_count": len(expected_tensors)}


def validate_adapter_checkpoint_binding(
    binding: object, progress: object, *,
    expected_phase_id: str, expected_step: int,
) -> dict[str, Any]:
    if not isinstance(binding, dict) or not isinstance(progress, dict):
        _fail("Adapter/checkpoint binding must use dict records")
    if type(expected_step) is not int or expected_step not in (1, 2):
        _fail("Reference step must be 1 or 2")
    if progress.get("schema") != PROGRESS_SCHEMA or type(progress.get("version")) is not int or progress["version"] != 1:
        _fail("Expected real checkpoint-progress v1 sidecar")
    if (binding.get("phase_id") != expected_phase_id
            or type(binding.get("logical_step")) is not int
            or binding["logical_step"] != expected_step):
        _fail("Adapter phase/step identity does not match current Worker phase")
    if not isinstance(binding.get("case_id"), str) or binding["case_id"] not in _CASES:
        _fail("Invalid Adapter case ID")
    checkpoint = progress.get("checkpoint_id")
    if not isinstance(checkpoint, str) or not checkpoint or binding.get("checkpoint_id") != checkpoint:
        _fail("Saved adapter does not match actual checkpoint identity")
    adapter = _real_path(binding.get("adapter_path"), name="adapter binding")
    if not adapter.name.endswith(f"-step{expected_step:08d}.safetensors"):
        _fail("Adapter filename does not identify the expected physical Step")
    for key in ("adapter_file_sha256", "adapter_tensor_inventory_sha256"):
        _require_sha(binding.get(key), name=key)
    return {"status": "pass", "checkpoint_id": checkpoint,
            "case_id": binding["case_id"], "logical_step": expected_step}


__all__ = [
    "FullBF16EvidenceError", "get_case_spec", "build_reference_policy",
    "validate_case_manifest", "validate_case_assets", "verify_case_inputs",
    "decoded_rgb_pixel_sha256", "validate_qualification_preconditions",
    "validate_qualification_lease_records", "issue_phase_ticket",
    "claim_phase_ticket", "publish_phase_result", "verify_adapter_file",
    "validate_adapter_checkpoint_binding",
]
