"""Generate isolated SD3.5 Medium Full-BF16 R1/R2 Host evidence cases.

B0 preparation only: no CUDA, no Torch import, no temporary qualification,
no model weight-value loading and no production-training subprocess.
Header/architecture and real GPU validation belong to later preflight/Worker.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mikazuki.parameter_policy import serialize_parameter_policy
from tools import sd3_full_bf16_evidence_support as support


def _source_commit() -> str:
    """Never attribute dirty Trainer/tool/submodule code to a clean Git SHA."""
    def git(*args: str) -> str:
        try:
            output = subprocess.run(
                ["git", *args], cwd=ROOT,
                capture_output=True, text=True, check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise support.FullBF16EvidenceError(
                f"Cannot establish clean B0 source identity: {args}: {exc}"
            ) from exc
        return output.stdout.strip()

    head = support._require_sha(
        git("rev-parse", "--verify", "HEAD"), name="source HEAD", commit=True,
    )
    if git("status", "--porcelain=v1", "--untracked-files=all",
           "--ignore-submodules=none"):
        raise support.FullBF16EvidenceError(
            "B0 evidence must not attribute modified/untracked source to a clean HEAD"
        )
    # An uninitialized checkout (-) is not used by this Host generator.
    # Any mismatched (+) or conflicted (U) gitlink is rejected; the later
    # GPU Runner must also require all in-use submodules to be initialized.
    for row in git("submodule", "status", "--recursive").splitlines():
        if row.startswith(("+", "U")):
            raise support.FullBF16EvidenceError(
                f"Submodule checkout differs from Gitlink: {row}"
            )
    return head


_SOURCE_SUBMODULES = frozenset({
    "frontend", "mikazuki/dataset-tag-editor", "sd-scripts",
})
_GITLINK_STATUS = re.compile(r"^([ +U-])([0-9a-f]{40}) ([^ ()]+)(?: \(.*\))?$")


def _source_submodule_inventory() -> dict[str, dict[str, str]]:
    """Record uninitialized Gitlinks without claiming them runtime-ready.

    This Generator is Host-only; later Runner/Worker MUST independently
    require every actual runtime-in-use submodule initialized and clean.
    """
    try:
        output = subprocess.run(
            ["git", "submodule", "status", "--recursive"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise support.FullBF16EvidenceError(
            f"Cannot inspect source submodule Gitlinks: {exc}"
        ) from exc
    result: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        match = _GITLINK_STATUS.fullmatch(line)
        if match is None:
            raise support.FullBF16EvidenceError(f"Invalid git submodule status row: {line!r}")
        status, commit, name = match.groups()
        if name in result or name not in _SOURCE_SUBMODULES:
            raise support.FullBF16EvidenceError(f"Unreviewed or duplicate source submodule: {name!r}")
        if status in ("+", "U"):
            raise support.FullBF16EvidenceError(f"Submodule mismatches Gitlink: {line!r}")
        result[name] = {
            "gitlink_sha": support._require_sha(commit, name=f"{name} Gitlink", commit=True),
            "checkout": "uninitialized" if status == "-" else "initialized",
        }
    if set(result) != _SOURCE_SUBMODULES:
        raise support.FullBF16EvidenceError("Missing expected source submodule Gitlinks")
    return dict(sorted(result.items()))


def _input_file(raw: Path, *, name: str) -> Path:
    path = support._real_path(str(Path(raw).expanduser().absolute()), name=name, must_exist=True)
    if not path.is_file():
        raise support.FullBF16EvidenceError(f"{name} must be an actual regular file: {path}")
    return path


def _image_sources(data_dir: Path, *, expected_count: int) -> list[tuple[Path, Path]]:
    src = support._real_path(str(Path(data_dir).expanduser().absolute()), name="input dataset", must_exist=True)
    if not src.is_dir():
        raise support.FullBF16EvidenceError("Input data_dir must be a directory")
    try:
        members = sorted(src.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise support.FullBF16EvidenceError(f"Cannot list source dataset: {exc}") from exc
    files: dict[str, Path] = {}
    for member in members:
        if member.is_symlink() or not member.is_file() or member.name.startswith("."):
            raise support.FullBF16EvidenceError(
                f"Non-regular, hidden or linked source dataset member: {member}"
            )
        if member.suffix.lower() not in (".png", ".caption"):
            raise support.FullBF16EvidenceError(
                f"Only exactly paired PNG/.caption reference files are allowed: {member}"
            )
        if member.name.lower() != member.name or member.suffix not in (".png", ".caption"):
            raise support.FullBF16EvidenceError(f"Mixed-case dataset extension is not allowed: {member}")
        files[member.name] = member
    image_names = sorted(x for x in files if x.endswith(".png"))
    captions = sorted(x for x in files if x.endswith(".caption"))
    if len(image_names) != expected_count or len(captions) != expected_count:
        raise support.FullBF16EvidenceError(
            f"Case requires exactly {expected_count} source PNG/caption pairs"
        )
    if {Path(x).stem for x in image_names} != {Path(x).stem for x in captions}:
        raise support.FullBF16EvidenceError(
            "Every source image must have one matching .caption, and no orphan caption"
        )
    candidates = [(files[png], files[Path(png).stem + ".caption"]) for png in image_names]
    images: set[str] = set()
    pixels: set[str] = set()
    captions_seen: set[str] = set()
    for image, caption in candidates:
        file_hash = support._sha256_file(image)
        pixel_hash = support.decoded_rgb_pixel_sha256(image)
        if file_hash in images or pixel_hash in pixels:
            raise support.FullBF16EvidenceError(
                "Distinct R2 source images must have different PNG bytes and RGB pixels"
            )
        images.add(file_hash)
        pixels.add(pixel_hash)
        try:
            raw = caption.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise support.FullBF16EvidenceError(f"Cannot read UTF-8 caption {caption}: {exc}") from exc
        clean = raw.strip()
        if (not clean or "\n" in clean or "\r" in clean
                or "*" in clean or "\x00" in clean):
            raise support.FullBF16EvidenceError(
                f"Caption is empty/multiline or uses unreviewed wildcard syntax: {caption}"
            )
        if clean in captions_seen:
            raise support.FullBF16EvidenceError("Reference captions must be pairwise distinct")
        captions_seen.add(clean)
    return candidates


def _external_models(external_models: dict[str, Path]) -> dict[str, Path]:
    flags = ("--clip_l", "--clip_g", "--t5xxl", "--vae")
    if not isinstance(external_models, dict) or set(external_models) != set(flags):
        raise support.FullBF16EvidenceError(
            "B0 reference requires all four explicit CLIP-L/CLIP-G/T5XXL/VAE sources"
        )
    return {key: _input_file(Path(external_models[key]), name=key) for key in flags}


def _reference_common_args(
    *, spec: dict[str, Any],
    model: Path, data_root: Path, policy_path: Path,
    external: dict[str, Path],
) -> list[str]:
    args = [
        support.TRAINER_ENTRYPOINT,
        "--pretrained_model_name_or_path", str(model),
        "--train_data_dir", str(data_root),
        "--parameter_policy_config", str(policy_path),
        "--output_name", spec["output_name"],
        "--max_train_steps", "2",
        "--save_every_n_steps", "1",
        "--save_state",
        "--mixed_precision", "bf16",
        "--full_bf16",
        "--optimizer_type", "AdamW",
        "--network_module", "networks.lora_sd3",
        "--network_train_unet_only",
        "--network_dim", "4",
        "--network_alpha", "4",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", str(spec["gradient_accumulation_steps"]),
        "--resolution", "512",
        "--save_model_as", "safetensors",
        "--lr_scheduler", "constant",
        "--seed", str(support.REFERENCE_SEED),
        "--max_data_loader_n_workers", "0",
        "--max_grad_norm", "1.0",
        "--cache_latents",
        "--cache_text_encoder_outputs",
    ]
    for name in ("--clip_l", "--clip_g", "--t5xxl", "--vae"):
        args.extend((name, str(external[name])))
    # Do NOT add --scale_weight_norms or --save_precision:
    # their real argparse defaults are None, frozen in Fix A-2.
    return args


def generate(
    *, case_id: str, model: Path, data_dir: Path, work_dir: Path,
    external_models: dict[str, Path],
) -> dict[str, str]:
    """Create one immutable external candidate case; refuse preexisting output.

    The input model need not load here: tiny fake model bytes are used in
    intentional RED Host fixtures. Real architecture+header checks happen
    later at validate_case_assets, before any Worker can start.
    """
    spec = support.get_case_spec(case_id)
    model = _input_file(Path(model), name="SD3.5 Medium checkpoint")
    external = _external_models(external_models)
    sources = _image_sources(Path(data_dir), expected_count=spec["effective_batches_per_epoch"])
    root = support._external_work_path(str(Path(work_dir).expanduser().absolute()), name="B0 work directory")
    if root.exists():
        raise support.FullBF16EvidenceError(f"B0 work directory already exists: {root}")
    # Avoid modifying original C1 R0 inputs or deriving evidence from a
    # mutable dataset path inside the new work root.
    input_root = support._real_path(str(Path(data_dir).expanduser().absolute()), name="input dataset", must_exist=True)
    if input_root == root or root in input_root.parents or input_root in root.parents:
        raise support.FullBF16EvidenceError("Input and output directories may not overlap")
    head = _source_commit()
    submodules = _source_submodule_inventory()
    root.mkdir(parents=True, exist_ok=False)
    subset = root / "data" / "1_dts_sd3c2b"
    subset.mkdir(parents=True, exist_ok=False)
    inventory: dict[str, dict[str, Any]] = {}
    for number, (source_image, source_caption) in enumerate(sources, 1):
        stem = f"sample_{number:04d}"
        image = subset / f"{stem}.png"
        caption = subset / f"{stem}.caption"
        shutil.copyfile(source_image, image)
        raw_caption = source_caption.read_text(encoding="utf-8").strip()
        marker = f"dts_c2b_{stem}"
        # Explicit, unique and stable through the reference no-shuffle,
        # no-dropout caption processing. Original caption is preserved after
        # the identity marker; never alter the already-accepted C1 dataset.
        caption.write_text(f"{marker}, {raw_caption}\n", encoding="utf-8")
        for kind, output in (("image", image), ("caption", caption)):
            relative = output.relative_to(root).as_posix()
            row: dict[str, Any] = {
                "kind": kind,
                "sha256": support._sha256_file(output),
            }
            if kind == "image":
                row["dimensions"] = [512, 512]
                row["rgb_pixel_sha256"] = support.decoded_rgb_pixel_sha256(output)
            else:
                row["identity"] = marker
            inventory[relative] = row
    # Recheck invariant after copy (including metadata-only duplicate images).
    pngs = [v for k, v in inventory.items() if k.endswith(".png")]
    if len({v["sha256"] for v in pngs}) != len(pngs) or len({v["rgb_pixel_sha256"] for v in pngs}) != len(pngs):
        raise support.FullBF16EvidenceError("Generated input identity collapsed")
    from mikazuki.parameter_policy import validate_parameter_policy
    policy = validate_parameter_policy(support.build_reference_policy(case_id))
    _name, canonical_text = serialize_parameter_policy(policy)
    policy_path = root / "policy.json"
    policy_path.write_text(canonical_text, encoding="utf-8")
    identity = {
        "model_sha256": support._sha256_file(model),
        "assets": {
            key[2:]: {"path": str(external[key]), "sha256": support._sha256_file(external[key])}
            for key in ("--clip_l", "--clip_g", "--t5xxl", "--vae")
        },
        "dataset_files": dict(sorted(inventory.items())),
        "dataset_signature": support._json_sha256(inventory),
        "policy_bytes_sha256": support._sha256_file(policy_path),
        "canonical_policy_hash": support._sha256(canonical_text.encode("utf-8")),
    }
    common = _reference_common_args(
        spec=spec, model=model, data_root=root / "data",
        policy_path=policy_path, external=external,
    )
    fresh, resumed = root / "fresh", root / "resume"
    output_name = spec["output_name"]
    payload = {
        "schema": support.SCHEMA,
        "version": support.VERSION,
        "case_id": case_id,
        "source_commit": head,
        "source_submodules": submodules,
        "work_root": str(root),
        "input_identity": identity,
        "reference": {
            "train_type": "sd3-lora", "model_variant": "sd35-medium",
            "resolution": [512, 512], "train_batch_size": 1,
            "effective_batches_per_epoch": spec["effective_batches_per_epoch"],
            "gradient_accumulation_steps": spec["gradient_accumulation_steps"],
            "max_train_steps": 2,
            "optimizer_profiles": spec["optimizer_types"],
        },
        "fresh_command": [*common, "--output_dir", str(fresh)],
        "resume_command": [
            *common, "--resume", str(fresh / f"{output_name}-step00000001-state"),
            "--output_dir", str(resumed),
        ],
        "fresh_checkpoint_dir": str(fresh / f"{output_name}-step00000001-state"),
        "resume_checkpoint_dir": str(resumed / f"{output_name}-step00000002-state"),
        "fresh_adapter_path": str(fresh / f"{output_name}-step00000001.safetensors"),
        "resume_adapter_path": str(resumed / f"{output_name}-step00000002.safetensors"),
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    # The normal manifest-only validator MUST succeed on the generated output.
    # It reads the policy and options, NOT fake SD3 header fixtures or CUDA.
    support.validate_case_manifest(payload)
    if fresh.exists() or resumed.exists():
        raise support.FullBF16EvidenceError("Generator must not precreate trainer output")
    return {
        "manifest": str(manifest),
        "policy": str(policy_path),
        "evidence": str(root / "result.json"),
        "case_id": case_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True, choices=sorted(support._CASES))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    for name in ("clip-l", "clip-g", "t5xxl", "vae"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    sources = {
        "--clip_l": args.clip_l, "--clip_g": args.clip_g,
        "--t5xxl": args.t5xxl, "--vae": args.vae,
    }
    try:
        result = generate(
            case_id=args.case_id, model=args.model, data_dir=args.data_dir,
            work_dir=args.work_dir, external_models=sources,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"SD3C2-B0 input generation failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
