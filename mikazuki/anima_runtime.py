"""Composable launch-time staging for optional Anima trainer features.

The pinned sd-scripts submodule is never modified.  A requested feature set is
materialized into one content-addressed isolated tree, patches are applied in a
fixed order, and the concrete trainer path is swapped only for that launch.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

from mikazuki.anima_qwen_config import trainer_supports_qwen_training
from mikazuki.anima_qwen_runtime import _extend_joint_block_swap_support
from tools import apply_anima_multi_caption_patch as multi_patch
from tools import apply_anima_parameter_policy_metadata_patch as metadata_patch
from tools import apply_anima_parameter_policy_runtime_patch as parameter_policy_runtime_patch
from tools import apply_anima_qwen3_sd_scripts_patch as qwen_patch


_CACHE_ROOT = Path("config") / "autosave" / "trainer-cache"
_RUNTIME_REVISION = "anima-runtime-features-v3"
_FEATURE_QWEN = "qwen3_joint"
_FEATURE_MULTI = "multi_caption"
_FEATURE_PARAMETER_POLICY_METADATA = "parameter_policy_metadata"
_FEATURE_PARAMETER_POLICY_RUNTIME = "parameter_policy_runtime"


def requested_runtime_features(prepared) -> tuple[str, ...]:
    features: list[str] = []
    if prepared.train_type == "anima-finetune" and bool(
        prepared.config.get("train_qwen3_text_encoder")
    ):
        features.append(_FEATURE_QWEN)
    if prepared.train_type in {"anima-lora", "anima-finetune"} and bool(
        prepared.config.get("multi_caption_config")
    ):
        features.append(_FEATURE_MULTI)
    if prepared.train_type == "anima-lora" and bool(
        prepared.config.get("parameter_policy_config")
    ):
        features.append(_FEATURE_PARAMETER_POLICY_METADATA)
    if prepared.train_type in {"anima-lora", "anima-finetune"} and bool(
        prepared.config.get("parameter_policy_config")
    ):
        features.append(_FEATURE_PARAMETER_POLICY_RUNTIME)
    return tuple(sorted(features))


def _cache_key(features: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(qwen_patch.EXPECTED_SD_SCRIPTS_HEAD.encode("ascii"))
    digest.update(_RUNTIME_REVISION.encode("ascii"))
    digest.update("\0".join(features).encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    if _FEATURE_QWEN in features:
        digest.update(Path(qwen_patch.__file__).read_bytes())
        # The block-swap extension lives in the compatibility runtime module.
        import mikazuki.anima_qwen_runtime as qwen_runtime
        digest.update(Path(qwen_runtime.__file__).read_bytes())
    if _FEATURE_MULTI in features:
        digest.update(Path(multi_patch.__file__).read_bytes())
        source = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "library" / "multi_caption.py"
        digest.update(source.read_bytes())
    if _FEATURE_PARAMETER_POLICY_METADATA in features:
        digest.update(Path(metadata_patch.__file__).read_bytes())
    if _FEATURE_PARAMETER_POLICY_RUNTIME in features:
        digest.update(Path(parameter_policy_runtime_patch.__file__).read_bytes())
        bridge = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "library" / "dts_parameter_policy_bridge.py"
        digest.update(bridge.read_bytes())
    return digest.hexdigest()[:16]


def _marker_payload(features: tuple[str, ...]) -> str:
    return json.dumps(
        {"revision": _RUNTIME_REVISION, "features": list(features)},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _has_multi_capability(target: Path) -> bool:
    required = {
        target / "library/multi_caption.py": "class MultiCaptionResolver",
        target / "library/args.py": "--multi_caption_config",
        target / "library/dataset.py": "multi_caption_resolver",
        target / "train_network.py": "configure_multi_caption_dataset_groups",
        target / "anima_train.py": "configure_multi_caption_dataset_groups",
    }
    for path, marker in required.items():
        if not path.is_file():
            return False
        if marker not in path.read_text(encoding="utf-8", errors="ignore"):
            return False
    return True


def _has_parameter_policy_metadata_capability(target: Path) -> bool:
    path = target / "networks/lora_anima.py"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    required = (
        metadata_patch.MARKER_ATTR,
        'target_root = "dit" if is_unet else ("qwen3" if text_encoder_idx == 0 else None)',
        "_attach_dts_parameter_policy_target(",
    )
    return all(marker in text for marker in required)


def _has_parameter_policy_runtime_capability(target: Path) -> bool:
    required = {
        target / "library/dts_parameter_policy_bridge.py": (
            "create_parameter_policy_session",
            "load_parameter_policy_file",
        ),
        target / "train_network.py": (
            "finalize_after_prepare",
            'phase="post_resume"',
            'phase="epoch_start"',
            "model_metadata()",
        ),
        target / "anima_train_network.py": ('return "anima-lora"',),
        target / "anima_train.py": (
            '"anima-finetune"',
            "finalize_after_prepare",
            'phase="post_resume"',
            'phase="epoch_start"',
            "_dts_parameter_policy_model_metadata",
        ),
        target / "library/anima_train_utils.py": (
            "_dts_parameter_policy_model_metadata",
        ),
    }
    for path, markers in required.items():
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker not in text for marker in markers):
            return False
    return True


def _is_valid_materialized_tree(target: Path, features: tuple[str, ...]) -> bool:
    marker = target / ".mikazuki-anima-runtime"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload != {"revision": _RUNTIME_REVISION, "features": list(features)}:
        return False
    if _FEATURE_QWEN in features:
        trainer = target / "anima_train.py"
        if not trainer_supports_qwen_training(str(trainer)):
            return False
        text = trainer.read_text(encoding="utf-8", errors="ignore")
        if "blocks_to_swap requires AdaFactor" not in text:
            return False
    if _FEATURE_MULTI in features and not _has_multi_capability(target):
        return False
    if (
        _FEATURE_PARAMETER_POLICY_METADATA in features
        and not _has_parameter_policy_metadata_capability(target)
    ):
        return False
    if (
        _FEATURE_PARAMETER_POLICY_RUNTIME in features
        and not _has_parameter_policy_runtime_capability(target)
    ):
        return False
    return True


def _apply_patch_map(patched: dict[Path, str]) -> None:
    for path, source in patched.items():
        ast.parse(source, filename=str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def materialize_anima_runtime_tree(
    features: tuple[str, ...],
    source_dir: str | os.PathLike[str] = "sd-scripts",
    cache_root: str | os.PathLike[str] | None = None,
) -> Path:
    features = tuple(sorted(set(features)))
    unknown = set(features) - {
        _FEATURE_QWEN,
        _FEATURE_MULTI,
        _FEATURE_PARAMETER_POLICY_METADATA,
        _FEATURE_PARAMETER_POLICY_RUNTIME,
    }
    if unknown:
        raise RuntimeError(f"Unknown Anima runtime features: {sorted(unknown)}")
    if not features:
        return Path(source_dir).resolve()

    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise RuntimeError(f"sd-scripts directory does not exist: {source}")
    if shutil.which("git") is None:
        raise RuntimeError(
            "Anima optional runtime features require Git so DTS can materialize the reviewed pinned "
            "sd-scripts commit. Install Git and initialize the sd-scripts submodule."
        )
    qwen_patch.verify_head(source)

    root = Path(cache_root) if cache_root is not None else _CACHE_ROOT
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep the staged path deliberately short for Windows installations near
    # MAX_PATH. Feature/head/revision details live in the marker, not the name.
    target = root / f"a-{_cache_key(features)}"
    if _is_valid_materialized_tree(target, features):
        return target

    temp = root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp.mkdir(parents=True, exist_ok=False)
        archive = temp / ".pinned-source.tar"
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "archive",
                "--format=tar",
                "--output",
                str(archive),
                qwen_patch.EXPECTED_SD_SCRIPTS_HEAD,
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Unable to archive pinned sd-scripts source: "
                + (completed.stderr.strip() or completed.stdout.strip() or "git archive failed")
            )
        with tarfile.open(archive, "r") as tf:
            tf.extractall(temp)
        archive.unlink(missing_ok=True)

        # Fixed order is part of the contract. Qwen owns its model/save blocks;
        # Multi-Caption owns Dataset/arg/dataset-construction blocks; metadata
        # annotates LoRA targets; Parameter Policy runtime is applied last so it
        # sees the final staged trainer shape and owns optimizer/runtime seams.
        if _FEATURE_QWEN in features:
            patched = qwen_patch.patch_files(temp)
            for path, source_text in list(patched.items()):
                if path == temp / "anima_train.py":
                    patched[path] = _extend_joint_block_swap_support(source_text)
            _apply_patch_map(patched)

        if _FEATURE_MULTI in features:
            _apply_patch_map(multi_patch.patch_files(temp))

        if _FEATURE_PARAMETER_POLICY_METADATA in features:
            _apply_patch_map(metadata_patch.patch_files(temp))

        if _FEATURE_PARAMETER_POLICY_RUNTIME in features:
            _apply_patch_map(parameter_policy_runtime_patch.patch_files(temp))

        (temp / ".mikazuki-anima-runtime").write_text(
            _marker_payload(features),
            encoding="utf-8",
        )

        if not _is_valid_materialized_tree(temp, features):
            raise RuntimeError(
                f"Materialized Anima runtime failed capability verification for {features}."
            )

        if target.exists():
            if _is_valid_materialized_tree(target, features):
                shutil.rmtree(temp, ignore_errors=True)
                return target
            shutil.rmtree(target)
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    if not _is_valid_materialized_tree(target, features):
        raise RuntimeError("Materialized Anima runtime failed final integrity verification.")
    return target


def prepare_runtime_trainer(prepared) -> None:
    """Swap only Anima jobs that explicitly request an optional runtime feature."""
    if prepared.train_type not in {"anima-lora", "anima-finetune"}:
        return
    features = requested_runtime_features(prepared)
    if not features:
        return

    tree = materialize_anima_runtime_tree(features)
    trainer_name = (
        "anima_train_network.py"
        if prepared.train_type == "anima-lora"
        else "anima_train.py"
    )
    prepared.trainer_file = str((tree / trainer_name).resolve())


__all__ = [
    "materialize_anima_runtime_tree",
    "prepare_runtime_trainer",
    "requested_runtime_features",
]
