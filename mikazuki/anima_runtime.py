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
import uuid

from mikazuki.anima_qwen_config import trainer_supports_qwen_training
from mikazuki.anima_qwen_runtime import _extend_joint_block_swap_support, _source_is_clean
from tools import apply_anima_multi_caption_patch as multi_patch
from tools import apply_anima_qwen3_sd_scripts_patch as qwen_patch


_CACHE_ROOT = Path("config") / "autosave" / "trainer-cache"
_RUNTIME_REVISION = "anima-runtime-features-v1"
_FEATURE_QWEN = "qwen3_joint"
_FEATURE_MULTI = "multi_caption"


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
    unknown = set(features) - {_FEATURE_QWEN, _FEATURE_MULTI}
    if unknown:
        raise RuntimeError(f"Unknown Anima runtime features: {sorted(unknown)}")
    if not features:
        return Path(source_dir).resolve()

    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise RuntimeError(f"sd-scripts directory does not exist: {source}")
    qwen_patch.verify_head(source)
    if not _source_is_clean(source):
        raise RuntimeError(
            "sd-scripts has tracked local modifications. Reset/update the submodule before starting "
            "an optional Anima runtime feature; staging is built only from the reviewed pinned source."
        )

    root = Path(cache_root) if cache_root is not None else _CACHE_ROOT
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    feature_slug = "-".join(features)
    target = root / (
        f"anima-{qwen_patch.EXPECTED_SD_SCRIPTS_HEAD[:8]}-{feature_slug}-{_cache_key(features)}"
    )
    if _is_valid_materialized_tree(target, features):
        return target

    temp = root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(
            source,
            temp,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )

        # Fixed order is part of the contract. Qwen owns optimizer/model/save
        # blocks; Multi-Caption owns Dataset/arg/dataset-construction blocks.
        if _FEATURE_QWEN in features:
            patched = qwen_patch.patch_files(temp)
            for path, source_text in list(patched.items()):
                if path == temp / "anima_train.py":
                    patched[path] = _extend_joint_block_swap_support(source_text)
            _apply_patch_map(patched)

        if _FEATURE_MULTI in features:
            _apply_patch_map(multi_patch.patch_files(temp))

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
