"""Launch-time materialization for the optional Anima Qwen3 joint trainer.

The project intentionally keeps the upstream ``sd-scripts`` submodule pinned and
clean.  When Qwen3 joint finetuning is requested, we copy that exact pinned tree
into an ignored cache directory, apply the reviewed patch there, extend it with
the block-swap/fused-Adafactor contract, and launch the isolated copy.

This makes the feature self-contained without mutating the submodule or requiring
an external fork to exist merely to carry four patched trainer files.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from mikazuki.anima_qwen_config import trainer_supports_qwen_training
from tools import apply_anima_qwen3_sd_scripts_patch as staging_patch


_CACHE_ROOT = Path("config") / "autosave" / "trainer-cache"
_PATCH_REVISION = "anima-qwen3-joint-fused-blockswap-v1"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Qwen3 runtime patch anchor {label!r} expected once, found {count}")
    return text.replace(old, new, 1)


def _extend_joint_block_swap_support(text: str) -> str:
    """Extend the staged joint patch with the proven block-swap optimizer path.

    Runtime smoke established that Anima block swapping cannot use a normal
    end-of-backward optimizer step: swapped DiT parameters can be on CPU while
    their gradients are still on CUDA.  The existing fused Adafactor path steps
    each parameter from its post-accumulate hook before it is swapped away.  The
    hook is installed for every optimizer parameter group, so it also covers the
    appended Qwen3 group and preserves its independent LR.
    """

    text = _replace_once(
        text,
        '''        if args.fused_backward_pass:\n            raise ValueError("Qwen3 joint finetuning does not support fused_backward_pass in the first implementation.")\n''',
        '''        if args.fused_backward_pass and str(args.optimizer_type or "AdamW").lower() != "adafactor":\n            raise ValueError("Qwen3 joint finetuning: fused_backward_pass currently requires AdaFactor.")\n''',
        "joint fused-backward guard",
    )

    text = _replace_once(
        text,
        '''            "sgdnesterov8bit",\n        }\n''',
        '''            "sgdnesterov8bit",\n            "adafactor",\n        }\n''',
        "joint optimizer allowlist",
    )

    text = _replace_once(
        text,
        '''        optimizer_name = str(args.optimizer_type or "AdamW").lower()\n        if optimizer_name not in supported_qwen_optimizers:\n''',
        '''        optimizer_name = str(args.optimizer_type or "AdamW").lower()\n        if (args.blocks_to_swap is not None and args.blocks_to_swap > 0) and not (\n            optimizer_name == "adafactor" and args.fused_backward_pass\n        ):\n            raise ValueError(\n                "Qwen3 joint finetuning with blocks_to_swap requires AdaFactor + fused_backward_pass."\n            )\n        if optimizer_name == "adafactor" and not args.fused_backward_pass:\n            raise ValueError("Qwen3 joint finetuning currently supports AdaFactor only with fused_backward_pass.")\n        if optimizer_name not in supported_qwen_optimizers:\n''',
        "joint block-swap optimizer guard",
    )

    return text


def _source_is_clean(source_dir: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _cache_key() -> str:
    tool_path = Path(staging_patch.__file__).resolve()
    this_path = Path(__file__).resolve()
    digest = hashlib.sha256()
    digest.update(staging_patch.EXPECTED_SD_SCRIPTS_HEAD.encode("ascii"))
    digest.update(_PATCH_REVISION.encode("ascii"))
    digest.update(tool_path.read_bytes())
    digest.update(this_path.read_bytes())
    return digest.hexdigest()[:16]


def _is_valid_materialized_tree(target: Path) -> bool:
    trainer = target / "anima_train.py"
    marker = target / ".mikazuki-qwen3-runtime"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != _PATCH_REVISION:
        return False
    return trainer_supports_qwen_training(str(trainer)) and "blocks_to_swap requires AdaFactor" in trainer.read_text(
        encoding="utf-8", errors="ignore"
    )


def materialize_qwen_joint_trainer(
    source_dir: str | os.PathLike[str] = "sd-scripts",
    cache_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return an isolated patched ``anima_train.py`` suitable for joint training.

    The source submodule must be exactly the reviewed pinned commit and have no
    tracked modifications.  The materialized tree is content-addressed by both
    the staging patch and this runtime extension, so code changes naturally
    create a new cache instead of reusing stale trainer files.
    """

    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise RuntimeError(f"sd-scripts directory does not exist: {source}")

    staging_patch.verify_head(source)
    if not _source_is_clean(source):
        raise RuntimeError(
            "sd-scripts has tracked local modifications. Reset/update the submodule before starting Qwen3 joint training; "
            "the runtime trainer is built only from the reviewed pinned source."
        )

    root = Path(cache_root) if cache_root is not None else _CACHE_ROOT
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"qwen3-{staging_patch.EXPECTED_SD_SCRIPTS_HEAD[:8]}-{_cache_key()}"
    if _is_valid_materialized_tree(target):
        return str((target / "anima_train.py").resolve())

    temp = root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(
            source,
            temp,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )

        patched = staging_patch.patch_files(temp)
        for path, source_text in patched.items():
            if path == temp / "anima_train.py":
                source_text = _extend_joint_block_swap_support(source_text)
            ast.parse(source_text, filename=str(path))
            path.write_text(source_text, encoding="utf-8")

        (temp / ".mikazuki-qwen3-runtime").write_text(_PATCH_REVISION + "\n", encoding="utf-8")
        if not trainer_supports_qwen_training(str(temp / "anima_train.py")):
            raise RuntimeError("Materialized Qwen3 trainer failed capability verification.")

        if target.exists():
            if _is_valid_materialized_tree(target):
                shutil.rmtree(temp, ignore_errors=True)
                return str((target / "anima_train.py").resolve())
            shutil.rmtree(target)
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    if not _is_valid_materialized_tree(target):
        raise RuntimeError("Materialized Qwen3 trainer failed final integrity verification.")
    return str((target / "anima_train.py").resolve())


def prepare_runtime_trainer(prepared) -> None:
    """Replace the trainer only for explicitly enabled Anima Full joint jobs."""

    if prepared.train_type != "anima-finetune":
        return
    if not bool(prepared.config.get("train_qwen3_text_encoder")):
        return
    prepared.trainer_file = materialize_qwen_joint_trainer()


__all__ = [
    "materialize_qwen_joint_trainer",
    "prepare_runtime_trainer",
]
