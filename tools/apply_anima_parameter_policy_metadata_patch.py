"""Patch pinned Anima LoRA creation with DTS Parameter Policy target metadata.

The pinned sd-scripts submodule is never modified by normal runtime staging.
This module exposes a pure source transformer used by mikazuki.anima_runtime and
an explicit --check CLI for reviewing the currently pinned upstream source.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


EXPECTED_SD_SCRIPTS_HEAD = "45dddfccb704b6b0591f65d98f0d695c997b3115"
MARKER_ATTR = "_dts_parameter_policy_target_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one source match, found {count}"
        )
    return text.replace(old, new, 1)


def patch_lora_anima(text: str) -> str:
    helper_anchor = '''logger = logging.getLogger(__name__)\n\n'''
    helper = helper_anchor + f'''DTS_PARAMETER_POLICY_TARGET_ATTR = "{MARKER_ATTR}"\n\n\ndef _attach_dts_parameter_policy_target(lora, target_root, target_path, target_module):\n    if target_root is None:\n        return\n    cls = target_module.__class__\n    module_name = getattr(cls, "__module__", "")\n    qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", ""))\n    target_type = f"{{module_name}}.{{qualname}}" if module_name else qualname\n    setattr(\n        lora,\n        DTS_PARAMETER_POLICY_TARGET_ATTR,\n        (target_root, target_path, target_type),\n    )\n\n\n'''
    text = replace_once(
        text,
        helper_anchor,
        helper,
        "Anima metadata helper",
    )

    root_anchor = '''            prefix = self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER\n\n            loras = []\n'''
    root_replacement = '''            prefix = self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER\n            target_root = "dit" if is_unet else ("qwen3" if text_encoder_idx == 0 else None)\n\n            loras = []\n'''
    text = replace_once(
        text,
        root_anchor,
        root_replacement,
        "Anima metadata root mapping",
    )

    attach_anchor = '''                            lora.original_name = original_name\n                            loras.append(lora)\n'''
    attach_replacement = '''                            lora.original_name = original_name\n                            _attach_dts_parameter_policy_target(\n                                lora,\n                                target_root,\n                                original_name,\n                                child_module,\n                            )\n                            loras.append(lora)\n'''
    text = replace_once(
        text,
        attach_anchor,
        attach_replacement,
        "Anima metadata attachment",
    )
    return text


def patch_files(sd_scripts_dir: Path) -> dict[Path, str]:
    path = sd_scripts_dir / "networks/lora_anima.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    original = path.read_text(encoding="utf-8-sig")
    return {path: patch_lora_anima(original)}


def validate_patch(sd_scripts_dir: Path) -> None:
    for path, source in patch_files(sd_scripts_dir).items():
        ast.parse(source, filename=str(path))
        required = (
            MARKER_ATTR,
            'target_root = "dit" if is_unet else ("qwen3" if text_encoder_idx == 0 else None)',
            "(target_root, target_path, target_type)",
            "lora.original_name = original_name",
        )
        for marker in required:
            if marker not in source:
                raise RuntimeError(
                    f"Patched {path} is missing required metadata contract marker: {marker}"
                )


def verify_head(sd_scripts_dir: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(sd_scripts_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if head != EXPECTED_SD_SCRIPTS_HEAD:
        raise RuntimeError(
            f"sd-scripts HEAD is {head}, expected pinned "
            f"{EXPECTED_SD_SCRIPTS_HEAD}. Re-review the metadata patch against "
            "the new upstream before applying it."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-scripts-dir", default="sd-scripts")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate pinned source anchors and patched Python syntax without writing.",
    )
    args = parser.parse_args()
    if not args.check:
        parser.error("only --check is supported; runtime staging never mutates the submodule")

    sd_scripts_dir = Path(args.sd_scripts_dir).resolve()
    verify_head(sd_scripts_dir)
    validate_patch(sd_scripts_dir)
    print("Anima Parameter Policy metadata patch check passed; no files were changed.")


if __name__ == "__main__":
    main()
