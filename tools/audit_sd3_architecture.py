#!/usr/bin/env python3
"""Read-only SD3/SD3.5 MMDiT checkpoint shape and optional LoRA target audit.

Does not run a trainer, modify a checkpoint, or qualify an optimizer/backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Enable direct script execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mikazuki.sd3_architecture_contract import (  # noqa: E402
    SD3ArchitectureError,
    inspect_sd3_architecture,
)
from mikazuki.sd3_component_audit import (  # noqa: E402
    SD3ComponentAuditError,
    audit_sd3_adapter_tensors,
)


def read_safetensors_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Read tensor shapes from the safetensors header, never materialize values."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SD3ArchitectureError("safetensors is required for --checkpoint") from exc
    result = {}
    with safe_open(str(path), framework="pt", device="cpu") as file:
        for key in file.keys():
            result[key] = tuple(file.get_slice(key).get_shape())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="SD3/SD3.5 safetensors checkpoint")
    source.add_argument(
        "--shape-index", type=Path,
        help="JSON tensor-name -> shape mapping (diagnostic only, not proof of real weights)",
    )
    parser.add_argument(
        "--adapter-targets", type=Path,
        help="Optional JSON list of original LoRA target metadata and parameter shapes",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args(argv)
    try:
        if args.checkpoint is not None:
            shapes = read_safetensors_shapes(args.checkpoint)
        else:
            data = json.loads(args.shape_index.read_text(encoding="utf-8"))
            shapes = data.get("shapes", data) if isinstance(data, dict) else data
        arch = inspect_sd3_architecture(shapes)
        result = {
            "schema": "dts.sd3-architecture-audit",
            "version": 1,
            "read_only": True,
            "qualifies_full_bf16": False,
            "architecture": arch.report(),
        }
        if args.adapter_targets is not None:
            adapter_targets = json.loads(args.adapter_targets.read_text(encoding="utf-8"))
            if not isinstance(adapter_targets, list):
                raise SD3ComponentAuditError("--adapter-targets JSON must contain a list")
            result["adapter_audit"] = audit_sd3_adapter_tensors(arch, adapter_targets)
        output = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output is not None:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.exit(2, f"SD3 architecture audit rejected input: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
