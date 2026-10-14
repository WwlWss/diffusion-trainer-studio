"""Read-only, torch-free structure contract for DTS SD3/SD3.5 MMDiT checkpoints.

This is an *audit*, not a model loader or an execution-qualification authority.
In particular, tensor shapes cannot distinguish SD3.5 Large from Large Turbo.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


class SD3ArchitectureError(ValueError):
    """The checkpoint does not satisfy a recognized DTS SD3 topology."""


# "sd35-large-topology" intentionally says nothing about distillation/release.
_KNOWN_TOPOLOGIES: dict[tuple[int, int, int, str | None, tuple[int, ...]], str] = {
    (24, 1536, 192, None, ()): "sd3-medium",
    (24, 1536, 384, "rms", tuple(range(13))): "sd35-medium",
    (38, 2432, 192, "rms", ()): "sd35-large-topology",
}
_MMDIT_PREFIX = "model.diffusion_model."
_BARE_MMDIT_PREFIXES = (
    "joint_blocks.", "x_embedder.", "y_embedder.", "context_embedder.",
    "t_embedder.", "final_layer.",
)
_BLOCK_INDEX = re.compile(r"^joint_blocks\.(\d+)\.")
_EXTRA_QKV = re.compile(r"^joint_blocks\.(\d+)\.x_block\.attn2\.qkv\.weight$")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SD3Architecture:
    """Immutable description of an audited checkpoint's MMDiT topology."""

    variant: str
    depth: int
    hidden_size: int
    num_heads: int
    head_dim: int
    patch_size: int
    pos_embed_grid: int
    qk_norm: str | None
    extra_attention_layers: tuple[int, ...]
    shape_index_sha256: str

    def identity(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "patch_size": self.patch_size,
            "pos_embed_grid": self.pos_embed_grid,
            "qk_norm": self.qk_norm,
            "extra_attention_layers": list(self.extra_attention_layers),
        }

    def report(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "architecture_signature_sha256": _canonical_hash(self.identity()),
            "shape_index_sha256": self.shape_index_sha256,
            "release_identity_verified": False,
            "note": (
                "Shape-only audit; not an optimizer/full-BF16 qualification. "
                "SD3.5 Large and Large Turbo cannot be distinguished by topology alone."
            ),
        }

    def block_kind(self, index: int) -> str:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.depth:
            raise SD3ArchitectureError(f"Invalid MMDiT block index: {index!r}")
        if index == self.depth - 1:
            return "context_pre_only"
        if index in self.extra_attention_layers:
            return "dual_attention"
        return "joint_attention"


def normalize_sd3_shape_index(shapes: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    """Normalize a full checkpoint or an already-stripped MMDiT shape index.

    Only the exact 'model.diffusion_model.' prefix is stripped, matching the
    repository's real sd3_utils.load_mmdit(). Text encoders and VAE are ignored.
    """
    if not isinstance(shapes, Mapping):
        raise SD3ArchitectureError("Tensor shape index must be a mapping")
    full = _MMDIT_PREFIX + "x_embedder.proj.weight" in shapes
    bare = "x_embedder.proj.weight" in shapes
    if full == bare:
        raise SD3ArchitectureError("Expected exactly one prefixed or bare SD3 MMDiT root")
    result: dict[str, tuple[int, ...]] = {}
    for key, shape in shapes.items():
        if not isinstance(key, str) or not key:
            raise SD3ArchitectureError("Shape index contains an invalid tensor name")
        if full:
            if not key.startswith(_MMDIT_PREFIX):
                continue
            name = key[len(_MMDIT_PREFIX):]
        else:
            if key.startswith(_MMDIT_PREFIX):
                raise SD3ArchitectureError("Mixed prefixed and unprefixed MMDiT tensors")
            if not (key.startswith("joint_blocks.") or key.startswith((
                "x_embedder.", "y_embedder.", "context_embedder.", "t_embedder.",
                "final_layer.", "pos_embed",
            ))):
                continue
            name = key
        if not isinstance(shape, (list, tuple)) or not shape or len(shape) > 8:
            raise SD3ArchitectureError(f"Invalid shape for {key!r}")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise SD3ArchitectureError(f"Invalid dimension for {key!r}: {shape!r}")
        if name in result:
            raise SD3ArchitectureError(f"Duplicate normalized tensor name: {name!r}")
        result[name] = tuple(shape)
    if not result:
        raise SD3ArchitectureError("No SD3 MMDiT tensors found")
    return result


def _is_bare_mmdit_key(key: str) -> bool:
    """Match known bare MMDiT roots, never similarly named external modules."""
    return key == "pos_embed" or key.startswith(_BARE_MMDIT_PREFIXES)


def _reject_mixed_mmdit_roots(shapes: Mapping[str, Sequence[int]]) -> None:
    """Reject bare MMDiT remnants beside a prefixed MMDiT in strict mode.

    The current DTS loader selects only model.diffusion_model.* keys; unrelated
    CLIP/T5/VAE namespaces remain legitimate and are never inspected here.
    Topology-only callers intentionally retain the historical normalization.
    """
    if _MMDIT_PREFIX + "x_embedder.proj.weight" not in shapes:
        return
    mixed = sorted(key for key in shapes if _is_bare_mmdit_key(key))
    if mixed:
        raise SD3ArchitectureError(
            "Mixed prefixed and bare MMDiT tensors: "
            f"examples={mixed[:5]!r}, count={len(mixed)}"
        )


def _expect(shapes: Mapping[str, tuple[int, ...]], key: str, shape: tuple[int, ...]) -> None:
    if shapes.get(key) != shape:
        raise SD3ArchitectureError(
            f"SD3 topology mismatch for {key!r}: expected {shape}, found {shapes.get(key)!r}"
        )


def inspect_sd3_architecture(shapes: Mapping[str, Sequence[int]]) -> SD3Architecture:
    """Recognize *only* known SD3/SD3.5 MMDiT structures, fail closed otherwise.

    Assumes names and shapes, not pretrained contents. This cannot authenticate
    weights, file origin, or a Large-vs-Turbo release identity.
    """
    idx = normalize_sd3_shape_index(shapes)
    x_embedder = idx.get("x_embedder.proj.weight", ())
    if len(x_embedder) != 4 or x_embedder[1] != 16 or x_embedder[2] != x_embedder[3]:
        raise SD3ArchitectureError("Invalid SD3 x_embedder Conv2d shape")
    hidden_size, _, patch_size, _ = x_embedder
    if patch_size != 2 or hidden_size % 64:
        raise SD3ArchitectureError("Unrecognized SD3 patch size / hidden width")
    depth = hidden_size // 64  # matches scripts/dev/library/sd3_models.py
    pos = idx.get("pos_embed", ())
    if len(pos) != 3 or pos[0] != 1 or pos[2] != hidden_size:
        raise SD3ArchitectureError("Invalid SD3 positional embedding shape")
    grid = math.isqrt(pos[1])
    if grid * grid != pos[1]:
        raise SD3ArchitectureError("SD3 pos_embed is not a square grid")
    context_shape = idx.get("context_embedder.weight", ())
    if len(context_shape) != 2 or context_shape[0] != hidden_size:
        raise SD3ArchitectureError("SD3 context_embedder does not match hidden width")

    found = {int(m.group(1)) for key in idx if (m := _BLOCK_INDEX.match(key))}
    if found != set(range(depth)):
        raise SD3ArchitectureError(
            f"MMDiT blocks must be contiguous from 0 to {depth-1}; observed {sorted(found)!r}"
        )
    extra = tuple(sorted(int(m.group(1)) for key in idx if (m := _EXTRA_QKV.match(key))))
    if len(extra) != len(set(extra)):
        raise SD3ArchitectureError("Duplicate SD3 additional attention blocks")
    qk_norm = "rms" if "joint_blocks.0.context_block.attn.ln_k.weight" in idx else None

    topology_key = (depth, hidden_size, grid, qk_norm, extra)
    variant = _KNOWN_TOPOLOGIES.get(topology_key)
    if variant is None:
        raise SD3ArchitectureError(
            "Unknown SD3 architecture (depth, hidden, pos_grid, qk_norm, extra_attention): "
            f"{topology_key!r}"
        )

    for i in range(depth):
        prefix = f"joint_blocks.{i}."
        for stream in ("x_block", "context_block"):
            root = prefix + stream
            _expect(idx, root + ".attn.qkv.weight", (hidden_size * 3, hidden_size))
            _expect(idx, root + ".adaLN_modulation.1.weight", (
                hidden_size * (2 if stream == "context_block" and i == depth-1
                               else 9 if stream == "x_block" and i in extra else 6),
                hidden_size,
            ))
            if stream == "context_block" and i == depth - 1:
                if root + ".attn.proj.weight" in idx or any(
                    key.startswith(root + ".mlp.") for key in idx
                ):
                    raise SD3ArchitectureError("Final SD3 context block must be pre-only")
            else:
                _expect(idx, root + ".attn.proj.weight", (hidden_size, hidden_size))
                _expect(idx, root + ".mlp.fc1.weight", (hidden_size * 4, hidden_size))
            for name in ("ln_q", "ln_k"):
                qk_key = root + f".attn.{name}.weight"
                if qk_norm == "rms":
                    _expect(idx, qk_key, (64,))
                elif qk_key in idx:
                    raise SD3ArchitectureError(f"Unexpected SD3 QKNorm tensor: {qk_key}")
        extra_root = prefix + "x_block.attn2"
        if i in extra:
            _expect(idx, extra_root + ".qkv.weight", (hidden_size * 3, hidden_size))
            _expect(idx, extra_root + ".proj.weight", (hidden_size, hidden_size))
            for name in ("ln_q", "ln_k"):
                _expect(idx, extra_root + f".{name}.weight", (64,))
        elif any(key.startswith(extra_root + ".") for key in idx):
            raise SD3ArchitectureError(f"Unexpected extra attention at block {i}")

    structural_index = [[key, list(shape)] for key, shape in sorted(idx.items())]
    return SD3Architecture(
        variant=variant,
        depth=depth,
        hidden_size=hidden_size,
        num_heads=depth,
        head_dim=64,
        patch_size=patch_size,
        pos_embed_grid=grid,
        qk_norm=qk_norm,
        extra_attention_layers=extra,
        shape_index_sha256=_canonical_hash(structural_index),
    )


@dataclass(frozen=True)
class SD3MMDiTShapeValidation:
    """Shape-only completeness of the MMDiT subtree; never a weight-integrity proof."""

    architecture: SD3Architecture
    checked_tensors: int
    loader_prefix_compatible: bool
    structurally_complete: bool = True

    def report(self) -> dict[str, Any]:
        return {
            "mmdit_structure_complete": self.structurally_complete,
            "dts_loader_prefix_compatible": self.loader_prefix_compatible,
            "checked_tensor_shapes": self.checked_tensors,
            "release_identity_verified": False,
            "qualifies_full_bf16": False,
        }


def expected_sd3_mmdit_shapes(architecture: SD3Architecture) -> dict[str, tuple[int, ...]]:
    """Exact state_dict key/shape manifest for DTS's three pinned MMDiT variants.

    This is intentionally tied to create_sd3_mmdit() with mlp_ratio=4,
    swiglu=False, qkv_bias=True, in_channels=16, and scale_mod_only=False.
    An unknown architecture/variant may not inherit this manifest.
    """
    if not isinstance(architecture, SD3Architecture):
        raise SD3ArchitectureError("Expected a recognized SD3Architecture")
    known = _KNOWN_TOPOLOGIES.get((
        architecture.depth, architecture.hidden_size, architecture.pos_embed_grid,
        architecture.qk_norm, architecture.extra_attention_layers,
    ))
    if known != architecture.variant or architecture.patch_size != 2 or architecture.head_dim != 64:
        raise SD3ArchitectureError("Unrecognized MMDiT shape-manifest variant")

    h = architecture.hidden_size
    p = architecture.patch_size
    shapes: dict[str, tuple[int, ...]] = {
        "x_embedder.proj.weight": (h, 16, p, p),
        "x_embedder.proj.bias": (h,),
        "pos_embed": (1, architecture.pos_embed_grid ** 2, h),
        "context_embedder.weight": (h, 4096),
        "context_embedder.bias": (h,),
        "final_layer.linear.weight": (p * p * 16, h),
        "final_layer.linear.bias": (p * p * 16,),
        "final_layer.adaLN_modulation.1.weight": (2 * h, h),
        "final_layer.adaLN_modulation.1.bias": (2 * h,),
    }

    def linear(prefix: str, output: int, input_: int) -> None:
        shapes[prefix + ".weight"] = (output, input_)
        shapes[prefix + ".bias"] = (output,)

    linear("t_embedder.mlp.0", h, 256)
    linear("t_embedder.mlp.2", h, h)
    linear("y_embedder.mlp.0", h, 2048)
    linear("y_embedder.mlp.2", h, h)

    for block in range(architecture.depth):
        for stream in ("x_block", "context_block"):
            prefix = f"joint_blocks.{block}.{stream}"
            pre_only = stream == "context_block" and block == architecture.depth - 1
            mod_count = 2 if pre_only else 9 if stream == "x_block" and block in architecture.extra_attention_layers else 6
            linear(prefix + ".attn.qkv", 3 * h, h)
            linear(prefix + ".adaLN_modulation.1", mod_count * h, h)
            if not pre_only:
                linear(prefix + ".attn.proj", h, h)
                linear(prefix + ".mlp.fc1", 4 * h, h)
                linear(prefix + ".mlp.fc2", h, 4 * h)
            if architecture.qk_norm == "rms":
                shapes[prefix + ".attn.ln_q.weight"] = (64,)
                shapes[prefix + ".attn.ln_k.weight"] = (64,)
        if block in architecture.extra_attention_layers:
            prefix = f"joint_blocks.{block}.x_block.attn2"
            linear(prefix + ".qkv", 3 * h, h)
            linear(prefix + ".proj", h, h)
            shapes[prefix + ".ln_q.weight"] = (64,)
            shapes[prefix + ".ln_k.weight"] = (64,)
    return shapes


def validate_sd3_mmdit_checkpoint(
    shapes: Mapping[str, Sequence[int]], *, require_loader_compatible_root: bool = False,
) -> SD3MMDiTShapeValidation:
    """Require all and only DTS MMDiT state_dict tensors with matching shapes.

    Text-encoder/VAE checkpoint namespaces are intentionally out of scope.
    Does not load tensor values, authenticate files, or authorize BF16 training.
    """
    architecture = inspect_sd3_architecture(shapes)
    _reject_mixed_mmdit_roots(shapes)
    actual = normalize_sd3_shape_index(shapes)
    expected = expected_sd3_mmdit_shapes(architecture)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        (key, expected[key], actual[key])
        for key in expected.keys() & actual.keys()
        if expected[key] != actual[key]
    )
    loader_root = _MMDIT_PREFIX + "x_embedder.proj.weight" in shapes
    if require_loader_compatible_root and not loader_root:
        raise SD3ArchitectureError("Checkpoint is not under the DTS loader's model.diffusion_model. root")
    if missing or unexpected or mismatched:
        raise SD3ArchitectureError(
            "SD3 MMDiT checkpoint structure incomplete or incompatible: "
            f"missing={missing[:8]!r} (count={len(missing)}), "
            f"unexpected={unexpected[:8]!r} (count={len(unexpected)}), "
            f"wrong_shapes={mismatched[:5]!r} (count={len(mismatched)})"
        )
    return SD3MMDiTShapeValidation(
        architecture=architecture,
        checked_tensors=len(expected),
        loader_prefix_compatible=loader_root,
    )
