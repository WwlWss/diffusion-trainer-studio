"""GPU-required ONNX Runtime session creation for local DTS taggers.

ONNX Runtime may deliberately place cheap shape/control-flow nodes on CPU even
when CUDA has higher priority. DTS therefore does not require every graph node
to live on CUDA. Instead we require a usable CUDA EP, record the actual graph
partition, and fail if no meaningful compute is assigned to CUDA. This prevents
a whole-model CPU fallback without breaking models that legitimately keep a few
shape-related nodes on CPU for performance.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


# Representative expensive operators used by the CNN/ViT taggers shipped with
# DTS. At least one of these must actually be assigned to CUDA. Shape, Gather,
# Unsqueeze, Concat, Reshape, etc. are intentionally not in this set because ORT
# may place those on CPU even when their CUDA kernels exist.
_GPU_COMPUTE_OPS = frozenset(
    {
        "Attention",
        "BatchNormalization",
        "BiasGelu",
        "Conv",
        "ConvTranspose",
        "Einsum",
        "FastGelu",
        "FusedConv",
        "FusedMatMul",
        "Gelu",
        "Gemm",
        "InstanceNormalization",
        "LayerNormalization",
        "MatMul",
        "MultiHeadAttention",
        "QLinearConv",
        "QLinearMatMul",
        "Softmax",
    }
)


def _summarize_ep_assignments(session):
    """Return node/operation counts for each EP recorded by ONNX Runtime."""
    assignments = session.get_provider_graph_assignment_info()
    node_counts: Counter[str] = Counter()
    op_counts: dict[str, Counter[str]] = {}

    for subgraph in assignments:
        ep_name = str(subgraph.ep_name)
        nodes = list(subgraph.get_nodes())
        node_counts[ep_name] += len(nodes)
        ep_ops = op_counts.setdefault(ep_name, Counter())
        ep_ops.update(str(node.op_type) for node in nodes)

    return node_counts, op_counts


def _format_op_counts(counter: Counter[str], limit: int = 12) -> str:
    if not counter:
        return "none"
    items = counter.most_common(limit)
    text = ", ".join(f"{name}={count}" for name, count in items)
    if len(counter) > limit:
        text += ", ..."
    return text


def create_cuda_onnx_session(model_path: str | Path):
    """Create a CUDA-preferred session and verify that real compute uses CUDA.

    CPUExecutionProvider remains available only for nodes that ONNX Runtime
    deliberately cannot or should not place on CUDA (commonly shape-related
    operators). A session where CUDA receives no substantial compute is rejected.
    """
    # Import torch first so its packaged CUDA/cuDNN DLLs are loaded on Windows.
    import torch
    import onnxruntime as ort

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This DTS ONNX tagger requires a usable CUDA GPU. PyTorch reports CUDA is unavailable; "
            "whole-model CPU tagging fallback is disabled."
        )

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime CUDA DLL preload failed: {exc}") from exc

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "ONNX Runtime CUDAExecutionProvider is unavailable. "
            f"Available providers: {available}. Whole-model CPU tagging fallback is disabled."
        )

    session_options = ort.SessionOptions()
    # ORT 1.24+ can report the exact graph partition. We use this rather than
    # session.disable_cpu_ep_fallback because ORT intentionally places some
    # shape-related nodes on CPU to improve performance.
    session_options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not create an ONNX Runtime CUDA session. "
            "Verify ONNX Runtime/CUDA/cuDNN compatibility. "
            f"Original error: {exc}"
        ) from exc

    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            f"CUDAExecutionProvider was not activated (active providers: {active}). "
            "Whole-model CPU tagging fallback is disabled."
        )

    try:
        node_counts, op_counts = _summarize_ep_assignments(session)
    except Exception as exc:
        raise RuntimeError(
            "CUDAExecutionProvider is active, but DTS could not verify ONNX graph assignment. "
            "The pinned DTS ONNX Runtime is expected to support EP graph-assignment reporting. "
            f"Original error: {exc}"
        ) from exc

    cuda_nodes = node_counts.get("CUDAExecutionProvider", 0)
    cuda_ops = op_counts.get("CUDAExecutionProvider", Counter())
    cuda_compute_nodes = sum(count for op, count in cuda_ops.items() if op in _GPU_COMPUTE_OPS)
    if cuda_nodes <= 0 or cuda_compute_nodes <= 0:
        raise RuntimeError(
            "ONNX Runtime loaded CUDAExecutionProvider but assigned no substantial tagger compute to CUDA. "
            f"EP node counts: {dict(node_counts)}. Whole-model CPU tagging fallback is disabled."
        )

    cpu_nodes = node_counts.get("CPUExecutionProvider", 0)
    cpu_ops = op_counts.get("CPUExecutionProvider", Counter())
    print(
        "ONNX tagger EP assignment: "
        f"CUDA={cuda_nodes} node(s) ({cuda_compute_nodes} compute node(s)), "
        f"CPU={cpu_nodes} node(s)."
    )
    if cpu_nodes:
        print(
            "ONNX Runtime kept some graph nodes on CPU (normally shape/control operations): "
            f"{_format_op_counts(cpu_ops)}"
        )

    return session
