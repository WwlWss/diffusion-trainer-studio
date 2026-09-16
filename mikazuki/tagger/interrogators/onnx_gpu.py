"""CUDA-only ONNX Runtime session creation for local DTS taggers."""

from __future__ import annotations

from pathlib import Path


def create_cuda_onnx_session(model_path: str | Path):
    """Create an ONNX session that cannot silently assign graph nodes to CPU."""
    # Import torch first so its packaged CUDA/cuDNN DLLs are loaded on Windows.
    import torch
    import onnxruntime as ort

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This DTS ONNX tagger requires a usable CUDA GPU. PyTorch reports CUDA is unavailable; "
            "CPU tagging fallback is disabled."
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
            f"Available providers: {available}. CPU tagging fallback is disabled."
        )

    session_options = ort.SessionOptions()
    # By default ORT may assign unsupported graph nodes to CPU even if CUDA is
    # explicitly requested. Disable that implicit CPU partition fallback.
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CUDAExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not create a CUDA-only ONNX Runtime tagger session. "
            "DTS will not fall back to CPU; verify ONNX Runtime/CUDA/cuDNN compatibility. "
            f"Original error: {exc}"
        ) from exc

    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            f"CUDAExecutionProvider was not activated (active providers: {active}). "
            "CPU tagging fallback is disabled."
        )
    return session
