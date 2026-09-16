"""GPU runtime validation shared by training launchers without web dependencies."""

from __future__ import annotations

from typing import Optional


def normalize_gpu_ids(gpu_ids: Optional[list]) -> list[str]:
    if not gpu_ids:
        return []
    normalized = [str(value).strip() for value in gpu_ids if str(value).strip()]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"GPU selection contains duplicate ids: {normalized}")
    return normalized


def validate_cuda_training_runtime(gpu_ids: Optional[list] = None) -> tuple[list[str], list[str]]:
    """Require CUDA so a DTS training job can never silently become CPU training.

    CPU remains valid for dataloading, checkpoint loading and explicit
    offload/block-swap features. The Accelerate training device itself must be
    CUDA.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Cannot import PyTorch for CUDA preflight: {exc}") from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError(
            "DTS training requires an NVIDIA CUDA GPU, but PyTorch reports no usable CUDA device. "
            "Training was not started; DTS will not fall back to CPU training."
        )

    selected = normalize_gpu_ids(gpu_ids)
    visible_count = int(torch.cuda.device_count())
    if selected:
        parsed: list[int] = []
        for value in selected:
            try:
                index = int(value)
            except ValueError as exc:
                raise RuntimeError(f"Invalid GPU id '{value}'; GPU ids must be integer device indices.") from exc
            if index < 0 or index >= visible_count:
                raise RuntimeError(
                    f"GPU id {index} is outside the visible CUDA device range 0..{visible_count - 1}."
                )
            parsed.append(index)
    else:
        parsed = list(range(visible_count))

    names = [str(torch.cuda.get_device_name(index)) for index in parsed]
    return selected, names
