"""Tooling-only tensor/state evidence helpers for Phase C GPU qualification."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy().tobytes()


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_fingerprint requires torch.Tensor.")
    detached = tensor.detach()
    finite = (
        bool(torch.isfinite(detached).all().item())
        if tensor.is_floating_point() or tensor.is_complex()
        else True
    )
    nonzero = bool(torch.count_nonzero(detached).item()) if tensor.numel() else False
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
        "finite": finite,
        "nonzero": nonzero,
    }


def canonical_state_evidence(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_fingerprint(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): canonical_state_evidence(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_state_evidence(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def state_fingerprint(value: Any) -> dict[str, Any]:
    canonical = canonical_state_evidence(value)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "value": canonical,
    }


def model_parameter_evidence(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: tensor_fingerprint(parameter)
        for name, parameter in sorted(model.named_parameters())
    }


def optimizer_state_evidence(optimizer: object) -> dict[str, Any]:
    state_dict = optimizer.state_dict()
    return state_fingerprint(state_dict)


def scheduler_state_evidence(scheduler: object | None) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    return state_fingerprint(scheduler.state_dict())


def gradient_evidence(parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, parameter in enumerate(parameters):
        grad = parameter.grad
        rows[str(index)] = None if grad is None else tensor_fingerprint(grad)
    return rows


__all__ = [
    "canonical_state_evidence",
    "gradient_evidence",
    "model_parameter_evidence",
    "optimizer_state_evidence",
    "scheduler_state_evidence",
    "state_fingerprint",
    "tensor_fingerprint",
]
