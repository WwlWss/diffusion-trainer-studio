#!/usr/bin/env python3
"""Real-CUDA Parameter Policy optimizer/runtime qualification matrix.

This probe intentionally uses tiny synthetic CUDA tensors. It qualifies the
shared Component optimizer facade, precision/scaler behavior, supported
optimizer implementations, mixed profiles, and optimizer save/resume without
requiring model checkpoints or datasets.

Backend/model-family smoke is a separate release axis documented in
docs/parameter-policy-step6f-plan.md.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

from mikazuki.optimizer_profiles import list_optimizer_capabilities
from mikazuki.parameter_policy_runtime import (
    OptimizerGroupSpec,
    OptimizerInstanceSpec,
    ParameterPolicyRuntimeSpec,
    RuntimeParameterSpec,
)
from mikazuki.parameter_policy_torch import build_parameter_policy_optimizer


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_parameter(parameter: torch.nn.Parameter, name: str, component: str):
    return RuntimeParameterSpec(
        parameter=parameter,
        parameter_id=id(parameter),
        canonical_name=name,
        shape=tuple(parameter.shape),
        numel=parameter.numel(),
        component_id=component,
        route_kind="primary",
        parameter_class="matrix_weight",
    )


def _optimizer_spec(
    profile: str,
    optimizer_type: str,
    parameter: torch.nn.Parameter,
    *,
    lr: float = 1e-3,
):
    item = _runtime_parameter(parameter, f"{profile}.weight", profile)
    group = OptimizerGroupSpec(learning_rate=lr, parameters=(item,))
    managed = optimizer_type in {
        "RAdamScheduleFree",
        "AdamWScheduleFree",
        "SGDScheduleFree",
    }
    return OptimizerInstanceSpec(
        profile_name=profile,
        optimizer_type=optimizer_type,
        optimizer_arguments_json="{}",
        supports_group_lr=True,
        uses_external_scheduler=not managed,
        lr_semantics="optimizer_managed" if managed else "normal",
        groups=(group,),
        topology_fingerprint=f"gpu-matrix:{profile}:{optimizer_type}",
    )


def _runtime_spec(types: list[str]):
    parameters = [
        torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.float32))
        for _ in types
    ]
    optimizers = tuple(
        _optimizer_spec(f"profile_{index}", optimizer_type, parameter)
        for index, (optimizer_type, parameter) in enumerate(zip(types, parameters))
    )
    spec = ParameterPolicyRuntimeSpec(
        version=1,
        optimizers=optimizers,
        topology_fingerprint="gpu-matrix:" + "+".join(types),
    )
    return spec, parameters


def _one_step(types: list[str], precision: str) -> dict:
    spec, parameters = _runtime_spec(types)
    optimizer = build_parameter_policy_optimizer(spec)
    before = [parameter.detach().clone() for parameter in parameters]

    if precision == "fp16":
        dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    elif precision == "bf16":
        dtype = torch.bfloat16
        scaler = None
    elif precision == "fp32":
        dtype = torch.float32
        scaler = None
    else:
        raise ValueError(f"unknown precision {precision!r}")

    optimizer.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=precision in {"fp16", "bf16"},
    ):
        loss = sum((parameter @ parameter.transpose(0, 1)).square().mean() for parameter in parameters)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    changed = [not torch.equal(old, parameter.detach()) for old, parameter in zip(before, parameters)]
    if not all(changed):
        raise AssertionError(f"optimizer did not update all owned tensors: {changed}")

    state = optimizer.state_dict()

    resumed_spec, resumed_parameters = _runtime_spec(types)
    resumed = build_parameter_policy_optimizer(resumed_spec)
    resumed.load_state_dict(state)
    resumed.train()
    resumed.zero_grad(set_to_none=True)
    resume_loss = sum(parameter.square().mean() for parameter in resumed_parameters)
    resume_loss.backward()
    resumed.step()
    torch.cuda.synchronize()

    return {
        "loss": float(loss.detach().float().cpu()),
        "resume_loss": float(resume_loss.detach().float().cpu()),
        "profiles": len(types),
        "state_children": sorted(state["children"]),
    }


def _supported_optimizer_types() -> list[str]:
    return [
        capability.name
        for capability in list_optimizer_capabilities()
        if capability.component_support == "supported"
    ]


def _cases() -> list[tuple[str, list[str], str]]:
    supported = _supported_optimizer_types()
    cases = [(f"optimizer:{name}", [name], "fp32") for name in supported]
    cases.extend(
        [
            ("precision:adamw-fp16", ["AdamW"], "fp16"),
            ("precision:adamw-bf16", ["AdamW"], "bf16"),
            ("mixed:adamw-schedulefree", ["AdamW", "AdamWScheduleFree"], "fp32"),
            ("mixed:muon-fallback", ["Muon", "AdamW"], "fp32"),
        ]
    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="parameter-policy-gpu-matrix.json")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("Parameter Policy GPU matrix requires a real CUDA device.")

    selected = set(args.case)
    cases = [
        case for case in _cases()
        if not selected or case[0] in selected
    ]
    if selected and len(cases) != len(selected):
        known = {case[0] for case in _cases()}
        unknown = sorted(selected - known)
        raise SystemExit("Unknown matrix case(s): " + ", ".join(unknown))

    evidence = {
        "schema_version": 1,
        "commit": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": {
            name: _package_version(name)
            for name in (
                "accelerate",
                "bitsandbytes",
                "lion-pytorch",
                "schedulefree",
                "pytorch-optimizer",
            )
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cases": [],
    }

    failed = False
    for name, optimizer_types, precision in cases:
        started = time.time()
        row = {
            "name": name,
            "optimizer_types": optimizer_types,
            "precision": precision,
        }
        try:
            row["details"] = _one_step(optimizer_types, precision)
            row["status"] = "pass"
        except Exception as exc:
            failed = True
            row["status"] = "fail"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["duration_seconds"] = round(time.time() - started, 4)
        evidence["cases"].append(row)
        print(f"{row['status'].upper():4} {name}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
