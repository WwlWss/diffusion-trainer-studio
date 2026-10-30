"""Restricted SD3C2-B0 Full-BF16 evidence Worker (Fix C2 admission).

This is not an alternate trainer, a qualification publisher, or a CUDA test.
Host admission imports no torch. The genuine ObservedSd3NetworkTrainer belongs
to C3/C4 and is deliberately not replaced by a fake or CPU fallback here.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from tools import sd3_full_bf16_evidence_support as host


@dataclass(frozen=True)
class WorkerInvocation:
    ticket_path: Path
    manifest_path: Path
    ticket_sha256: str
    case_id: str
    phase: str
    run_id: str
    phase_id: str
    source_commit: str
    output_dir: Path
    trainer_argv: list[str]
    expected_resume_checkpoint_id: str | None
    ticket_payload: dict[str, Any] = field(repr=False, compare=False)
    manifest_payload: dict[str, Any] = field(repr=False, compare=False)
    result_path: Path = field(repr=False, compare=False)


_TICKET_FIELDS = frozenset({
    "schema", "version", "run_id", "phase_id", "phase", "case_id",
    "source_commit", "manifest_sha256", "input_identity_sha256",
    "effective_argv_sha256", "expected_resume_checkpoint_id",
    "worker_entrypoint", "output_dir", "nonce",
})


def _reject(message: str) -> None:
    raise host.FullBF16EvidenceError(message)


def parse_worker_cli(argv: list[str] | None) -> Path:
    """One fixed invocation, never a general CLI training argument passthrough."""
    parser = argparse.ArgumentParser(
        description="DTS SD3C2-B0 restricted evidence Worker; no CUDA PASS authority",
        allow_abbrev=False,
    )
    parser.add_argument("--ticket", required=True)
    args = parser.parse_args(argv)
    if not Path(args.ticket).is_absolute():
        _reject("C2 Worker requires an absolute --ticket path")
    return host._external_work_path(args.ticket, name="Worker ticket")


def validate_ticket_manifest_binding(
    ticket: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_file_sha256: str,
) -> dict[str, Any]:
    """Independent exact Ticket/Manifest binding; full asset audit is separate.

    This API intentionally permits synthetic Host manifests for C2 RED tests.
    resolve_phase_invocation additionally enforces the full production manifest,
    physical input SHA256 and clean Git source before ticket consumption.
    """
    host._validate_ticket(ticket)
    if set(ticket) != _TICKET_FIELDS:
        _reject("Worker ticket has missing or unreviewed fields")
    if not isinstance(manifest, dict):
        _reject("Worker manifest must be a JSON object")
    host._require_sha(manifest_file_sha256, name="manifest file sha256")
    phase = ticket["phase"]
    case = ticket["case_id"]
    host.get_case_spec(case)
    if manifest.get("case_id") != case:
        _reject("Ticket and Manifest case ID mismatch")
    if manifest.get("source_commit") != ticket["source_commit"]:
        _reject("Ticket and Manifest source commit mismatch")
    if ticket["manifest_sha256"] != manifest_file_sha256:
        _reject("Ticket manifest full-file SHA256 mismatch")
    if ticket["input_identity_sha256"] != host._json_sha256(
        manifest.get("input_identity")
    ):
        _reject("Ticket input identity SHA256 mismatch")
    command_key = "fresh_command" if phase == "fresh" else "resume_command"
    command = manifest.get(command_key)
    options = host._parse_command(command)
    if ticket["effective_argv_sha256"] != host._json_sha256(command):
        _reject("Ticket effective argv SHA256 mismatch")
    root = host._external_work_path(manifest.get("work_root"), name="manifest work root")
    expected_output = root / phase
    actual_output = host._external_work_path(
        ticket["output_dir"], name="Ticket output dir",
    )
    if actual_output != expected_output:
        _reject("Ticket output directory differs from fixed external work root")
    if host._real_path(options.get("--output_dir"), name="trainer output") != expected_output:
        _reject("Ticket output differs from actual production trainer argument")
    if phase == "fresh" and "--resume" in options:
        _reject("Fresh command cannot request Resume")
    if phase == "resume":
        parent = host._real_path(options.get("--resume"), name="Resume source")
        if parent != host._real_path(
            manifest.get("fresh_checkpoint_dir"), name="declared Fresh checkpoint"
        ):
            _reject("Resume must target the declared Fresh Step-1 checkpoint")
    return {
        "case_id": case, "phase": phase, "source_commit": ticket["source_commit"],
        "output_dir": str(actual_output), "trainer_argv": list(command),
    }


def _verify_source_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on dirty code, mismatched Gitlinks or missing runtime submodule."""
    from tools.prepare_sd3_full_bf16_cases import (
        _source_commit, _source_submodule_inventory,
    )

    actual_head = _source_commit()
    if actual_head != manifest.get("source_commit"):
        _reject("B0 source HEAD differs from immutable candidate Manifest")
    observed = _source_submodule_inventory()
    declared = manifest.get("source_submodules")
    if observed != declared:
        _reject("B0 source Gitlinks/checkout states differ from Manifest")
    # The parent sd-scripts is an in-use runtime source. Other declared
    # Gitlinks may be uninitialized when genuinely unused by this Worker.
    runtime = observed.get("sd-scripts")
    if not isinstance(runtime, dict) or runtime.get("checkout") != "initialized":
        _reject("Worker requires initialized and clean runtime sd-scripts Gitlink")
    return {
        "source_commit": actual_head,
        "submodules": copy.deepcopy(observed),
        "source_clean": True,
    }


def resolve_phase_invocation(ticket_path: Path) -> WorkerInvocation:
    """Read-only, complete Host preflight. Does not consume Ticket or import Torch."""
    path = host._external_work_path(str(ticket_path), name="Worker ticket")
    if not path.is_file() or path.parent.name != "events":
        _reject("Worker ticket must exist under external work_root/events")
    root = path.parent.parent
    manifest_file = host._real_path(
        str(root / "manifest.json"), name="case Manifest", must_exist=True,
    )
    if not manifest_file.is_file():
        _reject("Worker Manifest must be an ordinary file")
    ticket = host._read_json(path)
    manifest = host._read_json(manifest_file)
    if path.name != f"{ticket.get('phase')}-{ticket.get('phase_id')}.ticket.json":
        _reject("Ticket filename must bind its exact phase and ID")
    if host._external_work_path(
        manifest.get("work_root"), name="Manifest work root"
    ) != root:
        _reject("Ticket and Manifest external roots disagree")
    bound = validate_ticket_manifest_binding(
        ticket, manifest,
        manifest_file_sha256=host._sha256_file(manifest_file),
    )
    # Full Host policy, whole input hashes, dataset pixels and genuine model
    # headers must pass BEFORE atomic consumption and CUDA import.
    host.validate_case_assets(manifest, phase=ticket["phase"])
    _verify_source_identity(manifest)

    if ticket["phase"] == "resume":
        # Physical independent restore evidence belongs to C4/Fix D.
        # C2 checks the claimed parent ID actually exists on disk, not that
        # this self-presented parent has already been accepted by Runner.
        progress = host._read_json(
            Path(manifest["fresh_checkpoint_dir"])
            / "dts_parameter_policy_progress.json"
        )
        if progress.get("checkpoint_id") != ticket["expected_resume_checkpoint_id"]:
            _reject("Resume ticket parent checkpoint ID differs from real Progress v1")
    result_path = path.parent / f"{ticket['phase']}-{ticket['phase_id']}.result.json"
    if result_path.exists() or path.with_name(path.name + ".consumed").exists():
        _reject("Stale result or already-consumed Ticket cannot start a Worker")
    return WorkerInvocation(
        ticket_path=path, manifest_path=manifest_file,
        ticket_sha256=host._sha256_file(path), case_id=bound["case_id"],
        phase=ticket["phase"], run_id=ticket["run_id"],
        phase_id=ticket["phase_id"], source_commit=ticket["source_commit"],
        output_dir=Path(bound["output_dir"]),
        trainer_argv=bound["trainer_argv"],
        expected_resume_checkpoint_id=ticket["expected_resume_checkpoint_id"],
        ticket_payload=copy.deepcopy(ticket),
        manifest_payload=copy.deepcopy(manifest),
        result_path=result_path,
    )


def _qualification_snapshot() -> dict[str, dict[str, Any]]:
    from mikazuki import parameter_policy_execution as execution
    return {
        "backends": dict(execution.FULL_BF16_BACKEND_QUALIFICATIONS),
        "optimizers": dict(execution.FULL_BF16_OPTIMIZER_QUALIFICATIONS),
    }


def run_with_temporary_qualification(
    case_id: str, callback: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Host-testable lease *mechanism*. Public CLI never bypasses CUDA checks.

    A callback supplies observations during a single process-local lease.
    This helper does not install a fake Trainer, bypass environment checks,
    issue certificates, or mutate qualifications persistently.
    """
    from mikazuki import parameter_policy_execution as execution
    from tools.parameter_policy_execution_gpu_support import (
        temporary_execution_qualification,
    )

    if not callable(callback):
        _reject("Worker lease requires an actual callback")
    spec = host.get_case_spec(case_id)
    names = tuple(spec["optimizer_types"])
    before = _qualification_snapshot()
    if "sd3-lora" not in before["backends"]:
        _reject("Missing Full-BF16 sd3-lora qualification record")
    if not set(names).issubset(before["optimizers"]):
        _reject("Missing pre-qualified Full-BF16 optimizer record")
    host.validate_qualification_preconditions(
        case_id, before["backends"]["sd3-lora"],
        {name: before["optimizers"][name] for name in names},
    )
    lease_id = "phase-c2b:worker-temp:" + case_id
    result: Any = None
    checked: dict[str, Any] | None = None
    try:
        with temporary_execution_qualification(
            backend="sd3-lora", optimizers=names, evidence_case_id=lease_id,
        ) as receipt:
            def verify_active():
                live = _qualification_snapshot()
                active = live["backends"].get("sd3-lora")
                if (active is None or active.status != "qualified"
                        or active.evidence_case_id != lease_id):
                    _reject("Unexpected active Backend qualification during Worker lease")
                if live["optimizers"] != before["optimizers"]:
                    _reject("Worker attempted to mutate Full-BF16 optimizer records")
                others = {k: v for k, v in live["backends"].items() if k != "sd3-lora"}
                prior = {k: v for k, v in before["backends"].items() if k != "sd3-lora"}
                if others != prior:
                    _reject("Worker attempted to mutate a non-target Backend record")
                return host.validate_qualification_lease_records(
                    case_id, receipt.get("records"),
                    {name: before["optimizers"][name] for name in names},
                    {name: live["optimizers"][name] for name in names},
                )

            checked = verify_active()
            result = callback(dict(checked))
            verify_active()
    finally:
        after = _qualification_snapshot()
        if after != before:
            _reject("Temporary Worker lease did not restore the complete qualification tables")
    if checked is None:
        _reject("Backend-only lease was never validated")
    return {"value": result, "lease": {**checked, "restored": True}}


def validate_runtime_environment(case_id: str) -> dict[str, Any]:
    """Runtime-only: called *after* Ticket Claim, no CPU/simulated-CUDA mode."""
    import torch
    import accelerate
    from tools.parameter_policy_gpu_qualification_environment import (
        validate_qualification_environment,
    )

    spec = host.get_case_spec(case_id)
    evidence = validate_qualification_environment(
        repo_root=host.ROOT,
        torch_module=torch,
        require_muon="Muon" in spec["optimizer_types"],
    )
    if accelerate.__version__ != "1.6.0":
        _reject("C2 Worker requires pinned accelerate==1.6.0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        _reject("Worker requires exactly one visible, selected CUDA device")
    for variable, expected in (
        ("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1"),
        ("RANK", "0"), ("LOCAL_RANK", "0"),
    ):
        observed = os.environ.get(variable)
        if observed is not None and observed != expected:
            _reject(f"Unsupported distributed Worker environment {variable}={observed!r}")
    if os.environ.get("ACCELERATE_USE_CPU", "").lower() in ("true", "1", "yes"):
        _reject("Worker cannot use the Accelerate CPU bypass")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _reject("Worker cannot enter an already-initialized distributed process group")
    return {
        **evidence, "accelerate_version": accelerate.__version__,
        "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0),
        "num_processes": 1, "candidate_case_id": case_id,
    }


def _production_namespace(invocation: WorkerInvocation):
    """Use the actual SD3 parser and production CLI normalization once."""
    scripts_dir = host.ROOT / "scripts" / "dev"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    trainer_module = importlib.import_module("sd3_train_network")
    source_file = host._real_path(
        str(Path(trainer_module.__file__).resolve()), name="real production trainer",
        must_exist=True,
    )
    if source_file != scripts_dir / "sd3_train_network.py":
        _reject("Worker imported a shadow production SD3 Trainer")
    from library import train_util
    parser = trainer_module.setup_parser()
    args = parser.parse_args(invocation.trainer_argv[1:])
    if getattr(args, "config_file", None) is not None:
        _reject("Worker cannot execute a config_file override")
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    options = host._parse_command(invocation.trainer_argv)
    for option, raw in options.items():
        action = parser._option_string_actions.get(option)
        if action is None or not hasattr(args, action.dest):
            _reject(f"Production Parser lacks reviewed option {option}")
        actual = getattr(args, action.dest)
        if raw is True:
            if actual is not True:
                _reject(f"Production Parser lost a true boolean option {option}")
        elif type(actual) is int or type(actual) is float:
            try:
                if Decimal(raw) != Decimal(str(actual)):
                    _reject(f"Production Parser changed effective option {option}")
            except InvalidOperation as exc:
                raise host.FullBF16EvidenceError(
                    f"Unparseable numeric effective option: {option}"
                ) from exc
        elif str(actual) != raw:
            _reject(f"Production Parser changed effective option {option}")
    for name in ("scale_weight_norms", "save_precision", "config_file"):
        if getattr(args, name, None) is not None:
            _reject(f"Unreviewed effective training default: {name}")
    if getattr(args, "full_bf16", None) is not True:
        _reject("Production Namespace lost explicit full_bf16=True")
    from mikazuki.sd3_full_bf16_scope import sd3_full_bf16_config_blockers
    issues = sd3_full_bf16_config_blockers(
        host.build_reference_policy(invocation.case_id), vars(args),
    )
    if issues:
        _reject("Parsed effective C2-A scope mismatch: " + "; ".join(issues))
    return args


def _invoke_real_observer(
    invocation: WorkerInvocation, args: Any, env: dict[str, Any],
    lease: dict[str, Any],
) -> dict[str, Any]:
    """C3 fills in the real ObservedSd3NetworkTrainer; NO synthetic fallback."""
    from tools.sd3_full_bf16_worker_observer import ObservedSd3NetworkTrainer

    trainer = ObservedSd3NetworkTrainer(
        invocation=invocation, cuda_environment=env, qualification_lease=lease,
    )
    trainer.train(args)  # exactly one inherited production training invocation
    observed = trainer.observer.finalize()
    if not isinstance(observed, dict) or observed.get("status") != "observed_complete":
        _reject("Worker Observer did not produce an accepted physical observation")
    return observed


def publish_failure_result(
    path: Path, identity: Mapping[str, Any], error: BaseException,
) -> dict[str, Any]:
    """Preserve the failure; publish once and never certify CUDA or release."""
    for field_name in ("run_id", "phase_id", "case_id", "phase", "source_commit"):
        if not isinstance(identity.get(field_name), str) or not identity[field_name]:
            _reject(f"Failure publication requires {field_name}")
    payload = {
        "schema": host.PHASE_SCHEMA, "version": 1,
        "status": "fail", "run_id": identity["run_id"],
        "phase_id": identity["phase_id"], "case_id": identity["case_id"],
        "phase": identity["phase"], "source_commit": identity["source_commit"],
        "error": {"type": type(error).__name__, "message": str(error)},
        "production_qualification_mutated": False,
        "backend_qualification_eligible": False,
    }
    host.publish_phase_result(path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """C2 can fail safely; C3/C4 must be completed before Worker is runnable."""
    path = parse_worker_cli(argv)
    invocation = resolve_phase_invocation(path)
    # A second read for identity before O_EXCL. Original bytes are pinned to
    # the preflight digest and expected Ticket; no replay after crash.
    if host._sha256_file(invocation.manifest_path) != invocation.ticket_payload["manifest_sha256"]:
        _reject("Manifest changed between Worker preflight and Ticket claim")
    claim = host.claim_phase_ticket(
        invocation.ticket_path, expected=invocation.ticket_payload,
    )
    try:
        if claim["ticket_sha256"] != invocation.ticket_sha256:
            _reject("Ticket mutated between preflight and exclusive claim")
        env = validate_runtime_environment(invocation.case_id)
        args = _production_namespace(invocation)
        leased = run_with_temporary_qualification(
            invocation.case_id,
            lambda lease: _invoke_real_observer(invocation, args, env, lease),
        )
        observed = leased["value"]
        if host._sha256_file(invocation.manifest_path) != invocation.ticket_payload["manifest_sha256"]:
            _reject("Manifest mutated during training")
        result = {
            "schema": host.PHASE_SCHEMA, "version": 1, "status": "worker_complete",
            "case_id": invocation.case_id, "phase": invocation.phase,
            "run_id": invocation.run_id, "phase_id": invocation.phase_id,
            "source_commit": invocation.source_commit,
            "ticket_sha256": claim["ticket_sha256"],
            "manifest_sha256": invocation.ticket_payload["manifest_sha256"],
            "cuda_environment": env,
            "qualification_lease": leased["lease"],
            "observations": observed,
            "production_qualification_mutated": False,
            "backend_qualification_eligible": False,
        }
        host.publish_phase_result(invocation.result_path, result)
        return 0
    except BaseException as exc:
        identity = {
            "run_id": invocation.run_id, "phase_id": invocation.phase_id,
            "case_id": invocation.case_id, "phase": invocation.phase,
            "source_commit": invocation.source_commit,
        }
        try:
            publish_failure_result(invocation.result_path, identity, exc)
        except BaseException as publish_error:
            print(
                f"C2 Worker FAILED: {type(exc).__name__}: {exc}; "
                f"failure publication also failed: {publish_error}",
                file=sys.stderr,
            )
        else:
            print(f"C2 Worker FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
