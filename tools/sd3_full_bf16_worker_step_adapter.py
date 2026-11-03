"""C4-A read-only bridge for an independently observed real physical Step.

C4-A does not wrap Accelerator.save_state/load_state, publish checkpoint files,
emit checkpoint_saved/checkpoint_loaded/train_completed, certify a Resume parent,
or release the C3PhaseEvidence.finalize() fail-closed barrier. C4-B/C/D must
supply authentic same-process Save/Load and cross-process acceptance evidence.
Importing this module never imports the production SD3 Trainer or Torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from tools import sd3_full_bf16_evidence_support as host


_ANCHOR_SCHEMA = "dts.sd3c2b.c4a.physical-step-anchor-v1"
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _fail(message: str) -> None:
    raise host.FullBF16EvidenceError("C4-A Step Adapter: " + message)


def canonical_c4_anchor_sha256(payload: dict[str, Any]) -> str:
    """Typed, canonical and process-address-free hash of the Anchor payload."""
    return host._json_sha256({"schema": _ANCHOR_SCHEMA, "anchor": payload})


@dataclass(frozen=True)
class C4PhysicalStepAnchor:
    case_id: str
    phase: str
    run_id: str
    phase_id: str
    source_commit: str
    logical_step: int
    accumulation: int
    samples: tuple[str, ...]
    child_profiles: tuple[str, ...]
    physical_scheduler_event_seq: int
    trained_adapter_parameter_sha256: str
    child_optimizer_state_sha256: tuple[tuple[str, str], ...]
    scheduler_state_sha256: str
    anchor_sha256: str


class C4StepAdapter:
    """An instance-bound, fail-closed pending-save classifier (C4-A only).

    The only Step proof is the already-installed C3 PhysicalStepProbe. No
    caller-provided metadata or manually emitted checkpoint event is accepted.
    This class intentionally performs NO save/load I/O or method replacement.
    """

    def __init__(
        self, *, probe: Any, accelerator: Any, network: Any,
        scheduler: Any, invocation: Any, manifest_payload: Any,
    ) -> None:
        # Delay the Torch/C3 import until a caller constructs the adapter.
        from tools.sd3_full_bf16_worker_observer import PhysicalStepProbe

        if type(probe) is not PhysicalStepProbe:
            _fail("PhysicalStepProbe must be the exact C3 physical authority")
        # SR1 Fix B: pin the class-defined exporter rather than an overridable
        # instance-bound attribute. The physical C3 implementation remains
        # the sole producer of Step evidence.
        reviewed_exporter = PhysicalStepProbe.export_completed_step_anchor
        if ("export_completed_step_anchor" in vars(probe)
                or not callable(reviewed_exporter)):
            _fail("Unreviewed Probe exporter override at C4 binding")
        if (probe.accelerator is not accelerator
                or probe.scheduler is not scheduler
                or probe.optimizer is not scheduler.optimizer):
            _fail("Accelerator, CompositeOptimizer and Scheduler identity mismatch")
        if probe._lora_network is not None and probe._lora_network is not network:
            _fail("Network differs from the C3 prepared LoRA root")
        if not callable(getattr(network, "named_parameters", None)):
            _fail("C4 requires the actual prepared Network")
        expected_ids = {id(parameter) for parameter in probe.params.values()}
        actual_ids = {id(parameter) for _, parameter in network.named_parameters()}
        if (len(expected_ids) != len(probe.params)
                or len(actual_ids) != len(expected_ids)
                or actual_ids != expected_ids):
            _fail("Prepared Network Parameter identity differs from the C3 Probe")

        identity_fields = (
            "case_id", "phase", "run_id", "phase_id", "source_commit",
        )
        identity = tuple(getattr(invocation, key, None) for key in identity_fields)
        if identity != tuple(getattr(probe, key) for key in identity_fields):
            _fail("Phase Ticket, Source and Probe identity mismatch")
        if (probe.phase not in ("fresh", "resume")
                or probe.start_step != (0 if probe.phase == "fresh" else 1)):
            _fail("Unreviewed Fresh/Resume Phase or start-step geometry")
        host._require_sha(probe.source_commit, name="C4 source commit", commit=True)
        if not isinstance(manifest_payload, dict):
            _fail("C4 requires a reviewed Case Manifest mapping")
        manifest = manifest_payload
        if manifest.get("case_id") != probe.case_id:
            _fail("C4 Manifest differs from exact Case ID")
        for key in ("phase", "run_id", "phase_id", "source_commit"):
            if key in manifest and manifest[key] != getattr(probe, key):
                _fail("C4 Manifest disagrees with the Phase Ticket: " + key)
        reference = manifest.get("reference")
        if not isinstance(reference, dict) or type(reference.get("max_train_steps")) is not int or reference["max_train_steps"] != 2:
            _fail("C4 requires fixed two-Step reference training")
        spec = host.get_case_spec(probe.case_id)
        if ("gradient_accumulation_steps" in reference
                and reference["gradient_accumulation_steps"] != probe.accumulation):
            _fail("C4 reference accumulation differs from physical C3 Probe")
        if probe.accumulation != spec["gradient_accumulation_steps"]:
            _fail("C4 physical accumulation differs from fixed R1/R2 geometry")

        # In production the filename comes from the fixed CaseSpec, never a
        # mutable Worker-supplied name. Tiny CPU fixtures may use their local
        # diagnostic prefix and do not confer CUDA reference authority.
        production = (len(probe.params) == 526 and accelerator.device.type == "cuda")
        output_name = manifest.get("output_name", spec["output_name"])
        if (not isinstance(output_name, str)
                or _OUTPUT_NAME.fullmatch(output_name) is None
                or (production and output_name != spec["output_name"])):
            _fail("C4 unreviewed output_name / production CaseSpec mismatch")
        work_root = manifest.get("work_root")
        if work_root is not None:
            root = host._real_path(str(work_root), name="C4 manifest work_root")
            expected_output = root / probe.phase
        else:
            expected_output = None
        declared_output = getattr(invocation, "output_dir", None)
        if declared_output is None:
            declared_output = manifest.get("output_dir")
        actual_output = host._real_path(
            str(declared_output), name="C4 phase output directory",
        )
        if expected_output is not None and actual_output != expected_output:
            _fail("C4 output path differs from immutable Manifest work_root/phase")
        if production and expected_output is None:
            _fail("Real CUDA requires independently verified Manifest work_root")
        if production:
            command_key = "fresh_command" if probe.phase == "fresh" else "resume_command"
            options = host._parse_command(manifest.get(command_key))
            if (host._real_path(options.get("--output_dir"), name="C4 argv output")
                    != actual_output
                    or options.get("--output_name") != output_name):
                _fail("C4 production command output path/name mismatch")

        self.probe = probe
        self.accelerator = accelerator
        self.network = network
        self.scheduler = scheduler
        self._invocation = invocation
        self._identity_fields = identity_fields
        self._identity = identity
        self._reviewed_exporter = reviewed_exporter
        self._expected_profiles = tuple(probe.child_profiles)
        self._expected_accumulation = probe.accumulation
        self._output_dir = actual_output
        self._output_name = output_name
        self._phase = probe.phase
        self._original_accelerator_slots = {
            name: (name in vars(accelerator), vars(accelerator).get(name))
            for name in ("save_state", "load_state")
        }
        self._installed = False
        self._closed = False
        self._reserved_steps: set[int] = set()
        self._pending_anchor: C4PhysicalStepAnchor | None = None

    def _check_bound_identity(self) -> None:
        if (self.probe.accelerator is not self.accelerator
                or self.probe.scheduler is not self.scheduler
                or self.probe.optimizer is not self.scheduler.optimizer
                or (self.probe._lora_network is not None
                    and self.probe._lora_network is not self.network)):
            _fail("C4 physical object identity changed after binding")
        if tuple(getattr(self._invocation, key, None)
                 for key in self._identity_fields) != self._identity:
            _fail("C4 external invocation identity changed after binding")
        if tuple(getattr(self.probe, key) for key in self._identity_fields) != self._identity:
            _fail("C4 C3 Probe identity changed after binding")
        if (tuple(self.probe.child_profiles) != self._expected_profiles
                or self.probe.accumulation != self._expected_accumulation):
            _fail("C4 physical Child/accumulation geometry changed after binding")
        if ("export_completed_step_anchor" in vars(self.probe)
                or type(self.probe).export_completed_step_anchor is not self._reviewed_exporter):
            _fail("Unreviewed Probe exporter override")
        # C4-A owns no save/load patches; reject an unreviewed replacement,
        # rather than silently delegating to an arbitrary method.
        for method, (had_instance, original) in self._original_accelerator_slots.items():
            if (method in vars(self.accelerator)) != had_instance:
                _fail("Accelerator " + method + " was replaced after C4 binding")
            if had_instance and vars(self.accelerator)[method] is not original:
                _fail("Accelerator " + method + " instance method drift")

    def install(self) -> None:
        if self._closed or self._installed:
            _fail("C4 Step Adapter cannot be installed twice or after close")
        if not self.probe._installed:
            _fail("C3 Probe must be installed before the Step Adapter")
        self._check_bound_identity()
        self._installed = True

    def _require_installed(self) -> None:
        if not self._installed or self._closed or not self.probe._installed:
            _fail("C4 Step Adapter is not attached to a live C3 Probe")
        self._check_bound_identity()

    def _validate_anchor(
        self, anchor: C4PhysicalStepAnchor, *, logical_step: int,
    ) -> None:
        """Independently validate immutable payload, geometry and canonical SHA.

        This validator does not mint physical Step evidence; only the pinned
        C3 class exporter is allowed to produce it.
        """
        if type(logical_step) is not int or logical_step not in (1, 2):
            _fail("Unreviewed logical Step requested")
        if type(anchor) is not C4PhysicalStepAnchor:
            _fail("C4 exporter returned an unreviewed Anchor type")
        if ((anchor.case_id, anchor.phase, anchor.run_id,
             anchor.phase_id, anchor.source_commit) != self._identity):
            _fail("C4 Anchor Phase Ticket identity mismatch")
        if (type(anchor.logical_step) is not int
                or anchor.logical_step != logical_step):
            _fail("C4 Anchor logical Step mismatch")
        if (type(anchor.accumulation) is not int
                or anchor.accumulation != self._expected_accumulation
                or anchor.child_profiles != self._expected_profiles
                or type(anchor.child_profiles) is not tuple):
            _fail("C4 Anchor physical Child/accumulation geometry mismatch")
        if ((anchor.phase == "fresh" and anchor.logical_step not in (1, 2))
                or (anchor.phase == "resume" and anchor.logical_step != 2)):
            _fail("C4 Anchor has an impossible phase/Step pair")
        if (type(anchor.samples) is not tuple
                or len(anchor.samples) != self._expected_accumulation
                or len(set(anchor.samples)) != len(anchor.samples)
                or any(type(sample) is not str
                       or re.fullmatch(r"dts_c2b_sample_[0-9]{4}", sample) is None
                       for sample in anchor.samples)):
            _fail("C4 Anchor physical sample/microbatch identity mismatch")
        if (type(anchor.physical_scheduler_event_seq) is not int
                or anchor.physical_scheduler_event_seq < 1):
            _fail("C4 Anchor genuine Scheduler event sequence is missing")
        if (type(anchor.child_optimizer_state_sha256) is not tuple
                or len(anchor.child_optimizer_state_sha256) != len(self._expected_profiles)):
            _fail("C4 Anchor physical Child state inventory mismatch")
        for expected, record in zip(
            self._expected_profiles, anchor.child_optimizer_state_sha256
        ):
            if (type(record) is not tuple or len(record) != 2
                    or record[0] != expected):
                _fail("C4 Anchor ordered Child Profile state mismatch")
            host._require_sha(record[1], name="C4 Anchor Child state digest")
        host._require_sha(
            anchor.trained_adapter_parameter_sha256, name="C4 Anchor Adapter digest"
        )
        host._require_sha(
            anchor.scheduler_state_sha256, name="C4 Anchor Scheduler digest"
        )
        host._require_sha(anchor.anchor_sha256, name="C4 Anchor digest")
        from dataclasses import fields
        payload = {
            field.name: getattr(anchor, field.name)
            for field in fields(C4PhysicalStepAnchor)
            if field.name != "anchor_sha256"
        }
        if canonical_c4_anchor_sha256(payload) != anchor.anchor_sha256:
            _fail("C4 Anchor digest mismatch")

    def completed_step_anchor(self, *, logical_step: int) -> C4PhysicalStepAnchor:
        self._require_installed()
        if type(logical_step) is not int:
            _fail("Logical Step must be an integer")
        # Call the exact reviewed C3 class method; never resolve a mutable
        # instance attribute or treat caller metadata as a physical Step.
        anchor = self._reviewed_exporter(self.probe, logical_step)
        self._validate_anchor(anchor, logical_step=logical_step)
        return anchor

    def reserve_step_save(self, *, logical_step: int) -> C4PhysicalStepAnchor:
        self._require_installed()
        if logical_step in self._reserved_steps or self._pending_anchor is not None:
            _fail("Duplicate or unresolved C4 Step-N Save reservation")
        anchor = self.completed_step_anchor(logical_step=logical_step)
        self._reserved_steps.add(logical_step)
        self._pending_anchor = anchor
        # C4-A MUST NOT emit checkpoint_saved; C4-B will only clear pending
        # after successful original save_state and on-disk byte attestation.
        return anchor

    def classify_state_save(self, *, state_dir: Path | str) -> str:
        self._require_installed()
        actual = host._real_path(str(state_dir), name="C4 save_state target")
        if actual == self._output_dir / f"{self._output_name}-state":
            return "ancillary_last"
        permitted = (1, 2) if self._phase == "fresh" else (2,)
        for step in permitted:
            exact = self._output_dir / f"{self._output_name}-step{step:08d}-state"
            if actual == exact:
                if (self._pending_anchor is None
                        or self._pending_anchor.logical_step != step):
                    _fail("Step-named Save has no one-use real Step Anchor reservation")
                return f"authoritative_step_{step}"
        _fail("Unknown, cross-phase or unreviewed C4 Save path")

    def close(self) -> None:
        # No Save/Load hooks are installed until C4-B/C. In particular this
        # method never replaces accelerator methods or publishes success.
        self._pending_anchor = None
        self._installed = False
        self._closed = True


__all__ = ["C4PhysicalStepAnchor", "C4StepAdapter"]
