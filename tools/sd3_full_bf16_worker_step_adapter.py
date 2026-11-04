"""C4-A physical Step Anchor and C4-B same-instance real Save observation.

C4-B may certify only after original Accelerator.save_state returns AND
complete same-Step on-disk production evidence verifies. Tiny pinned CPU
diagnostics deliberately never receive a production Step receipt.
C4-C Resume and C4-D/finalize/worker_complete remain fail-closed.
Importing this module never eagerly imports the production SD3 Trainer.
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


@dataclass(frozen=True)
class C4BSaveAttempt:
    classification: str
    logical_step: int | None
    original_call_count: int
    original_returned: bool
    qualified: bool
    failure_reason: str | None


@dataclass(frozen=True)
class C4BVerifiedStepSave:
    case_id: str
    phase: str
    run_id: str
    phase_id: str
    source_commit: str
    logical_step: int
    anchor_sha256: str
    checkpoint_id: str
    adapter_path: str
    adapter_file_sha256: str
    tensor_inventory_sha256: str
    state_dir: str
    state_file_sha256: tuple[tuple[str, str], ...]


class C4StepAdapter:
    """Instance-bound physical Step + genuine Save, with no Resume admission.

    Only the true C3 Probe establishes a physical Step. Only the pinned original
    Accelerator.save_state can create a disk-backed Save attempt. Caller data
    cannot mint a Step Anchor, an accepted checkpoint ID or worker_complete.
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
        self._reviewed_save_method = type(accelerator).save_state
        self._original_save_state = getattr(accelerator, "save_state")
        if (not callable(self._original_save_state)
                or not callable(self._reviewed_save_method)
                or "save_state" in vars(accelerator)):
            _fail("C4-B requires unshadowed original Accelerator.save_state")
        self._managed_save_wrapper = None
        self._production_save = (
            self.accelerator.device.type == "cuda"
            and len(self.probe.params) == 526
        )
        self._lora_writer = None
        self._save_call_count = 0
        self._attempted_steps: set[int] = set()
        self._last_save_attempt: C4BSaveAttempt | None = None
        self._verified_step_records: dict[int, C4BVerifiedStepSave] = {}

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
        # Only our exact managed Save wrapper is a trusted instance override.
        # All other replacements and ANY Load override remain fail-closed.
        if type(self.accelerator).save_state is not self._reviewed_save_method:
            _fail("Original Accelerator.save_state implementation drift")
        if self._production_save and self._installed:
            if self._lora_writer is None:
                _fail("C4-B production original LoRA Writer observer missing")
            self._lora_writer._check_method()
        # Fixed Accelerate 1.6.0 otherwise redirects to checkpoints/checkpoint_N
        # and returns a location different from the authorized Step directory.
        if (getattr(self.accelerator.project_configuration,
                    "automatic_checkpoint_naming", None) is not False):
            _fail("C4-B automatic checkpoint naming is not an authorized Save")
        for method, (had_instance, original) in self._original_accelerator_slots.items():
            expected_present = had_instance
            expected_obj = original
            if method == "save_state" and self._installed:
                expected_present = True
                expected_obj = self._managed_save_wrapper
            if (method in vars(self.accelerator)) != expected_present:
                _fail("Accelerator " + method + " was replaced after C4 binding")
            if expected_present and vars(self.accelerator)[method] is not expected_obj:
                _fail("Accelerator " + method + " instance method drift")

    def install(self) -> None:
        if self._closed or self._installed:
            _fail("C4 Step Adapter cannot be installed twice or after close")
        if not self.probe._installed:
            _fail("C3 Probe must be installed before the Step Adapter")
        self._check_bound_identity()
        def managed_save_state(*args, **kwargs):
            return self._observed_save_state(*args, **kwargs)
        self._managed_save_wrapper = managed_save_state
        try:
            if self._production_save:
                from tools.sd3_full_bf16_worker_save_verifier import (
                    C4BLoRAWriteObserver,
                )
                def actual_writer_step():
                    step = self.probe._last_completed_step
                    if type(step) is not int:
                        _fail("Original LoRA Writer before real physical Step")
                    self.completed_step_anchor(logical_step=step)
                    return step
                self._lora_writer = C4BLoRAWriteObserver(
                    network=self.network, output_dir=self._output_dir,
                    output_name=self._output_name,
                    step_provider=actual_writer_step,
                )
                self._lora_writer.install()
            vars(self.accelerator)["save_state"] = managed_save_state
            self._installed = True
            self._check_bound_identity()
        except BaseException:
            self._restore_save_slot()
            if self._lora_writer is not None:
                self._lora_writer.close()
            self._lora_writer = None
            self._installed = False
            self._managed_save_wrapper = None
            raise

    def _restore_save_slot(self) -> None:
        had, original = self._original_accelerator_slots["save_state"]
        if had:
            vars(self.accelerator)["save_state"] = original
        else:
            vars(self.accelerator).pop("save_state", None)

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
                if self.probe._last_completed_step != step:
                    _fail("Stale Step Save reservation: physical Step has advanced")
                # Re-export from the real C3 physical boundary at classification
                # time; a once-valid reservation is not a perpetual capability.
                # This does NOT publish checkpoint_saved or mark Save successful.
                current = self.completed_step_anchor(logical_step=step)
                if current.anchor_sha256 != self._pending_anchor.anchor_sha256:
                    _fail("Stale Step Save reservation: Anchor state changed")
                return f"authoritative_step_{step}"
        _fail("Unknown, cross-phase or unreviewed C4 Save path")

    def _observed_save_state(self, *args, **kwargs):
        """Original Save exactly once; authority only after post-return proof."""
        self._require_installed()
        # Never return an earlier Step's success record for a new failing
        # attempt, including a failure in the pre-original authorization.
        self._last_save_attempt = None
        if not args and "output_dir" not in kwargs:
            _fail("C4-B Save requires an explicit original output_dir")
        if args and "output_dir" in kwargs:
            _fail("C4-B duplicate original Save output_dir")
        location = args[0] if args else kwargs["output_dir"]
        if not isinstance(location, (str, Path)):
            _fail("C4-B cannot bind unreviewed output_dir type")
        # Resolve is used by the C4-A path classifier, but a symlink alias
        # must NEVER become authorization for the original filesystem write.
        # Reject both linked target and linked parent / traversal aliases.
        literal_path = Path(location).expanduser().absolute()
        resolved_path = literal_path.resolve(strict=False)
        if (literal_path != resolved_path or literal_path.is_symlink()):
            _fail("C4-B linked/symlink/noncanonical Save path before original")
        state_dir = host._real_path(str(location), name="C4-B Save directory")
        label = "unclassified"
        step = None
        try:
            # Classification must precede ORIGINAL pre-save hooks or filesystem
            # writes: even a genuine Accelerator cannot write an unknown path.
            if state_dir == self._output_dir / f"{self._output_name}-state":
                label = self.classify_state_save(state_dir=state_dir)
            else:
                permitted = (1, 2) if self._phase == "fresh" else (2,)
                for candidate in permitted:
                    exact = (self._output_dir /
                             f"{self._output_name}-step{candidate:08d}-state")
                    if state_dir == exact:
                        step = candidate
                        if (self._pending_anchor is None
                                and candidate not in self._reserved_steps):
                            self.reserve_step_save(logical_step=candidate)
                        label = self.classify_state_save(state_dir=state_dir)
                        break
                if step is None:
                    _fail("C4-B unreviewed Step Save path before original write")
            if step is not None and step in self._attempted_steps:
                _fail("C4-B original Save already attempted: retry/duplicate prohibited")
            if step is not None and state_dir.exists():
                _fail("C4-B authoritative Step destination already exists before Save")
            if self._phase == "fresh" and step == 2:
                # A completed *physical* Step-1 is not a saved parent. Fresh
                # Step-2 requires the authentic, still-byte-valid Step-1
                # post-return receipt. Checking the saved bytes AGAIN also
                # rejects Step-1 files tampered with before Step-2 Save.
                self.verified_step_save(logical_step=1)
        except BaseException as exc:
            self._last_save_attempt = C4BSaveAttempt(
                classification=label, logical_step=step,
                original_call_count=0, original_returned=False,
                qualified=False, failure_reason=type(exc).__name__,
            )
            raise

        production = (self.accelerator.device.type == "cuda"
                      and len(self.probe.params) == 526)
        before = None
        if step is not None:
            anchor = self.completed_step_anchor(logical_step=step)
            if anchor.anchor_sha256 != self._pending_anchor.anchor_sha256:
                _fail("C4-B Anchor changed before original Save")
            if production:
                from tools.sd3_full_bf16_worker_save_verifier import (
                    snapshot_serialized_tensor_inventory,
                )
                from tools.sd3_full_bf16_worker_observer import (
                    fingerprint_state_tree, snapshot_rng_fingerprint,
                )
                adapter_path = (self._output_dir /
                                f"{self._output_name}-step{step:08d}.safetensors")
                # The production Trainer writes the external LoRA Step file
                # BEFORE it calls accelerator.save_state. Missing file is an
                # immediate pre-save reject; no fabricated artifact is allowed.
                if not adapter_path.is_file() or adapter_path.is_symlink():
                    _fail("C4-B production Step LoRA Adapter missing before Save")
                if self._lora_writer is None:
                    _fail("C4-B genuine original LoRA Writer not installed")
                observed_write = self._lora_writer.completed_step_write(
                    logical_step=step,
                )
                if (observed_write.output_path != str(adapter_path)
                        or observed_write.file_sha256
                           != host._sha256_file(adapter_path)):
                    _fail("C4-B original LoRA Writer path/bytes do not match Step")
                before = {
                    "writer": observed_write,
                    "anchor": anchor,
                    "adapter_path": adapter_path,
                    "tensors": snapshot_serialized_tensor_inventory(
                        self.network, diagnostic_cpu=False,
                    ),
                    "optimizer": fingerprint_state_tree(
                        self.probe.optimizer.state_dict(),
                    ),
                    "scheduler": fingerprint_state_tree(
                        self.probe.scheduler.state_dict(),
                    ),
                    "rng": snapshot_rng_fingerprint(),
                }

        if step is not None:
            self._attempted_steps.add(step)
        self._save_call_count += 1
        self._last_save_attempt = C4BSaveAttempt(
            classification=label, logical_step=step,
            original_call_count=1, original_returned=False,
            qualified=False, failure_reason=None,
        )
        try:
            # Preserve exact original args/kwargs and its original exceptions.
            result = self._original_save_state(*args, **kwargs)
        except BaseException as exc:
            self._last_save_attempt = C4BSaveAttempt(
                classification=label, logical_step=step,
                original_call_count=1, original_returned=False,
                qualified=False, failure_reason=type(exc).__name__,
            )
            raise
        # Accelerate 1.6.0 returns the *actual* location (and can redirect it).
        # The real original return must match the approved destination, even
        # when the current run is only a non-authoritative CPU diagnostic.
        self._last_save_attempt = C4BSaveAttempt(
            classification=label, logical_step=step,
            original_call_count=1, original_returned=True,
            qualified=False, failure_reason="pending actual-location verification",
        )
        if not isinstance(result, (str, Path)):
            _fail("C4-B original Save returned no actual save_location")
        actual_return_dir = host._real_path(
            str(result), name="C4-B actual original Save location",
            must_exist=True,
        )
        if actual_return_dir != state_dir:
            _fail("C4-B actual Save destination differs from approved Step directory")
        self._last_save_attempt = C4BSaveAttempt(
            classification=label, logical_step=step,
            original_call_count=1, original_returned=True,
            qualified=False, failure_reason="ancillary or CPU unqualified",
        )
        if step is None or not production:
            # A genuine CPU Save can prove the original four-file operation
            # but NEVER the full 263-target 789-tensor/9-file CUDA reference.
            return result
        try:
            from tools.sd3_full_bf16_worker_save_verifier import (
                verify_original_post_save,
            )
            proof = verify_original_post_save(
                network=self.network, probe=self.probe,
                anchor=before["anchor"], state_dir=state_dir,
                adapter_path=before["adapter_path"],
                before_tensors=before["tensors"],
                before_optimizer_sha=before["optimizer"],
                before_scheduler_sha=before["scheduler"],
                before_rng=before["rng"],
            )
            if self.completed_step_anchor(logical_step=step).anchor_sha256 != anchor.anchor_sha256:
                _fail("C4-B physical Step drift after original Save")
            receipt = C4BVerifiedStepSave(
                case_id=anchor.case_id, phase=anchor.phase,
                run_id=anchor.run_id, phase_id=anchor.phase_id,
                source_commit=anchor.source_commit, logical_step=step,
                anchor_sha256=anchor.anchor_sha256,
                checkpoint_id=proof["checkpoint_id"],
                adapter_path=str(before["adapter_path"]),
                adapter_file_sha256=proof["adapter_file_sha256"],
                tensor_inventory_sha256=proof["tensor_inventory_sha256"],
                state_dir=str(state_dir),
                state_file_sha256=tuple(sorted(proof["state_file_sha256"].items())),
            )
            if step in self._verified_step_records:
                _fail("C4-B duplicate verified Step")
            # Publish a Save receipt ONLY after the real Probe lifecycle
            # append succeeds. If _record rejects, no receipt can escape.
            # C4-C/D still gate the entire Phase / worker_complete.
            self.probe._record("checkpoint_saved", details={
                "checkpoint_id": proof["checkpoint_id"],
                "anchor_sha256": anchor.anchor_sha256,
            })
            self._verified_step_records[step] = receipt
            self._pending_anchor = None
            self._last_save_attempt = C4BSaveAttempt(
                classification=label, logical_step=step,
                original_call_count=1, original_returned=True,
                qualified=True, failure_reason=None,
            )
        except BaseException as exc:
            self._last_save_attempt = C4BSaveAttempt(
                classification=label, logical_step=step,
                original_call_count=1, original_returned=True,
                qualified=False, failure_reason=type(exc).__name__,
            )
            raise
        return result

    def save_attempt_observation(self) -> C4BSaveAttempt:
        if self._last_save_attempt is None:
            _fail("No original Accelerator.save_state attempt was observed")
        return self._last_save_attempt

    def verified_step_save(self, *, logical_step: int) -> C4BVerifiedStepSave:
        """Read-only authenticated receipt; late file tamper invalidates it."""
        if type(logical_step) is not int:
            _fail("Invalid verified Step request")
        receipt = self._verified_step_records.get(logical_step)
        if receipt is None:
            _fail("C4-B unqualified: no genuine post-return verified Step Save")
        state = Path(receipt.state_dir)
        if not state.is_dir() or state.is_symlink():
            _fail("Previously verified Step state directory was replaced/linked")
        now = host._checkpoint_file_inventory(state)
        if tuple(sorted(now.items())) != receipt.state_file_sha256:
            _fail("Verified state-file inventory drifted after Save")
        adapter_file = Path(receipt.adapter_path)
        if (not adapter_file.is_file() or adapter_file.is_symlink()
                or host._sha256_file(adapter_file)
                != receipt.adapter_file_sha256):
            _fail("Verified Step Adapter file drifted after Save")
        return receipt

    def close(self) -> None:
        # Restore *exactly* the prior per-instance Save slot shape. Do not
        # mask a real original Save error in Trainer's outer finally.
        self._restore_save_slot()
        if self._lora_writer is not None:
            self._lora_writer.close()
        self._lora_writer = None
        self._managed_save_wrapper = None
        self._pending_anchor = None
        self._installed = False
        self._closed = True


__all__ = [
    "C4PhysicalStepAnchor", "C4BSaveAttempt",
    "C4BVerifiedStepSave", "C4StepAdapter",
]
