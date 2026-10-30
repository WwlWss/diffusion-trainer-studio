"""C3 production SD3 observer subclass, no second training implementation.

Every inherited Trainer operation runs exactly once. Physical C3 information
can be gathered, but C4 Save/Resume evidence is still mandatory before the
parent Worker may emit worker_complete.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys
from typing import Any

from tools import sd3_full_bf16_evidence_support as host
from tools.sd3_full_bf16_worker_observer import (
    PhysicalStepProbe, observe_prepared_ownership, verify_prepared_identity,
)


_SCRIPTS = host.ROOT / "scripts" / "dev"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_real_sd3 = importlib.import_module("sd3_train_network")
_real_source = host._real_path(
    str(Path(_real_sd3.__file__).resolve()),
    name="production SD3 Trainer for Worker", must_exist=True,
)
if _real_source != _SCRIPTS / "sd3_train_network.py":
    raise host.FullBF16EvidenceError("C3 observer imported unreviewed shadow SD3 Trainer")
Sd3NetworkTrainer = _real_sd3.Sd3NetworkTrainer


class C3PhaseEvidence:
    """Physical observation store. C4 must attest checkpoint save and load."""
    def __init__(self, *, probe: PhysicalStepProbe, parameter_ownership: dict[str, Any]):
        self.probe = probe
        self.parameter_ownership = parameter_ownership
        self.training_completed = False

    def finalize(self):
        if not self.training_completed:
            raise host.FullBF16EvidenceError("C3 Trainer did not complete real training")
        # Deliberately fail closed: a physical-step stream alone is not the
        # matching Saved Adapter / Accelerator checkpoint / Fresh→Resume proof.
        raise host.FullBF16EvidenceError(
            "C4 checkpoint and Resume physical evidence is required before worker_complete"
        )


class ObservedSd3NetworkTrainer(Sd3NetworkTrainer):
    """Adds per-instance observation without altering the production loop."""

    def __init__(self, *, invocation: Any, cuda_environment: dict, qualification_lease: dict):
        super().__init__()
        self._c3_invocation = invocation
        self._c3_cuda_environment = cuda_environment
        self._c3_qualification_lease = qualification_lease
        self._c3_session = None
        self._c3_scheduler = None
        self._c3_build_original = None
        self._c3_build_had_instance = False
        self._c3_build_saved = None
        self._c3_build_count = 0
        self._c3_probe: PhysicalStepProbe | None = None
        self._c3_expected_samples = set()
        self.observer: C3PhaseEvidence | None = None

    def configure_parameter_policy_training(self, args, session, text_encoders):
        if self._c3_session is not None:
            raise host.FullBF16EvidenceError("C3 Session configured more than once")
        result = super().configure_parameter_policy_training(
            args, session, text_encoders,
        )
        self._c3_session = session
        self._c3_build_original = session.build_scheduler
        self._c3_build_had_instance = "build_scheduler" in vars(session)
        self._c3_build_saved = vars(session).get("build_scheduler")
        original = self._c3_build_original

        def capture_scheduler(*args, **kwargs):
            if self._c3_build_count != 0:
                raise host.FullBF16EvidenceError(
                    "C3 Session constructed a Scheduler more than once"
                )
            scheduler = original(*args, **kwargs)
            from mikazuki.parameter_policy_torch import CompositeLRScheduler
            if not isinstance(scheduler, CompositeLRScheduler):
                raise host.FullBF16EvidenceError("C3 requires real CompositeLRScheduler")
            self._c3_build_count += 1
            self._c3_scheduler = scheduler
            return scheduler

        session.build_scheduler = capture_scheduler
        return result

    def register_additional_state_hooks(
        self, *, accelerator, args, unet, train_dataloader,
        current_epoch, current_step,
    ):
        # The original SD3 Forward Semantics and Resume cursor hooks MUST run
        # first. No copy/paste or replacement of production hooks.
        super().register_additional_state_hooks(
            accelerator=accelerator, args=args, unet=unet,
            train_dataloader=train_dataloader,
            current_epoch=current_epoch, current_step=current_step,
        )
        session = self._c3_session
        scheduler = self._c3_scheduler
        if session is None or scheduler is None or self._c3_build_count != 1:
            raise host.FullBF16EvidenceError(
                "C3 expected exactly one genuine Session Scheduler construction"
            )
        verify_prepared_identity(
            session=session, accelerator=accelerator, scheduler=scheduler,
        )
        roots = [
            reference.reference()
            for reference in getattr(session, "execution_root_refs", ())
            if reference.scan_key == "network"
        ]
        if len(roots) != 1 or roots[0] is None:
            raise host.FullBF16EvidenceError("C3 cannot recover real Session LoRA root")
        network = roots[0]
        registered_models = getattr(accelerator, "_models", ())
        if not any(accelerator.unwrap_model(item) is network
                   for item in registered_models):
            raise host.FullBF16EvidenceError(
                "C3 Session live LoRA root is not in actual prepared Accelerator models"
            )
        ownership = observe_prepared_ownership(
            session=session, accelerator=accelerator, network=network,
            train_dataloader=train_dataloader, case_id=self._c3_invocation.case_id,
        )
        identity = self._c3_invocation.manifest_payload["input_identity"]
        dataset_files = identity.get("dataset_files", {})
        samples = {
            item.get("identity") for item in dataset_files.values()
            if isinstance(item, dict) and item.get("kind") == "caption"
        }
        expected = self._c3_invocation.manifest_payload["reference"]["effective_batches_per_epoch"]
        if (len(samples) != expected or
                any(not isinstance(item,str)
                    or re.fullmatch(r"dts_c2b_sample_[0-9]{4}", item) is None
                    for item in samples)):
            raise host.FullBF16EvidenceError("C3 Caption markers differ from immutable Dataset identity")
        self._c3_expected_samples = samples
        params = {"network." + name: param for name,param in network.named_parameters()}
        probe = PhysicalStepProbe(
            accelerator=accelerator,
            composite_optimizer=session.optimizer,
            composite_scheduler=scheduler,
            parameter_names=params, case_id=self._c3_invocation.case_id,
            run_id=self._c3_invocation.run_id,
            phase_id=self._c3_invocation.phase_id,
            source_commit=self._c3_invocation.source_commit,
            phase=self._c3_invocation.phase,
            start_step=1 if self._c3_invocation.phase == "resume" else 0,
        )
        self._c3_probe = probe
        self.observer = C3PhaseEvidence(
            probe=probe, parameter_ownership=ownership,
        )
        probe.install()

    def on_step_start(
        self, args, accelerator, network, text_encoders, unet,
        batch, weight_dtype, is_train=True,
    ):
        super().on_step_start(
            args, accelerator, network, text_encoders, unet,
            batch, weight_dtype, is_train=is_train,
        )
        if not is_train:
            return
        if self._c3_probe is None:
            raise host.FullBF16EvidenceError("C3 Batch reached training before physical Hook registration")
        captions = batch.get("captions")
        if not isinstance(captions, (list, tuple)) or len(captions) != 1 or not isinstance(captions[0], str):
            raise host.FullBF16EvidenceError("C3 production Batch must have one actual caption")
        match = re.match(r"^(dts_c2b_sample_[0-9]{4})(?:, |$)", captions[0])
        if match is None or match.group(1) not in self._c3_expected_samples:
            raise host.FullBF16EvidenceError("C3 actual Batch Caption is outside immutable input identity")
        self._c3_probe.begin_batch(match.group(1))

    def train(self, args):
        try:
            # Exactly one production NetworkTrainer.train invocation.
            result = super().train(args)
            if self._c3_probe is None or self.observer is None:
                raise host.FullBF16EvidenceError("C3 physical Probe was never installed")
            self._c3_probe.assert_step_complete()
            self.observer.training_completed = True
            return result
        finally:
            if self._c3_probe is not None:
                self._c3_probe.close()
            if self._c3_session is not None and self._c3_build_original is not None:
                if self._c3_build_had_instance:
                    self._c3_session.build_scheduler = self._c3_build_saved
                elif "build_scheduler" in vars(self._c3_session):
                    delattr(self._c3_session, "build_scheduler")


__all__ = ["ObservedSd3NetworkTrainer", "C3PhaseEvidence"]
