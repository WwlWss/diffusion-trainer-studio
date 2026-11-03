# D1-SD3C2-B0 — C4-A Step Adapter source-level contract and Tests-only RED plan

> **Status: C4-A CONTRACT V1 / DESIGN-ONLY.** Frozen against PR #65 source baseline `1e2393718aa6f816125f66d4b01eb5913b405ad5` (F11-D scoped PASS, 452 real Host/CPU tests PASS + 6 intentional Host skips). **NO C4 implementation, RED execution, checkpoint/Resume GREEN, actual R1/R2 CUDA qualification, worker_complete, C3 Ready or release claim is made here.** PR #65 remains Draft/unmerged. The C3 `C3PhaseEvidence.finalize()` fail-closed barrier must remain in place until all C4 requirements are proved.

## 1. Purpose, exact execution authority and separation

C4-A establishes a minimal **read-only evidence bridge** between an *already validated real physical Step* and the production-owned Checkpoint Save/Load calls. It must not implement a second training loop, simulate a GPU, perform an extra Backward/Step, force the Scheduler forward, modify optimizer/LoRA weight state, invoke another Save/Load, alter any production Checkpoint ABI, or lift the pending Full-BF16 backend qualification.

The authoritative production order (single Worker process):

1. `scripts/dev/train_network.py:1532-1556`: actual prepared `Accelerator.accumulate`, SD3 `process_batch`, `accelerator.backward`.
2. `train_network.py:1557-1574`: clipping, prepared Optimizer and Scheduler steps, zero-grad. `PhysicalStepProbe` observes actual Child `optimizer.step()` and `CompositeLRScheduler.step()`.
3. `tools/sd3_full_bf16_worker_observer.py` Scheduler wrapper, currently around lines 1854–1886: after all actual Children and Scheduler validate, it records `physical_scheduler_stepped`, freezes `_last_completed_step` and accepted full trained-Parameter SHA256. **This is the first boundary at which a Step may become eligible for a checkpoint.**
4. `train_network.py:1600-1625`: optional sampling then, at each reference Step, `save_model(ckpt_name,...)` performs the actual `network.save_weights` and only then `train_util.save_and_remove_state_stepwise` invokes the real `accelerator.save_state(step_state_dir)`. The two writes are distinct physical operations.
5. `train_network.py:1835-1843`: training end also calls `save_state_on_train_end`, normally writing `<output_name>-state`, and last adapter. **This is a permitted ancillary tail, not another physical Step or authoritative Step-1/Step-2 evidence.**
6. Resume: `train_network.py:1046-1057` registers production save/load hooks and calls real `accelerator.load_state(args.resume)` **after** `ObservedSd3NetworkTrainer.register_additional_state_hooks()`. Resume restoration must be observed *after the original load returns*, but *before* `plan_backend_resume_cursor` and the first resumed Backward.

The true SD3 and Parameter Policy pre-save/pre-load hooks, Step Adapter and Probe must all target the **same physical Accelerator instance** and prepared CompositeOptimizer/Scheduler/Network references; no substitute model or synthetic state is acceptable.

## 2. Fixed scope and reference source

- Case IDs: `r1-adamw-accum1` and `r2-muon-adamw-fallback-accum2` only.
- Phase IDs: `fresh`, `resume`; independent OS processes, run/phase/source IDs verified through existing one-shot Tickets, not from mutable Probe output.
- Both production commands retain total `max_train_steps=2` (never truncate Fresh to 1); Fresh physically completes Steps 1 **and 2**. Resume uses the **Fresh Step-1** checkpoint in a fresh process, restores *before* Backward, then physically executes Step 2.
- Actual R1 reference: one AdamW Child, one prepared Batch / Step. Actual R2: two independent disjoint Muon+AdamW Children, two different physically observed Batches / Step, `sync_gradients=(False,True)`, no early optimizer/scheduler update. Existing C3 Probe guards remain the physical authority.
- Pinned real CPU diagnostic environment is Torch 2.7.0 + Accelerate 1.6.0 + pytorch-optimizer 3.10.0. CPU fixtures test mechanics only. Production B1/B2 requires a genuine single CUDA BF16 device, complete 526 trained BF16 LoRA Parameters and 263 Alpha Buffers, and complete fixed R1/R2 assets.

## 3. Proposed internal module and typed interfaces (C4-A only)

**New module** `tools/sd3_full_bf16_worker_step_adapter.py` — importable by real pinned CPU tests without eagerly importing the full SD3 training graph. The existing `tools/sd3_full_bf16_observed_trainer.py` remains the **sole** production Trainer subclass. The existing `tools/sd3_full_bf16_worker_observer.py` retains *physical* Backward/Child/Scheduler authority; C4 must not duplicate its proof.

Proposed interfaces, names frozen for upcoming RED tests; exact implementation can only change after an explicit contract review:

~~~python
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
    anchor_sha256: str   # canonical hash of the above without itself

class C4StepAdapter:
    def __init__(self, *, probe, accelerator, network,
                 scheduler, invocation, manifest_payload): ...
    def completed_step_anchor(self, *, logical_step: int) -> C4PhysicalStepAnchor: ...
    def install(self) -> None: ...
    def close(self) -> None: ...
~~~

The **only** permitted origin of an Anchor is the prepared `PhysicalStepProbe` after its actual `CompositeLRScheduler.step()` validation. Introduce one narrow **read-only** Probe method (e.g. `export_completed_step_anchor(step)`) that verifies all underlying counters/invariants and returns primitive immutable evidence. The Adapter must not inspect arbitrary mutable child objects and proclaim a new baseline, nor change private Probe counters. `tuple` fields avoid mutable nested mappings. A cross-process record uses canonical Parameter/Profile names and typed SHA256, **never** `id(parameter)`, Python repr, object addresses, or a GPU pointer.

The adapter may create a one-use *pending* Save candidate from an accepted Step, but **C4-A does not implement or certify** successful Checkpoint file binding. No generic caller-provided `emit("checkpoint_saved", ...)` or manual `mark_success()` API is permitted. C4-B/C will add fixed, private instance-wrapper success paths tied to the original Accelerator operations and independently verified files.

**Identity freeze:** bind the actual `probe`, `accelerator`, prepared `network`, CompositeOptimizer and CompositeLRScheduler objects once. The reference source is independently audited C3 Prepared ownership, exact known fixed paths from `manifest_payload`, and Phase Ticket. Never recompute authority from a post-mutation live Group, a newly loaded file supplied by the Worker, or a self-declared `checkpoint_id`.

## 4. Step eligibility, snapshot construction and lifecycle

An Anchor is eligible iff all hold simultaneously:

- `probe._installed` and no incomplete R2 accumulation, no retained non-sync step, and no missing Backward.
- Expected physical Child profile set is exact, **each** Child count for the requested Step equals 1, and its real Scheduler count equals 1. No extra Child/Scheduler event, unknown Profile, duplicate event or out-of-order Step.
- The requested Step is exactly 1/2 in Fresh or 2 in Resume. In Resume, Step-2 is not eligible until an authentic successful load has been observed; the parent must be the Host-accepted Fresh Step-1 (Fix D ancestry is separately required).
- `probe._last_completed_step == requested_step`, the C3 accepted full Adapter SHA equals an immediate new read-only fingerprint of **the same canonical prepared trainable Parameters**, and ordered Group/Forward/Parameter Hook invariants still hold.
- The physical event stream has one `physical_scheduler_stepped` for this Step preceded by all required `physical_child_stepped` events and the exact R1/R2 real Backward/microbatch/sample identities. A Step Anchor is not created by fabricated event JSON alone.
- A Step-N Save candidate can be *reserved* at most once and cannot be upgraded to `checkpoint_saved` merely because the pre-save hook ran or an expected directory name exists.

The Step Anchor is only a **completed physical Step checkpoint candidate**. Distinct C4-B evidence must capture the complete **prepared Network state_dict** (all fixed trainable LoRA weights **and** Alpha buffers) by canonical keys, shape, dtype and full Tensor SHA256, one Tensor at a time. Do not mistake `_adapter_parameter_sha256()` (trained 526 Parameters) for the **serialized 789-tensor** LoRA Adapter inventory, or assume it includes Alpha. Do not retain full-model Tensor clones on CUDA.

C4-B must also capture actual per-Profile Child optimizer state, Scheduler state, RNG fingerprint (`snapshot_rng_fingerprint`: Python/NumPy/Torch CPU/active CUDA) and the real state directory files, with typed hashes and identity. Use existing `fingerprint_state_tree`, `verify_reference_parameter_ownership`, `verify_adapter_checkpoint_binding` as downstream primitives where applicable; reusing them does not convert a Host metadata fixture into physical proof.

### 4.1 Authoritative output paths and ancillary tail

Use `manifest_payload` plus validated production `args.output_dir`/`args.output_name`; no filename substring or basename-only acceptance:

| Phase | Authoritative step artifact | Authoritative state directory | Ancillary |
| --- | --- | --- | --- |
| Fresh Step 1 | `fresh/<output_name>-step00000001.safetensors` | `fresh/<output_name>-step00000001-state` | none |
| Fresh Step 2 | `fresh/<output_name>-step00000002.safetensors` | `fresh/<output_name>-step00000002-state` | none |
| Resume Step 2 | `resume/<output_name>-step00000002.safetensors` | `resume/<output_name>-step00000002-state` | none |
| Training end | `<phase>/<output_name>.safetensors` | `<phase>/<output_name>-state` | allowed `last` save, **never authoritative** |

This mirrors `train_util.STEP_FILE_NAME`, `STEP_STATE_NAME`, `LAST_STATE_NAME` and the existing Host CaseManifest exact-path gate. Any unknown, repeated, cross-phase or linked/reparse path fails closed. A normal final `last` state Save still delegates the real `save_state` **exactly once**, with a separate ancillary classification, but cannot create a Step record.

### 4.2 Phase event sequencing without false admission

The existing C3 `PhysicalStepProbe._record()` requires `_current_step is not None`; it **cannot** record `checkpoint_loaded` before a Resume Batch. Do not fake a Batch or set the private step index just to use `_record`.

C4 must introduce a narrow phase-lifecycle event appender sharing **the same** `probe.events` append-only sequence and `event_seq=len(events)+1`. Event entrypoints are exclusively the Step Adapter's **verified** real callbacks, not public arbitrary JSON injection:

- `checkpoint_loaded`: only Resume, exactly first event, only after the real `Accelerator.load_state` succeeds and actual state restoration is checked; `logical_step=1`, genuine Fresh parent ID from external accepted evidence.
- `checkpoint_saved`: only after an authorized physical Step-N, actual step-named `Accelerator.save_state` returns, full same-Step adapter/state files verify and real `checkpoint_id` is **read from** the production `dts_parameter_policy_progress.json`. Never invent `checkpoint_id`.
- `train_completed`: only after actual inherited `Sd3NetworkTrainer.train` returns, `probe.assert_step_complete()` succeeds, required step saves are accepted, and no C4 unresolved state remains. Existing `C3PhaseEvidence.finalize()` must **continue to throw** until C4-D completes the full proof and Fix D still independently validates Worker process/Ticket ancestry.

`validate_worker_event_stream` is a pure Host **metadata validator** returning `timeline_metadata_valid`, not a CUDA PASS or proof that any save/load actually ran. Add fields/validators narrowly; retain its existing ordering and no-event-after-completion checks.

## 5. Real Save/Load extension seams (reserved for C4-B/C; do not implement in A)

Install observation only **after** `super().register_additional_state_hooks(...)`, the C3 prepared identity/ownership audit, and `probe.install()`. Wrap precisely the **same Accelerator instance's** `save_state` and `load_state` methods; freeze method identity, forward exact original args/kwargs and call original **once**. Restore original per-instance attributes with the established Probe-style `vars(obj)` tracking in every exit path. Avoid overriding `network.save_weights`, modifying `train_network.py` or replacing the production registered pre-save/pre-load hooks.

**Save, C4-B**: classify expected Step state-dir versus normal final `last`; require one full Step Anchor and the expected step-named Adapter file before touching the original Save; capture actual pre-Save Step/Network/Child/Scheduler/RNG snapshot; delegate original `accelerator.save_state` once; **only after successful return**, read the full actual filesystem inventory and production sidecars and compare the on-disk Step-N Adapter/Accelerate `model.safetensors` against the *same-step* prepared Network snapshot, exact typed file digests, child states, scheduler and real `checkpoint_id`. Fail if original Save raises, paths vanish/swap, state files are partial, any snapshot drifts or adapter is stale. A Save pre-hook completing alone is not `checkpoint_saved`.

**Resume, C4-C**: only one exact `--resume` path to Host-accepted Fresh Step-1, independent new process, original `accelerator.load_state` called once. Capture pre-load state before original; after it returns and **before any resumed Backward**, check restored canonical BF16 LoRA weights/Alpha, per-Profile optimizer moments/state, constant Scheduler, validated Progress v1 and SD3 cursor, and applicable RNG fingerprints against the independent Fresh Step-1 Save record. Restore equality applies to saved Step-1, **not** to the unrelated Fresh Step-2. A Worker-owned `expected_parent_checkpoint_id` alone is not sufficient; Fix D binds parent to the Host-accepted Fresh result. Only then emit pre-Batch `checkpoint_loaded`.

**Safety/cleanup**: exceptions preserve the original failure; no retry inside a consumed Ticket or deletion/replacement of failed evidence; never call save/load twice for verification. Probe/Step Adapter/Session wrappers are restored in `finally`; the temporary backend lease is restored regardless. Save/Load verification is observational, not a new Checkpoint format.

## 6. C4-A Tests-only RED plan (new code ONLY in a later explicit RED stage)

First create `tests/test_sd3_full_bf16_worker_c4a_cpu.py` against the **unchanged old Observer/Trainer** and introduce the new exact discovery CI gate, without adding `C4StepAdapter` implementation or weakening existing C3. Use the real `tests.test_sd3_full_bf16_worker_c3_f9_cpu._lora_bundle`, actual LoRAModule patched target, BF16 gradients, Accelerate 1.6.0 and actual Muon+independent AdamW. Real R1 fixture from F8 is separate. All tests must be discovered and run; **zero ERROR / zero SKIP / zero expectedFailure / zero unexpectedSuccess**.

Proposed **12 pinned CPU cases**:

| ID | Contract | Old-code RED / unchanged positive |
| --- | --- | --- |
| A01 | Real R1 one-Child physical Step can expose a correctly bound read-only Step Anchor | positive C3 physical baseline; new Adapter API initially missing |
| A02 | Real R2 first unsynced Microbatch cannot produce an accepted Step Anchor | targeted RED |
| A03 | Real R2 second synced Microbatch with two disjoint Children and one Scheduler creates exactly one eligible Anchor | positive existing physical baseline; new Adapter API initially missing |
| A04 | No completed real Child Step cannot produce an Anchor, even with fabricated metadata | targeted RED |
| A05 | Missing or duplicate physical Child Step must fail before Snapshot | targeted RED |
| A06 | Missing or duplicate physical Scheduler Step must fail | targeted RED |
| A07 | Wrong logical Step, phase or source/Phase ID cannot be re-bound to valid evidence | targeted RED |
| A08 | A Step-N checkpoint candidate cannot be reserved/accepted twice | targeted RED |
| A09 | After an Anchor, a real BF16 Adapter/Optimizer mutation before Save invalidates the candidate | targeted RED |
| A10 | A valid `last` Save classification never emits an extra `checkpoint_saved` event | targeted RED |
| A11 | Resume Step-1 `checkpoint_loaded` needs a separate pre-Batch event path; no fake `begin_batch()` | targeted RED |
| A12 | A failed installation or aborted Step Adapter restores its instance-local method wrappers and never marks `worker_complete` | positive legacy C3 fail-closed + targeted RED for new cleanup |

Note: where the new API does not exist yet, tests should use **explicit, targeted assertion failures** for the absent contract, not accidental ImportError/AttributeError as a purported meaningful RED. Actual *false acceptance* negative tests should use real BF16 behavior, original-method call counters and minimal tamper controls. After green implementation, all **12** must PASS with no skips; do not accept a generic mocked `Step complete` flag as proof.

Second, add `tests/test_sd3_full_bf16_worker_c4a_host.py` with **8 pure Host contract tests** for event sequence and phase fields. Reuse `validate_worker_event_stream` to prove it *only* grants `timeline_metadata_valid`, and reject missing load-first, nonmonotonic sequence, wrong Step, duplicate save, fake R2 second Batch, early Child/Scheduler step, unrelated phase/run IDs and tail-as-authoritative Step. These tests may already PASS on the old Host validator: **do not mislabel existing PASS as RED**. Any added Host validation API must itself have targeted RED first.

**Future separate RED suites, not C4-A**:

- `c4b_cpu`: actual pinned `Accelerator.save_state` success/failure, production-style pre-save hooks, genuine full LoRA Alpha+weight safetensors, complete file digests and same-step snapshots, allowed `last` tail.
- `c4c_cpu`: fresh Accelerator, actual `load_state` restoration of BF16 Adapter, both optimizer Children, Scheduler and RNG; wrong Fresh parent, changed/missing sidecars, no valid pre-Backward restored proof.
- `c4d_host`: combined full phase event stream / Worker incomplete-versus-complete semantics; no self-certified Resume parent; Fix D still authenticates external process and Tickets.

## 7. CI and exact acceptance gates for the next RED implementation

- Add **exactly** `12` discovered tests in the pinned CPU C4-A gate, and `8` pure Host tests in separate Ubuntu/Windows host gates once written. Reject wrong discovery counts, Failures, Errors, Skips, ExpectedFailures or UnexpectedSuccesses.
- Extend paths and invocations **without deleting** F6–F11, legacy C1–C3, C2-A and R0 tests in `.github/workflows/sd3-c2b0-host.yml`; extend both push and PR path filters in `.github/workflows/sd3-c2b0-fix-c-red.yml`. Fix D later also extends the general `anima-qwen3-review.yml` review CI.
- Preserve a source SHA and Actions link for **Tests-only RED** proving failing C4-A-specific assertions, not a broken dependency, import or fixture. Then minimal GREEN; run both Host OS and pinned real CPU, and separately review changes to the production ownership/evidence call chain.
- The C4-A gate can say **Step Adapter Host/CPU GREEN** only. It may not turn `C3PhaseEvidence.finalize()` into a pass or permit `worker_complete` without completed C4-B/C/D file and cross-process proof.
- Do not execute full R1/R2 CUDA qualification until C4, Fix D and independent strict review are complete. Any optional GPU diagnosis before that is clearly nonqualifying.

## 8. Hard stop / ownership register

| Area | Source and authority | Status after C4-A contract |
| --- | --- | --- |
| Actual Forward/Backward/physical Child/Scheduler | C3 `PhysicalStepProbe`, real production SD3 Trainer | F11-D pinned Host/CPU GREEN only |
| Completed-Step Anchor interface | Proposed C4-A Step Adapter + narrow Probe exporter | **Design-only, RED tests not yet run** |
| Same-Step Adapter and Accelerator checkpoint bytes | C4-B, genuine `save_state` post-return file validation | **Not implemented** |
| Fresh Step-1 → new-process Resume restored state | C4-C and Host-accepted parent | **Not implemented** |
| Full-phase event / `C3PhaseEvidence.finalize` | C4-D | **Fail-closed by design** |
| Ticket/claim/Worker process authentication and aggregate | Fix D independent Runner | **Not implemented** |
| Physical RTX 4090 Laptop CUDA R1/R2 and backend release | B1/B2/B3, distinct eligibility PR | **Not qualified, pending** |

**Next authorized action:** implement only the C4-A Tests-only RED suites and exact CI discovery/path pins, run pinned CPU and Host RED, examine every failure, then perform minimal Step Adapter GREEN. Do not bundle C4-B Save and C4-C Resume implementations in the first C4-A code patch.
